// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pir/internal/inner_product_hwy.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "pir/canonical_status_payload_uris.h"
#include "pir/private_information_retrieval.pb.h"

// Guard the following definition of inline functions to make sure they are
// defined only once, since hwy/foreach_target.h will include this .cc file
// multiple times (once for each target archtechture).
#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_INTERNAL_INNER_PRODUCT_HWY_INLINE_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_INTERNAL_INNER_PRODUCT_HWY_INLINE_

namespace distributed_point_functions {
namespace pir_internal {

constexpr int kBitsPerBlock = 8 * sizeof(absl::uint128);

// XOR `value` with `result`, where the output is written back to `result`.
// Note that `result` must point to an array of size >= value.size().
inline void XorStringInPlaceNoHwy(const absl::string_view value, char* result) {
  // If either `value` or `result` are not aligned to 128-bit boundary, then we
  // fallback to the slow path.
  if (reinterpret_cast<size_t>(result) % alignof(absl::uint128) ||
      reinterpret_cast<size_t>(value.data()) % alignof(absl::uint128)) {
    for (int i = 0; i < value.size(); ++i) {
      result[i] ^= value[i];
    }
    return;
  }

  // Main part: XOR each 128-bit block inside the byte arrays `value` and
  // `result`. First we compute the number of 128-bit blocks in `value`, and
  // we use the index i to access them.
  const absl::uint128* const source =
      reinterpret_cast<const absl::uint128*>(value.data());
  absl::uint128* const dest = reinterpret_cast<absl::uint128*>(result);
  const size_t value_size_in_uint128 = value.size() / sizeof(absl::uint128);
  int i = 0;
  for (; i < value_size_in_uint128; ++i) {
    dest[i] ^= source[i];
  }
  // Remaining part: XOR the remaining bytes
  // We now convert i to an index into bytes starting from the last accessed
  // 128-bit location, and loop over the remaining of `value` and `result`.
  i *= sizeof(absl::uint128);
  for (; i < value.size(); ++i) {
    result[i] ^= value[i];
  }
}

}  // namespace pir_internal
}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_INTERNAL_INNER_PRODUCT_HWY_INLINE_

// Highway implementations.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "pir/internal/inner_product_hwy.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// clang-format on

// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace distributed_point_functions::pir_internal {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

#if HWY_TARGET == HWY_SCALAR

absl::StatusOr<std::vector<std::string>> InnerProductHwy(
    absl::Span<const absl::string_view> values,
    absl::Span<const std::vector<BlockType>> selections,
    int64_t max_value_size) {
  return InnerProductNoHwy(values, selections, max_value_size);
}

#else

// Dummy struct to get HWY_ALIGN as a number, for testing if an array of
// BlockType is aligned.
struct HWY_ALIGN AlignedBlockType {
  BlockType _;
};

// Get the alignment of absl::uint128
constexpr size_t kHwyAlignment = alignof(AlignedBlockType);

// For each pointer in `result_ptrs`, computes the XOR of two byte strings at
// `result_ptrs[i] + pos` and `value + pos` that have `remaining` bytes, and
// writes the output to `result_ptrs[i] + pos`. The template parameter `D` must
// be a hwy vector tag type of N lanes such that `result_ptrs[i] + pos` points
// to the last `remaining & (N - 1)` bytes of `result_ptrs[i]`. Since N is
// guaranteed to be a power of two, this condition makes sure that the array
// `result_ptr` can be decomposed into hwy vectors of N/2, N/4, ... lanes.
// Similar condition holds for `value_ptr` and `value`.
template <int kPow2>
inline void XorPartialString(const uint8_t* value_ptr, int64_t pos,
                             const int64_t remaining,
                             absl::Span<uint8_t* const> result_ptrs) {
  const hn::ScalableTag<uint8_t, kPow2 - 1> d_half;
  const int64_t lanes_half = hn::Lanes(d_half);
  // `lanes_half` = N/2 is guaranteed to be a power of two, and the following
  // block computes on a sub-array of length N/2 if remaining >= N/2.
  if (remaining & lanes_half) {
    auto vec = hn::Load(d_half, value_ptr + pos);
    for (int i = 0; i < result_ptrs.size(); ++i) {
      auto vec2 = hn::Xor(vec, hn::Load(d_half, result_ptrs[i] + pos));
      hn::Store(vec2, d_half, result_ptrs[i] + pos);
    }
    pos += lanes_half;
  }

  // Continue to process the remaining bytes using fewer lanes unless we have
  // reached the base case with a single lane.
  if (lanes_half > 1) {
    XorPartialString<kPow2 - 1>(value_ptr, pos, remaining, result_ptrs);
  }
}

// Terminates the compile-time recursion (very low kPow2 will cause compile
// errors).
template <>
inline void XorPartialString<HWY_MIN_POW2>(
    const uint8_t* /*value_ptr*/, int64_t /*pos*/, int64_t /*remaining*/,
    absl::Span<uint8_t* const> /*result_ptrs*/) {}

absl::StatusOr<std::vector<std::string>> InnerProductHwy(
    absl::Span<const absl::string_view> values,
    absl::Span<const std::vector<BlockType>> selections,
    int64_t max_value_size) {
  // Vector type used throughout this function: Largest byte vector available.
  const hn::ScalableTag<uint8_t> d8;
  const int N = hn::Lanes(d8);
  // Do not run the highway version if
  // - the number of bytes in a hwy vector is less than 16, or
  // - the number of bytes in a hwy vector is not a multiple of 16.
  if (ABSL_PREDICT_FALSE(N < 16 || N % 16 != 0)) {
    return InnerProductNoHwy(values, selections, max_value_size);
  }

  // Allocate aligned buffers to hold the intermediate values of inner products.
  std::vector<hwy::AlignedFreeUniquePtr<uint8_t[]>> aligned_results;
  aligned_results.reserve(selections.size());
  for (int i = 0; i < selections.size(); ++i) {
    aligned_results.emplace_back(hwy::AllocateAligned<uint8_t>(max_value_size));
    if (aligned_results[i] == nullptr) {
      return absl::ResourceExhaustedError("memory allocation error");
    }
    std::fill_n(aligned_results[i].get(), max_value_size, '\0');
  }

  // Compute the inner products using highway instructions. Assume all selection
  // vectors have the same size.
  const int64_t selection_vector_size = selections[0].size();
  std::vector<uint8_t*> result_ptrs;
  result_ptrs.reserve(aligned_results.size());
  for (int64_t i = 0; i < selection_vector_size; ++i) {
    // The selection bits are packed in blocks, so we go over them next.
    const int base = i * kBitsPerBlock;
    for (int j = 0; j < kBitsPerBlock; ++j) {
      const int index = base + j;
      if (index >= values.size()) {
        break;  // end of values reached
      }
      const int64_t value_size = values[index].size();
      // Pointer alias for the current value to be read.
      const uint8_t* const value_ptr =
          reinterpret_cast<const uint8_t*>(values[index].data());
      // Do not run the highway version if this value isn't aligned.
      // The alignment is a power of 2, so the following check is equivalent
      // to
      //   (size_t)value_ptr % kHwyAlignment == 0
      const bool is_value_aligned =
          (reinterpret_cast<uintptr_t>(value_ptr) & (kHwyAlignment - 1)) == 0;

      // Go over each selection vector. If the current bit is 0, ignore it. If
      // the value is unaligned, XOR it directly without HWY. Otherwise add the
      // corresponding result pointer to result_ptrs, to be processed with
      // Highway in parallel.
      result_ptrs.clear();
      for (int k = 0; k < selections.size(); ++k) {
        const absl::uint128& selection_block = selections[k][i].value();
        if ((selection_block & (absl::uint128{1} << j)) == 0) {
          // Skip this value since the selection bit at the index (base + j) is
          // 0. Theoretically this may introduce the possibility of timing
          // attacks, i.e. if an attacker can precisely measure the execution
          // time of this function then it may infer the distribution of
          // selection bits. However, this doesn't seem to be a realistic attack
          // scenatio in the context of 2-party DPF-based PIR.
          continue;
        }
        if (!is_value_aligned) {
          // Compute XOR(value, result) using the non-highway version.
          XorStringInPlaceNoHwy(
              values[index], reinterpret_cast<char*>(aligned_results[k].get()));
          continue;
        }
        result_ptrs.push_back(aligned_results[k].get());
      }
      if (result_ptrs.empty()) {
        continue;
      }

      // Compute XOR(value, result) using the Highway instructions.
      // First, one hwy vector at a time.
      int64_t pos = 0;
      for (; pos + N <= value_size; pos += N) {
        auto vec = hn::Load(d8, value_ptr + pos);
        for (int k = 0; k < result_ptrs.size(); ++k) {
          auto vec2 = hn::Xor(vec, hn::Load(d8, result_ptrs[k] + pos));
          hn::Store(vec2, d8, result_ptrs[k] + pos);
        }
      }
      // Remaining bytes that are shorter than a full hwy vector.
      if (pos < value_size) {
        XorPartialString<0>(value_ptr, pos, value_size - pos, result_ptrs);
      }
    }
  }

  // Copy into unaligned strings and return.
  std::vector<std::string> result(aligned_results.size());
  for (int i = 0; i < aligned_results.size(); ++i) {
    result[i] = std::string(reinterpret_cast<char*>(aligned_results[i].get()),
                            max_value_size);
  }
  return result;
}

#endif  // HWY_TARGET == HWY_SCALAR

}  // namespace HWY_NAMESPACE
}  // namespace distributed_point_functions::pir_internal
HWY_AFTER_NAMESPACE();

#if HWY_ONCE || HWY_IDE
namespace distributed_point_functions {
namespace pir_internal {

absl::StatusOr<std::vector<std::string>> InnerProductNoHwy(
    absl::Span<const absl::string_view> values,
    absl::Span<const std::vector<BlockType>> selections,
    int64_t max_value_size) {
  std::vector<std::string> result(selections.size(),
                                  std::string(max_value_size, '\0'));
  const int64_t selection_vector_size = selections[0].size();
  for (int i = 0; i < selection_vector_size; ++i) {
    int base = i * kBitsPerBlock;
    for (int j = 0; j < kBitsPerBlock; ++j) {
      if (base + j >= values.size()) {
        break;
      }
      absl::string_view value = values[base + j];
      for (int k = 0; k < selections.size(); ++k) {
        const BlockType& selection_block = selections[k][i];
        if ((selection_block.value() & (absl::uint128{1} << j)) == 0) {
          // Skip this value since the selection bit at the index (base + j) is
          // 0.
          continue;
        }
        XorStringInPlaceNoHwy(value, &result[k][0]);
      }
    }
  }
  return result;
}

HWY_EXPORT(InnerProductHwy);

absl::StatusOr<std::vector<std::string>> InnerProduct(
    absl::Span<const absl::string_view> values,
    absl::Span<const std::vector<BlockType>> selections,
    int64_t max_value_size) {
  // Do error handling once here, and assume inputs are valid in the actual
  // implementations.
  if (selections.empty()) {
    return std::vector<std::string>();
  }
  int first_selection_vector_size = selections[0].size();
  for (int i = 0; i < selections.size(); ++i) {
    if ((selections[i].size() * kBitsPerBlock) < values.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "`selections[", i, "]` contains insufficient number of bits: ",
          selections[i].size() * kBitsPerBlock, ", expected: ", values.size()));
    }
    if (selections[i].size() != first_selection_vector_size) {
      return absl::InvalidArgumentError(absl::StrCat(
          "`selections[", i,
          "].size()` does not match `selections[0].size()`: actual",
          selections[i].size(), ", expected ", first_selection_vector_size));
    }
    if (max_value_size <= 0) {
      absl::Status status =
          absl::InvalidArgumentError("`max_value_size` must be positive");
      CanonicalPirError payload;
      payload.set_code(CanonicalPirError::MAX_VALUE_SIZE_IS_ZERO);
      status.SetPayload(kPirInternalErrorUri,
                        std::move(payload).SerializeAsCord());
      return status;
    }
  }
  for (int i = 0; i < values.size(); ++i) {
    if (values[i].size() > max_value_size) {
      return absl::InvalidArgumentError(
          absl::StrCat("`values[", i, "]` is larger than `max_value_size`"));
    }
  }
  return HWY_DYNAMIC_DISPATCH(InnerProductHwy)(values, selections,
                                               max_value_size);
}

}  // namespace pir_internal
}  // namespace distributed_point_functions
#endif  // HWY_ONCE || HWY_IDE
