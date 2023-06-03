/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DISTRIBUTED_POINT_FUNCTIONS_DCF_DISTRIBUTED_COMPARISON_FUNCTION_H_
#define DISTRIBUTED_POINT_FUNCTIONS_DCF_DISTRIBUTED_COMPARISON_FUNCTION_H_

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/meta/type_traits.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dcf/distributed_comparison_function.pb.h"
#include "dpf/distributed_point_function.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/internal/maybe_deref_span.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

namespace distributed_point_functions {

class DistributedComparisonFunction {
 public:
  static absl::StatusOr<std::unique_ptr<DistributedComparisonFunction>> Create(
      const DcfParameters& parameters);

  // Creates keys for a DCF that evaluates to shares of `beta` on any input x <
  // `alpha`, and shares of 0 otherwise.
  //
  // Returns INVALID_ARGUMENT if `alpha` or `beta` do not match the
  // DcfParameters passed at construction.
  //
  // Overload for explicit Value proto.
  absl::StatusOr<std::pair<DcfKey, DcfKey>> GenerateKeys(absl::uint128 alpha,
                                                         const Value& beta);

  // Template for automatic conversion to Value proto. Disabled if the argument
  // is convertible to `absl::uint128` or `Value` to make overloading
  // unambiguous.
  template <typename T, typename = absl::enable_if_t<
                            !std::is_convertible<T, Value>::value &&
                            is_supported_type_v<T>>>
  absl::StatusOr<std::pair<DcfKey, DcfKey>> GenerateKeys(absl::uint128 alpha,
                                                         const T& beta) {
    absl::StatusOr<Value> value = dpf_->ToValue(beta);
    if (!value.ok()) {
      return value.status();
    }
    return GenerateKeys(alpha, *value);
  }

  // Evaluates a DcfKey at the given point `x`.
  //
  // Returns INVALID_ARGUMENT if `key` or `x` do not match the parameters passed
  // at construction.
  template <typename T>
  inline absl::StatusOr<T> Evaluate(const DcfKey& key, absl::uint128 x) {
    T result{};
    absl::Status status = BatchEvaluate<T>(absl::MakeConstSpan(&key, 1),
                                           absl::MakeConstSpan(&x, 1),
                                           absl::MakeSpan(&result, 1));
    if (!status.ok()) {
      return status;
    }
    return result;
  }

  // Evaluates `keys[i]` at `evaluation_points[i]` for all i. `keys` can be any
  // container convertible to absl::Span<const DcfKey> or absl::Span<const
  // DcfKey* const>.
  //
  // Returns INVALID_ARGUMENT if `keys` and `evaluation_points` have different
  // sizes, or if any element of `keys` is invalid or any element of
  // `evaluation_points` is out of scope.
  template <typename T>
  inline absl::StatusOr<std::vector<T>> BatchEvaluate(
      dpf_internal::MaybeDerefSpan<const DcfKey> keys,
      absl::Span<const absl::uint128> evaluation_points) {
    std::vector<T> result(keys.size());
    absl::Status status =
        BatchEvaluate<T>(keys, evaluation_points, absl::MakeSpan(result));
    if (!status.ok()) {
      return status;
    }
    return result;
  }

  // As the two-argument version above, but writes to `output` instead of
  // allocating a std::vector.
  //
  // Returns INVALID_ARGUMENT `output`, `keys`, and `evaluation_points` don't
  // have the same size.
  template <typename T>
  absl::Status BatchEvaluate(dpf_internal::MaybeDerefSpan<const DcfKey> keys,
                             absl::Span<const absl::uint128> evaluation_points,
                             absl::Span<T> output);

  // DistributedComparisonFunction is neither copyable nor movable.
  DistributedComparisonFunction(const DistributedComparisonFunction&) = delete;
  DistributedComparisonFunction& operator=(
      const DistributedComparisonFunction&) = delete;

 private:
  DistributedComparisonFunction(DcfParameters parameters,
                                std::unique_ptr<DistributedPointFunction> dpf);

  const DcfParameters parameters_;
  const std::unique_ptr<DistributedPointFunction> dpf_;
};

// Implementation details.

template <typename T>
absl::Status DistributedComparisonFunction::BatchEvaluate(
    dpf_internal::MaybeDerefSpan<const DcfKey> keys,
    absl::Span<const absl::uint128> evaluation_points, absl::Span<T> output) {
  if (keys.size() != evaluation_points.size()) {
    // Different error message for the two-argument version.
    return absl::InvalidArgumentError(
        "`keys` and `evaluation_points` must have the same size");
  }
  if (output.size() != keys.size()) {
    return absl::InvalidArgumentError(
        "`keys`, `evaluation_points`, and `output` must have the same size");
  }

  const int log_domain_size = parameters_.parameters().log_domain_size();
  const int num_keys = keys.size();
  int hierarchy_level = 0;
  absl::Status status = absl::OkStatus();
  auto accumulator = [&status, &hierarchy_level, num_keys, log_domain_size,
                      output, evaluation_points](absl::Span<const T> values) {
    if (values.size() != num_keys) {
      status = absl::InternalError(
          "The size of the span passed to `accumulator` does not match the "
          "number of batched keys");
      return false;
    }
    const absl::uint128 mask =
        (absl::uint128{1} << (log_domain_size - hierarchy_level - 1));
    for (int i = 0; i < num_keys; ++i) {
      const auto current_bit =
          static_cast<int>((evaluation_points[i] & mask) != 0);
      if (current_bit == 0) {
        output[i] += values[i];
      }
    }
    ++hierarchy_level;
    return true;
  };

  // Create vector of pointers to DPF keys.
  std::vector<const DpfKey*> dpf_keys(num_keys);
  for (int i = 0; i < num_keys; ++i) {
    dpf_keys[i] = &(keys[i].key());
  }

  // We don't evaluate on the least-significant bit, since there the output only
  // depends on alpha. See Algorith m 7 in https://eprint.iacr.org/2022/866.pdf.
  auto prefixes = hwy::AllocateAligned<absl::uint128>(num_keys);
  if (prefixes == nullptr) {
    return absl::ResourceExhaustedError("Memory allocation error");
  }
  for (int i = 0; i < num_keys; ++i) {
    prefixes[i] = evaluation_points[i] >> 1;
  }

  std::fill(output.begin(), output.end(), T{});
  absl::Status status2 = dpf_->EvaluateAndApply<T>(
      dpf_keys, absl::MakeConstSpan(prefixes.get(), num_keys),
      std::move(accumulator));
  if (!status2.ok()) return status2;
  return status;
}

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DCF_DISTRIBUTED_COMPARISON_FUNCTION_H_
