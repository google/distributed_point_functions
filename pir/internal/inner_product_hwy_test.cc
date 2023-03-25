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

#include <string>
#include <utility>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "dpf/internal/status_matchers.h"
#include "gtest/gtest.h"
#include "hwy/aligned_allocator.h"
#include "hwy/detect_targets.h"

// clang-format off
#define HWY_IS_TEST 1
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "pir/internal/inner_product_hwy_test.cc"  // NOLINT
#include "hwy/foreach_target.h"
// clang-format on

#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace distributed_point_functions {
namespace pir_internal {
namespace HWY_NAMESPACE {

using distributed_point_functions::dpf_internal::StatusIs;
using testing::HasSubstr;

// The number of bits packed in each block.
constexpr int kBitsPerBlock = 8 * sizeof(BlockType);
// The byte size of each aligned entry.
constexpr int kMaxValueSize = 128;
// The maximal number of values in inner product computation.
constexpr int kMaxNumValues = 3 * kBitsPerBlock;

// Hold the input data maximally aligned to use hwy instructions.
struct AlignedEntry {
  hwy::AlignedFreeUniquePtr<uint8_t[]> data;
};

class TestInnerProduct {
 public:
  void SetUp() {
    // Sizes (in bytes) of the values. We use a mix of multiples of 16 and also
    // values slightly smaller/larger; all sizes are smaller than kMaxValueSize.
    constexpr int kNumSizes = 12;
    int sizes[kNumSizes] = {0, 1, 3, 7, 16, 17, 31, 32, 63, 64, 80, 81};
    for (int i = 0; i < kMaxNumValues; ++i) {
      value_sizes_.push_back(sizes[i % kNumSizes]);
    }

    // Generate some random bytes, and store them as blocks aligned to 128-bit.
    absl::BitGen gen;
    entries_.reserve(value_sizes_.size());
    for (int i = 0; i < value_sizes_.size(); ++i) {
      AlignedEntry entry;
      entry.data = hwy::AllocateAligned<uint8_t>(kMaxValueSize);
      // Fill in `entry` with random bytes. Since kMaxEntriSize is larger than
      // all value sizes, we are safe to let unaligned values point to the
      // second byte of each entry.
      for (int j = 0; j < kMaxValueSize; ++j) {
        entry.data[j] = absl::Uniform<uint8_t>(gen);
      }
      entries_.push_back(std::move(entry));
    }
    // Create aligned values.
    aligned_values_.reserve(value_sizes_.size());
    for (int i = 0; i < value_sizes_.size(); ++i) {
      aligned_values_.push_back(
          {reinterpret_cast<char*>(entries_[i].data.get()), value_sizes_[i]});
    }
    // Create unaligned values, by shifting one byte from aligned entries.
    unaligned_values_.reserve(value_sizes_.size());
    for (int i = 0; i < value_sizes_.size(); ++i) {
      unaligned_values_.push_back(
          {reinterpret_cast<char*>(&entries_[i].data[1]), value_sizes_[i]});
    }
  }

  void InnerProductAlignedFailsWithEmptySelection() {
    std::vector<bool> selections;
    std::vector<BlockType> packed_selections =
        this->PackSelectionBits(selections);
    ASSERT_EQ(packed_selections.size(), 0);
    EXPECT_THAT(
        InnerProduct(this->aligned_values_, {packed_selections}, kMaxValueSize),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("insufficient number of bits")));
  }

  void InnerProductUnalignedFailsWithEmptySelection() {
    std::vector<bool> selections;
    std::vector<BlockType> packed_selections =
        this->PackSelectionBits(selections);
    ASSERT_EQ(packed_selections.size(), 0);
    EXPECT_THAT(InnerProduct(this->unaligned_values_, {packed_selections},
                             kMaxValueSize),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("insufficient number of bits")));
  }

  void InnerProductAlignedFailsWithZeroMaxValueSize() {
    std::vector<bool> selections(this->aligned_values_.size(), true);
    std::vector<BlockType> packed_selections =
        this->PackSelectionBits(selections);
    EXPECT_THAT(InnerProduct(this->aligned_values_, {packed_selections}, 0),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("`max_value_size` must be positive")));
  }

  void InnerProductUnalignedFailsWithZeroMaxValueSize() {
    std::vector<bool> selections(this->unaligned_values_.size(), true);
    std::vector<BlockType> packed_selections =
        this->PackSelectionBits(selections);
    EXPECT_THAT(InnerProduct(this->unaligned_values_, {packed_selections}, 0),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("`max_value_size` must be positive")));
  }

  // `InnerProduct` should returns an error if there is a value of size larger
  // than the given `max_value_size` argument.
  void InnerProductAlignedFailsWithOutOfBoundValueSize() {
    std::vector<bool> selections(this->aligned_values_.size(), true);
    std::vector<BlockType> packed_selections =
        this->PackSelectionBits(selections);
    EXPECT_THAT(InnerProduct(this->aligned_values_, {packed_selections}, 1),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("larger than `max_value_size`")));
  }

  // `InnerProduct` should returns an error if there is a value of size larger
  // than the given `max_value_size` argument.
  void InnerProductUnalignedFailsWithOutOfBoundValueSize() {
    std::vector<bool> selections(this->unaligned_values_.size(), true);
    std::vector<BlockType> packed_selections =
        this->PackSelectionBits(selections);
    EXPECT_THAT(InnerProduct(this->unaligned_values_, {packed_selections}, 1),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("larger than `max_value_size`")));
  }

  void InnerProductAlignedFailsWithInsufficientSelectionBits() {
    // The number of blocks needed to store a selection vector.
    int num_blocks = this->NumberOfBlocksFor(this->aligned_values_.size());
    ASSERT_GE(num_blocks, 2);  // must have at least two blocks
    // A selection vector packed into fewer blocks.
    num_blocks -= 1;
    std::vector<bool> selections(num_blocks * kBitsPerBlock, false);
    std::vector<BlockType> packed_selections = PackSelectionBits(selections);
    ASSERT_EQ(packed_selections.size(), num_blocks);
    EXPECT_THAT(
        InnerProduct(this->aligned_values_, {packed_selections}, kMaxValueSize),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("insufficient number of bits")));
  }

  void InnerProductUnalignedFailsWithInsufficientSelectionBits() {
    // The number of blocks needed to store a selection vector.
    int num_blocks = this->NumberOfBlocksFor(this->unaligned_values_.size());
    ASSERT_GE(num_blocks, 2);  // must have at least two blocks
    // A selection vector packed into fewer blocks.
    num_blocks -= 1;
    std::vector<bool> selections(num_blocks * kBitsPerBlock, false);
    std::vector<BlockType> packed_selections = PackSelectionBits(selections);
    ASSERT_EQ(packed_selections.size(), num_blocks);
    EXPECT_THAT(InnerProduct(this->unaligned_values_, {packed_selections},
                             kMaxValueSize),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("insufficient number of bits")));
  }

  // `InnerProduct` should return a string of 0's when selection bits are unset,
  void InnerProductOfAlignedValueAndZeroSelectionBits() {
    const std::string target(kMaxValueSize, '\0');
    DPF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::string> result0,
        InnerProduct({this->aligned_values_[0]},
                     {this->PackSelectionBits({false})}, kMaxValueSize));
    ASSERT_EQ(result0.size(), 1);
    ASSERT_EQ(result0[0], target);

    std::vector<bool> selections1(this->aligned_values_.size(), false);
    DPF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::string> result1,
        InnerProduct(this->aligned_values_,
                     {this->PackSelectionBits(selections1)}, kMaxValueSize));
    ASSERT_EQ(result1.size(), 1);
    ASSERT_EQ(result1[0], target);
  }

  // Same as "InnerProductOfAlignedValueAndZeroSelectionBits" but with unaligned
  // values.
  void InnerProductOfUnalignedValueAndZeroSelectionBits() {
    const std::string target(kMaxValueSize, '\0');
    DPF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::string> result0,
        InnerProduct({this->unaligned_values_[0]},
                     {this->PackSelectionBits({false})}, kMaxValueSize));
    ASSERT_EQ(result0.size(), 1);
    ASSERT_EQ(result0[0], target);

    std::vector<bool> selections1(this->aligned_values_.size(), false);
    DPF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::string> result1,
        InnerProduct(this->unaligned_values_,
                     {this->PackSelectionBits(selections1)}, kMaxValueSize));
    ASSERT_EQ(result1.size(), 1);
    ASSERT_EQ(result1[0], target);
  }

  // Tests `InnerProduct` on k aligned values where k is smaller than a full
  // block
  void InnerProductOfAlignedValuesAndPartialBlockSelectionVector() {
    const int num_values = kBitsPerBlock - 1;
    absl::Span<const absl::string_view> partial_aligned_values =
        absl::MakeSpan(this->aligned_values_).subspan(0, num_values);
    std::vector<bool> selections(num_values, true);
    DPF_ASSERT_OK_AND_ASSIGN(
        std::string target,
        this->InnerProductForTest(partial_aligned_values, selections));
    DPF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::string> result,
        InnerProduct(partial_aligned_values,
                     {this->PackSelectionBits(selections)}, kMaxValueSize));
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0], target);
  }

  // Same as "InnerProductOfAlignedValuesAndPartialBlockSelectionVector" but
  // with unaligned values.
  void InnerProductOfUnalignedValuesAndPartialBlockSelectionVector() {
    const int num_values = kBitsPerBlock - 1;
    absl::Span<const absl::string_view> partial_unaligned_values =
        absl::MakeSpan(this->unaligned_values_).subspan(0, num_values);
    std::vector<bool> selections(num_values, true);
    DPF_ASSERT_OK_AND_ASSIGN(
        std::string target,
        this->InnerProductForTest(partial_unaligned_values, selections));
    DPF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::string> result,
        InnerProduct(partial_unaligned_values,
                     {this->PackSelectionBits(selections)}, kMaxValueSize));
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0], target);
  }

  // Tests `InnerProduct` on aligned values with a selection vector of multiple
  // blocks and random bits.
  void InnerProductOfAlignedValuesAndLongSelectionVector() {
    std::vector<bool> selections;
    absl::BitGen bitgen;
    uint64_t bits = absl::Uniform<uint64_t>(bitgen);
    for (int i = 0; i < this->aligned_values_.size(); ++i) {
      uint64_t mask = uint64_t{1} << (i % 64);
      selections.push_back(bits & mask);
    }
    DPF_ASSERT_OK_AND_ASSIGN(
        std::string target,
        this->InnerProductForTest(this->aligned_values_, selections));
    DPF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::string> result,
        InnerProduct(this->aligned_values_,
                     {this->PackSelectionBits(selections)}, kMaxValueSize));
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0], target);
  }

  // Same as "InnerProductOfAlignedValuesAndLongSelectionVector" but with
  // unaligned values.
  void InnerProductOfUnalignedValuesAndLongSelectionVector() {
    std::vector<bool> selections;
    absl::BitGen bitgen;
    uint64_t bits = absl::Uniform<uint64_t>(bitgen);
    for (int i = 0; i < this->aligned_values_.size(); ++i) {
      uint64_t mask = uint64_t{1} << (i % 64);
      selections.push_back(bits & mask);
    }
    DPF_ASSERT_OK_AND_ASSIGN(
        std::string target,
        this->InnerProductForTest(this->unaligned_values_, selections));
    DPF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::string> result,
        InnerProduct(this->unaligned_values_,
                     {this->PackSelectionBits(selections)}, kMaxValueSize));
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0], target);
  }

  // Tests `InnerProduct` on aligned values with a selection vector of all 1s.
  void InnerProductOfAllAlignedValues() {
    std::vector<bool> selections(this->value_sizes_.size(), true);
    DPF_ASSERT_OK_AND_ASSIGN(
        std::string target,
        this->InnerProductForTest(this->aligned_values_, selections));
    DPF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::string> result,
        InnerProduct(this->aligned_values_,
                     {this->PackSelectionBits(selections)}, kMaxValueSize));
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0], target);
  }

  // Same as "InnerProductOfAllAlignedValues" but with unaligned values.
  void InnerProductOfAllUnalignedValues() {
    std::vector<bool> selections(this->value_sizes_.size(), true);
    DPF_ASSERT_OK_AND_ASSIGN(
        std::string target,
        this->InnerProductForTest(this->unaligned_values_, selections));
    DPF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::string> result,
        InnerProduct(this->unaligned_values_,
                     {this->PackSelectionBits(selections)}, kMaxValueSize));
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0], target);
  }

 protected:
  // Returns the number of blocks needed to pack `n` bits.
  static int NumberOfBlocksFor(int n) {
    if (n <= kBitsPerBlock) {
      return 1;
    }
    return (n + kBitsPerBlock - 1) / kBitsPerBlock;
  }

  // Returns a vector of packed bit blocks of the bits in `selections`.
  static std::vector<BlockType> PackSelectionBits(
      const std::vector<bool>& selections) {
    int num_blocks = (selections.size() + kBitsPerBlock - 1) / kBitsPerBlock;
    std::vector<BlockType> blocks;
    blocks.reserve(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
      absl::uint128 block{0};
      int base = i * kBitsPerBlock;
      for (int j = 0; j < kBitsPerBlock; ++j) {
        if (base + j >= selections.size()) {
          break;  // reached the last partial block
        }
        if (selections[base + j]) {
          block |= absl::uint128{1} << j;
        }
      }
      blocks.push_back(BlockType(block));
    }
    return blocks;
  }

  // Returns the inner product between `values` and `selections` for testing
  // the implementation with packed selection bits.
  static absl::StatusOr<std::string> InnerProductForTest(
      absl::Span<const absl::string_view> values,
      const std::vector<bool>& selections) {
    if (values.size() != selections.size()) {
      return absl::InvalidArgumentError("operands dimension mismatch");
    }

    std::string result(kMaxValueSize, 0);
    for (int i = 0; i < values.size(); ++i) {
      if (!selections[i]) {
        continue;
      }
      for (int j = 0; j < values[i].size(); ++j) {
        result[j] ^= values[i][j];
      }
    }
    return result;
  }

  std::vector<size_t> value_sizes_;
  std::vector<AlignedEntry> entries_;
  std::vector<absl::string_view> aligned_values_;
  std::vector<absl::string_view> unaligned_values_;
};

void TestAll() {
  TestInnerProduct test;
  test.SetUp();
  test.InnerProductAlignedFailsWithEmptySelection();
  test.InnerProductUnalignedFailsWithEmptySelection();
  test.InnerProductAlignedFailsWithZeroMaxValueSize();
  test.InnerProductUnalignedFailsWithZeroMaxValueSize();
  test.InnerProductAlignedFailsWithOutOfBoundValueSize();
  test.InnerProductUnalignedFailsWithOutOfBoundValueSize();
  test.InnerProductAlignedFailsWithInsufficientSelectionBits();
  test.InnerProductUnalignedFailsWithInsufficientSelectionBits();
  test.InnerProductOfAlignedValueAndZeroSelectionBits();
  test.InnerProductOfUnalignedValueAndZeroSelectionBits();
  test.InnerProductOfAlignedValuesAndPartialBlockSelectionVector();
  test.InnerProductOfUnalignedValuesAndPartialBlockSelectionVector();
  test.InnerProductOfAlignedValuesAndLongSelectionVector();
  test.InnerProductOfUnalignedValuesAndLongSelectionVector();
  test.InnerProductOfAllAlignedValues();
  test.InnerProductOfAllUnalignedValues();
}

}  // namespace HWY_NAMESPACE
}  // namespace pir_internal
}  // namespace distributed_point_functions
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace distributed_point_functions {
namespace pir_internal {
HWY_BEFORE_TEST(InnerProductHwyTest);
HWY_EXPORT_AND_TEST_P(InnerProductHwyTest, TestAll);
}  // namespace pir_internal
}  // namespace distributed_point_functions

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif
