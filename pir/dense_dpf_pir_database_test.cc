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

#include "pir/dense_dpf_pir_database.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "dpf/internal/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "pir/testing/mock_pir_database.h"
#include "pir/testing/pir_selection_bits.h"

namespace distributed_point_functions {
namespace {

using dpf_internal::IsOkAndHolds;
using dpf_internal::StatusIs;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::StartsWith;
using ::testing::WhenDynamicCastTo;
using BlockType = DenseDpfPirDatabase::BlockType;
using InterfacePtr = std::unique_ptr<DenseDpfPirDatabase::Interface>;
using DatabasePtr = std::unique_ptr<DenseDpfPirDatabase>;

constexpr int kNumValues = 300;
constexpr int kMaxValueBytes = 80;
constexpr int kLevelsToPack = 7;
constexpr int kBitsPerBlock = 8 * sizeof(BlockType);

// Use default setting to build a database.
TEST(DenseDpfPirDatabaseBuilder, CreateWithAllDefaultArguments) {
  DPF_ASSERT_OK_AND_ASSIGN(InterfacePtr database,
                           DenseDpfPirDatabase::Builder().Build());
  DenseDpfPirDatabase* dense_database =
      dynamic_cast<DenseDpfPirDatabase*>(database.get());
  ASSERT_NE(dense_database, nullptr);
  EXPECT_EQ(dense_database->max_value_size_in_bytes(), 0);
  EXPECT_EQ(dense_database->content().size(), 0);
}

// This test fixture is for exercising the Append() member function.
class DenseDpfPirDatabaseBuilderInsertTest : public ::testing::Test {
 protected:
  void SetUp() override {
    append_value_sizes_test_cases_ = {
        {0},        // an empty string
        {0, 0},     // two consecutive empty strings
        {0, 1},     // 1 byte after an empty string
        {1},        // 1 byte
        {1, 0},     // 1 byte and then an empty string
        {1, 0, 1},  // an empty string in the middle
        {15, 1},    // first string smaller than one block
        {15, 16},   // same as above but second string fits in a block
        {15, 32},   // same as above but second string fits more than a block
        {16},       // first string fits in a block
        {16, 0},    // an empty after a full-block size string
        {16, 1},    // a small string after a full-block size string
        {16, 32},   // a large string after a full-block size string
        {32},       // a large string that fits in 2 blocks
        {32, 32},   // two large strings
    };

    rebuild_content_view_test_cases_ = {
        15,    // smaller than one block
        16,    // one block
        17,    // one full and one partial block
        32,    // two blocks
        8192,  // many blocks
    };
  }

  // Matcher that checks if the tested database has contents equal to `values`.
  ::testing::Matcher<InterfacePtr> IsContentEqual(
      const std::vector<std::string>& values) {
    std::vector<absl::string_view> value_views(values.begin(), values.end());
    return Property(
        &InterfacePtr::get,
        WhenDynamicCastTo<DatabasePtr::pointer>(Pointee(Property(
            &DenseDpfPirDatabase::content, ElementsAreArray(value_views)))));
  }

  // Test cases for invoking Append(), where each test case is defined by a
  // sequence of sizes of random values to be inserted into the database.
  std::vector<std::vector<int>> append_value_sizes_test_cases_;

  // Test cases for checking the `content` view can be updated after inserting
  // a value that causes buffer reallocation. A test case is defined by a value
  // size s, where the test appends values of size s, 2s, 3s, etc.
  std::vector<int> rebuild_content_view_test_cases_;
};

TEST_F(DenseDpfPirDatabaseBuilderInsertTest, BuildFailsIfAlreadyBuilt) {
  DenseDpfPirDatabase::Builder builder;
  DPF_EXPECT_OK(builder.Build());
  EXPECT_THAT(builder.Build(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                        HasSubstr("already built")));
  EXPECT_THAT(builder.Clone()->Build(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("already built")));
}

// Values can be correctly inserted to the database.
TEST_F(DenseDpfPirDatabaseBuilderInsertTest,
       AppendWithEmptyDatabaseBufferSucceeds) {
  for (auto const& value_sizes : this->append_value_sizes_test_cases_) {
    DenseDpfPirDatabase::Builder builder;

    // Insert random values to the database.
    DPF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> values,
                             pir_testing::GenerateRandomStrings(value_sizes));
    for (const std::string& value : values) {
      builder.Insert(value);
    }
    EXPECT_THAT(builder.Build(), IsOkAndHolds(IsContentEqual(values)));
  }
}

// Checks that the content view of the database, accessed via `content()`, is
// correct and contains all the inserted values after inserting values of
// different sizes.
TEST_F(DenseDpfPirDatabaseBuilderInsertTest, ContentViewRebuiltSucceeds) {
  for (auto value_size : this->rebuild_content_view_test_cases_) {
    DenseDpfPirDatabase::Builder builder;
    // Insert a random value to the database.
    DPF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> values,
                             pir_testing::GenerateRandomStrings({value_size}));
    builder.Insert(values.back());
    EXPECT_THAT(builder.Clone()->Build(), IsOkAndHolds(IsContentEqual(values)));

    // Insert a value of 2x the size of the first value.
    DPF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::string> additional_values1,
        pir_testing::GenerateRandomStrings({value_size * 2}));
    values.push_back(additional_values1[0]);
    builder.Insert(values.back());
    EXPECT_THAT(builder.Clone()->Build(), IsOkAndHolds(IsContentEqual(values)));
    // Insert another value of 2x the size of the first value, which should
    // cause buffer reallocation.
    DPF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::string> additional_values2,
        pir_testing::GenerateRandomStrings({value_size * 2}));
    values.push_back(additional_values2[0]);
    builder.Insert(values.back());
    EXPECT_THAT(builder.Clone()->Build(), IsOkAndHolds(IsContentEqual(values)));

    // Insert a value of 5x the size of the first value, ie larger than
    // the size of the entire database, to trigger another buffer reallocation.
    DPF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::string> additional_values3,
        pir_testing::GenerateRandomStrings({value_size * 5}));
    values.push_back(additional_values3[0]);
    builder.Insert(values.back());
    EXPECT_THAT(builder.Clone()->Build(), IsOkAndHolds(IsContentEqual(values)));
  }
}

TEST_F(DenseDpfPirDatabaseBuilderInsertTest,
       AppendLongerValueThanMaxValueSize) {
  DenseDpfPirDatabase::Builder builder;
  std::vector<std::string> values;
  // Append a value whose length is smaller than the default max value size.
  int small_value_size = kMaxValueBytes - 1;
  DPF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::string> values1,
      pir_testing::GenerateRandomStrings({small_value_size}));
  builder.Insert(values1.back());
  values.push_back(values1.back());
  DPF_ASSERT_OK_AND_ASSIGN(InterfacePtr database1, builder.Clone()->Build());
  EXPECT_THAT(database1, IsContentEqual(values));
  EXPECT_EQ(dynamic_cast<DatabasePtr::pointer>(database1.get())
                ->max_value_size_in_bytes(),
            small_value_size);

  // Append a value whose length is larger than current max value size.
  int large_value_size = kMaxValueBytes * 2;
  DPF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::string> values2,
      pir_testing::GenerateRandomStrings({large_value_size}));
  builder.Insert(values2.back());
  values.push_back(values2.back());
  DPF_ASSERT_OK_AND_ASSIGN(InterfacePtr database2, builder.Clone()->Build());
  EXPECT_THAT(database2, IsContentEqual(values));
  EXPECT_EQ(dynamic_cast<DatabasePtr::pointer>(database2.get())
                ->max_value_size_in_bytes(),
            large_value_size);
}

// This test fixture is for testing `InnerProductWith` member function.
class DenseDpfPirDatabaseInnerProductTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Insert random values into the database.
    absl::BitGen bitgen;
    std::vector<int> value_sizes;
    for (int i = 0; i < kNumValues; ++i) {
      size_t value_size =
          absl::Uniform(bitgen, 0u, static_cast<size_t>(kMaxValueBytes));
      value_sizes.push_back(value_size);
    }
    DPF_ASSERT_OK_AND_ASSIGN(values_,
                             pir_testing::GenerateRandomStrings(value_sizes));
    for (const std::string& value : values_) {
      builder_.Insert(value);
    }
    DPF_ASSERT_OK_AND_ASSIGN(database_, builder_.Clone()->Build());
  }

  DenseDpfPirDatabase::Builder builder_;
  InterfacePtr database_;
  std::vector<std::string> values_;
};

// `InnerProductWith` fails if the selection vector is empty.
TEST_F(DenseDpfPirDatabaseInnerProductTest,
       InnerProductFailsWithEmptySelection) {
  std::vector<bool> selections;
  std::vector<BlockType> packed_selections =
      pir_testing::PackSelectionBits<BlockType>(selections);
  ASSERT_EQ(packed_selections.size(), 0);
  EXPECT_THAT(this->database_->InnerProductWith({packed_selections}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("insufficient number of bits")));
}

// `InnerProductWith` fails if the selection vector doesn't contain enough
// bits.
TEST_F(DenseDpfPirDatabaseInnerProductTest,
       InnerProductFailsWithInsufficientSelectionBits) {
  // A selection vector packed into a smaller number of blocks.
  int num_blocks = (kNumValues >> kLevelsToPack) - 1;
  std::vector<bool> selections(num_blocks << kLevelsToPack, false);
  std::vector<BlockType> packed_selections =
      pir_testing::PackSelectionBits<BlockType>(selections);
  ASSERT_EQ(packed_selections.size(), num_blocks);
  EXPECT_THAT(this->database_->InnerProductWith({packed_selections}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("insufficient number of bits")));
}

TEST_F(DenseDpfPirDatabaseInnerProductTest,
       InnerProductWithoutSelectionVectorSucceeds) {
  EXPECT_THAT(this->database_->InnerProductWith({}), IsOkAndHolds(IsEmpty()));
}

// `InnerProductWith` should return a correct inner product.
TEST_F(DenseDpfPirDatabaseInnerProductTest, InnerProductIsCorrect) {
  // Sample random selection bits
  absl::BitGen bitgen;
  std::vector<bool> selections(kNumValues, false);
  for (int i = 0; i < kNumValues; ++i) {
    selections[i] = absl::Uniform(bitgen, 0u, 2u);
  }
  std::vector<BlockType> packed_selections =
      pir_testing::PackSelectionBits<BlockType>(selections);

  // Compute the inner product.
  DPF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::string> result,
      this->database_->InnerProductWith({packed_selections}));
  // Compute the inner product using test implementation.
  DPF_ASSERT_OK_AND_ASSIGN(
      std::string target,
      pir_testing::InnerProductWithUnpacked(selections, values_));
  ASSERT_EQ(result.size(), 1);
  EXPECT_GE(result[0].size(), target.size());
  EXPECT_EQ(result[0],  // Result might be padded with zero bytes.
            target + std::string(result[0].size() - target.size(), '\0'));
}

TEST_F(DenseDpfPirDatabaseInnerProductTest,
       InnerProductWithTwoSelectionVectorsIsCorrect) {
  // Sample random selection bits.
  absl::BitGen bitgen;
  std::vector<bool> selections1(kNumValues, false);
  std::vector<bool> selections2 = selections1;
  for (int i = 0; i < kNumValues; ++i) {
    selections1[i] = absl::Uniform(bitgen, 0u, 2u);
    selections2[i] = absl::Uniform(bitgen, 0u, 2u);
  }
  std::vector<BlockType> packed_selections1 =
      pir_testing::PackSelectionBits<BlockType>(selections1);
  std::vector<BlockType> packed_selections2 =
      pir_testing::PackSelectionBits<BlockType>(selections2);

  // Compute the inner product using test implementation.
  DPF_ASSERT_OK_AND_ASSIGN(
      std::string target1,
      pir_testing::InnerProductWithUnpacked(selections1, values_));
  DPF_ASSERT_OK_AND_ASSIGN(
      std::string target2,
      pir_testing::InnerProductWithUnpacked(selections2, values_));

  // Compute the inner product and check that it is correct.
  EXPECT_THAT(
      this->database_->InnerProductWith(
          {packed_selections1, packed_selections2}),
      IsOkAndHolds(ElementsAre(StartsWith(target1), StartsWith(target2))));
}

TEST_F(DenseDpfPirDatabaseInnerProductTest,
       InnerProductWithClonedBuilderIsTheSame) {
  auto selections =
      pir_testing::GenerateRandomPackedSelectionBits<BlockType>(kNumValues);

  // Compute first inner product with database from SetUp().
  DPF_ASSERT_OK_AND_ASSIGN(auto inner_product_1,
                           database_->InnerProductWith({selections}));

  // Build second database and compute inner product again.
  DPF_ASSERT_OK_AND_ASSIGN(auto database_2, builder_.Build());
  DPF_ASSERT_OK_AND_ASSIGN(auto inner_product_2,
                           database_2->InnerProductWith({selections}));

  EXPECT_EQ(inner_product_1, inner_product_2);
}

}  // namespace
}  // namespace distributed_point_functions
