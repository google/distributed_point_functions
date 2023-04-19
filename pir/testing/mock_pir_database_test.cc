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

#include "pir/testing/mock_pir_database.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dpf/internal/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "pir/dense_dpf_pir_database.h"

namespace distributed_point_functions {
namespace pir_testing {
namespace {

using dpf_internal::StatusIs;
using ::testing::HasSubstr;

TEST(GenerateCountingStrings, FailsWithNegativeNumElements) {
  EXPECT_THAT(
      GenerateCountingStrings(-1, ""),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("negative")));
}

TEST(GenerateRandomStrings, FailsWithNegativeSize) {
  EXPECT_THAT(
      GenerateRandomStrings({-1}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("negative")));
}

TEST(GenerateRandomStringsEqualSize, FailsWithNegativeNumElements) {
  EXPECT_THAT(
      GenerateRandomStringsEqualSize(-1, 0),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("num_elements")));
}

TEST(GenerateRandomStringsEqualSize, FailsWithNegativeSize) {
  EXPECT_THAT(GenerateRandomStringsEqualSize(0, -1),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("size")));
}

TEST(GenerateRandomStringsVariableSize, FailsWithNegativeNumElements) {
  EXPECT_THAT(
      GenerateRandomStringsVariableSize(-1, 0, 0),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("num_elements")));
}

TEST(GenerateRandomStringsVariableSize, FailsWithNegativeAvgElementSize) {
  EXPECT_THAT(GenerateRandomStringsVariableSize(0, -1, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("avg_element_size")));
}

TEST(GenerateRandomStringsVariableSize, FailsWithNegativeMaxSizeDiff) {
  EXPECT_THAT(
      GenerateRandomStringsVariableSize(0, 0, -1),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("max_size_diff")));
}

TEST(GenerateRandomStrings, ElementSizesAreCorrect) {
  std::vector<int> element_sizes = {0, 1, 8};
  const int num_elements = element_sizes.size();
  DPF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> elements,
                           GenerateRandomStrings(element_sizes));

  // Check the generated elements.
  EXPECT_EQ(elements.size(), num_elements);
  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(elements[i].size(), element_sizes[i]);
  }
}

TEST(GenerateRandomStringsEqualSize, ElementSizesAreCorrect) {
  constexpr int num_elements = 12;
  constexpr int element_size = 8;

  DPF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::string> elements,
      GenerateRandomStringsEqualSize(num_elements, element_size));

  // Check the appended elements
  EXPECT_EQ(elements.size(), num_elements);
  for (int i = 0; i < elements.size(); ++i) {
    EXPECT_EQ(elements[i].size(), element_size);
  }
}

TEST(GenerateRandomStringsVariableSize, ElementSizesAreCorrect) {
  constexpr int num_elements = 12;
  constexpr int element_size = 8;
  constexpr int max_size_diff = 2;

  DPF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> elements,
                           GenerateRandomStringsVariableSize(
                               num_elements, element_size, max_size_diff));

  // Check the appended elements
  EXPECT_EQ(elements.size(), num_elements);
  for (int i = 0; i < elements.size(); ++i) {
    EXPECT_GE(elements[i].size(), element_size - max_size_diff);
    EXPECT_LE(elements[i].size(), element_size + max_size_diff);
  }
}

TEST(CreateFakeDatabase, InsertselementsCorrectly) {
  constexpr int num_elements = 1234;
  DPF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> elements,
                           GenerateCountingStrings(num_elements, "Element "));
  DPF_ASSERT_OK_AND_ASSIGN(auto database,
                           CreateFakeDatabase<DenseDpfPirDatabase>(elements));
  EXPECT_EQ(database->size(), num_elements);
  DenseDpfPirDatabase* dense_database =
      dynamic_cast<DenseDpfPirDatabase*>(database.get());
  ASSERT_NE(dense_database, nullptr);
  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(elements[i], dense_database->content()[i]);
  }
}

}  // namespace
}  // namespace pir_testing
}  // namespace distributed_point_functions
