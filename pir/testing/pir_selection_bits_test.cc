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

#include "pir/testing/pir_selection_bits.h"

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

using BlockType = DenseDpfPirDatabase::BlockType;
using dpf_internal::IsOkAndHolds;
using dpf_internal::StatusIs;
using ::testing::HasSubstr;

constexpr int kBitsPerBlock = 8 * sizeof(BlockType);

// Should be packed to one block if the number of bits is smaller than a full
// block size.
TEST(PackSelectionBits, ReturnsCorrectNumberOfBlocksSmall) {
  std::vector<bool> selections(1, true);
  std::vector<BlockType> packed_selections =
      PackSelectionBits<BlockType>(selections);
  EXPECT_EQ(packed_selections.size(), 1);
}

// Should be packed to multiple blocks if the number of bits is more than one
// block size.
TEST(PackSelectionBits, ReturnsCorrectNumberOfBlocksLarge) {
  std::vector<bool> selections(kBitsPerBlock + 1, true);
  std::vector<BlockType> packed_selections =
      PackSelectionBits<BlockType>(selections);
  EXPECT_EQ(packed_selections.size(), 2);
}

TEST(GenerateRandomPackedSelectionBits, ReturnsCorrectNumberofBlocksSmall) {
  std::vector<BlockType> packed_selections =
      GenerateRandomPackedSelectionBits<BlockType>(1);
  EXPECT_EQ(packed_selections.size(), 1);
}

TEST(GenerateRandomPackedSelectionBits, ReturnsCorrectNumberofBlocksLarge) {
  std::vector<BlockType> packed_selections =
      GenerateRandomPackedSelectionBits<BlockType>(kBitsPerBlock + 1);
  EXPECT_EQ(packed_selections.size(), 2);
}

TEST(InnerProductWithUnpacked, FailsWhenSizesDontMatch) {
  EXPECT_THAT(InnerProductWithUnpacked({}, {"a"}),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("size")));
}

TEST(InnerProductWithUnpacked, ComputesInnerProductCorrectly) {
  std::vector<bool> selections({1, 0, 1});
  std::vector<std::string> values({"a", "b", "c"});
  EXPECT_THAT(InnerProductWithUnpacked(selections, values),
              IsOkAndHolds(std::string(1, 'a' ^ 'c')));
}

}  // namespace
}  // namespace pir_testing
}  // namespace distributed_point_functions
