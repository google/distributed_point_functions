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

#include "pir/hashing/hash_family.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "dpf/internal/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace distributed_point_functions {

namespace {

using dpf_internal::StatusIs;
using ::testing::StartsWith;

class HashTableTest : public ::testing::Test {
 public:
  testing::MockFunction<int(absl::string_view, int)> mock_hash_function_;
  testing::MockFunction<HashFunction(absl::string_view)> mock_hash_family_;
};

TEST_F(HashTableTest, FailsIfNumHashFunctionsNegative) {
  EXPECT_CALL(mock_hash_function_, Call).Times(0);
  EXPECT_THAT(CreateHashFunctions(mock_hash_family_.AsStdFunction(), -1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       StartsWith("num_hash_functions must not be negative")));
}

TEST_F(HashTableTest, WrapWithSeedPrependsSeed) {
  constexpr absl::string_view kSeed1 = "kSeed1";
  constexpr absl::string_view kSeed2 = "kSeed2";
  constexpr absl::string_view kInput = "kInput";
  constexpr int kUpperBound = 42, kResult = 23;
  std::string arg = absl::StrCat(kSeed1, kSeed2);
  EXPECT_CALL(mock_hash_family_, Call(absl::string_view(arg)))
      .WillOnce(::testing::Return(mock_hash_function_.AsStdFunction()));
  EXPECT_CALL(mock_hash_function_, Call(kInput, kUpperBound))
      .WillOnce(::testing::Return(kResult));

  EXPECT_EQ(WrapWithSeed(mock_hash_family_.AsStdFunction(), kSeed1)(kSeed2)(
                kInput, kUpperBound),
            kResult);
}

}  // namespace

}  // namespace distributed_point_functions
