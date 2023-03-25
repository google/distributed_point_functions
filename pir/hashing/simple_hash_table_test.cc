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

#include "pir/hashing/simple_hash_table.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "benchmark/benchmark.h"
#include "dpf/internal/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "pir/hashing/farm_hash_family.h"
#include "pir/hashing/sha256_hash_family.h"

namespace distributed_point_functions {

namespace {

using dpf_internal::StatusIs;
using ::testing::StartsWith;

const int kNumBuckets = 10;
const int kNumHashFunctions = 3;

class SimpleHashTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    DPF_ASSERT_OK_AND_ASSIGN(
        table_, SimpleHashTable::Create(FarmHashFamily{}, kNumBuckets,
                                        kNumHashFunctions));
  }
  std::unique_ptr<SimpleHashTable> table_;
};

TEST_F(SimpleHashTableTest, TestInsert) {
  for (int i = 0; i < 1000; i++) {
    DPF_ASSERT_OK(table_->Insert(absl::StrCat("Element number ", i)));
  }
  int count = 0;
  for (int i = 0; i < table_->GetTable().size(); i++) {
    const std::vector<std::string>& current_table = table_->GetTable()[i];
    for (int j = 0; j < current_table.size(); j++) {
      EXPECT_THAT(current_table[j], StartsWith("Element number "));
    }
    count += current_table.size();
  }
  EXPECT_EQ(count, 1000 * kNumHashFunctions);
}

TEST_F(SimpleHashTableTest, TestOverflow) {
  // Create new table with limited bucket size
  DPF_ASSERT_OK_AND_ASSIGN(
      table_, SimpleHashTable::Create(FarmHashFamily{}, kNumBuckets,
                                      kNumHashFunctions, 3));
  absl::Status status = absl::OkStatus();
  for (int i = 0; status.ok(); i++) {
    status = table_->Insert(absl::StrCat("Element number ", i));
  }
  EXPECT_THAT(
      status,
      StatusIs(
          absl::StatusCode::kInternal,
          StartsWith("Cannot insert element: maximum bucket size reached")));
}

TEST(SimpleHashTable, FailsIfNumBucketsNegative) {
  EXPECT_THAT(SimpleHashTable::Create(FarmHashFamily{}, 0, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       StartsWith("num_buckets must be positive")));
}

TEST(SimpleHashTable, FailsIfNumHashZero) {
  EXPECT_THAT(SimpleHashTable::Create(FarmHashFamily{}, 1, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       StartsWith("hash_functions must not be empty")));
}

TEST(SimpleHashTable, FailsIfMaxBucketSizeNegative) {
  EXPECT_THAT(SimpleHashTable::Create(FarmHashFamily{}, 1, 1, -1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       StartsWith("max_bucket_size must be positive")));
}

void BM_Insert(benchmark::State& state) {
  for (auto _ : state) {
    DPF_ASSERT_OK_AND_ASSIGN(
        auto table, SimpleHashTable::Create(SHA256HashFamily{}, state.range(1),
                                            kNumHashFunctions));
    for (int i = 0; i < state.range(0); i++) {
      DPF_ASSERT_OK(table->Insert(absl::StrCat(i)));
    }
    ::benchmark::DoNotOptimize(table);
  }
}
// Benchmark hashing with number of elements between 1 and 1<<20 and number of
// buckets between 1 and 1<<10.
BENCHMARK(BM_Insert)->RangePair(1, 1 << 20, 1, 1 << 10);

}  // namespace

}  // namespace distributed_point_functions
