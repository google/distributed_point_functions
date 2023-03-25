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

#include "pir/hashing/cuckoo_hash_table.h"

#include <memory>
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

const int kNumBuckets = 100;
const int kNumHashFunctions = 3;
const int kMaxRelocations = 50;
const int kMaxStashSize = 3;

class CuckooHashTableTest : public ::testing::Test {
 protected:
  CuckooHashTableTest() : mock_hash_functions_(kNumHashFunctions) {}
  void SetUp() override {
    int i = 0;
    // We expect the constructor to call the hash family kNumHashFunctions
    // times.
    EXPECT_CALL(mock_hash_family_, Call)
        .Times(kNumHashFunctions)
        .WillRepeatedly([this, &i]() {
          // Add a real implementation to each mock function so that Cuckoo
          // Hashing works.
          ON_CALL(mock_hash_functions_[i], Call(::testing::_, ::testing::_))
              .WillByDefault([i](absl::string_view input, int bound) {
                return FarmHashFamily{}(absl::StrCat(i))(input, bound);
              });
          return mock_hash_functions_[i++].AsStdFunction();
        });
    DPF_ASSERT_OK_AND_ASSIGN(
        table_, CuckooHashTable::Create(mock_hash_family_.AsStdFunction(),
                                        kNumBuckets, kNumHashFunctions,
                                        kMaxRelocations, kMaxStashSize));
  }
  std::unique_ptr<CuckooHashTable> table_;
  std::vector<testing::MockFunction<int(absl::string_view, int)>>
      mock_hash_functions_;
  testing::MockFunction<HashFunction(absl::string_view)> mock_hash_family_;
};

TEST_F(CuckooHashTableTest, TestInsert) {
  auto element = "Hello Cuckoo";
  DPF_ASSERT_OK(table_->Insert(element));
  // Test if element was inserted exactly once.
  bool found = false;
  for (int i = 0; i < kNumBuckets; i++) {
    if (!found && table_->GetTable()[i]) {
      EXPECT_EQ(element, table_->GetTable()[i]);
      found = true;
    } else {
      EXPECT_FALSE(table_->GetTable()[i]);
    }
  }
}

TEST_F(CuckooHashTableTest, TestStashLimit) {
  // Add expectation that all hash functions are called at least once.
  for (auto& mock : mock_hash_functions_) {
    EXPECT_CALL(mock, Call).Times(::testing::AtLeast(1));
  }
  int num_elements = 0;
  while (true) {
    absl::Status status =
        table_->Insert(absl::StrCat("Element number ", num_elements));
    if (status.ok()) {
      num_elements++;
    } else {
      EXPECT_THAT(status,
                  StatusIs(absl::StatusCode::kInternal,
                           StartsWith("Cannot insert element: stash is full")));
      break;
    }
  }
  EXPECT_EQ(kMaxStashSize, table_->GetStash().size());

  // Make sure each element we inserted is either in the table or on the stash.
  int count = 0;
  for (int i = 0; i < kNumBuckets; i++) {
    if (table_->GetTable()[i]) {
      count++;
      EXPECT_THAT(*table_->GetTable()[i], StartsWith("Element number "));
    }
  }
  EXPECT_EQ(count + table_->GetStash().size(), num_elements);
  for (int i = 0; i < table_->GetStash().size(); i++) {
    EXPECT_THAT(table_->GetStash()[i], StartsWith("Element number "));
  }
}

TEST(CuckooHashTable, TestDefaultUnlimitedStash) {
  DPF_ASSERT_OK_AND_ASSIGN(
      auto table, CuckooHashTable::Create(FarmHashFamily{}, kNumBuckets,
                                          kNumHashFunctions, kMaxRelocations));
  for (int i = 0; i < 1000; i++) {
    DPF_ASSERT_OK(table->Insert(absl::StrCat("Element number ", i)));
  }
  EXPECT_GE(table->GetStash().size(), 1000 - kNumBuckets);
}

TEST(CuckooHashTable, FailsIfNumBucketsNegative) {
  EXPECT_THAT(CuckooHashTable::Create(FarmHashFamily{}, 0, 0, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       StartsWith("num_buckets must be positive")));
}

TEST(CuckooHashTable, FailsIfNumHashFunctionsLessThanTwo) {
  EXPECT_THAT(CuckooHashTable::Create(FarmHashFamily{}, 1, 1, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       StartsWith("hash_functions.size() must be at least 2")));
}

TEST(CuckooHashTable, FailsIfMaxRelocationsNegative) {
  EXPECT_THAT(CuckooHashTable::Create(FarmHashFamily{}, 1, 2, -1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       StartsWith("max_relocations must be non-negative")));
}

TEST(CuckooHashTable, FailsIfMaxStashSizeNegative) {
  EXPECT_THAT(CuckooHashTable::Create(FarmHashFamily{}, 1, 2, 0, -1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       StartsWith("max_stash_size must be non-negative")));
}

void BM_Insert(benchmark::State& state) {
  for (auto _ : state) {
    DPF_ASSERT_OK_AND_ASSIGN(
        auto table,
        CuckooHashTable::Create(SHA256HashFamily{}, 1.5 * state.range(0),
                                kNumHashFunctions, state.range(0)));
    for (int i = 0; i < state.range(0); i++) {
      DPF_ASSERT_OK(table->Insert(absl::StrCat(i)));
    }
    ::benchmark::DoNotOptimize(table);
  }
}
// Benchmark hashing with number of elements between 1 and 1<<20, three hash
// functions and 1.5 times as many buckets as elements.
BENCHMARK(BM_Insert)->Range(1, 1 << 20);

}  // namespace

}  // namespace distributed_point_functions
