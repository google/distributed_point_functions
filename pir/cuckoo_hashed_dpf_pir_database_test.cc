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

#include "pir/cuckoo_hashed_dpf_pir_database.h"

#include <stddef.h>

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dpf/internal/status_matchers.h"
#include "dpf/xor_wrapper.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "pir/dense_dpf_pir_database.h"
#include "pir/hashing/hash_family_config.pb.h"
#include "pir/private_information_retrieval.pb.h"
#include "pir/testing/mock_pir_database.h"
#include "pir/testing/pir_selection_bits.h"

namespace distributed_point_functions {
namespace {

constexpr int kNumDatabaseElements = 1234;
constexpr int kNumBuckets = static_cast<int>(1.5 * kNumDatabaseElements);
constexpr int kNumHashFunctions = 3;
constexpr int kDatabaseElementSize = 80;

using dpf_internal::IsOk;
using dpf_internal::IsOkAndHolds;
using dpf_internal::StatusIs;
using ::testing::AtLeast;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::FieldsAre;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StartsWith;
using ::testing::Truly;
using Database = CuckooHashedDpfPirDatabase::Interface;
using DenseDatabase = CuckooHashedDpfPirDatabase::DenseDatabase;
using MockDenseDatabase =
    pir_testing::MockPirDatabase<XorWrapper<absl::uint128>, std::string>;
using MockDenseBuilder = MockDenseDatabase::Builder;

TEST(CuckooHashedDpfPirDatabaseBuilder, SetParamsFailsIfNumBucketsIsZero) {
  CuckooHashedDpfPirDatabase::Builder builder;
  CuckooHashingParams params;
  params.set_num_buckets(0);
  params.set_num_hash_functions(1);

  EXPECT_THAT(
      builder.SetParams(params).Build(),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("num_buckets")));
}

TEST(CuckooHashedDpfPirDatabaseBuilder,
     SetParamsFailsIfNumHashFunctionsIsZero) {
  CuckooHashedDpfPirDatabase::Builder builder;
  CuckooHashingParams params;
  params.set_num_buckets(1);
  params.set_num_hash_functions(0);

  EXPECT_THAT(builder.SetParams(params).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("num_hash_functions")));
}

class CuckooHashedDpfPirDatabaseBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up builder with cuckoo hashing params.
    CuckooHashingParams params;
    params.set_num_buckets(kNumBuckets);
    params.set_num_hash_functions(kNumHashFunctions);
    params.mutable_hash_family_config()->set_hash_family(
        HashFamilyConfig::HASH_FAMILY_SHA256);
    params.mutable_hash_family_config()->set_seed("A seed");
    builder_.SetParams(params);

    // Set up dense mock builders for keys and values.
    mock_key_builder_ = std::make_unique<MockDenseBuilder>();
    mock_value_builder_ = std::make_unique<MockDenseBuilder>();
    ON_CALL(*mock_key_builder_, Insert)
        .WillByDefault(ReturnRef(*mock_key_builder_));
    ON_CALL(*mock_key_builder_, Clear)
        .WillByDefault(ReturnRef(*mock_key_builder_));
    ON_CALL(*mock_value_builder_, Insert)
        .WillByDefault(ReturnRef(*mock_value_builder_));
    ON_CALL(*mock_value_builder_, Clear)
        .WillByDefault(ReturnRef(*mock_value_builder_));

    // Set up mock databases.
    mock_key_database_ = std::make_unique<MockDenseDatabase>();
    mock_value_database_ = std::make_unique<MockDenseDatabase>();
    ON_CALL(*mock_key_database_, num_selection_bits)
        .WillByDefault(Return(kNumBuckets));
    ON_CALL(*mock_value_database_, num_selection_bits)
        .WillByDefault(Return(kNumBuckets));
  }

  void InsertElements() {
    DPF_ASSERT_OK_AND_ASSIGN(keys_, pir_testing::GenerateCountingStrings(
                                        kNumDatabaseElements, "Key "));
    DPF_ASSERT_OK_AND_ASSIGN(values_,
                             pir_testing::GenerateRandomStringsEqualSize(
                                 kNumDatabaseElements, kDatabaseElementSize));
    for (int i = 0; i < kNumDatabaseElements; ++i) {
      builder_.Insert({keys_[i], values_[i]});
    }
  }

  std::vector<std::string> keys_, values_;
  CuckooHashedDpfPirDatabase::Builder builder_;
  std::unique_ptr<MockDenseBuilder> mock_key_builder_, mock_value_builder_;
  std::unique_ptr<MockDenseDatabase> mock_key_database_, mock_value_database_;
};

TEST_F(CuckooHashedDpfPirDatabaseBuilderTest,
       SetKeyDatabaseBuilderClearsBuilder) {
  EXPECT_CALL(*mock_key_builder_, Clear).Times(1);
  builder_.SetKeyDatabaseBuilder(std::move(mock_key_builder_));
}

TEST_F(CuckooHashedDpfPirDatabaseBuilderTest,
       SetValueDatabaseBuilderClearsBuilder) {
  EXPECT_CALL(*mock_value_builder_, Clear).Times(1);
  builder_.SetValueDatabaseBuilder(std::move(mock_value_builder_));
}

TEST_F(CuckooHashedDpfPirDatabaseBuilderTest,
       ClearCallsClearOnWrappedBuilders) {
  // Called two times because `Set(Key|Value)DatabaseBuilder` also calls Clear.
  EXPECT_CALL(*mock_key_builder_, Clear).Times(2);
  EXPECT_CALL(*mock_value_builder_, Clear).Times(2);
  builder_.SetKeyDatabaseBuilder(std::move(mock_key_builder_))
      .SetValueDatabaseBuilder(std::move(mock_value_builder_));
  builder_.Clear();
}

TEST_F(CuckooHashedDpfPirDatabaseBuilderTest, BuildsEmptyDatabase) {
  EXPECT_THAT(
      builder_.Build(),
      IsOkAndHolds(Truly([](auto& db) -> bool { return db->size() == 0; })));
}

TEST_F(CuckooHashedDpfPirDatabaseBuilderTest, FailsToBuildWithEmptyKey) {
  EXPECT_THAT(builder_.Insert({"", "Value"}).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("empty")));
}

TEST_F(CuckooHashedDpfPirDatabaseBuilderTest, InsertsOneElementCorrectly) {
  const std::string key = "Key 1";
  const std::string value = "Value 1";
  builder_.Insert({key, value});
  EXPECT_THAT(builder_.Build(),
              IsOkAndHolds(Pointee(Property(&Database::size, Eq(1)))));
}

TEST_F(CuckooHashedDpfPirDatabaseBuilderTest, InsertsOneEmptyElementCorrectly) {
  const std::string key = "Key 1";
  const std::string value = "";
  builder_.Insert({key, value});
  EXPECT_THAT(builder_.Build(),
              IsOkAndHolds(Pointee(Property(&Database::size, Eq(1)))));
}

TEST_F(CuckooHashedDpfPirDatabaseBuilderTest, UsesKeyBuilderCorrectly) {
  const std::string key = "Key 1";
  const std::string value = "Value 1";

  EXPECT_CALL(*mock_key_builder_, Insert(key)).Times(1);
  EXPECT_CALL(*mock_key_builder_, Insert("")).Times(kNumBuckets - 1);
  EXPECT_CALL(*mock_key_builder_, Build)
      .WillOnce(Return(std::move(mock_key_database_)));

  builder_.Insert({key, value})
      .SetKeyDatabaseBuilder(std::move(mock_key_builder_));

  EXPECT_THAT(builder_.Build(), IsOkAndHolds(NotNull()));
}

TEST_F(CuckooHashedDpfPirDatabaseBuilderTest, UsesValueBuilderCorrectly) {
  const std::string key = "Key 1";
  const std::string value = "Value 1";

  EXPECT_CALL(*mock_value_builder_, Insert(value)).Times(1);
  EXPECT_CALL(*mock_value_builder_, Insert("")).Times(kNumBuckets - 1);
  EXPECT_CALL(*mock_value_builder_, Build)
      .WillOnce(Return(std::move(mock_value_database_)));

  builder_.Insert({key, value})
      .SetValueDatabaseBuilder(std::move(mock_value_builder_));

  EXPECT_THAT(builder_.Build(), IsOkAndHolds(NotNull()));
}

TEST_F(CuckooHashedDpfPirDatabaseBuilderTest,
       CloneCallsDenseDatabaseCloneCorrectly) {
  EXPECT_CALL(*mock_key_builder_, Clone).Times(1);
  EXPECT_CALL(*mock_value_builder_, Clone).Times(1);

  builder_.SetKeyDatabaseBuilder(std::move(mock_key_builder_))
      .SetValueDatabaseBuilder(std::move(mock_value_builder_));

  EXPECT_THAT(builder_.Clone()->Build(), IsOkAndHolds(NotNull()));
}

TEST_F(CuckooHashedDpfPirDatabaseBuilderTest, FailsToBuildDatabaseTwice) {
  builder_.Insert({"Key 1", "Value 1"});
  EXPECT_THAT(builder_.Build(), IsOk());
  EXPECT_THAT(builder_.Build(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                         HasSubstr("already built")));
  builder_.SetKeyDatabaseBuilder(nullptr);
  builder_.SetValueDatabaseBuilder(nullptr);
  EXPECT_THAT(builder_.Clone()->Build(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("already built")));
}

TEST_F(CuckooHashedDpfPirDatabaseBuilderTest, CanBuildAgainAfterCallingClear) {
  builder_.Insert({"Key 1", "Value 1"});
  EXPECT_THAT(builder_.Build(), IsOk());
  EXPECT_THAT(builder_.Build(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                         HasSubstr("already built")));

  builder_.Clear();
  EXPECT_THAT(builder_.Build(), IsOkAndHolds(NotNull()));
}

TEST_F(CuckooHashedDpfPirDatabaseBuilderTest, InsertsManyElementsCorrectly) {
  InsertElements();
  EXPECT_THAT(builder_.Build(),
              IsOkAndHolds(Pointee(
                  Property(&Database::size, Eq(kNumDatabaseElements)))));
}

TEST_F(CuckooHashedDpfPirDatabaseBuilderTest, ComputesInnerProductCorrectly) {
  InsertElements();
  DPF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Database> database,
                           builder_.Build());

  // Generate selection vector with all bits set to 1.
  std::vector<Database::BlockType> selections =
      pir_testing::PackSelectionBits<Database::BlockType>(
          std::vector<bool>(database->num_selection_bits(), true));

  // We expect the result to be the XOR of all keys, padded with zeros.
  std::string key, value;
  for (int i = 0; i < kNumDatabaseElements; ++i) {
    key.resize(std::max(key.size(), keys_[i].size()), '\0');
    value.resize(std::max(value.size(), values_[i].size()), '\0');
    for (int j = 0; j < keys_[i].size(); ++j) {
      key[j] ^= keys_[i][j];
    }
    for (int j = 0; j < values_[i].size(); ++j) {
      value[j] ^= values_[i][j];
    }
  }

  EXPECT_THAT(
      database->InnerProductWith({selections}),
      IsOkAndHolds(ElementsAre(FieldsAre(StartsWith(key), StartsWith(value)))));
}

TEST_F(CuckooHashedDpfPirDatabaseBuilderTest, CallsInnerProductWithCorrectly) {
  InsertElements();

  std::vector<Database::BlockType> selections =
      pir_testing::PackSelectionBits<Database::BlockType>(
          std::vector<bool>(kNumBuckets, true));

  EXPECT_CALL(*mock_key_database_, InnerProductWith(ElementsAre(selections)))
      .WillOnce(Return(std::vector<std::string>{"dummy key"}));
  EXPECT_CALL(*mock_value_database_, InnerProductWith(ElementsAre(selections)))
      .WillOnce(Return(std::vector<std::string>{"dummy value"}));
  EXPECT_CALL(*mock_key_builder_, Build)
      .WillOnce(Return(std::move(mock_key_database_)));
  EXPECT_CALL(*mock_value_builder_, Build)
      .WillOnce(Return(std::move(mock_value_database_)));

  DPF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Database> database,
      builder_.SetKeyDatabaseBuilder(std::move(mock_key_builder_))
          .SetValueDatabaseBuilder(std::move(mock_value_builder_))
          .Build())

  EXPECT_THAT(database->InnerProductWith({selections}),
              IsOkAndHolds(ElementsAre(FieldsAre(StartsWith("dummy key"),
                                                 StartsWith("dummy value")))));
}

TEST_F(CuckooHashedDpfPirDatabaseBuilderTest,
       InnerProductFromClonedBuilderIsTheSame) {
  InsertElements();
  auto builder2 = builder_.Clone();
  DPF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Database> database1,
                           builder_.Build());
  DPF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Database> database2,
                           builder2->Build());
  auto selections =
      pir_testing::GenerateRandomPackedSelectionBits<Database::BlockType>(
          kNumBuckets);

  DPF_ASSERT_OK_AND_ASSIGN(auto inner_product_1,
                           database1->InnerProductWith({selections}));
  DPF_ASSERT_OK_AND_ASSIGN(auto inner_product_2,
                           database2->InnerProductWith({selections}));

  EXPECT_EQ(inner_product_1, inner_product_2);
}

}  // namespace
}  // namespace distributed_point_functions
