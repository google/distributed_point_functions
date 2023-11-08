// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pir/simple_hashed_dpf_pir_database.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "dpf/internal/status_matchers.h"
#include "dpf/xor_wrapper.h"
#include "gmock/gmock.h"
#include "google/protobuf/io/coded_stream.h"
#include "gtest/gtest.h"
#include "pir/hashing/hash_family.h"
#include "pir/hashing/hash_family_config.h"
#include "pir/hashing/hash_family_config.pb.h"
#include "pir/private_information_retrieval.pb.h"
#include "pir/testing/mock_pir_database.h"
#include "pir/testing/pir_selection_bits.h"

namespace distributed_point_functions {
namespace {

constexpr int kNumDatabaseElements = 1234;
constexpr int kNumBuckets = 123;
constexpr int kDatabaseElementSize = 80;

using dpf_internal::IsOk;
using dpf_internal::IsOkAndHolds;
using dpf_internal::StatusIs;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::Truly;
using Database = SimpleHashedDpfPirDatabase::Interface;
using DenseDatabase = SimpleHashedDpfPirDatabase::DenseDatabase;
using MockDenseDatabase =
    pir_testing::MockPirDatabase<XorWrapper<absl::uint128>, std::string>;
using MockDenseBuilder = MockDenseDatabase::Builder;

TEST(SimpleHashedDpfPirDatabaseBuilder, SetParamsFailsIfNumBucketsIsZero) {
  SimpleHashedDpfPirDatabase::Builder builder;
  SimpleHashingParams params;
  params.set_num_buckets(0);

  EXPECT_THAT(
      builder.SetParams(params).Build(),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("num_buckets")));
}

class SimpleHashedDpfPirDatabaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    params_.set_num_buckets(kNumBuckets);
    params_.mutable_hash_family_config()->set_hash_family(
        HashFamilyConfig::HASH_FAMILY_SHA256);
    params_.mutable_hash_family_config()->set_seed("A seed");
    builder_.SetParams(params_);

    // Set up dense mock builder.
    mock_dense_builder_ = std::make_unique<MockDenseBuilder>();
    ON_CALL(*mock_dense_builder_, Insert)
        .WillByDefault(ReturnRef(*mock_dense_builder_));
    ON_CALL(*mock_dense_builder_, Clear)
        .WillByDefault(ReturnRef(*mock_dense_builder_));

    // Set up mock database.
    mock_dense_database_ = std::make_unique<MockDenseDatabase>();
    ON_CALL(*mock_dense_database_, num_selection_bits)
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
  SimpleHashedDpfPirDatabase::Builder builder_;
  std::unique_ptr<MockDenseBuilder> mock_dense_builder_;
  std::unique_ptr<MockDenseDatabase> mock_dense_database_;
  SimpleHashingParams params_;
};

TEST_F(SimpleHashedDpfPirDatabaseTest, SetDenseDatabaseBuilderClearsBuilder) {
  EXPECT_CALL(*mock_dense_builder_, Clear).Times(1);
  builder_.SetDenseDatabaseBuilder(std::move(mock_dense_builder_));
}

TEST_F(SimpleHashedDpfPirDatabaseTest, ClearCallsClearOnWrappedBuilders) {
  // Called two times because `SetDenseDatabaseBuilder` also calls Clear.
  EXPECT_CALL(*mock_dense_builder_, Clear).Times(2);
  builder_.SetDenseDatabaseBuilder(std::move(mock_dense_builder_));
  builder_.Clear();
}

TEST_F(SimpleHashedDpfPirDatabaseTest, BuildsEmptyDatabase) {
  EXPECT_THAT(
      builder_.Build(),
      IsOkAndHolds(Truly([](auto& db) -> bool { return db->size() == 0; })));
}

TEST_F(SimpleHashedDpfPirDatabaseTest, FailsToBuildWithEmptyKey) {
  EXPECT_THAT(builder_.Insert({"", "Value"}).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("empty")));
}

TEST_F(SimpleHashedDpfPirDatabaseTest, InsertsOneElementCorrectly) {
  const std::string key = "Key 1";
  const std::string value = "Value 1";
  builder_.Insert({key, value});
  EXPECT_THAT(builder_.Build(),
              IsOkAndHolds(Pointee(Property(&Database::size, Eq(1)))));
}

TEST_F(SimpleHashedDpfPirDatabaseTest, InsertsOneEmptyElementCorrectly) {
  const std::string key = "Key 1";
  const std::string value = "";
  builder_.Insert({key, value});
  EXPECT_THAT(builder_.Build(),
              IsOkAndHolds(Pointee(Property(&Database::size, Eq(1)))));
}

TEST_F(SimpleHashedDpfPirDatabaseTest, UsesDenseBuilderCorrectly) {
  const std::string key = "Key 1";
  const std::string value = "Value 1";

  EXPECT_CALL(*mock_dense_builder_, Insert(Truly([&key,
                                                  &value](std::string proto) {
    HashedPirDatabaseBucket bucket;
    return bucket.ParseFromString(proto) && bucket.keys(0) == key &&
           bucket.values(0) == value;
  }))).Times(1);
  EXPECT_CALL(*mock_dense_builder_, Insert("")).Times(kNumBuckets - 1);
  EXPECT_CALL(*mock_dense_builder_, Build)
      .WillOnce(Return(std::move(mock_dense_database_)));

  builder_.Insert({key, value})
      .SetDenseDatabaseBuilder(std::move(mock_dense_builder_));

  EXPECT_THAT(builder_.Build(),
              IsOkAndHolds(Pointee(Property(&Database::size, Eq(1)))));
}

TEST_F(SimpleHashedDpfPirDatabaseTest, CloneCallsDenseDatabaseCloneCorrectly) {
  EXPECT_CALL(*mock_dense_builder_, Clone).Times(1);
  builder_.SetDenseDatabaseBuilder(std::move(mock_dense_builder_));

  EXPECT_THAT(builder_.Clone()->Build(), IsOkAndHolds(NotNull()));
}

TEST_F(SimpleHashedDpfPirDatabaseTest, FailsToBuildDatabaseTwice) {
  builder_.Insert({"Key 1", "Value 1"});
  EXPECT_THAT(builder_.Build(), IsOk());
  EXPECT_THAT(builder_.Build(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                         HasSubstr("already built")));
  builder_.SetDenseDatabaseBuilder(nullptr);
  EXPECT_THAT(builder_.Clone()->Build(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("already built")));
}

TEST_F(SimpleHashedDpfPirDatabaseTest, CanBuildAgainAfterCallingClear) {
  builder_.Insert({"Key 1", "Value 1"});
  EXPECT_THAT(builder_.Build(), IsOk());
  EXPECT_THAT(builder_.Build(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                         HasSubstr("already built")));

  builder_.Clear();
  EXPECT_THAT(builder_.Build(), IsOkAndHolds(NotNull()));
}

TEST_F(SimpleHashedDpfPirDatabaseTest, InsertsManyElementsCorrectly) {
  InsertElements();
  EXPECT_THAT(builder_.Build(),
              IsOkAndHolds(Pointee(
                  Property(&Database::size, Eq(kNumDatabaseElements)))));
}

TEST_F(SimpleHashedDpfPirDatabaseTest, ClonedBuilderRetainsRecords) {
  InsertElements();
  EXPECT_THAT(builder_.Clone()->Build(),
              IsOkAndHolds(Pointee(
                  Property(&Database::size, Eq(kNumDatabaseElements)))));
}

TEST_F(SimpleHashedDpfPirDatabaseTest, CallsInnerProductWithCorrectly) {
  InsertElements();

  std::vector<Database::BlockType> selections =
      pir_testing::PackSelectionBits<Database::BlockType>(
          std::vector<bool>(kNumBuckets, true));

  EXPECT_CALL(*mock_dense_database_, InnerProductWith(ElementsAre(selections)))
      .WillOnce(Return(std::vector<std::string>{"dummy result"}));
  EXPECT_CALL(*mock_dense_builder_, Build)
      .WillOnce(Return(std::move(mock_dense_database_)));

  DPF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Database> database,
      builder_.SetDenseDatabaseBuilder(std::move(mock_dense_builder_)).Build())

  EXPECT_THAT(database->InnerProductWith({selections}),
              IsOkAndHolds(ElementsAre("dummy result")));
}

TEST_F(SimpleHashedDpfPirDatabaseTest,
       BucketReturnedByInnerProductWithIsCorrect) {
  InsertElements();

  DPF_ASSERT_OK_AND_ASSIGN(auto hash_family, CreateHashFamilyFromConfig(
                                                 params_.hash_family_config()));
  DPF_ASSERT_OK_AND_ASSIGN(auto hash_functions,
                           CreateHashFunctions(std::move(hash_family), 1));
  std::vector<std::vector<std::string>> expected_keys(kNumBuckets);
  std::vector<std::vector<std::string>> expected_values(kNumBuckets);
  for (int i = 0; i < kNumDatabaseElements; ++i) {
    expected_keys[hash_functions[0](keys_[i], kNumBuckets)].push_back(keys_[i]);
    expected_values[hash_functions[0](keys_[i], kNumBuckets)].push_back(
        values_[i]);
  }

  DPF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Database> database,
                           builder_.Build());
  for (int i = 0; i < kNumBuckets; ++i) {
    std::vector<bool> selections(kNumBuckets, false);
    selections[i] = true;
    DPF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::string> response_str,
        database->InnerProductWith(
            {pir_testing::PackSelectionBits<Database::BlockType>(selections)}));
    HashedPirDatabaseBucket result;
    ::google::protobuf::io::CodedInputStream coded_stream(
        reinterpret_cast<const uint8_t*>(response_str[0].data()),
        response_str[0].size());
    EXPECT_TRUE(result.ParseFromCodedStream(&coded_stream));

    EXPECT_THAT(result.keys(), ElementsAreArray(expected_keys[i]));
    EXPECT_THAT(result.values(), ElementsAreArray(expected_values[i]));
  }
}

TEST_F(SimpleHashedDpfPirDatabaseTest, ReturnsEmptyBucketCorrectly) {
  std::string key = "Key 0";
  std::string value = "Value 0";
  builder_.Insert({key, value});

  DPF_ASSERT_OK_AND_ASSIGN(auto hash_family, CreateHashFamilyFromConfig(
                                                 params_.hash_family_config()));
  DPF_ASSERT_OK_AND_ASSIGN(auto hash_functions,
                           CreateHashFunctions(std::move(hash_family), 1));
  int bucket = hash_functions[0](key, kNumBuckets);
  int wrong_bucket = (bucket + 1) % kNumBuckets;

  DPF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Database> database,
                           builder_.Build());
  std::vector<bool> selections(kNumBuckets, false);
  selections[wrong_bucket] = true;
  DPF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::string> response_str,
      database->InnerProductWith(
          {pir_testing::PackSelectionBits<Database::BlockType>(selections)}));
  HashedPirDatabaseBucket result;
  ::google::protobuf::io::CodedInputStream coded_stream(
      reinterpret_cast<const uint8_t*>(response_str[0].data()),
      response_str[0].size());
  EXPECT_TRUE(result.ParseFromCodedStream(&coded_stream));

  EXPECT_TRUE(result.keys().empty());
  EXPECT_TRUE(result.values().empty());
}

}  // namespace
}  // namespace distributed_point_functions
