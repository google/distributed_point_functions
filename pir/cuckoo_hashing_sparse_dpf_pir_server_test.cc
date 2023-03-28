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

#include "pir/cuckoo_hashing_sparse_dpf_pir_server.h"

#include "absl/algorithm/container.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "dpf/internal/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "pir/cuckoo_hashed_dpf_pir_database.h"
#include "pir/hashing/hash_family.h"
#include "pir/hashing/hash_family_config.h"
#include "pir/hashing/hash_family_config.pb.h"
#include "pir/private_information_retrieval.pb.h"
#include "pir/testing/encrypt_decrypt.h"
#include "pir/testing/mock_pir_database.h"
#include "pir/testing/request_generator.h"

namespace distributed_point_functions {
namespace {

constexpr int kNumElements = 1234;
constexpr int kValueSize = 16;
constexpr HashFamilyConfig::HashFamily kHashFamily =
    HashFamilyConfig::HASH_FAMILY_SHA256;

using dpf_internal::IsOkAndHolds;
using dpf_internal::StatusIs;
using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::testing::StartsWith;
using ::testing::Truly;
using Database = CuckooHashingSparseDpfPirServer::Database;

TEST(CuckooHashingSparseDpfPirServer, GenerateParamsFailsWhenConfigIsInvalid) {
  PirConfig config;

  EXPECT_THAT(CuckooHashingSparseDpfPirServer::GenerateParams(config),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("valid CuckooHashingSparseDpfPirConfig")));
}

class CuckooHashingSparseDpfPirServerTest : public testing::Test {
 protected:
  void SetUpConfig() {
    config_.mutable_cuckoo_hashing_sparse_dpf_pir_config()->set_num_elements(
        kNumElements);
    config_.mutable_cuckoo_hashing_sparse_dpf_pir_config()->set_hash_family(
        kHashFamily);
  }

  void SetUpParams() {
    SetUpConfig();
    DPF_ASSERT_OK_AND_ASSIGN(
        params_, CuckooHashingSparseDpfPirServer::GenerateParams(config_));
  }

  void GenerateKeyValuePairs() {
    SetUpParams();
    DPF_ASSERT_OK_AND_ASSIGN(
        keys_, pir_testing::GenerateCountingStrings(kNumElements, "Key "));
    DPF_ASSERT_OK_AND_ASSIGN(
        values_,
        pir_testing::GenerateRandomStringsEqualSize(kNumElements, kValueSize));
  }

  void SetUpDatabase() {
    if (keys_.empty() || values_.empty()) {
      GenerateKeyValuePairs();
    }
    std::vector<Database::RecordType> pairs(kNumElements);
    for (int i = 0; i < kNumElements; ++i) {
      pairs[i] = {keys_[i], values_[i]};
    }
    CuckooHashedDpfPirDatabase::Builder builder;
    builder.SetParams(params_);
    DPF_ASSERT_OK_AND_ASSIGN(
        database_, pir_testing::CreateFakeDatabase<CuckooHashedDpfPirDatabase>(
                       pairs, &builder));
  }

  void SetUpServer() {
    SetUpDatabase();
    DPF_ASSERT_OK_AND_ASSIGN(server_,
                             CuckooHashingSparseDpfPirServer::CreatePlain(
                                 params_, std::move(database_)));
  }

  PirConfig config_;
  CuckooHashingParams params_;
  std::vector<std::string> keys_, values_;
  std::unique_ptr<Database> database_;
  std::unique_ptr<CuckooHashingSparseDpfPirServer> server_;
};

TEST_F(CuckooHashingSparseDpfPirServerTest, GenerateParamsReturnsValidParams) {
  SetUpConfig();
  config_.mutable_cuckoo_hashing_sparse_dpf_pir_config()->set_num_elements(
      kNumElements);
  config_.mutable_cuckoo_hashing_sparse_dpf_pir_config()->set_hash_family(
      HashFamilyConfig::HASH_FAMILY_SHA256);

  EXPECT_THAT(CuckooHashingSparseDpfPirServer::GenerateParams(config_),
              IsOkAndHolds(Truly([](auto& params) {
                return params.num_buckets() > kNumElements &&
                       params.num_hash_functions() > 1 &&
                       !params.hash_family_config().seed().empty() &&
                       !absl::c_all_of(params.hash_family_config().seed(),
                                       [](char c) { return c == '\0'; });
              })));
}

TEST_F(CuckooHashingSparseDpfPirServerTest,
       CreatePlainFailsWhenNumBucketsIsZero) {
  SetUpDatabase();

  params_.set_num_buckets(0);

  EXPECT_THAT(
      CuckooHashingSparseDpfPirServer::CreatePlain(params_,
                                                   std::move(database_)),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("num_buckets")));
}

TEST_F(CuckooHashingSparseDpfPirServerTest,
       CreatePlainFailsWhenNumHashFunctionsIsZero) {
  SetUpDatabase();

  params_.set_num_hash_functions(0);

  EXPECT_THAT(CuckooHashingSparseDpfPirServer::CreatePlain(
                  params_, std::move(database_)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("num_hash_functions")));
}

TEST_F(CuckooHashingSparseDpfPirServerTest,
       CreatePlainFailsWhenHashFamilyIsUnspecified) {
  SetUpDatabase();

  params_.mutable_hash_family_config()->set_hash_family(
      HashFamilyConfig::default_instance().hash_family());

  EXPECT_THAT(
      CuckooHashingSparseDpfPirServer::CreatePlain(params_,
                                                   std::move(database_)),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("hash_family")));
}

TEST_F(CuckooHashingSparseDpfPirServerTest,
       CreatePlainFailsWhenDatabaseIsNull) {
  SetUpParams();

  EXPECT_THAT(CuckooHashingSparseDpfPirServer::CreatePlain(params_, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("null")));
}

TEST_F(CuckooHashingSparseDpfPirServerTest,
       CreatePlainFailsWhenumBucketsDoesNotMatchNumSelectionBits) {
  SetUpDatabase();

  params_.set_num_buckets(database_->num_selection_bits() + 1);

  EXPECT_THAT(CuckooHashingSparseDpfPirServer::CreatePlain(
                  params_, std::move(database_)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("selection bits")));
}

TEST_F(CuckooHashingSparseDpfPirServerTest, CreateLeaderSucceeds) {
  SetUpDatabase();
  auto dummy_sender =
      [](const PirRequest& request,
         std::function<void()> while_waiting) -> absl::StatusOr<PirResponse> {
    return absl::UnimplementedError("Dummy");
  };

  EXPECT_THAT(CuckooHashingSparseDpfPirServer::CreateLeader(
                  params_, std::move(database_), std::move(dummy_sender)),
              IsOkAndHolds(NotNull()));
}

TEST_F(CuckooHashingSparseDpfPirServerTest, CreateHelperSucceeds) {
  SetUpDatabase();
  DPF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<const crypto::tink::HybridDecrypt> hybrid_decrypt,
      pir_testing::CreateFakeHybridDecrypt());
  auto decrypter = [&hybrid_decrypt](absl::string_view ciphertext,
                                     absl::string_view context_info) {
    return hybrid_decrypt->Decrypt(ciphertext, context_info);
  };

  EXPECT_THAT(CuckooHashingSparseDpfPirServer::CreateHelper(
                  params_, std::move(database_), decrypter),
              IsOkAndHolds(NotNull()));
}

TEST_F(CuckooHashingSparseDpfPirServerTest, CreatePlainSucceeds) {
  SetUpDatabase();

  EXPECT_THAT(CuckooHashingSparseDpfPirServer::CreatePlain(
                  params_, std::move(database_)),
              IsOkAndHolds(NotNull()));
}

TEST_F(CuckooHashingSparseDpfPirServerTest, HandleRequestSucceeds) {
  // Create two servers.
  SetUpDatabase();
  DPF_ASSERT_OK_AND_ASSIGN(auto server1,
                           CuckooHashingSparseDpfPirServer::CreatePlain(
                               params_, std::move(database_)));
  SetUpDatabase();
  DPF_ASSERT_OK_AND_ASSIGN(auto server2,
                           CuckooHashingSparseDpfPirServer::CreatePlain(
                               params_, std::move(database_)));

  // Hash the client's query with each hash function.
  DPF_ASSERT_OK_AND_ASSIGN(
      HashFamily hash_family,
      CreateHashFamilyFromConfig(params_.hash_family_config()));
  DPF_ASSERT_OK_AND_ASSIGN(std::vector<HashFunction> hash_functions,
                           CreateHashFunctions(std::move(hash_family),
                                               params_.num_hash_functions()));
  ASSERT_EQ(hash_functions.size(), params_.num_hash_functions());
  constexpr absl::string_view query = "Key 42";
  std::vector<int> indices;
  for (const HashFunction& hash_function : hash_functions) {
    indices.push_back(hash_function(query, params_.num_buckets()));
  }

  // Generate plain requests for `indices`.
  DPF_ASSERT_OK_AND_ASSIGN(
      auto request_generator,
      pir_testing::RequestGenerator::Create(
          params_.num_buckets(),
          CuckooHashingSparseDpfPirServer::kEncryptionContextInfo));
  PirRequest request1, request2;
  DPF_ASSERT_OK_AND_ASSIGN(
      std::tie(*request1.mutable_dpf_pir_request()->mutable_plain_request(),
               *request2.mutable_dpf_pir_request()->mutable_plain_request()),
      request_generator->CreateDpfPirPlainRequests(indices));

  // Obtain a response from each server and add them up.
  PirResponse response1, response2;
  std::vector<std::string> response_keys;
  DPF_ASSERT_OK_AND_ASSIGN(response1, server1->HandleRequest(request1));
  DPF_ASSERT_OK_AND_ASSIGN(response2, server2->HandleRequest(request2));
  ASSERT_EQ(response1.dpf_pir_response().masked_response_size(),
            response2.dpf_pir_response().masked_response_size());
  ASSERT_EQ(response1.dpf_pir_response().masked_response_size(),
            2 * indices.size());
  for (int i = 0; i < response1.dpf_pir_response().masked_response_size();
       i++) {
    ASSERT_EQ(response1.dpf_pir_response().masked_response(i).size(),
              response2.dpf_pir_response().masked_response(i).size());
    if (i % 2 == 0) {  // Keys and values are interleaved. We only check if the
                       // key we wanted is in the response.
      response_keys.emplace_back(
          response1.dpf_pir_response().masked_response(i).size(), '\0');
      for (int j = 0;
           j < response1.dpf_pir_response().masked_response(i).size(); ++j) {
        response_keys.back()[j] =
            response1.dpf_pir_response().masked_response(i)[j] ^
            response2.dpf_pir_response().masked_response(i)[j];
      }
    }
  }

  EXPECT_THAT(response_keys, Contains(StartsWith(query)));
}

TEST_F(CuckooHashingSparseDpfPirServerTest,
       HandlePlainRequestFailsWhenRequestIsNotDpfPirRequest) {
  SetUpServer();
  PirRequest request;

  EXPECT_THAT(server_->HandleRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("valid DpfPirRequest")));
}

TEST_F(CuckooHashingSparseDpfPirServerTest,
       HandlePlainRequestFailsWhenRequestIsNotPlainRequest) {
  SetUpServer();
  PirRequest request;
  request.mutable_dpf_pir_request();

  EXPECT_THAT(server_->HandleRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("valid DpfPirRequest::PlainRequest")));
}

TEST_F(CuckooHashingSparseDpfPirServerTest,
       HandlePlainRequestFailsWhenDpfKeyIsEmpty) {
  SetUpServer();
  PirRequest request;
  request.mutable_dpf_pir_request()->mutable_plain_request();

  EXPECT_THAT(
      server_->HandleRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("dpf_key")));
}

TEST_F(CuckooHashingSparseDpfPirServerTest,
       GetPublicParamsReturnsParamsPassedAtConstruction) {
  SetUpServer();

  EXPECT_THAT(
      server_->GetPublicParams().cuckoo_hashing_sparse_dpf_pir_server_params(),
      Truly([this](const auto& params) {
        return params.num_buckets() == params_.num_buckets() &&
               params.num_hash_functions() == params_.num_hash_functions() &&
               params.hash_family_config().hash_family() ==
                   params_.hash_family_config().hash_family() &&
               params.hash_family_config().seed() ==
                   params_.hash_family_config().seed();
      }));
}

}  // namespace
}  // namespace distributed_point_functions
