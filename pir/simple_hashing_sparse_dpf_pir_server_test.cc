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

#include "pir/simple_hashing_sparse_dpf_pir_server.h"

#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "dpf/internal/status_matchers.h"
#include "gmock/gmock.h"
#include "google/protobuf/io/coded_stream.h"
#include "gtest/gtest.h"
#include "pir/hashing/hash_family.h"
#include "pir/hashing/hash_family_config.h"
#include "pir/hashing/hash_family_config.pb.h"
#include "pir/private_information_retrieval.pb.h"
#include "pir/simple_hashed_dpf_pir_database.h"
#include "pir/testing/encrypt_decrypt.h"
#include "pir/testing/mock_pir_database.h"
#include "pir/testing/request_generator.h"

namespace distributed_point_functions {
namespace {

constexpr int kNumElements = 123;
constexpr int kNumBuckets = 11;
constexpr int kValueSize = 16;
constexpr HashFamilyConfig::HashFamily kHashFamily =
    HashFamilyConfig::HASH_FAMILY_SHA256;

using dpf_internal::IsOkAndHolds;
using dpf_internal::StatusIs;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::testing::Truly;
using Database = SimpleHashingSparseDpfPirServer::Database;

TEST(SimpleHashingSparseDpfPirServer, GenerateParamsFailsWhenConfigIsInvalid) {
  PirConfig config;

  EXPECT_THAT(SimpleHashingSparseDpfPirServer::GenerateParams(config),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("valid SimpleHashingSparseDpfPirConfig")));
}

TEST(SimpleHashingSparseDpfPirServer,
     GenerateParamsFailsWhenHashFamilyIsNotSet) {
  PirConfig config;
  config.mutable_simple_hashing_sparse_dpf_pir_config()->set_num_buckets(
      kNumBuckets);

  EXPECT_THAT(
      SimpleHashingSparseDpfPirServer::GenerateParams(config),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("hash_family")));
}

TEST(SimpleHashingSparseDpfPirServer,
     GenerateParamsFailsWhenNumBucketsIsNotSet) {
  PirConfig config;
  config.mutable_simple_hashing_sparse_dpf_pir_config()->set_hash_family(
      kHashFamily);

  EXPECT_THAT(
      SimpleHashingSparseDpfPirServer::GenerateParams(config),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("num_buckets")));
}

class SimpleHashingSparseDpfPirServerTest : public testing::Test {
 protected:
  void SetUpConfig() {
    config_.mutable_simple_hashing_sparse_dpf_pir_config()->set_num_buckets(
        kNumBuckets);
    config_.mutable_simple_hashing_sparse_dpf_pir_config()->set_hash_family(
        kHashFamily);
  }

  void SetUpParams() {
    SetUpConfig();
    DPF_ASSERT_OK_AND_ASSIGN(
        params_, SimpleHashingSparseDpfPirServer::GenerateParams(config_));
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
    SimpleHashedDpfPirDatabase::Builder builder;
    builder.SetParams(params_);
    DPF_ASSERT_OK_AND_ASSIGN(
        database_, pir_testing::CreateFakeDatabase<SimpleHashedDpfPirDatabase>(
                       pairs, &builder));
  }

  void SetUpServer() {
    SetUpDatabase();
    DPF_ASSERT_OK_AND_ASSIGN(server_,
                             SimpleHashingSparseDpfPirServer::CreatePlain(
                                 params_, std::move(database_)));
  }

  PirConfig config_;
  SimpleHashingParams params_;
  std::vector<std::string> keys_, values_;
  std::unique_ptr<Database> database_;
  std::unique_ptr<SimpleHashingSparseDpfPirServer> server_;
};

TEST_F(SimpleHashingSparseDpfPirServerTest, GenerateParamsReturnsValidParams) {
  SetUpConfig();
  config_.mutable_simple_hashing_sparse_dpf_pir_config()->set_num_buckets(
      kNumBuckets);
  config_.mutable_simple_hashing_sparse_dpf_pir_config()->set_hash_family(
      HashFamilyConfig::HASH_FAMILY_SHA256);

  EXPECT_THAT(SimpleHashingSparseDpfPirServer::GenerateParams(config_),
              IsOkAndHolds(Truly([](auto& params) {
                return params.num_buckets() == kNumBuckets &&
                       !params.hash_family_config().seed().empty() &&
                       !absl::c_all_of(params.hash_family_config().seed(),
                                       [](char c) { return c == '\0'; });
              })));
}

TEST_F(SimpleHashingSparseDpfPirServerTest,
       CreatePlainFailsWhenNumBucketsIsZero) {
  SetUpDatabase();

  params_.set_num_buckets(0);

  EXPECT_THAT(
      SimpleHashingSparseDpfPirServer::CreatePlain(params_,
                                                   std::move(database_)),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("num_buckets")));
}

TEST_F(SimpleHashingSparseDpfPirServerTest,
       CreatePlainFailsWhenHashFamilyIsUnspecified) {
  SetUpDatabase();

  params_.mutable_hash_family_config()->set_hash_family(
      HashFamilyConfig::default_instance().hash_family());

  EXPECT_THAT(
      SimpleHashingSparseDpfPirServer::CreatePlain(params_,
                                                   std::move(database_)),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("hash_family")));
}

TEST_F(SimpleHashingSparseDpfPirServerTest,
       CreatePlainFailsWhenDatabaseIsNull) {
  SetUpParams();

  EXPECT_THAT(SimpleHashingSparseDpfPirServer::CreatePlain(params_, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("null")));
}

TEST_F(SimpleHashingSparseDpfPirServerTest,
       CreatePlainFailsWhenumBucketsDoesNotMatchNumSelectionBits) {
  SetUpDatabase();

  params_.set_num_buckets(database_->num_selection_bits() + 1);

  EXPECT_THAT(SimpleHashingSparseDpfPirServer::CreatePlain(
                  params_, std::move(database_)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("selection bits")));
}

TEST_F(SimpleHashingSparseDpfPirServerTest, CreateLeaderSucceeds) {
  SetUpDatabase();
  auto dummy_sender = [](const PirRequest& request,
                         absl::AnyInvocable<void()> while_waiting)
      -> absl::StatusOr<PirResponse> {
    return absl::UnimplementedError("Dummy");
  };

  EXPECT_THAT(SimpleHashingSparseDpfPirServer::CreateLeader(
                  params_, std::move(database_), std::move(dummy_sender)),
              IsOkAndHolds(NotNull()));
}

TEST_F(SimpleHashingSparseDpfPirServerTest, CreateHelperSucceeds) {
  SetUpDatabase();
  DPF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<const crypto::tink::HybridDecrypt> hybrid_decrypt,
      pir_testing::CreateFakeHybridDecrypt());
  auto decrypter = [&hybrid_decrypt](absl::string_view ciphertext,
                                     absl::string_view context_info) {
    return hybrid_decrypt->Decrypt(ciphertext, context_info);
  };

  EXPECT_THAT(SimpleHashingSparseDpfPirServer::CreateHelper(
                  params_, std::move(database_), decrypter),
              IsOkAndHolds(NotNull()));
}

TEST_F(SimpleHashingSparseDpfPirServerTest, CreatePlainSucceeds) {
  SetUpDatabase();

  EXPECT_THAT(SimpleHashingSparseDpfPirServer::CreatePlain(
                  params_, std::move(database_)),
              IsOkAndHolds(NotNull()));
}

TEST_F(SimpleHashingSparseDpfPirServerTest,
       HandlRequestFailsWhenSeedFingerprintDoesNotMatch) {
  // Create two servers.
  SetUpDatabase();
  DPF_ASSERT_OK_AND_ASSIGN(auto server1,
                           SimpleHashingSparseDpfPirServer::CreatePlain(
                               params_, std::move(database_)));
  // Generate a request.
  DPF_ASSERT_OK_AND_ASSIGN(
      auto request_generator,
      pir_testing::RequestGenerator::Create(
          params_.num_buckets(),
          SimpleHashingSparseDpfPirServer::kEncryptionContextInfo));
  PirRequest request;
  DPF_ASSERT_OK_AND_ASSIGN(
      std::tie(*request.mutable_dpf_pir_request()->mutable_plain_request(),
               std::ignore),
      request_generator->CreateDpfPirPlainRequests({0}));

  int wrong_fingerprint = 123;
  request.mutable_dpf_pir_request()
      ->mutable_plain_request()
      ->set_seed_fingerprint(wrong_fingerprint);

  EXPECT_THAT(server1->HandleRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("seed_fingerprint")));
}

TEST_F(SimpleHashingSparseDpfPirServerTest, HandleRequestSucceeds) {
  // Create two servers.
  SetUpDatabase();
  DPF_ASSERT_OK_AND_ASSIGN(auto server1,
                           SimpleHashingSparseDpfPirServer::CreatePlain(
                               params_, std::move(database_)));
  SetUpDatabase();
  DPF_ASSERT_OK_AND_ASSIGN(auto server2,
                           SimpleHashingSparseDpfPirServer::CreatePlain(
                               params_, std::move(database_)));

  // Hash the client's query with each hash function.
  DPF_ASSERT_OK_AND_ASSIGN(
      HashFamily hash_family,
      CreateHashFamilyFromConfig(params_.hash_family_config()));
  DPF_ASSERT_OK_AND_ASSIGN(std::vector<HashFunction> hash_functions,
                           CreateHashFunctions(std::move(hash_family), 1));
  constexpr int query_index = 42;
  int index = hash_functions[0](keys_[query_index], params_.num_buckets());

  // Generate plain requests for `indices`.
  DPF_ASSERT_OK_AND_ASSIGN(
      auto request_generator,
      pir_testing::RequestGenerator::Create(
          params_.num_buckets(),
          SimpleHashingSparseDpfPirServer::kEncryptionContextInfo));
  PirRequest request1, request2;
  DPF_ASSERT_OK_AND_ASSIGN(
      std::tie(*request1.mutable_dpf_pir_request()->mutable_plain_request(),
               *request2.mutable_dpf_pir_request()->mutable_plain_request()),
      request_generator->CreateDpfPirPlainRequests({index}));

  // Obtain a response from each server and add them up.
  PirResponse response1, response2;
  std::vector<std::string> response_keys;
  DPF_ASSERT_OK_AND_ASSIGN(response1, server1->HandleRequest(request1));
  DPF_ASSERT_OK_AND_ASSIGN(response2, server2->HandleRequest(request2));
  ASSERT_EQ(response1.dpf_pir_response().masked_response_size(),
            response2.dpf_pir_response().masked_response_size());
  ASSERT_EQ(response1.dpf_pir_response().masked_response_size(), 1);
  ASSERT_EQ(response1.dpf_pir_response().masked_response(0).size(),
            response2.dpf_pir_response().masked_response(0).size());
  std::string response_str = response1.dpf_pir_response().masked_response(0);
  for (int i = 0; i < response_str.size(); ++i) {
    response_str[i] ^= response2.dpf_pir_response().masked_response(0)[i];
  }

  // Deserialize the result and check that key and value match. We need to use a
  // CodedInputStream here to handle the null bytes at the end of the string.
  HashedPirDatabaseBucket result;
  ::google::protobuf::io::CodedInputStream coded_stream(
      reinterpret_cast<const uint8_t*>(response_str.data()),
      response_str.size());
  EXPECT_TRUE(result.ParseFromCodedStream(&coded_stream));
  ASSERT_EQ(result.keys_size(), result.values_size());

  bool found = false;
  for (int i = 0; i < result.keys_size(); ++i) {
    if (result.keys(i) == keys_[query_index]) {
      found = true;
      EXPECT_EQ(result.values(i), values_[query_index]);
    }
  }
  EXPECT_TRUE(found);
}

TEST_F(SimpleHashingSparseDpfPirServerTest,
       HandlePlainRequestCanBeCalledConcurrently) {
  SetUpServer();
  std::vector<int> indices = {0};
  constexpr int kNumThreads = 1024;

  // Create plain request for `indices`.
  DPF_ASSERT_OK_AND_ASSIGN(
      auto request_generator,
      pir_testing::RequestGenerator::Create(
          params_.num_buckets(),
          SimpleHashingSparseDpfPirServer::kEncryptionContextInfo));
  PirRequest request;
  DPF_ASSERT_OK_AND_ASSIGN(
      std::tie(*request.mutable_dpf_pir_request()->mutable_plain_request(),
               std::ignore),
      request_generator->CreateDpfPirPlainRequests(indices));

  auto do_handle_request = [&request, &server = server_]() {
    DPF_ASSERT_OK_AND_ASSIGN(PirResponse response,
                             server->HandleRequest(request));
  };

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(do_handle_request);
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(SimpleHashingSparseDpfPirServerTest,
       HandlePlainRequestFailsWhenRequestIsNotDpfPirRequest) {
  SetUpServer();
  PirRequest request;

  EXPECT_THAT(server_->HandleRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("valid DpfPirRequest")));
}

TEST_F(SimpleHashingSparseDpfPirServerTest,
       HandlePlainRequestFailsWhenRequestIsNotPlainRequest) {
  SetUpServer();
  PirRequest request;
  request.mutable_dpf_pir_request();

  EXPECT_THAT(server_->HandleRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("valid DpfPirRequest::PlainRequest")));
}

TEST_F(SimpleHashingSparseDpfPirServerTest,
       HandlePlainRequestFailsWhenDpfKeyIsEmpty) {
  SetUpServer();
  PirRequest request;
  request.mutable_dpf_pir_request()->mutable_plain_request();

  EXPECT_THAT(
      server_->HandleRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("dpf_key")));
}

TEST_F(SimpleHashingSparseDpfPirServerTest,
       GetPublicParamsReturnsParamsPassedAtConstruction) {
  SetUpServer();

  EXPECT_THAT(
      server_->GetPublicParams().simple_hashing_sparse_dpf_pir_server_params(),
      Truly([this](const auto& params) {
        return params.num_buckets() == params_.num_buckets() &&
               params.hash_family_config().hash_family() ==
                   params_.hash_family_config().hash_family() &&
               params.hash_family_config().seed() ==
                   params_.hash_family_config().seed();
      }));
}

}  // namespace
}  // namespace distributed_point_functions
