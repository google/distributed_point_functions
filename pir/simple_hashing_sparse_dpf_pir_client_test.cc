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

#include "pir/simple_hashing_sparse_dpf_pir_client.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "dpf/internal/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "pir/hashing/hash_family_config.pb.h"
#include "pir/private_information_retrieval.pb.h"
#include "pir/simple_hashed_dpf_pir_database.h"
#include "pir/simple_hashing_sparse_dpf_pir_server.h"
#include "pir/testing/encrypt_decrypt.h"
#include "pir/testing/mock_pir_database.h"
#include "tink/hybrid_encrypt.h"

namespace distributed_point_functions {
namespace {

constexpr int kTestDatabaseNumElements = 1234;
constexpr int kTestDatabaseNumBuckets = 123;
inline constexpr absl::string_view kTestHashFamilySeed = "kTestHashFamilySeed";

using dpf_internal::IsOkAndHolds;
using dpf_internal::StatusIs;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::testing::Optional;
using ::testing::StartsWith;

PirServerPublicParams GetDefaultParams() {
  SimpleHashingParams sh_params;
  sh_params.set_num_buckets(kTestDatabaseNumBuckets);
  *sh_params.mutable_hash_family_config()->mutable_seed() =
      std::string(kTestHashFamilySeed);
  sh_params.mutable_hash_family_config()->set_hash_family(
      HashFamilyConfig::HASH_FAMILY_SHA256);
  PirServerPublicParams params;
  *params.mutable_simple_hashing_sparse_dpf_pir_server_params() =
      std::move(sh_params);
  return params;
}

SimpleHashingSparseDpfPirClient::EncryptHelperRequestFn GetEncrypter() {
  static const auto hybrid_encrypt =
      pir_testing::CreateFakeHybridEncrypt().value();
  auto encrypter = [singleton = hybrid_encrypt.get()](
                       absl::string_view plain_pir_request,
                       absl::string_view context_info) {
    return singleton->Encrypt(plain_pir_request, context_info);
  };
  return encrypter;
}

TEST(SimpleHashingSparseDpfPirClient, CreateFailsIfEncrypterIsNull) {
  PirServerPublicParams params = GetDefaultParams();

  EXPECT_THAT(SimpleHashingSparseDpfPirClient::Create(params, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("null")));
}

TEST(SimpleHashingSparseDpfPirClient, CreateFailsIfParamsNotValid) {
  PirServerPublicParams params;

  EXPECT_THAT(SimpleHashingSparseDpfPirClient::Create(params, GetEncrypter()),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("valid")));
}

TEST(SimpleHashingSparseDpfPirClient, CreateFailsIfNumBucketsIsZero) {
  PirServerPublicParams params = GetDefaultParams();
  params.mutable_simple_hashing_sparse_dpf_pir_server_params()->set_num_buckets(
      0);

  EXPECT_THAT(
      SimpleHashingSparseDpfPirClient::Create(params, GetEncrypter()),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("positive")));
}

TEST(SimpleHashingSparseDpfPirClient, CreateSucceeds) {
  EXPECT_THAT(SimpleHashingSparseDpfPirClient::Create(GetDefaultParams(),
                                                      GetEncrypter()),
              IsOkAndHolds(NotNull()));
}

class SimpleHashingSparseDpfPirClientTest : public ::testing::Test {
 protected:
  void SetUp() override {
    PirConfig config;
    config.mutable_simple_hashing_sparse_dpf_pir_config()->set_hash_family(
        HashFamilyConfig::HASH_FAMILY_SHA256);
    config.mutable_simple_hashing_sparse_dpf_pir_config()->set_num_buckets(
        kTestDatabaseNumBuckets);
    DPF_ASSERT_OK_AND_ASSIGN(
        SimpleHashingParams params,
        SimpleHashingSparseDpfPirServer::GenerateParams(config));
    DPF_ASSERT_OK_AND_ASSIGN(keys_, pir_testing::GenerateCountingStrings(
                                        kTestDatabaseNumElements, "Key "));
    DPF_ASSERT_OK_AND_ASSIGN(values_, pir_testing::GenerateCountingStrings(
                                          kTestDatabaseNumElements, "Value "));
    std::vector<std::pair<std::string, std::string>> pairs;
    for (int i = 0; i < kTestDatabaseNumElements; ++i) {
      pairs.emplace_back(keys_[i], values_[i]);
    }
    SimpleHashedDpfPirDatabase::Builder builder, builder1;
    builder.SetParams(params);
    builder1.SetParams(params);
    DPF_ASSERT_OK_AND_ASSIGN(
        auto database,
        pir_testing::CreateFakeDatabase<SimpleHashedDpfPirDatabase>(pairs,
                                                                    &builder));
    DPF_ASSERT_OK_AND_ASSIGN(
        auto database1,
        pir_testing::CreateFakeDatabase<SimpleHashedDpfPirDatabase>(pairs,
                                                                    &builder1));
    DPF_ASSERT_OK_AND_ASSIGN(leader_,
                             SimpleHashingSparseDpfPirServer::CreateLeader(
                                 params, std::move(database),
                                 [this](auto request, auto while_waiting) {
                                   while_waiting();
                                   return helper_->HandleRequest(request);
                                 }));

    DPF_ASSERT_OK_AND_ASSIGN(auto decrypter,
                             pir_testing::CreateFakeHybridDecrypt());
    DPF_ASSERT_OK_AND_ASSIGN(
        helper_, SimpleHashingSparseDpfPirServer::CreateHelper(
                     params, std::move(database1),
                     [decrypter = std::move(decrypter)](auto encrypted_request,
                                                        auto context_string) {
                       return decrypter->Decrypt(encrypted_request,
                                                 context_string);
                     }));
    DPF_ASSERT_OK_AND_ASSIGN(client_,
                             SimpleHashingSparseDpfPirClient::Create(
                                 leader_->GetPublicParams(), GetEncrypter()));
  }
  std::unique_ptr<SimpleHashingSparseDpfPirClient> client_;
  std::unique_ptr<SimpleHashingSparseDpfPirServer> leader_, helper_;
  std::vector<std::string> keys_, values_;
};

TEST_F(SimpleHashingSparseDpfPirClientTest,
       FailsIfResponseIsNotADpfPirResponse) {
  PirResponse response;
  PirRequestClientState client_state;
  client_state.mutable_simple_hashing_sparse_dpf_pir_request_client_state();

  EXPECT_THAT(client_->HandleResponse(response, client_state),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("valid DpfPirResponse")));
}

TEST_F(SimpleHashingSparseDpfPirClientTest,
       FailsIfResponseIsNotASimpleHashingSparseDpfPirRequestClientState) {
  PirResponse response;
  response.mutable_dpf_pir_response();
  PirRequestClientState client_state;

  EXPECT_THAT(
      client_->HandleResponse(response, client_state),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("valid SimpleHashingSparseDpfPirRequestClientState")));
}

TEST_F(SimpleHashingSparseDpfPirClientTest, FailsIfNumberOfResponsesIsWrong) {
  std::vector<std::string> queries = {"Key 1", "Key 2"};
  PirRequest request;
  PirRequestClientState client_state;
  DPF_ASSERT_OK_AND_ASSIGN(std::tie(request, client_state),
                           client_->CreateRequest(queries));
  DPF_ASSERT_OK_AND_ASSIGN(PirResponse response,
                           leader_->HandleRequest(request));

  response.mutable_dpf_pir_response()->mutable_masked_response()->RemoveLast();

  EXPECT_THAT(client_->HandleResponse(response, client_state),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Number of responses")));
}

TEST_F(SimpleHashingSparseDpfPirClientTest, EndToEndSucceeds) {
  std::vector<std::string> queries = {"Key 1", "Key", "Key 42"};

  PirRequest request;
  PirRequestClientState client_state;
  DPF_ASSERT_OK_AND_ASSIGN(std::tie(request, client_state),
                           client_->CreateRequest(queries));
  DPF_ASSERT_OK_AND_ASSIGN(PirResponse response,
                           leader_->HandleRequest(request));
  DPF_ASSERT_OK_AND_ASSIGN(std::vector<absl::optional<std::string>> result,
                           client_->HandleResponse(response, client_state));

  EXPECT_EQ(result.size(), queries.size());
  EXPECT_THAT(result[0], Optional(StartsWith(values_[1])));
  EXPECT_EQ(result[1], absl::nullopt);
  EXPECT_THAT(result[2], Optional(StartsWith(values_[42])));
}

TEST_F(SimpleHashingSparseDpfPirClientTest,
       EndToEndSucceedsWithoutFingerprintForBackwardsCompatibility) {
  std::vector<std::string> queries = {"Key 1", "Key", "Key 42"};

  PirRequest request;
  PirRequestClientState client_state;
  DPF_ASSERT_OK_AND_ASSIGN(std::tie(request, client_state),
                           client_->CreateRequest(queries));

  request.mutable_dpf_pir_request()
      ->mutable_leader_request()
      ->mutable_plain_request()
      ->clear_seed_fingerprint();

  DPF_ASSERT_OK_AND_ASSIGN(PirResponse response,
                           leader_->HandleRequest(request));
  DPF_ASSERT_OK_AND_ASSIGN(std::vector<absl::optional<std::string>> result,
                           client_->HandleResponse(response, client_state));

  EXPECT_EQ(result.size(), queries.size());
  EXPECT_THAT(result[0], Optional(StartsWith(values_[1])));
  EXPECT_EQ(result[1], absl::nullopt);
  EXPECT_THAT(result[2], Optional(StartsWith(values_[42])));
}

}  // namespace
}  // namespace distributed_point_functions
