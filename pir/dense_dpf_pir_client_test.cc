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

#include "pir/dense_dpf_pir_client.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "dpf/internal/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "pir/dense_dpf_pir_database.h"
#include "pir/dense_dpf_pir_server.h"
#include "pir/private_information_retrieval.pb.h"
#include "pir/prng/aes_128_ctr_seeded_prng.h"
#include "pir/testing/encrypt_decrypt.h"
#include "pir/testing/mock_pir_database.h"

namespace distributed_point_functions {
namespace {

using dpf_internal::IsOkAndHolds;
using dpf_internal::StatusIs;
using pir_testing::CreateFakeHybridDecrypt;
using pir_testing::CreateFakeHybridEncrypt;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::testing::StartsWith;

constexpr int kTestDatabaseElements = 1234;

TEST(DenseDpfPirClient, CreateFailsIfEncrypterIsNull) {
  PirConfig config;
  config.mutable_dense_dpf_pir_config()->set_num_elements(
      kTestDatabaseElements);

  EXPECT_THAT(DenseDpfPirClient::Create(config, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("null")));
}

TEST(DenseDpfPirClient, CreateFailsIfConfigNotValid) {
  PirConfig config;

  EXPECT_THAT(DenseDpfPirClient::Create(config, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("valid DenseDpfPirConfig")));
}

TEST(DenseDpfPirClient, CreateFailsIfNumElementsIsZero) {
  PirConfig config;
  config.mutable_dense_dpf_pir_config()->set_num_elements(0);
  DPF_ASSERT_OK_AND_ASSIGN(auto hybrid_encrypt,
                           pir_testing::CreateFakeHybridEncrypt());
  auto encrypter = [&hybrid_encrypt](absl::string_view plain_pir_request,
                                     absl::string_view context_info) {
    return hybrid_encrypt->Encrypt(plain_pir_request, context_info);
  };

  EXPECT_THAT(
      DenseDpfPirClient::Create(config, encrypter),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("positive")));
}

TEST(DenseDpfPirClient, CreateSucceeds) {
  PirConfig config;
  config.mutable_dense_dpf_pir_config()->set_num_elements(
      kTestDatabaseElements);
  DPF_ASSERT_OK_AND_ASSIGN(auto hybrid_encrypt, CreateFakeHybridEncrypt());
  auto encrypter = [&hybrid_encrypt](absl::string_view plain_pir_request,
                                     absl::string_view context_info) {
    return hybrid_encrypt->Encrypt(plain_pir_request, context_info);
  };

  EXPECT_THAT(DenseDpfPirClient::Create(config, encrypter),
              IsOkAndHolds(NotNull()));
}

class DenseDpfPirClientTest : public ::testing::Test {
 protected:
  void SetUp() override {
    PirConfig config;
    config.mutable_dense_dpf_pir_config()->set_num_elements(
        kTestDatabaseElements);
    DPF_ASSERT_OK_AND_ASSIGN(hybrid_decrypt_, CreateFakeHybridDecrypt());
    DPF_ASSERT_OK_AND_ASSIGN(hybrid_encrypt_, CreateFakeHybridEncrypt());
    auto encrypter = [this](absl::string_view plain_pir_request,
                            absl::string_view context_info) {
      return hybrid_encrypt_->Encrypt(plain_pir_request, context_info);
    };
    DPF_ASSERT_OK_AND_ASSIGN(client_,
                             DenseDpfPirClient::Create(config, encrypter));
    DPF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> elements,
                             pir_testing::GenerateCountingStrings(
                                 kTestDatabaseElements, "Element "));
    DPF_ASSERT_OK_AND_ASSIGN(
        auto database1,
        pir_testing::CreateFakeDatabase<DenseDpfPirDatabase>(elements));
    auto sender = [this](const PirRequest& helper_request,
                         absl::AnyInvocable<void()> while_waiting) {
      while_waiting();
      return helper_->HandleRequest(helper_request);
    };
    DPF_ASSERT_OK_AND_ASSIGN(
        leader_, DenseDpfPirServer::CreateLeader(config, std::move(database1),
                                                 std::move(sender)));
    DPF_ASSERT_OK_AND_ASSIGN(
        auto database2,
        pir_testing::CreateFakeDatabase<DenseDpfPirDatabase>(elements));
    auto decrypter = [this](absl::string_view ciphertext,
                            absl::string_view context_info) {
      return hybrid_decrypt_->Decrypt(ciphertext, context_info);
    };
    DPF_ASSERT_OK_AND_ASSIGN(
        helper_, DenseDpfPirServer::CreateHelper(config, std::move(database2),
                                                 std::move(decrypter)));
  }

  std::unique_ptr<const crypto::tink::HybridDecrypt> hybrid_decrypt_;
  std::unique_ptr<const crypto::tink::HybridEncrypt> hybrid_encrypt_;
  std::unique_ptr<DenseDpfPirClient> client_;
  std::unique_ptr<DenseDpfPirServer> leader_;
  std::unique_ptr<DenseDpfPirServer> helper_;
};

TEST_F(DenseDpfPirClientTest, CreateRequestFailsIfIndexOutOfBounds) {
  EXPECT_THAT(
      client_->CreateRequest({kTestDatabaseElements}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("out of bounds")));
}

TEST_F(DenseDpfPirClientTest, CreateRequestFailsIfIndexIsNegative) {
  EXPECT_THAT(
      client_->CreateRequest({-1}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("negative")));
}

TEST_F(DenseDpfPirClientTest, HandleResponseFailsIfResponseHasWrongType) {
  PirResponse response;
  PirRequestClientState request_client_state;
  DPF_ASSERT_OK_AND_ASSIGN(
      *(request_client_state.mutable_dense_dpf_pir_request_client_state()
            ->mutable_one_time_pad_seed()),
      Aes128CtrSeededPrng::GenerateSeed());

  EXPECT_THAT(client_->HandleResponse(response, request_client_state),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("valid DpfPirResponse")));
}

TEST_F(DenseDpfPirClientTest, HandleResponseFailsIfPrivateKeyHasWrongType) {
  PirResponse response;
  response.mutable_dpf_pir_response()->add_masked_response("Test");
  PirRequestClientState request_client_state;

  EXPECT_THAT(client_->HandleResponse(response, request_client_state),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("valid DenseDpfPirRequestClientState")));
}

TEST_F(DenseDpfPirClientTest, HandleResponseFailsIfSeedIsMissing) {
  PirResponse response;
  response.mutable_dpf_pir_response()->add_masked_response("Test");
  PirRequestClientState request_client_state;
  request_client_state.mutable_dense_dpf_pir_request_client_state();

  EXPECT_THAT(client_->HandleResponse(response, request_client_state),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("one_time_pad_seed")));
}

TEST_F(DenseDpfPirClientTest, HandleResponseFailsIfNoResponse) {
  PirResponse response;
  response.mutable_dpf_pir_response();
  PirRequestClientState request_client_state;
  request_client_state.mutable_dense_dpf_pir_request_client_state()
      ->set_one_time_pad_seed("Test");

  EXPECT_THAT(client_->HandleResponse(response, request_client_state),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("masked_response")));
}

TEST_F(DenseDpfPirClientTest, HandleResponseAppliesOneTimePadCorrectly) {
  // Create OTP seed and PRG.
  DPF_ASSERT_OK_AND_ASSIGN(std::string one_time_pad_seed,
                           Aes128CtrSeededPrng::GenerateSeed());
  DPF_ASSERT_OK_AND_ASSIGN(auto prg,
                           Aes128CtrSeededPrng::Create(one_time_pad_seed));
  PirRequestClientState request_client_state;
  request_client_state.mutable_dense_dpf_pir_request_client_state()
      ->set_one_time_pad_seed(one_time_pad_seed);

  // Mask database with OTP and save result in a PirResponse.
  std::vector<std::string> database = {"Element 1", "Element 23"};
  PirResponse response;
  for (int i = 0; i < database.size(); ++i) {
    std::string masked_response = prg->GetRandomBytes(database[i].size());
    for (int j = 0; j < masked_response.size(); ++j) {
      masked_response[j] ^= database[i][j];
    }
    *(response.mutable_dpf_pir_response()->mutable_masked_response()->Add()) =
        std::move(masked_response);
  }

  // HandleResponse and check that the result equals `database`.
  EXPECT_THAT(client_->HandleResponse(response, request_client_state),
              IsOkAndHolds(ElementsAreArray(database)));
}

TEST_F(DenseDpfPirClientTest, TestPirEndToEnd) {
  PirRequest request;
  PirRequestClientState request_client_state;
  DPF_ASSERT_OK_AND_ASSIGN(std::tie(request, request_client_state),
                           client_->CreateRequest({23, 42}));
  DPF_ASSERT_OK_AND_ASSIGN(PirResponse response,
                           leader_->HandleRequest(request));
  DPF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::string> result,
      client_->HandleResponse(response, request_client_state));

  // Using StartsWith because of trailing null bytes.
  EXPECT_THAT(result, testing::ElementsAre(StartsWith("Element 23"),
                                           StartsWith("Element 42")));
}

}  // namespace
}  // namespace distributed_point_functions
