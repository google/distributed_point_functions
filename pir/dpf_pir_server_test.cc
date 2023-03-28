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

#include "pir/dpf_pir_server.h"

#include <memory>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "dpf/internal/status_matchers.h"
#include "dpf/status_macros.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "pir/dense_dpf_pir_database.h"
#include "pir/private_information_retrieval.pb.h"
#include "pir/prng/aes_128_ctr_seeded_prng.h"
#include "pir/testing/encrypt_decrypt.h"
#include "pir/testing/mock_pir_database.h"
#include "pir/testing/request_generator.h"

namespace distributed_point_functions {
namespace {

using dpf_internal::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::StartsWith;

constexpr int kTestDatabaseElements = 1234;
inline constexpr absl::string_view kEncryptionContextInfo = "DpfPirHelperTest";

// Inherit from DpfPirServer to expose protected methods for testing.
class DpfPirServerTestBase : public DpfPirServer {
 public:
  using DpfPirServer::MakeHelper, DpfPirServer::MakeLeader;
  using Database = DenseDpfPirDatabase::Interface;

  virtual void SetIndices(absl::Span<const int> indices) { indices_ = indices; }

 protected:
  static absl::StatusOr<const DenseDpfPirDatabase*> DatabaseSingleton() {
    static const auto database =
        []() -> absl::StatusOr<std::unique_ptr<Database>> {
      DPF_ASSIGN_OR_RETURN(std::vector<std::string> elements,
                           pir_testing::GenerateCountingStrings(
                               kTestDatabaseElements, "Element "));
      DPF_ASSIGN_OR_RETURN(
          std::unique_ptr<Database> database,
          pir_testing::CreateFakeDatabase<DenseDpfPirDatabase>(elements));
      return database;
    }();
    if (!database.ok()) {
      return database.status();
    }
    DenseDpfPirDatabase* dense_database =
        dynamic_cast<DenseDpfPirDatabase*>(database->get());
    if (!dense_database) {
      return absl::InternalError(
          "CreateFakeDatabase<DenseDpfPirDatabase> did not return a "
          "DenseDpfPirDatabase");
    }
    return dense_database;
  }

  absl::StatusOr<PirResponse> HandlePlainRequest(
      const PirRequest& request) const override {
    DPF_ASSIGN_OR_RETURN(const DenseDpfPirDatabase* database,
                         DatabaseSingleton());
    PirResponse response;
    for (int i = 0;
         i < request.dpf_pir_request().plain_request().dpf_key_size(); ++i) {
      if (indices_.size() <= i) {
        return absl::InternalError(
            "SetIndices must be called before handling any requests");
      }
      *(response.mutable_dpf_pir_response()->mutable_masked_response()->Add()) =
          std::string(database->content()[indices_[i]]);
    }
    return response;
  }

  const PirServerPublicParams& GetPublicParams() const override {
    return PirServerPublicParams::default_instance();
  }

  absl::Span<const int> indices_;
};

class DpfPirServerTest : public ::testing::Test, public DpfPirServerTestBase {};

TEST_F(DpfPirServerTest, MakeLeaderFailsIfSenderIsNull) {
  EXPECT_THAT(this->MakeLeader(nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("null")));
}

TEST_F(DpfPirServerTest, MakeLeaderSucceeds) {
  ForwardHelperRequestFn dummy_sender =
      [](const PirRequest& request,
         std::function<void()> while_waiting) -> absl::StatusOr<PirResponse> {
    return absl::UnimplementedError("Dummy");
  };

  EXPECT_THAT(this->MakeLeader(dummy_sender), StatusIs(absl::StatusCode::kOk));
  EXPECT_EQ(this->role(), Role::kLeader);
}

TEST_F(DpfPirServerTest, MakeHelperFailsIfDecrypterIsNull) {
  EXPECT_THAT(this->MakeHelper(nullptr, kEncryptionContextInfo),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("null")));
}

TEST_F(DpfPirServerTest, MakeHelperSucceeds) {
  DPF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<const crypto::tink::HybridDecrypt> hybrid_decrypt,
      pir_testing::CreateFakeHybridDecrypt());
  auto decrypter = [&hybrid_decrypt](absl::string_view ciphertext,
                                     absl::string_view context_info) {
    return hybrid_decrypt->Decrypt(ciphertext, context_info);
  };

  EXPECT_THAT(this->MakeHelper(decrypter, kEncryptionContextInfo),
              StatusIs(absl::StatusCode::kOk));
  EXPECT_EQ(this->role(), Role::kHelper);
}

class DpfPirLeaderTest : public DpfPirServerTest {
 protected:
  void SetUp() override {
    helper_ = std::make_unique<DpfPirServerTestBase>();
    DPF_ASSERT_OK_AND_ASSIGN(hybrid_decrypt_,
                             pir_testing::CreateFakeHybridDecrypt());
    auto decrypter = [this](absl::string_view ciphertext,
                            absl::string_view context_info) {
      return hybrid_decrypt_->Decrypt(ciphertext, context_info);
    };
    DPF_ASSERT_OK(helper_->MakeHelper(decrypter, kEncryptionContextInfo));
    auto default_sender = [this](const PirRequest& request,
                                 std::function<void()> while_waiting) {
      while_waiting();
      return helper_->HandleRequest(request);
    };
    SetSender(default_sender);
    DPF_ASSERT_OK_AND_ASSIGN(
        request_generator_, pir_testing::RequestGenerator::Create(
                                kTestDatabaseElements, kEncryptionContextInfo));
  }

  void SetIndices(absl::Span<const int> indices) override {
    helper_->SetIndices(indices);
    this->DpfPirServerTestBase::SetIndices(indices);
  }

  absl::StatusOr<PirResponse> HandlePlainRequest(
      const PirRequest& request) const override {
    DPF_ASSIGN_OR_RETURN(
        PirResponse response,
        this->DpfPirServerTestBase::HandlePlainRequest(request));
    // Let the Leader always return a string of 'X's of the correct length.
    // Used to check that the responses are XORed together correctly.
    for (int i = 0; i < response.dpf_pir_response().masked_response_size();
         ++i) {
      *(response.mutable_dpf_pir_response()->mutable_masked_response(i)) =
          std::string(response.dpf_pir_response().masked_response(i).size(),
                      'X');
    }
    return response;
  }

  void SetSender(ForwardHelperRequestFn sender) {
    DPF_ASSERT_OK(this->MakeLeader(sender));
  }

  std::unique_ptr<const crypto::tink::HybridDecrypt> hybrid_decrypt_;
  std::unique_ptr<DpfPirServerTestBase> helper_;
  std::unique_ptr<pir_testing::RequestGenerator> request_generator_;
};

TEST_F(DpfPirLeaderTest, HandleRequestFailsIfRequestNotLeaderRequest) {
  PirRequest request;
  request.mutable_dpf_pir_request();

  EXPECT_THAT(
      this->HandleRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("LeaderRequest")));
}

TEST_F(DpfPirLeaderTest, HandleRequestFailsIfPlainRequestMissing) {
  PirRequest request;
  request.mutable_dpf_pir_request()
      ->mutable_leader_request()
      ->mutable_encrypted_helper_request();

  EXPECT_THAT(
      this->HandleRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("plain_request")));
}

TEST_F(DpfPirLeaderTest, HandleRequestFailsIfEncryptedHelperRequestMissing) {
  PirRequest request;
  request.mutable_dpf_pir_request()
      ->mutable_leader_request()
      ->mutable_plain_request();

  EXPECT_THAT(this->HandleRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("encrypted_helper_request")));
}

TEST_F(DpfPirLeaderTest, HandleRequestFailsIfProcessPlainRequestNotCalled) {
  std::vector<int> indices{23};
  SetIndices(indices);
  PirRequest request;
  DPF_ASSERT_OK_AND_ASSIGN(
      *(request.mutable_dpf_pir_request()->mutable_leader_request()),
      request_generator_->CreateDpfPirLeaderRequest(indices));

  // Dummy sender that does not actually call `while_waiting`. Create should
  // succeed, but HandleRequest should not.
  ForwardHelperRequestFn dummy_sender =
      [this](
          const PirRequest& request,
          std::function<void()> while_waiting) -> absl::StatusOr<PirResponse> {
    return helper_->HandleRequest(request);
  };
  SetSender(dummy_sender);

  // Require this to fail, to ensure the class is used correctly.
  EXPECT_THAT(this->HandleRequest(request),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("while_waiting")));
}

TEST_F(DpfPirLeaderTest, HandleRequestFailsIfNumbersOfResponsesDontMatch) {
  std::vector<int> indices{23, 24};
  SetIndices(indices);
  PirRequest request;
  DPF_ASSERT_OK_AND_ASSIGN(
      *(request.mutable_dpf_pir_request()->mutable_leader_request()),
      // Create two requests so we can drop one.
      request_generator_->CreateDpfPirLeaderRequest(indices));

  ForwardHelperRequestFn corrupted_sender =
      [this](
          const PirRequest& helper_request,
          std::function<void()> while_waiting) -> absl::StatusOr<PirResponse> {
    while_waiting();
    DPF_ASSIGN_OR_RETURN(PirResponse helper_response,
                         helper_->HandleRequest(helper_request));
    // Remove last response from helper.
    helper_response.mutable_dpf_pir_response()
        ->mutable_masked_response()
        ->RemoveLast();
    return helper_response;
  };
  SetSender(corrupted_sender);

  // Handle the request, and check that the internal error is caught.
  EXPECT_THAT(
      this->HandleRequest(request),
      StatusIs(absl::StatusCode::kInternal,
               AllOf(HasSubstr("number of responses"), HasSubstr("Helper (=1)"),
                     HasSubstr("Leader (=2)"))));
}

TEST_F(DpfPirLeaderTest, HandleRequestFailsIfResponseSizeDoesntMatch) {
  std::vector<int> indices{23};
  SetIndices(indices);
  PirRequest request;
  DPF_ASSERT_OK_AND_ASSIGN(
      *(request.mutable_dpf_pir_request()->mutable_leader_request()),
      request_generator_->CreateDpfPirLeaderRequest(indices));

  ForwardHelperRequestFn corrupted_sender =
      [this](
          const PirRequest& helper_request,
          std::function<void()> while_waiting) -> absl::StatusOr<PirResponse> {
    while_waiting();
    DPF_ASSIGN_OR_RETURN(PirResponse helper_response,
                         helper_->HandleRequest(helper_request));
    // Remove last character from Helper's response.
    helper_response.mutable_dpf_pir_response()
        ->mutable_masked_response(0)
        ->pop_back();
    return helper_response;
  };
  SetSender(corrupted_sender);

  // Handle the request, and check that the internal error is caught.
  EXPECT_THAT(this->HandleRequest(request),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("size mismatch at index 0")));
}

TEST_F(DpfPirLeaderTest, HandleRequestSucceeds) {
  std::vector<int> indices{23, 24};
  SetIndices(indices);
  PirRequest request;
  DPF_ASSERT_OK_AND_ASSIGN(
      *(request.mutable_dpf_pir_request()->mutable_leader_request()),
      request_generator_->CreateDpfPirLeaderRequest(indices));

  // Run the PIR and check that the response is as expected.
  DPF_ASSERT_OK_AND_ASSIGN(const DenseDpfPirDatabase* database,
                           DatabaseSingleton());
  DPF_ASSERT_OK_AND_ASSIGN(PirResponse response, this->HandleRequest(request));
  std::vector<std::string> wanted(indices_.size());
  for (int i = 0; i < wanted.size(); ++i) {
    wanted[i] = std::string(database->content()[indices_[i]]);
  }
  DPF_ASSERT_OK_AND_ASSIGN(
      auto prg, Aes128CtrSeededPrng::Create(request_generator_->otp_seed()));
  for (int i = 0; i < response.dpf_pir_response().masked_response_size(); ++i) {
    std::string current_response =
        response.dpf_pir_response().masked_response(i);
    ASSERT_GE(current_response.size(),
              wanted[i].size());  // May be padded with null bytes.
    std::string otp = prg->GetRandomBytes(current_response.size());
    for (int j = 0; j < current_response.size(); ++j) {
      current_response[j] ^= otp[j] ^ 'X';
      if (j > wanted[i].size()) {
        EXPECT_EQ(current_response[j], '\0');
      }
    }
    EXPECT_THAT(current_response, StartsWith(wanted[i]));
  }
}

class DpfPirHelperTest : public DpfPirServerTest {
 protected:
  void SetUp() override {
    DPF_ASSERT_OK_AND_ASSIGN(hybrid_decrypt_,
                             pir_testing::CreateFakeHybridDecrypt());
    auto decrypter = [this](absl::string_view ciphertext,
                            absl::string_view context_info) {
      return hybrid_decrypt_->Decrypt(ciphertext, context_info);
    };
    DPF_ASSERT_OK(this->MakeHelper(decrypter, kEncryptionContextInfo));
  }

  std::unique_ptr<const crypto::tink::HybridDecrypt> hybrid_decrypt_;
};

TEST_F(DpfPirHelperTest,
       HandleHelperRequestFailsWhenRequestIsNotEncryptedHelperRequest) {
  PirRequest request;
  request.mutable_dpf_pir_request();

  EXPECT_THAT(this->HandleRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("EncryptedHelperRequest")));
}

TEST_F(DpfPirHelperTest, HandleRequestFailsWhenDecryptionFails) {
  PirRequest request;
  request.mutable_dpf_pir_request()
      ->mutable_encrypted_helper_request()
      ->set_encrypted_request("not a valid ciphertext");

  EXPECT_FALSE(this->HandleRequest(request).ok());
}

TEST_F(DpfPirHelperTest,
       HandleRequestFailsWhenCiphertextDoesntContainHelperRequest) {
  const absl::string_view not_a_proto = "Not a valid HelperRequest";
  DPF_ASSERT_OK_AND_ASSIGN(auto encrypter,
                           pir_testing::CreateFakeHybridEncrypt());

  PirRequest request;
  DPF_ASSERT_OK_AND_ASSIGN(
      *(request.mutable_dpf_pir_request()
            ->mutable_encrypted_helper_request()
            ->mutable_encrypted_request()),
      encrypter->Encrypt(not_a_proto, kEncryptionContextInfo));

  EXPECT_THAT(
      this->HandleRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("HelperRequest")));
}

TEST_F(DpfPirHelperTest, HandleRequestSucceeds) {
  DPF_ASSERT_OK_AND_ASSIGN(auto request_generator,
                           pir_testing::RequestGenerator::Create(
                               kTestDatabaseElements, kEncryptionContextInfo));
  std::vector<int> indices{23, 24};
  SetIndices(indices);

  // Set up encrypted inner request. Create two batched requests (to check that
  // a single seed works correctly for multiple batched requests).
  DpfPirRequest::HelperRequest inner_request;
  DPF_ASSERT_OK_AND_ASSIGN(
      std::tie(*(inner_request.mutable_plain_request()), std::ignore),
      request_generator->CreateDpfPirPlainRequests(indices));
  DPF_ASSERT_OK_AND_ASSIGN(std::string otp_seed,
                           Aes128CtrSeededPrng::GenerateSeed());
  inner_request.set_one_time_pad_seed(otp_seed);

  // Encrypt inner request.
  DPF_ASSERT_OK_AND_ASSIGN(auto encrypter,
                           pir_testing::CreateFakeHybridEncrypt());
  PirRequest request;
  DPF_ASSERT_OK_AND_ASSIGN(*(request.mutable_dpf_pir_request()
                                 ->mutable_encrypted_helper_request()
                                 ->mutable_encrypted_request()),
                           encrypter->Encrypt(inner_request.SerializeAsString(),
                                              kEncryptionContextInfo));

  // Set up a plain request for checking the result.
  PirRequest plain_request;
  *(plain_request.mutable_dpf_pir_request()->mutable_plain_request()) =
      inner_request.plain_request();

  // Check that `request` results in the same response as `plain_request`,
  // except with the expanded one time pad added to the result.
  DPF_ASSERT_OK_AND_ASSIGN(PirResponse result, this->HandleRequest(request));
  DPF_ASSERT_OK_AND_ASSIGN(PirResponse plain_result,
                           this->HandlePlainRequest(plain_request));
  DPF_ASSERT_OK_AND_ASSIGN(auto prng, Aes128CtrSeededPrng::Create(otp_seed));
  for (int i = 0; i < result.dpf_pir_response().masked_response_size(); ++i) {
    std::string expected = plain_result.dpf_pir_response().masked_response(i);
    std::string otp = prng->GetRandomBytes(expected.size());
    for (int j = 0; j < expected.size(); ++j) {
      expected[j] ^= otp[j];
    }
    EXPECT_EQ(result.dpf_pir_response().masked_response()[i], expected);
  }
}

}  // namespace
}  // namespace distributed_point_functions
