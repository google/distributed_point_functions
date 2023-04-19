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

#include "pir/dense_dpf_pir_server.h"

#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "dpf/distributed_point_function.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/internal/status_matchers.h"
#include "dpf/xor_wrapper.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "pir/private_information_retrieval.pb.h"
#include "pir/testing/encrypt_decrypt.h"
#include "pir/testing/mock_pir_database.h"

namespace distributed_point_functions {
namespace {

using dpf_internal::IsOkAndHolds;
using dpf_internal::StatusIs;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::testing::Return;
using MockDenseDpfPirDatbase =
    pir_testing::MockPirDatabase<XorWrapper<absl::uint128>, std::string>;

constexpr int kTestDatabaseElements = 1234;
constexpr int kBitsPerBlock = 128;

TEST(DenseDpfPirServer, CreateSucceeds) {
  PirConfig config;
  config.mutable_dense_dpf_pir_config()->set_num_elements(
      kTestDatabaseElements);
  auto database = std::make_unique<MockDenseDpfPirDatbase>();

  EXPECT_CALL(*database, size()).WillOnce(Return(kTestDatabaseElements));

  absl::StatusOr<std::unique_ptr<DenseDpfPirServer>> server =
      DenseDpfPirServer::CreatePlain(config, std::move(database));

  DPF_EXPECT_OK(server);
}

TEST(DenseDpfPirServer, CreateLeaderSucceeds) {
  PirConfig config;
  config.mutable_dense_dpf_pir_config()->set_num_elements(
      kTestDatabaseElements);
  auto database = std::make_unique<MockDenseDpfPirDatbase>();

  EXPECT_CALL(*database, size()).WillOnce(Return(kTestDatabaseElements));

  auto dummy_sender = [](const PirRequest& request,
                         absl::AnyInvocable<void()> while_waiting)
      -> absl::StatusOr<PirResponse> {
    return absl::UnimplementedError("Dummy");
  };

  EXPECT_THAT(DenseDpfPirServer::CreateLeader(config, std::move(database),
                                              std::move(dummy_sender)),
              IsOkAndHolds(NotNull()));
}

TEST(DenseDpfPirServer, CreateHelperSucceeds) {
  PirConfig config;
  config.mutable_dense_dpf_pir_config()->set_num_elements(
      kTestDatabaseElements);
  auto database = std::make_unique<MockDenseDpfPirDatbase>();

  EXPECT_CALL(*database, size()).WillOnce(Return(kTestDatabaseElements));

  DPF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<const crypto::tink::HybridDecrypt> hybrid_decrypt,
      pir_testing::CreateFakeHybridDecrypt());
  auto decrypter = [&hybrid_decrypt](absl::string_view ciphertext,
                                     absl::string_view context_info) {
    return hybrid_decrypt->Decrypt(ciphertext, context_info);
  };

  EXPECT_THAT(DenseDpfPirServer::CreateHelper(config, std::move(database),
                                              std::move(decrypter)),
              IsOkAndHolds(NotNull()));
}

TEST(DenseDpfPirServer, CreateFailsIfConfigUninitialized) {
  PirConfig config;
  auto database = std::make_unique<MockDenseDpfPirDatbase>();

  EXPECT_THAT(DenseDpfPirServer::CreatePlain(config, std::move(database)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("DenseDpfPirConfig")));
}

TEST(DenseDpfPirServer, CreateFailsIfDatabaseIsNull) {
  PirConfig config;
  config.mutable_dense_dpf_pir_config()->set_num_elements(
      kTestDatabaseElements);

  EXPECT_THAT(DenseDpfPirServer::CreatePlain(config, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("null")));
}

TEST(DenseDpfPirServer, CreateFailsIfDatabaseSizeIsZero) {
  PirConfig config;
  config.mutable_dense_dpf_pir_config()->set_num_elements(0);
  auto database = std::make_unique<MockDenseDpfPirDatbase>();

  EXPECT_THAT(
      DenseDpfPirServer::CreatePlain(config, std::move(database)),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("positive")));
}

TEST(DenseDpfPirServer, CreateFailsIfDatabaseSizeDoesNotMatchConfig) {
  PirConfig config;
  config.mutable_dense_dpf_pir_config()->set_num_elements(
      kTestDatabaseElements);
  auto database = std::make_unique<MockDenseDpfPirDatbase>();

  EXPECT_CALL(*database, size()).WillOnce(Return(kTestDatabaseElements + 1));

  EXPECT_THAT(DenseDpfPirServer::CreatePlain(config, std::move(database)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("size does not match")));
}

class DenseDpfPirServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a config and a database.
    PirConfig config;
    config.mutable_dense_dpf_pir_config()->set_num_elements(
        kTestDatabaseElements);
    content_.reserve(kTestDatabaseElements);
    content_views_.reserve(kTestDatabaseElements);
    for (int i = 0; i < kTestDatabaseElements; ++i) {
      content_.push_back(absl::StrCat("Element ", i));
      content_views_.push_back(content_.back());
    }

    // Create a DPF instance.
    DpfParameters parameters;
    parameters.mutable_value_type()->mutable_xor_wrapper()->set_bitsize(
        kBitsPerBlock);
    parameters.set_log_domain_size(
        static_cast<int>(std::ceil(std::log2(kTestDatabaseElements))));
    DPF_ASSERT_OK_AND_ASSIGN(dpf_,
                             DistributedPointFunction::Create(parameters));

    // Set up the mock database to use the test implementation.
    auto database = std::make_unique<MockDenseDpfPirDatbase>();
    ON_CALL(*database, size()).WillByDefault(Return(content_views_.size()));
    ON_CALL(*database, InnerProductWith(::testing::_))
        .WillByDefault(
            [this](auto x) -> auto { return this->InnerProductWith(x); });

    // Create the server.
    DPF_ASSERT_OK_AND_ASSIGN(
        server_, DenseDpfPirServer::CreatePlain(config, std::move(database)));

    // Expose database so we can set expectations in tests.
    database_ =
        dynamic_cast<const MockDenseDpfPirDatbase*>(&(server_->database()));
    ASSERT_NE(database_, nullptr);
  }

  void SetupFakeRequest(int index, PirRequest& request) const {
    absl::uint128 alpha = index / kBitsPerBlock;
    XorWrapper<absl::uint128> beta(absl::uint128{1} << (index % kBitsPerBlock));
    DPF_ASSERT_OK_AND_ASSIGN(std::tie(*(request.mutable_dpf_pir_request()
                                            ->mutable_plain_request()
                                            ->mutable_dpf_key()
                                            ->Add()),
                                      std::ignore),
                             dpf_->GenerateKeys(alpha, beta));
  }

  // Inner product implementation for testing.
  std::vector<std::string> InnerProductWith(
      absl::Span<const std::vector<XorWrapper<absl::uint128>>> selections)
      const {
    std::vector<std::string> result;
    result.reserve(selections.size());
    for (const auto& current_selections : selections) {
      std::string current_result;
      for (int i = 0; i < current_selections.size(); ++i) {
        for (int j = 0; j < kBitsPerBlock; ++j) {
          if (i * kBitsPerBlock + j >= content_.size()) {
            break;
          }
          if (current_selections[i].value() & (absl::uint128{1} << j) != 0) {
            current_result = XorStrings(std::move(current_result),
                                        content_[i * kBitsPerBlock + j]);
          }
        }
      }
      result.push_back(std::move(current_result));
    }
    return result;
  }

  std::unique_ptr<DenseDpfPirServer> server_;
  const MockDenseDpfPirDatbase* database_;
  std::vector<std::string> content_;
  std::vector<absl::string_view> content_views_;
  std::unique_ptr<DistributedPointFunction> dpf_;

 private:
  // Helper function that XORs b onto a and returns a.
  std::string XorStrings(std::string a, const std::string& b) const {
    if (b.size() > a.size()) {
      a.resize(b.size(), '\0');
    }
    for (int i = 0; i < b.size(); ++i) {
      a[i] ^= b[i];
    }
    return a;
  }
};

TEST_F(DenseDpfPirServerTest, HandleRequestFailsIfRequestIsNotDpfPirRequest) {
  PirRequest request;

  EXPECT_THAT(
      server_->HandleRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("DpfPirRequest")));
}

TEST_F(DenseDpfPirServerTest, HandleRequestFailsIfRequestIsNotPlainRequest) {
  PirRequest request;
  request.mutable_dpf_pir_request();

  EXPECT_THAT(
      server_->HandleRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("PlainRequest")));
}

TEST_F(DenseDpfPirServerTest, HandleRequestFailsIfRequestIsEmpty) {
  PirRequest request;
  request.mutable_dpf_pir_request()->mutable_plain_request();

  EXPECT_THAT(server_->HandleRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("empty")));
}

TEST_F(DenseDpfPirServerTest, HandleRequestFailsIfRequestKeyHasWrongSize) {
  PirRequest request;
  SetupFakeRequest(123, request);

  request.mutable_dpf_pir_request()
      ->mutable_plain_request()
      ->mutable_dpf_key(0)
      ->mutable_correction_words()
      ->Add();

  EXPECT_THAT(
      server_->HandleRequest(request),
      StatusIs(
          absl::StatusCode::kInvalidArgument));  // Status message depends on
                                                 // DPF implementation, so not
                                                 // checking that here.
}

TEST_F(DenseDpfPirServerTest, HandleRequestSucceeds) {
  PirRequest request;
  SetupFakeRequest(123, request);

  // We expect one call to InnerProductWith.
  EXPECT_CALL(*database_, InnerProductWith(::testing::SizeIs(1))).Times(1);

  // Handle the request, and check that it matches the expectation.
  DPF_ASSERT_OK_AND_ASSIGN(
      auto ctx, dpf_->CreateEvaluationContext(
                    request.dpf_pir_request().plain_request().dpf_key(0)));
  DPF_ASSERT_OK_AND_ASSIGN(
      auto expansion, dpf_->EvaluateNext<XorWrapper<absl::uint128>>({}, ctx));
  DPF_ASSERT_OK_AND_ASSIGN(PirResponse result, server_->HandleRequest(request));
  ASSERT_EQ(result.dpf_pir_response().masked_response_size(), 1);
  EXPECT_EQ(result.dpf_pir_response().masked_response(0),
            InnerProductWith({expansion})[0]);
}

TEST_F(DenseDpfPirServerTest,
       BatchedRequestWithoutOtpEquivalentToSingleRequests) {
  // Set up three requests, where the third is the same as the first two
  // batched.
  PirRequest request1, request2, request3;
  SetupFakeRequest(123, request1);
  SetupFakeRequest(456, request2);
  *(request3.mutable_dpf_pir_request()
        ->mutable_plain_request()
        ->mutable_dpf_key()
        ->Add()) = request1.dpf_pir_request().plain_request().dpf_key(0);
  *(request3.mutable_dpf_pir_request()
        ->mutable_plain_request()
        ->mutable_dpf_key()
        ->Add()) = request2.dpf_pir_request().plain_request().dpf_key(0);

  // Expect two single calls and one batched call to InnerProductWith.
  EXPECT_CALL(*database_, InnerProductWith(::testing::SizeIs(1))).Times(2);
  EXPECT_CALL(*database_, InnerProductWith(::testing::SizeIs(2))).Times(1);

  // Handle the requests, and check that the result of the third is the
  // concatenation of the first two.
  DPF_ASSERT_OK_AND_ASSIGN(PirResponse result1,
                           server_->HandleRequest(request1));
  DPF_ASSERT_OK_AND_ASSIGN(PirResponse result2,
                           server_->HandleRequest(request2));
  DPF_ASSERT_OK_AND_ASSIGN(PirResponse result3,
                           server_->HandleRequest(request3));
  ASSERT_EQ(result1.dpf_pir_response().masked_response_size(), 1);
  ASSERT_EQ(result2.dpf_pir_response().masked_response_size(), 1);
  ASSERT_EQ(result3.dpf_pir_response().masked_response_size(), 2);
  EXPECT_EQ(result1.dpf_pir_response().masked_response(0),
            result3.dpf_pir_response().masked_response(0));
  EXPECT_EQ(result2.dpf_pir_response().masked_response(0),
            result3.dpf_pir_response().masked_response(1));
}

}  // namespace
}  // namespace distributed_point_functions
