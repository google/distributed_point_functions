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

#include "pir/testing/request_generator.h"

#include "absl/status/status.h"
#include "dpf/distributed_point_function.h"
#include "dpf/internal/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "pir/private_information_retrieval.pb.h"
#include "pir/testing/encrypt_decrypt.h"

namespace distributed_point_functions::pir_testing {
namespace {

using dpf_internal::IsOkAndHolds;
using dpf_internal::StatusIs;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::testing::Truly;

constexpr int kDatabaseSize = 1234;
constexpr int kBitsPerBlock = 128;
inline constexpr absl::string_view kEncryptionContextInfo =
    "RequestGeneratorTest";

void CheckRequestsAreConsistent(const DpfPirRequest::PlainRequest& request1,
                                const DpfPirRequest::PlainRequest& request2,
                                int index) {
  // Expand requests separately.
  DpfParameters parameters;
  parameters.mutable_value_type()->mutable_xor_wrapper()->set_bitsize(
      kBitsPerBlock);
  parameters.set_log_domain_size(
      static_cast<int>(std::ceil(std::log2(kDatabaseSize))));
  DPF_ASSERT_OK_AND_ASSIGN(auto dpf,
                           DistributedPointFunction::Create(parameters));
  std::vector<XorWrapper<absl::uint128>> expansion1, expansion2;
  DPF_ASSERT_OK_AND_ASSIGN(EvaluationContext ctx1,
                           dpf->CreateEvaluationContext(request1.dpf_key(0)));
  DPF_ASSERT_OK_AND_ASSIGN(EvaluationContext ctx2,
                           dpf->CreateEvaluationContext(request2.dpf_key(0)));
  DPF_ASSERT_OK_AND_ASSIGN(
      expansion1, dpf->EvaluateNext<XorWrapper<absl::uint128>>({}, ctx1))
  DPF_ASSERT_OK_AND_ASSIGN(
      expansion2, dpf->EvaluateNext<XorWrapper<absl::uint128>>({}, ctx2));

  // Check that they add up to correct shares of the selection vector.
  EXPECT_EQ(expansion1.size(), expansion2.size());
  EXPECT_EQ((expansion1[0] + expansion2[0]).value(), absl::uint128{1} << index);
  for (int i = 1; i < expansion1.size(); ++i) {
    EXPECT_EQ((expansion1[i] + expansion2[i]).value(), absl::uint128{0});
  }
}

TEST(RequestGenerator, CreateSucceeds) {
  EXPECT_THAT(RequestGenerator::Create(1, kEncryptionContextInfo),
              IsOkAndHolds(NotNull()));
}

TEST(RequestGenerator, CreateFailsIfDatabaseSizeIsZero) {
  EXPECT_THAT(
      RequestGenerator::Create(0, kEncryptionContextInfo),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("positive")));
}

TEST(RequestGenerator, CreatePlainDpfRequestSucceeds) {
  DPF_ASSERT_OK_AND_ASSIGN(
      auto request_generator,
      RequestGenerator::Create(kDatabaseSize, kEncryptionContextInfo));
  int index = 23;

  DpfPirRequest::PlainRequest request1, request2;
  DPF_ASSERT_OK_AND_ASSIGN(
      std::tie(request1, request2),
      request_generator->CreateDpfPirPlainRequests({index}));

  CheckRequestsAreConsistent(request1, request2, index);
}

TEST(RequestGenerator, CreatePlainDpfRequestSucceedsWithTwoIndices) {
  DPF_ASSERT_OK_AND_ASSIGN(
      auto request_generator,
      RequestGenerator::Create(kDatabaseSize, kEncryptionContextInfo));

  EXPECT_THAT(request_generator->CreateDpfPirPlainRequests({0, 1}),
              IsOkAndHolds(Truly([](const auto& x) {
                return x.first.dpf_key_size() == 2 &&
                       x.second.dpf_key_size() == 2;
              })));
}

TEST(RequestGenerator, CreatePlainDpfRequestFailsIfIndexIsNegative) {
  DPF_ASSERT_OK_AND_ASSIGN(auto request_generator,
                           RequestGenerator::Create(1, kEncryptionContextInfo));
  EXPECT_THAT(
      request_generator->CreateDpfPirPlainRequests({-1}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("negative")));
}

TEST(RequestGenerator, CreatePlainDpfRequestFailsIfIndexIsTooLarge) {
  DPF_ASSERT_OK_AND_ASSIGN(auto request_generator,
                           RequestGenerator::Create(1, kEncryptionContextInfo));
  EXPECT_THAT(
      request_generator->CreateDpfPirPlainRequests({1}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("less than")));
}

TEST(RequestGenerator, CreateLeaderRequestSucceeds) {
  DPF_ASSERT_OK_AND_ASSIGN(
      auto request_generator,
      RequestGenerator::Create(kDatabaseSize, kEncryptionContextInfo));
  int index = 23;

  DPF_ASSERT_OK_AND_ASSIGN(
      DpfPirRequest::LeaderRequest request,
      request_generator->CreateDpfPirLeaderRequest({index}));

  // Decrypt helper request and check that requests are consistent.
  DPF_ASSERT_OK_AND_ASSIGN(auto decrypter, CreateFakeHybridDecrypt());
  DPF_ASSERT_OK_AND_ASSIGN(
      std::string serialized_helper_request,
      decrypter->Decrypt(request.encrypted_helper_request().encrypted_request(),
                         kEncryptionContextInfo));
  DpfPirRequest::HelperRequest helper_request;
  ASSERT_TRUE(helper_request.ParseFromString(serialized_helper_request));
  CheckRequestsAreConsistent(helper_request.plain_request(),
                             request.plain_request(), index);
}

}  // namespace
}  // namespace distributed_point_functions::pir_testing
