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

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "dpf/distributed_point_function.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/status_macros.h"
#include "pir/private_information_retrieval.pb.h"
#include "pir/prng/aes_128_ctr_seeded_prng.h"
#include "pir/testing/encrypt_decrypt.h"

namespace distributed_point_functions {
namespace pir_testing {

constexpr int kBitsPerBlock = 128;

RequestGenerator::RequestGenerator(
    std::unique_ptr<DistributedPointFunction> dpf, std::string otp_seed,
    std::string encryption_context_info, int database_size)
    : dpf_(std::move(dpf)),
      otp_seed_(std::move(otp_seed)),
      encryption_context_info_(std::move(encryption_context_info)),
      database_size_(database_size) {}

absl::StatusOr<std::unique_ptr<RequestGenerator>> RequestGenerator::Create(
    int database_size, absl::string_view encryption_context_info) {
  if (database_size <= 0) {
    return absl::InvalidArgumentError("`database_size` must be positive");
  }
  DpfParameters parameters;
  parameters.mutable_value_type()->mutable_xor_wrapper()->set_bitsize(
      kBitsPerBlock);
  parameters.set_log_domain_size(
      static_cast<int>(std::ceil(std::log2(database_size))));
  DPF_ASSIGN_OR_RETURN(auto dpf, DistributedPointFunction::Create(parameters));
  DPF_ASSIGN_OR_RETURN(std::string otp_seed,
                       Aes128CtrSeededPrng::GenerateSeed());
  return absl::WrapUnique(new RequestGenerator(
      std::move(dpf), std::move(otp_seed), std::string(encryption_context_info),
      database_size));
}

absl::StatusOr<
    std::pair<DpfPirRequest::PlainRequest, DpfPirRequest::PlainRequest>>
RequestGenerator::CreateDpfPirPlainRequests(
    absl::Span<const int> indices) const {
  DpfPirRequest::PlainRequest request1, request2;
  for (int i = 0; i < indices.size(); ++i) {
    if (indices[i] < 0) {
      return absl::InvalidArgumentError("`indices` must be non-negative");
    }
    if (indices[i] >= database_size_) {
      return absl::InvalidArgumentError(
          "`indices` must be less than `database_size`");
    }
    absl::uint128 alpha = indices[i] / kBitsPerBlock;
    XorWrapper<absl::uint128> beta(absl::uint128{1}
                                   << (indices[i] % kBitsPerBlock));
    DPF_ASSIGN_OR_RETURN(std::tie(*(request1.mutable_dpf_key()->Add()),
                                  *(request2.mutable_dpf_key()->Add())),
                         dpf_->GenerateKeys(alpha, beta));
  }
  return std::make_pair(std::move(request1), std::move(request2));
}

absl::StatusOr<DpfPirRequest::LeaderRequest>
RequestGenerator::CreateDpfPirLeaderRequest(
    absl::Span<const int> indices) const {
  DpfPirRequest::LeaderRequest leader_request;
  DpfPirRequest::HelperRequest helper_request;
  helper_request.set_one_time_pad_seed(otp_seed_);
  DPF_ASSIGN_OR_RETURN(std::tie(*(leader_request.mutable_plain_request()),
                                *(helper_request.mutable_plain_request())),
                       CreateDpfPirPlainRequests(indices));

  // Encrypt Helper request and add it to leader request.
  DPF_ASSIGN_OR_RETURN(auto encrypter, CreateFakeHybridEncrypt());
  DPF_ASSIGN_OR_RETURN(*(leader_request.mutable_encrypted_helper_request()
                             ->mutable_encrypted_request()),
                       encrypter->Encrypt(helper_request.SerializeAsString(),
                                          encryption_context_info_));
  return leader_request;
}

}  // namespace pir_testing
}  // namespace distributed_point_functions
