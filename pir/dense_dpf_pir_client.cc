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

#include <cmath>
#include <memory>
#include <new>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/memory/memory.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "dpf/distributed_point_function.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/status_macros.h"
#include "pir/dense_dpf_pir_server.h"
#include "pir/private_information_retrieval.pb.h"
#include "pir/prng/aes_128_ctr_seeded_prng.h"

namespace distributed_point_functions {

DenseDpfPirClient::DenseDpfPirClient(
    std::unique_ptr<DistributedPointFunction> dpf,
    EncryptHelperRequestFn encrypter, std::string encryption_context_info,
    int database_size)
    : dpf_(std::move(dpf)),
      encrypter_(std::move(encrypter)),
      encryption_context_info_(std::move(encryption_context_info)),
      database_size_(database_size) {}

absl::StatusOr<std::unique_ptr<DenseDpfPirClient>> DenseDpfPirClient::Create(
    const PirConfig& config, EncryptHelperRequestFn encrypter,
    absl::string_view encryption_context_info) {
  if (config.wrapped_pir_config_case() != PirConfig::kDenseDpfPirConfig) {
    return absl::InvalidArgumentError(
        "`config` does not contain a valid DenseDpfPirConfig");
  }
  if (config.dense_dpf_pir_config().num_elements() <= 0) {
    return absl::InvalidArgumentError("`num_elements` must be positive");
  }
  if (encrypter == nullptr) {
    return absl::InvalidArgumentError("`encrypter` must not be null");
  }

  DpfParameters parameters;
  parameters.set_log_domain_size(static_cast<int>(
      std::ceil(std::log2(config.dense_dpf_pir_config().num_elements()))));
  parameters.mutable_value_type()->mutable_xor_wrapper()->set_bitsize(
      kBitsPerBlock);
  DPF_ASSIGN_OR_RETURN(auto dpf, DistributedPointFunction::Create(parameters));

  return absl::WrapUnique(
      new DenseDpfPirClient(std::move(dpf), std::move(encrypter),
                            std::string(encryption_context_info),
                            config.dense_dpf_pir_config().num_elements()));
}

absl::StatusOr<std::pair<PirRequest, PirRequestClientState>>
DenseDpfPirClient::CreateRequest(absl::Span<const int> query_indices) const {
  for (const int query : query_indices) {
    if (query < 0) {
      return absl::InvalidArgumentError(
          "All `query_indices` must be non-negative");
    }
    if (query >= database_size_) {
      return absl::InvalidArgumentError("All `query_indices` out of bounds");
    }
  }

  // Generate plain requests for each index.
  DpfPirRequest::LeaderRequest leader_request;
  DpfPirRequest::HelperRequest helper_request;
  for (int i = 0; i < query_indices.size(); ++i) {
    absl::uint128 alpha = query_indices[i] / kBitsPerBlock;
    XorWrapper<absl::uint128> beta(absl::uint128{1}
                                   << (query_indices[i] % kBitsPerBlock));
    DPF_ASSIGN_OR_RETURN(
        std::tie(
            *(leader_request.mutable_plain_request()->mutable_dpf_key()->Add()),
            *(helper_request.mutable_plain_request()
                  ->mutable_dpf_key()
                  ->Add())),
        dpf_->GenerateKeys(alpha, beta));
  }

  // Generate OTP seed.
  DPF_ASSIGN_OR_RETURN(*(helper_request.mutable_one_time_pad_seed()),
                       Aes128CtrSeededPrng::GenerateSeed());

  // Encrypt helper_request.
  DPF_ASSIGN_OR_RETURN(
      *(leader_request.mutable_encrypted_helper_request()
            ->mutable_encrypted_request()),
      encrypter_(helper_request.SerializeAsString(), encryption_context_info_));

  // Assemble result.
  std::pair<PirRequest, PirRequestClientState> result;
  *(result.first.mutable_dpf_pir_request()->mutable_leader_request()) =
      std::move(leader_request);
  *(result.second.mutable_dense_dpf_pir_request_client_state()
        ->mutable_one_time_pad_seed()) = helper_request.one_time_pad_seed();
  return result;
}

absl::StatusOr<std::vector<std::string>> DenseDpfPirClient::HandleResponse(
    const PirResponse& pir_response,
    const PirRequestClientState& request_client_state) const {
  if (pir_response.wrapped_pir_response_case() !=
      PirResponse::kDpfPirResponse) {
    return absl::InvalidArgumentError(
        "`response` does not contain a valid DpfPirResponse");
  }
  if (request_client_state.wrapped_pir_request_client_state_case() !=
      PirRequestClientState::kDenseDpfPirRequestClientState) {
    return absl::InvalidArgumentError(
        "`request_client_state` does not contain a valid "
        "DenseDpfPirRequestClientState");
  }
  if (pir_response.dpf_pir_response().masked_response().empty()) {
    return absl::InvalidArgumentError("`masked_response` must not be empty");
  }
  if (request_client_state.dense_dpf_pir_request_client_state()
          .one_time_pad_seed()
          .empty()) {
    return absl::InvalidArgumentError("`one_time_pad_seed` must not be empty");
  }

  DPF_ASSIGN_OR_RETURN(
      auto prng, Aes128CtrSeededPrng::Create(
                     request_client_state.dense_dpf_pir_request_client_state()
                         .one_time_pad_seed()));
  std::vector<std::string> result(
      pir_response.dpf_pir_response().masked_response_size());
  for (int i = 0; i < result.size(); ++i) {
    result[i] = pir_response.dpf_pir_response().masked_response(i);
    std::string mask = prng->GetRandomBytes(result[i].size());
    for (int j = 0; j < result[i].size(); ++j) {
      result[i][j] ^= mask[j];
    }
  }
  return result;
}

}  // namespace distributed_point_functions
