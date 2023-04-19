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

#include "pir/cuckoo_hashing_sparse_dpf_pir_client.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "dpf/status_macros.h"
#include "pir/dense_dpf_pir_client.h"
#include "pir/hashing/hash_family.h"
#include "pir/hashing/hash_family_config.h"
#include "pir/private_information_retrieval.pb.h"

namespace distributed_point_functions {

namespace {

// Helper function that checks whether `input` is equal to `prefix` padded with
// zero bytes.
bool IsPrefixPaddedWithZeros(absl::string_view input,
                             absl::string_view prefix) {
  for (int i = 0; i < input.size(); ++i) {
    if (i < prefix.size()) {
      if (input[i] != prefix[i]) return false;
    } else {
      if (input[i] != '\0') return false;
    }
  }
  return true;
}

}  // namespace

CuckooHashingSparseDpfPirClient::CuckooHashingSparseDpfPirClient(
    std::unique_ptr<DenseDpfPirClient> wrapped_client,
    std::vector<HashFunction> hash_functions, int num_buckets)
    : wrapped_client_(std::move(wrapped_client)),
      hash_functions_(std::move(hash_functions)),
      num_buckets_(num_buckets) {}

absl::StatusOr<std::unique_ptr<CuckooHashingSparseDpfPirClient>>
CuckooHashingSparseDpfPirClient::Create(
    const PirServerPublicParams& params, EncryptHelperRequestFn encrypter,
    absl::string_view encryption_context_info) {
  if (encrypter == nullptr) {
    return absl::InvalidArgumentError("`enrypter` may not be null");
  }
  if (params.wrapped_pir_server_public_params_case() !=
      PirServerPublicParams::kCuckooHashingSparseDpfPirServerParams) {
    return absl::InvalidArgumentError(
        "`params` does not contain valid valid "
        "CuckooHashingSparseDpfPirServerParams");
  }
  if (params.cuckoo_hashing_sparse_dpf_pir_server_params().num_buckets() <= 0) {
    return absl::InvalidArgumentError("`num_buckets` must be positive");
  }
  if (params.cuckoo_hashing_sparse_dpf_pir_server_params()
          .num_hash_functions() <= 0) {
    return absl::InvalidArgumentError("`num_hash_functions` must be positive");
  }

  DPF_ASSIGN_OR_RETURN(HashFamily hash_family,
                       CreateHashFamilyFromConfig(
                           params.cuckoo_hashing_sparse_dpf_pir_server_params()
                               .hash_family_config()));

  DPF_ASSIGN_OR_RETURN(
      std::vector<HashFunction> hash_functions,
      CreateHashFunctions(std::move(hash_family),
                          params.cuckoo_hashing_sparse_dpf_pir_server_params()
                              .num_hash_functions()));

  PirConfig wrapped_client_config;
  wrapped_client_config.mutable_dense_dpf_pir_config()->set_num_elements(
      params.cuckoo_hashing_sparse_dpf_pir_server_params().num_buckets());
  DPF_ASSIGN_OR_RETURN(
      std::unique_ptr<DenseDpfPirClient> wrapped_client,
      DenseDpfPirClient::Create(wrapped_client_config, std::move(encrypter),
                                encryption_context_info));

  return absl::WrapUnique(new CuckooHashingSparseDpfPirClient(
      std::move(wrapped_client), std::move(hash_functions),
      params.cuckoo_hashing_sparse_dpf_pir_server_params().num_buckets()));
}

absl::StatusOr<std::pair<PirRequest, PirRequestClientState>>
CuckooHashingSparseDpfPirClient::CreateRequest(
    absl::Span<const std::string> query) const {
  std::vector<int> indices;
  indices.reserve(hash_functions_.size() * query.size());
  for (int i = 0; i < query.size(); ++i) {
    for (int j = 0; j < hash_functions_.size(); ++j) {
      indices.push_back(hash_functions_[j](query[i], num_buckets_));
    }
  }
  PirRequest request;
  PirRequestClientState request_client_state;
  DPF_ASSIGN_OR_RETURN(std::tie(request, request_client_state),
                       wrapped_client_->CreateRequest(indices));
  std::string otp_seed =
      request_client_state.dense_dpf_pir_request_client_state()
          .one_time_pad_seed();
  request_client_state
      .mutable_cuckoo_hashing_sparse_dpf_pir_request_client_state()
      ->set_one_time_pad_seed(std::move(otp_seed));
  for (int i = 0; i < query.size(); ++i) {
    request_client_state
        .mutable_cuckoo_hashing_sparse_dpf_pir_request_client_state()
        ->add_query_strings(query[i]);
  }
  return std::make_pair(std::move(request), std::move(request_client_state));
}

absl::StatusOr<std::vector<absl::optional<std::string>>>
CuckooHashingSparseDpfPirClient::HandleResponse(
    const PirResponse& pir_response,
    const PirRequestClientState& request_client_state) const {
  if (pir_response.wrapped_pir_response_case() !=
      PirResponse::kDpfPirResponse) {
    return absl::InvalidArgumentError(
        "`pir_response` does not contain a valid DpfPirResponse");
  }
  if (request_client_state.wrapped_pir_request_client_state_case() !=
      PirRequestClientState::kCuckooHashingSparseDpfPirRequestClientState) {
    return absl::InvalidArgumentError(
        "`request_client_state` does not contain a valid "
        "CuckooHashingSparseDpfPirRequestClientState");
  }
  if (request_client_state.cuckoo_hashing_sparse_dpf_pir_request_client_state()
              .query_strings_size() *
          hash_functions_.size() * 2 !=
      pir_response.dpf_pir_response().masked_response_size()) {
    // We should get two responses for each query, one for the key and one for
    // the value.
    return absl::InvalidArgumentError(
        "Number of responses must be equal to the number of queries times the "
        "number of hash functions times 2");
  }

  PirRequestClientState wrapped_client_state;
  wrapped_client_state.mutable_dense_dpf_pir_request_client_state()
      ->set_one_time_pad_seed(
          request_client_state
              .cuckoo_hashing_sparse_dpf_pir_request_client_state()
              .one_time_pad_seed());
  DPF_ASSIGN_OR_RETURN(
      std::vector<std::string> raw_responses,
      wrapped_client_->HandleResponse(pir_response, wrapped_client_state));
  std::vector<absl::optional<std::string>> result(
      raw_responses.size() / hash_functions_.size() / 2, absl::nullopt);
  for (int i = 0; i < result.size(); ++i) {
    for (int j = 0; j < hash_functions_.size(); ++j) {
      int raw_index = 2 * (hash_functions_.size() * i + j);
      if (!result[i].has_value() &&
          IsPrefixPaddedWithZeros(
              raw_responses[raw_index],
              request_client_state
                  .cuckoo_hashing_sparse_dpf_pir_request_client_state()
                  .query_strings(i))) {
        result[i] = raw_responses[raw_index + 1];
      }
    }
  }
  return result;
}

}  // namespace distributed_point_functions
