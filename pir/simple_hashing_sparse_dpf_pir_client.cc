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

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "dpf/status_macros.h"
#include "google/protobuf/io/coded_stream.h"
#include "pir/dense_dpf_pir_client.h"
#include "pir/dpf_pir_client.h"
#include "pir/hashing/hash_family.h"
#include "pir/hashing/hash_family_config.h"
#include "pir/hashing/sha256_hash_family.h"
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

// Always returns an InternalError. Used in the constructor for the wrapped PIR
// client, which should never be called directly.
absl::StatusOr<std::string> DummyEncrypter(absl::string_view plaintext,
                                           absl::string_view context_info) {
  return absl::InternalError(
      "This PIR client is wrapped by a SimpleHashingSparseDpfPirClient and "
      "should never be called directly");
}

}  // namespace

SimpleHashingSparseDpfPirClient::SimpleHashingSparseDpfPirClient(
    EncryptHelperRequestFn encrypter, std::string encryption_context_info,
    std::unique_ptr<DenseDpfPirClient> wrapped_client,
    HashFunction hash_function, int num_buckets, int seed_fingerprint)
    : DpfPirClient(std::move(encrypter), std::move(encryption_context_info)),
      wrapped_client_(std::move(wrapped_client)),
      hash_function_(std::move(hash_function)),
      num_buckets_(num_buckets),
      seed_fingerprint_(seed_fingerprint) {}

absl::StatusOr<std::unique_ptr<SimpleHashingSparseDpfPirClient>>
SimpleHashingSparseDpfPirClient::Create(
    const PirServerPublicParams& params, EncryptHelperRequestFn encrypter,
    absl::string_view encryption_context_info) {
  if (encrypter == nullptr) {
    return absl::InvalidArgumentError("`enrypter` may not be null");
  }
  if (params.wrapped_pir_server_public_params_case() !=
      PirServerPublicParams::kSimpleHashingSparseDpfPirServerParams) {
    return absl::InvalidArgumentError(
        "`params` does not contain valid valid "
        "SimpleHashingSparseDpfPirServerParams");
  }
  if (params.simple_hashing_sparse_dpf_pir_server_params().num_buckets() <= 0) {
    return absl::InvalidArgumentError("`num_buckets` must be positive");
  }

  DPF_ASSIGN_OR_RETURN(HashFamily hash_family,
                       CreateHashFamilyFromConfig(
                           params.simple_hashing_sparse_dpf_pir_server_params()
                               .hash_family_config()));

  // The first 31 bits of the SHA256 hash of the seed. Used to check that client
  // and both servers use the same key.
  int seed_fingerprint = SHA256HashFunction("")(
      params.simple_hashing_sparse_dpf_pir_server_params()
          .hash_family_config()
          .seed(),
      std::numeric_limits<int>::max());

  DPF_ASSIGN_OR_RETURN(std::vector<HashFunction> hash_functions,
                       CreateHashFunctions(std::move(hash_family), 1));
  HashFunction& hash_function = hash_functions.back();

  PirConfig wrapped_client_config;
  wrapped_client_config.mutable_dense_dpf_pir_config()->set_num_elements(
      params.simple_hashing_sparse_dpf_pir_server_params().num_buckets());
  DPF_ASSIGN_OR_RETURN(
      std::unique_ptr<DenseDpfPirClient> wrapped_client,
      DenseDpfPirClient::Create(wrapped_client_config, DummyEncrypter, ""));

  return absl::WrapUnique(new SimpleHashingSparseDpfPirClient(
      std::move(encrypter), std::string(encryption_context_info),
      std::move(wrapped_client), std::move(hash_function),
      params.simple_hashing_sparse_dpf_pir_server_params().num_buckets(),
      seed_fingerprint));
}
absl::StatusOr<std::tuple<DpfPirRequest::PlainRequest,
                          DpfPirRequest::HelperRequest, PirRequestClientState>>
SimpleHashingSparseDpfPirClient::CreatePlainRequests(
    absl::Span<const std::string> query) const {
  std::vector<int> indices;
  indices.reserve(query.size());
  for (int i = 0; i < query.size(); ++i) {
    indices.push_back(hash_function_(query[i], num_buckets_));
  }
  DpfPirRequest::PlainRequest leader_request;
  DpfPirRequest::HelperRequest helper_request;
  PirRequestClientState request_client_state;
  DPF_ASSIGN_OR_RETURN(
      std::tie(leader_request, helper_request, request_client_state),
      wrapped_client_->CreatePlainRequests(indices));
  std::string otp_seed =
      request_client_state.dense_dpf_pir_request_client_state()
          .one_time_pad_seed();
  request_client_state
      .mutable_simple_hashing_sparse_dpf_pir_request_client_state()
      ->set_one_time_pad_seed(std::move(otp_seed));
  for (int i = 0; i < query.size(); ++i) {
    request_client_state
        .mutable_simple_hashing_sparse_dpf_pir_request_client_state()
        ->add_query_strings(query[i]);
  }
  leader_request.set_seed_fingerprint(seed_fingerprint_);
  helper_request.mutable_plain_request()->set_seed_fingerprint(
      seed_fingerprint_);
  return std::make_tuple(std::move(leader_request), std::move(helper_request),
                         std::move(request_client_state));
}

absl::StatusOr<std::vector<absl::optional<std::string>>>
SimpleHashingSparseDpfPirClient::HandleResponse(
    const PirResponse& pir_response,
    const PirRequestClientState& request_client_state) const {
  if (pir_response.wrapped_pir_response_case() !=
      PirResponse::kDpfPirResponse) {
    return absl::InvalidArgumentError(
        "`pir_response` does not contain a valid DpfPirResponse");
  }
  if (request_client_state.wrapped_pir_request_client_state_case() !=
      PirRequestClientState::kSimpleHashingSparseDpfPirRequestClientState) {
    return absl::InvalidArgumentError(
        "`request_client_state` does not contain a valid "
        "SimpleHashingSparseDpfPirRequestClientState");
  }
  if (request_client_state.simple_hashing_sparse_dpf_pir_request_client_state()
          .query_strings_size() !=
      pir_response.dpf_pir_response().masked_response_size()) {
    return absl::InvalidArgumentError(
        "Number of responses must be equal to the number of queries");
  }

  PirRequestClientState wrapped_client_state;
  wrapped_client_state.mutable_dense_dpf_pir_request_client_state()
      ->set_one_time_pad_seed(
          request_client_state
              .simple_hashing_sparse_dpf_pir_request_client_state()
              .one_time_pad_seed());
  DPF_ASSIGN_OR_RETURN(
      std::vector<std::string> raw_responses,
      wrapped_client_->HandleResponse(pir_response, wrapped_client_state));
  std::vector<absl::optional<std::string>> result(raw_responses.size(),
                                                  absl::nullopt);
  for (int i = 0; i < result.size(); ++i) {
    // We need to use a CodedInputStream here to handle the null bytes at the
    // end of the string.
    HashedPirDatabaseBucket bucket;
    ::google::protobuf::io::CodedInputStream coded_stream(
        reinterpret_cast<const uint8_t*>(raw_responses[i].data()),
        raw_responses[i].size());
    bucket.ParseFromCodedStream(&coded_stream);
    absl::string_view query =
        request_client_state
            .simple_hashing_sparse_dpf_pir_request_client_state()
            .query_strings(i);
    for (int j = 0; j < bucket.keys_size(); ++j) {
      if (bucket.keys(j) == query) {
        result[i] = bucket.values(j);
        break;
      }
    }
  }
  return result;
}

}  // namespace distributed_point_functions
