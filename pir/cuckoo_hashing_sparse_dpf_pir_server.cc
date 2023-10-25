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

#include "pir/cuckoo_hashing_sparse_dpf_pir_server.h"

#include <cmath>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/status_macros.h"
#include "openssl/rand.h"
#include "pir/hashing/hash_family_config.pb.h"
#include "pir/hashing/sha256_hash_family.h"
#include "pir/private_information_retrieval.pb.h"

namespace distributed_point_functions {
namespace {
static constexpr int kNumHashFunctions = 3;
static constexpr double kBucketsPerElement = 1.5;
}  // namespace

CuckooHashingSparseDpfPirServer::CuckooHashingSparseDpfPirServer(
    PirServerPublicParams params, std::unique_ptr<DistributedPointFunction> dpf,
    std::unique_ptr<Database> database, int seed_fingerprint)
    : params_(std::move(params)),
      dpf_(std::move(dpf)),
      database_(std::move(database)),
      seed_fingerprint_(seed_fingerprint) {}

absl::StatusOr<CuckooHashingParams>
CuckooHashingSparseDpfPirServer::GenerateParams(const PirConfig& config) {
  if (config.wrapped_pir_config_case() !=
      PirConfig::kCuckooHashingSparseDpfPirConfig) {
    return absl::InvalidArgumentError(
        "`config` must be a valid CuckooHashingSparseDpfPirConfig");
  }
  CuckooHashingParams params;
  std::string seed(kHashFunctionSeedLengthBytes, '\0');
  RAND_bytes(reinterpret_cast<uint8_t*>(&seed[0]), seed.size());
  params.mutable_hash_family_config()->set_seed(std::move(seed));
  params.mutable_hash_family_config()->set_hash_family(
      config.cuckoo_hashing_sparse_dpf_pir_config().hash_family());
  params.set_num_hash_functions(kNumHashFunctions);
  params.set_num_buckets(
      kBucketsPerElement *
      config.cuckoo_hashing_sparse_dpf_pir_config().num_elements());
  return params;
}

absl::StatusOr<std::unique_ptr<CuckooHashingSparseDpfPirServer>>
CuckooHashingSparseDpfPirServer::CreateLeader(
    CuckooHashingParams params, std::unique_ptr<Database> database,
    ForwardHelperRequestFn sender) {
  DPF_ASSIGN_OR_RETURN(auto leader, CreatePlain(params, std::move(database)));
  DPF_RETURN_IF_ERROR(leader->MakeLeader(std::move(sender)));
  return leader;
}

absl::StatusOr<std::unique_ptr<CuckooHashingSparseDpfPirServer>>
CuckooHashingSparseDpfPirServer::CreateHelper(
    CuckooHashingParams params, std::unique_ptr<Database> database,
    DecryptHelperRequestFn decrypter) {
  DPF_ASSIGN_OR_RETURN(auto helper, CreatePlain(params, std::move(database)));
  DPF_RETURN_IF_ERROR(
      helper->MakeHelper(std::move(decrypter), kEncryptionContextInfo));
  return helper;
}

absl::StatusOr<std::unique_ptr<CuckooHashingSparseDpfPirServer>>
CuckooHashingSparseDpfPirServer::CreatePlain(
    CuckooHashingParams params, std::unique_ptr<Database> database) {
  if (params.num_buckets() <= 0) {
    return absl::InvalidArgumentError("`num_buckets` must be positive");
  }
  if (params.num_hash_functions() <= 0) {
    return absl::InvalidArgumentError("`num_hash_functions` must be positive");
  }
  if (params.hash_family_config().hash_family() ==
      HashFamilyConfig::HASH_FAMILY_UNSPECIFIED) {
    return absl::InvalidArgumentError(
        "params.hash_family_config.hash_family must be set");
  }
  if (database == nullptr) {
    return absl::InvalidArgumentError("`database` cannot be null");
  }
  if (database->num_selection_bits() != params.num_buckets()) {
    return absl::InvalidArgumentError(
        "Number of selection bits in the database does not match "
        "`params.num_buckets`");
  }

  DpfParameters dpf_parameters;
  dpf_parameters.set_log_domain_size(
      static_cast<int>(std::ceil(std::log2(params.num_buckets()))));
  dpf_parameters.mutable_value_type()->mutable_xor_wrapper()->set_bitsize(
      kDpfBlockSizeBits);
  DPF_ASSIGN_OR_RETURN(auto dpf,
                       DistributedPointFunction::Create(dpf_parameters));

  // The first 31 bits of the SHA256 hash of the seed. Used to check that client
  // and both servers use the same key.
  int seed_fingerprint = SHA256HashFunction("")(
      params.hash_family_config().seed(), std::numeric_limits<int>::max());

  PirServerPublicParams server_params;
  *(server_params.mutable_cuckoo_hashing_sparse_dpf_pir_server_params()) =
      std::move(params);

  return absl::WrapUnique(new CuckooHashingSparseDpfPirServer(
      std::move(server_params), std::move(dpf), std::move(database),
      seed_fingerprint));
}

// Computes the response to the client's `request`.
absl::StatusOr<PirResponse> CuckooHashingSparseDpfPirServer::HandlePlainRequest(
    const PirRequest& request) const {
  if (request.wrapped_pir_request_case() != PirRequest::kDpfPirRequest) {
    return absl::InvalidArgumentError(
        "`request` does not contain a valid DpfPirRequest");
  }
  if (request.dpf_pir_request().wrapped_request_case() !=
      DpfPirRequest::kPlainRequest) {
    return absl::InvalidArgumentError(
        "`request` does not contain a valid DpfPirRequest::PlainRequest");
  }
  const DpfPirRequest::PlainRequest& plain_request =
      request.dpf_pir_request().plain_request();
  if (plain_request.dpf_key_size() == 0) {
    return absl::InvalidArgumentError("`dpf_key` must not be empty");
  }
  if (plain_request.seed_fingerprint() != 0 &&
      plain_request.seed_fingerprint() != seed_fingerprint_) {
    return absl::InvalidArgumentError(
        "`seed_fingerprint` does not match. Please ensure that all servers and "
        "the client are initialized with the same parameters.");
  }

  std::vector<std::vector<XorWrapper<absl::uint128>>> selections(
      plain_request.dpf_key_size());
  for (int i = 0; i < plain_request.dpf_key_size(); ++i) {
    // Evaluate DPF and compute inner product with the database.
    DPF_ASSIGN_OR_RETURN(
        auto ctx, dpf_->CreateEvaluationContext(plain_request.dpf_key(i)));
    DPF_ASSIGN_OR_RETURN(
        selections[i], dpf_->EvaluateNext<XorWrapper<absl::uint128>>({}, ctx));
  }

  DPF_ASSIGN_OR_RETURN(std::vector<Database::RecordType> inner_products,
                       database_->InnerProductWith(selections));
  PirResponse response;
  response.mutable_dpf_pir_response()->mutable_masked_response()->Reserve(
      2 * inner_products.size());
  for (int i = 0; i < inner_products.size(); ++i) {
    *(response.mutable_dpf_pir_response()->add_masked_response()) =
        inner_products[i].first;
    *(response.mutable_dpf_pir_response()->add_masked_response()) =
        inner_products[i].second;
  }
  return response;
}

}  // namespace distributed_point_functions
