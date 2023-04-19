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
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "dpf/distributed_point_function.h"
#include "dpf/status_macros.h"
#include "pir/private_information_retrieval.pb.h"

namespace distributed_point_functions {

DenseDpfPirServer::DenseDpfPirServer(
    std::unique_ptr<DistributedPointFunction> dpf,
    std::unique_ptr<Database> database)
    : dpf_(std::move(dpf)), database_(std::move(database)) {}

absl::StatusOr<std::unique_ptr<DenseDpfPirServer>>
DenseDpfPirServer::CreateLeader(const PirConfig& config,
                                std::unique_ptr<Database> database,
                                ForwardHelperRequestFn sender) {
  DPF_ASSIGN_OR_RETURN(auto leader, CreatePlain(config, std::move(database)));
  DPF_RETURN_IF_ERROR(leader->MakeLeader(std::move(sender)));
  return leader;
}

absl::StatusOr<std::unique_ptr<DenseDpfPirServer>>
DenseDpfPirServer::CreateHelper(const PirConfig& config,
                                std::unique_ptr<Database> database,
                                DecryptHelperRequestFn decrypter) {
  DPF_ASSIGN_OR_RETURN(auto helper, CreatePlain(config, std::move(database)));
  DPF_RETURN_IF_ERROR(
      helper->MakeHelper(std::move(decrypter), kEncryptionContextInfo));
  return helper;
}

absl::StatusOr<std::unique_ptr<DenseDpfPirServer>>
DenseDpfPirServer::CreatePlain(const PirConfig& config,
                               std::unique_ptr<Database> database) {
  if (config.wrapped_pir_config_case() != PirConfig::kDenseDpfPirConfig) {
    return absl::InvalidArgumentError(
        "`config` does not contain a valid DenseDpfPirConfig");
  }
  if (database == nullptr) {
    return absl::InvalidArgumentError("`database` cannot be null");
  }
  if (config.dense_dpf_pir_config().num_elements() <= 0) {
    return absl::InvalidArgumentError("`num_elements` must be positive");
  }
  if (database->size() != config.dense_dpf_pir_config().num_elements()) {
    return absl::InvalidArgumentError(
        "Database size does not match the config size");
  }

  DpfParameters parameters;
  parameters.set_log_domain_size(static_cast<int>(
      std::ceil(std::log2(config.dense_dpf_pir_config().num_elements()))));
  parameters.mutable_value_type()->mutable_xor_wrapper()->set_bitsize(
      kDpfBlockSize);
  DPF_ASSIGN_OR_RETURN(auto dpf, DistributedPointFunction::Create(parameters));

  return absl::WrapUnique(
      new DenseDpfPirServer(std::move(dpf), std::move(database)));
}

const PirServerPublicParams& DenseDpfPirServer::GetPublicParams() const {
  return PirServerPublicParams::default_instance();
}

// Computes the response to the client's `request`.
absl::StatusOr<PirResponse> DenseDpfPirServer::HandlePlainRequest(
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

  std::vector<std::vector<XorWrapper<absl::uint128>>> selections(
      plain_request.dpf_key_size());
  for (int i = 0; i < plain_request.dpf_key_size(); ++i) {
    // Evaluate DPF and compute inner product with the database.
    DPF_ASSIGN_OR_RETURN(
        auto ctx, dpf_->CreateEvaluationContext(plain_request.dpf_key(i)));
    DPF_ASSIGN_OR_RETURN(
        selections[i], dpf_->EvaluateNext<XorWrapper<absl::uint128>>({}, ctx));
  }

  DPF_ASSIGN_OR_RETURN(std::vector<std::string> inner_products,
                       database_->InnerProductWith(selections));
  PirResponse response;
  for (int i = 0; i < inner_products.size(); ++i) {
    *(response.mutable_dpf_pir_response()->add_masked_response()) =
        std::move(inner_products[i]);
  }
  return response;
}

}  // namespace distributed_point_functions
