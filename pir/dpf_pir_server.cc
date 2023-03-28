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

#include "absl/status/status.h"
#include "dpf/status_macros.h"
#include "pir/prng/aes_128_ctr_seeded_prng.h"
#include "tink/hybrid_decrypt.h"

namespace distributed_point_functions {

DpfPirServer::DpfPirServer() : role_(Role::kPlain) {}

absl::Status DpfPirServer::MakeLeader(ForwardHelperRequestFn sender) {
  if (sender == nullptr) {
    return absl::InvalidArgumentError("`sender` may not be null");
  }
  role_storage_ = DpfPirLeader{std::move(sender)};
  role_ = Role::kLeader;
  return absl::OkStatus();
}

absl::Status DpfPirServer::MakeHelper(
    DecryptHelperRequestFn decrypter,
    absl::string_view encryption_context_info) {
  if (decrypter == nullptr) {
    return absl::InvalidArgumentError("`decrypter` may not be null");
  }
  role_storage_ = DpfPirHelper{std::move(decrypter), encryption_context_info};
  role_ = Role::kHelper;
  return absl::OkStatus();
}

absl::StatusOr<PirResponse> DpfPirServer::HandleRequest(
    const PirRequest& request) const {
  switch (role_) {
    case Role::kPlain:
      return HandlePlainRequest(request);
    case Role::kLeader:
      return HandleLeaderRequest(request);
    case Role::kHelper:
      return HandleHelperRequest(request);
  }
}

absl::StatusOr<PirResponse> DpfPirServer::HandleLeaderRequest(
    const PirRequest& request) const {
  if (request.dpf_pir_request().wrapped_request_case() !=
      DpfPirRequest::kLeaderRequest) {
    return absl::InvalidArgumentError(
        "`request` must be a valid DpfPirRequest::LeaderRequest");
  }
  const DpfPirRequest::LeaderRequest& leader_request =
      request.dpf_pir_request().leader_request();
  if (!leader_request.has_plain_request()) {
    return absl::InvalidArgumentError("`plain_request` must be set");
  }
  if (!leader_request.has_encrypted_helper_request()) {
    return absl::InvalidArgumentError("`encrypted_helper_request` must be set");
  }

  // Split up request into one for the wrapped server, one for the helper.
  PirRequest plain_request, helper_request;
  *(plain_request.mutable_dpf_pir_request()->mutable_plain_request()) =
      leader_request.plain_request();
  *(helper_request.mutable_dpf_pir_request()
        ->mutable_encrypted_helper_request()) =
      leader_request.encrypted_helper_request();

  // Lambda for the callback executed by `sender`. We make sure that it is
  // actually run, to avoid accidental misuse.
  const DpfPirLeader* leader = absl::get_if<DpfPirLeader>(&role_storage_);
  if (leader == nullptr || role_ != Role::kLeader) {
    return absl::InternalError(
        "`HandleLeaderRequest` called when DpfPirServer was not initialized as "
        "a Leader. This should never happen.");
  }
  bool has_run = false;
  absl::StatusOr<PirResponse> leader_response;
  auto while_waiting = [&has_run, &leader_response, &plain_request, this] {
    leader_response = this->HandlePlainRequest(plain_request);
    has_run = true;
  };

  // Call the sender function with the helper request and the callback. Ensure
  // that `while_waiting` is called, and check both statuses.
  DPF_ASSIGN_OR_RETURN(PirResponse helper_response,
                       leader->sender(helper_request, while_waiting));
  if (!has_run) {
    return absl::FailedPreconditionError(
        "HandleRequest: `while_waiting` was not called from `sender` passed at "
        "construction.");
  }
  DPF_RETURN_IF_ERROR(leader_response.status());

  // Combine the results, checking that both have the same sizes.
  if (helper_response.dpf_pir_response().masked_response_size() !=
      leader_response->dpf_pir_response().masked_response_size()) {
    return absl::InternalError(absl::StrCat(
        "Number of responses from Helper (=",
        helper_response.dpf_pir_response().masked_response_size(),
        ")  does not match the number of responses from Leader (=",
        leader_response->dpf_pir_response().masked_response_size(), ")"));
  }
  for (int i = 0; i < helper_response.dpf_pir_response().masked_response_size();
       ++i) {
    std::string& current_helper_response = *(
        helper_response.mutable_dpf_pir_response()->mutable_masked_response(i));
    std::string& current_leader_response =
        *(leader_response->mutable_dpf_pir_response()->mutable_masked_response(
            i));
    if (current_helper_response.size() != current_leader_response.size()) {
      return absl::InternalError(
          absl::StrCat("Response size mismatch at index ", i, ": Got ",
                       current_helper_response.size(), " (Helper) vs. ",
                       current_leader_response.size(), " (Leader)"));
    }
    for (int j = 0; j < current_helper_response.size(); ++j) {
      current_leader_response[j] ^= current_helper_response[j];
    }
  }
  return leader_response;
}

absl::StatusOr<PirResponse> DpfPirServer::HandleHelperRequest(
    const PirRequest& request) const {
  if (request.dpf_pir_request().wrapped_request_case() !=
      DpfPirRequest::kEncryptedHelperRequest) {
    return absl::InvalidArgumentError(
        "`request` must be a valid EncryptedHelperRequest");
  }

  const DpfPirHelper* helper = absl::get_if<DpfPirHelper>(&role_storage_);
  if (helper == nullptr || role_ != Role::kHelper) {
    return absl::InternalError(
        "`HandleHelperRequest` called when DpfPirServer was not initialized as "
        "a Helper. This should never happen.");
  }

  // Decrypt the request.
  DPF_ASSIGN_OR_RETURN(std::string decrypted_request,
                       helper->decrypter(request.dpf_pir_request()
                                             .encrypted_helper_request()
                                             .encrypted_request(),
                                         helper->encryption_context_info));
  DpfPirRequest::HelperRequest inner_request;
  if (!inner_request.ParseFromString(decrypted_request)) {
    return absl::InvalidArgumentError(
        "`request` does not encrypt a valid DpfPirRequest::HelperRequest");
  }

  // Pass the plain request to server_common_.
  PirRequest plain_request;
  *(plain_request.mutable_dpf_pir_request()->mutable_plain_request()) =
      std::move(*(inner_request.mutable_plain_request()));
  DPF_ASSIGN_OR_RETURN(auto response, this->HandlePlainRequest(plain_request));

  // Expand one-time-pad and XOR with the response.
  DPF_ASSIGN_OR_RETURN(auto prng, Aes128CtrSeededPrng::Create(
                                      inner_request.one_time_pad_seed()));
  for (int i = 0; i < response.dpf_pir_response().masked_response_size(); ++i) {
    std::string& current_response =
        *(response.mutable_dpf_pir_response()->mutable_masked_response(i));
    const std::string one_time_pad =
        prng->GetRandomBytes(current_response.size());
    for (int j = 0; j < current_response.size(); ++j) {
      current_response[j] ^= one_time_pad[j];
    }
  }
  return response;
}

}  // namespace distributed_point_functions
