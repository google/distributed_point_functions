/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_DPF_PIR_CLIENT_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_DPF_PIR_CLIENT_H_

#include <string>
#include <tuple>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "pir/pir_client.h"
#include "pir/private_information_retrieval.pb.h"

namespace distributed_point_functions {

template <typename QueryType, typename ResponseType>
class DpfPirClient : public PirClient<QueryType, ResponseType> {
 public:
  // Function type for the client to encrypt a PIR request to the helper.
  // This function has the same parameter and return types as
  // `crypto::tink::HybridEncrypt::Encrypt()`: it takes `plain_helper_request`
  // storing the PIR request and `encryption_context_info` to be passed to the
  // helper to correctly decrypt the encrypted PIR request, and it returns the
  // result of the encryption.
  //
  // The client stores a function object of this type because in some cases a
  // HybridEncrypt object may have to be refreshed before being invoked on a
  // PIR request.
  // Using this wrapper allows the underlying HybridEncrypt to change between
  // creation of a client and a call to `CreateRequest`.
  using EncryptHelperRequestFn = absl::AnyInvocable<absl::StatusOr<std::string>(
      absl::string_view plain_helper_request,
      absl::string_view encryption_context_info) const>;

  // Creates a pair of plain DPF PIR requests, one for each server, as well as
  // the client's private state. Useful to wrap PIR protocols in other
  // protocols, or to send queries to servers directly. For the communication
  // pattern described in dpf_pir_server.h, please use CreateRequest instead.
  //
  virtual absl::StatusOr<
      std::tuple<DpfPirRequest::PlainRequest, DpfPirRequest::HelperRequest,
                 PirRequestClientState>>
  CreatePlainRequests(QueryType query) const = 0;

  // Creates a new PIR request for the given `query`, which is to be sent to the
  // Leader server. If successful, returns the request together with the
  // client's private state needed to decrypt the server's response.
  absl::StatusOr<std::pair<PirRequest, PirRequestClientState>> CreateRequest(
      QueryType query) const final {
    absl::StatusOr<
        std::tuple<DpfPirRequest::PlainRequest, DpfPirRequest::HelperRequest,
                   PirRequestClientState>>
        plain_requests = CreatePlainRequests(query);
    if (!plain_requests.ok()) {
      return plain_requests.status();
    }
    absl::StatusOr<PirRequest> leader_request = PlainRequestsToLeaderRequest(
        std::move(std::get<0>(plain_requests.value())),
        std::move(std::get<1>(plain_requests.value())));
    if (!leader_request.ok()) {
      return leader_request.status();
    }
    return std::make_pair(
        std::move(leader_request.value()),
        std::move(std::get<PirRequestClientState>(plain_requests.value())));
  }

  virtual ~DpfPirClient() = default;

 protected:
  // Constructor to be called by subclasses. `encrypter` and
  // `encryption_context_info` will be used to encrypt the helper's PIR request.
  DpfPirClient(EncryptHelperRequestFn encrypter,
               std::string encryption_context_info)
      : encrypter_(std::move(encrypter)),
        encryption_context_info_(std::move(encryption_context_info)) {}

 private:
  absl::StatusOr<PirRequest> PlainRequestsToLeaderRequest(
      DpfPirRequest::PlainRequest leader_plain_request,
      DpfPirRequest::HelperRequest helper_plain_request) const;

  EncryptHelperRequestFn encrypter_;
  std::string encryption_context_info_;
};

template <typename QueryType, typename ResponseType>
absl::StatusOr<PirRequest>
DpfPirClient<QueryType, ResponseType>::PlainRequestsToLeaderRequest(
    DpfPirRequest::PlainRequest leader_plain_request,
    DpfPirRequest::HelperRequest helper_plain_request) const {
  // Encrypt helper_request.
  absl::StatusOr<std::string> encrypted_helper_request = encrypter_(
      helper_plain_request.SerializeAsString(), encryption_context_info_);
  if (!encrypted_helper_request.ok()) {
    return encrypted_helper_request.status();
  }

  // Assemble and return result.
  PirRequest leader_request;
  *(leader_request.mutable_dpf_pir_request()
        ->mutable_leader_request()
        ->mutable_encrypted_helper_request()
        ->mutable_encrypted_request()) =
      std::move(encrypted_helper_request.value());
  *(leader_request.mutable_dpf_pir_request()
        ->mutable_leader_request()
        ->mutable_plain_request()) = std::move(leader_plain_request);
  return leader_request;
}

}  // namespace distributed_point_functions
#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_DPF_PIR_CLIENT_H_
