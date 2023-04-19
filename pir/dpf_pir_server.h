/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_DPF_PIR_SERVER_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_DPF_PIR_SERVER_H_

#include <string>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "pir/pir_server.h"
#include "pir/private_information_retrieval.pb.h"

namespace distributed_point_functions {

// Common superclass of all DPF-based two-server PIR schemes. Two-server PIR
// supports two architecture flavors. First, the plain model, in which both
// servers directly receive requests from the client, compute a response, and
// return it to the client. Second, the Leader/Helper model. Here, the Leader
// receives both requests, with one of them being encrypted with the Helper's
// public key. The Leader forwards one request to the Helper, and both servers
// compute a response. The Helpler's response is again sent via the Leader,
// masked with a one-time-pad provided by the client. The Leader then combines
// both server's responses and returns the masked response to the client.
//
// Overall, the communication pattern looks as follows:
//
// Client                  Leader                        Helper
//   |                       |                             |
//   | CreateRequest();      |                             |
//   |                       |                             |
//   |     LeaderRequest     |                             |
//   |---------------------->|                             |
//   |                       |                             |
//   |                       |    EncryptedHelperRequest   |
//   |                       |---------------------------->|
//   |                       |                             |
//   |                       | HandleRequest();            | HandleRequest();
//   |                       |                             |
//   |                       |        DpfPirResponse       |
//   |                       |<----------------------------|
//   |                       |                             |
//   |                       | CombineResponses();         |
//   |                       |                             |
//   |     DpfPirResponse    |                             |
//   |<----------------------|                             |
//   |                       |                             |
//   | ProcessResponse();    |                             |
//   |                       |                             |
//
// All two-server DPF PIR schemes should inherit from this class. This class
// cannot be instantiated directly, only via its sub-class. It does not perform
// any actual request handling, and only provides the functionality needed to
// instantiat Leader of Helper servers. The sub-class's factory function should
// call MakeLeader or MakeHelper after construction, as appropriate.
// HandlePlainRequest should contain the main request handler, which will be
// invoked from HandleRequest at the appropriate time, depending on whether the
// server is instantiated as Leader, Helper, or as plain server.
class DpfPirServer : public PirServer {
 public:
  // As described above, DPF PIR servers can run as three possible roles: Plain
  // servers (if deployed without Leader/Helper architecture), Leader servers,
  // or Helper servers.
  enum class Role {
    kPlain = 0,
    kLeader,
    kHelper,
  };

  // Function type for the `sender` argument to `MakeLeader`. Takes a PirRequest
  // to be sent to the Helper, as well as a callback that should be called while
  // waiting for the response. The callback `while_waiting` is provided by this
  // class, whereas the the `sender` function should be provided by the caller
  // of `MakeLeader` (or the factory function of the derived class). Should
  // return the result of the RPC call to the Helper's HandleRequest.
  using ForwardHelperRequestFn = absl::AnyInvocable<absl::StatusOr<PirResponse>(
      const PirRequest& helper_request,
      absl::AnyInvocable<void()> while_waiting) const>;

  // Function type for the helper to decrypt the encrypted helper request. This
  // function has the same parameter and return types as
  // `crypto::tink::HybridDecrypt::Decrypt()`: it takes a byte array
  // `encrypted_helper_request` storing the request ciphertext, and
  // `encryption_context_info` that must match the context info used by the
  // client side, and it should return the result of the decryption.
  //
  // The helper server stores a function object of this type because in some
  // cases a HybridDecrypt object may have shorter lifespan than the PIR server,
  // Using this wrapper allows the underlying HybridDecrypt to change between
  // requests, without special handling via the `HandleRequest` interface.
  using DecryptHelperRequestFn = absl::AnyInvocable<absl::StatusOr<std::string>(
      absl::string_view encrypted_helper_request,
      absl::string_view encryption_context_info) const>;

  // Returns this server's role.
  inline Role role() { return role_; }

  // Handles a client's request. If this is a Leader server, forward the
  // EncryptedHelperRequest to the Helper, and comptutes the Leader response
  // while waiting for a response from the Helper. If this is a Helper server,
  // decrypts the EncryptedHelperRequest and returns a response masked with the
  // one-time-pad derived from the seed contained therein. If this is a plain
  // server, simply computes the response to the given PlainRequest.
  //
  // Returns INVALID_ARGUMENT if the request does not have the right type, or is
  // malformed.
  absl::StatusOr<PirResponse> HandleRequest(
      const PirRequest& request) const final;

 protected:
  // Protected constructor for derived classes.
  DpfPirServer();

  // Implementation of HandleRequest to be provided by the derived class. Should
  // assume the request is a PlainRequest (and throw an error if not). Will
  // either be called directly, or after unpacking the LeaderRequest or
  // EncryptedHelperRequest in case of the Leader/Helper model.
  virtual absl::StatusOr<PirResponse> HandlePlainRequest(
      const PirRequest& request) const = 0;

  // To be called by the derived class if this server should act as a Leader.
  // `sender` should be a function that forwards the EncryptedHelperRequest to
  // the Helper, and executes its callback while waiting for the response (which
  // will in turn compute the Leader's response).
  //
  // Returns INVALID_ARGUMENT if `sender` is NULL.
  absl::Status MakeLeader(ForwardHelperRequestFn sender);

  // To be called by the derived class if this server should act as a Helper.
  // `decrypter` should be a lambda that creates a `crypto::tink::HybridDecrypt`
  // object, for which the client has the public key, and calls `Decrypt()` on
  // the helper's request. `encryption_context_info` is passed as the second
  // argument to `decrypter` and must match the context info used on the client
  // side.
  //
  // Returns INVALID_ARGUMENT if `decrypter` is NULL.
  absl::Status MakeHelper(DecryptHelperRequestFn decrypter,
                          absl::string_view encryption_context_info);

 private:
  struct DpfPirPlain {};
  struct DpfPirLeader {
    ForwardHelperRequestFn sender;
  };
  struct DpfPirHelper {
    DecryptHelperRequestFn decrypter;
    absl::string_view encryption_context_info;
  };

  virtual absl::StatusOr<PirResponse> HandleLeaderRequest(
      const PirRequest& request) const;

  virtual absl::StatusOr<PirResponse> HandleHelperRequest(
      const PirRequest& request) const;

  absl::variant<DpfPirPlain, DpfPirLeader, DpfPirHelper> role_storage_;
  Role role_;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_DPF_PIR_SERVER_H_
