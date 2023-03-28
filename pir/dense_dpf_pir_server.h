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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_DENSE_DPF_PIR_SERVER_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_DENSE_DPF_PIR_SERVER_H_

#include <memory>

#include "absl/numeric/int128.h"
#include "absl/strings/string_view.h"
#include "dpf/distributed_point_function.h"
#include "pir/dpf_pir_server.h"
#include "pir/pir_database_interface.h"
#include "pir/private_information_retrieval.pb.h"

namespace distributed_point_functions {

// Implements the server of a DPF-based two-server PIR scheme with a dense
// database that is indexed by numbers in [0, database_size). This class can be
// instantiated as a Leader, Helper, or plain server. See the documentation of
// DpfPirServer for details.
class DenseDpfPirServer : public DpfPirServer {
 public:
  // The Database interface used for dense DPF pir. Block type is
  // XorWrapper<absl::uint128> (same as used by DPFs), and values are strings.
  using Database = PirDatabaseInterface<XorWrapper<absl::uint128>, std::string>;

  // Function type for the `sender` argument passed to CreateLeader. See
  // DpfPirServer documentation for details.
  using DpfPirServer::ForwardHelperRequestFn;

  // Function type for the `decrypter` argument passed to CreateHelper. See
  // DpfPirServer documentation for details.
  using DpfPirServer::DecryptHelperRequestFn;

  // Context Info passed to the decrypter when created as Helper. Should be the
  // same as used on the client for encryption.
  static inline constexpr absl::string_view kEncryptionContextInfo =
      "DenseDpfPirServer";

  // Creates a new DenseDpfPirServer instance with the given PirConfig and
  // Database, acting as a Leader server. `sender` should be a function that
  // forwards the EncryptedHelperRequest to the Helper, and executes its
  // callback while waiting for the response (which will in turn compute the
  // Leader's response).
  //
  // Returns INVALID_ARGUMENT if `sender` or `database` is NULL, or if `config`
  // is invalid.
  static absl::StatusOr<std::unique_ptr<DenseDpfPirServer>> CreateLeader(
      const PirConfig& config, std::unique_ptr<Database> database,
      ForwardHelperRequestFn sender);

  // Creates a new DenseDpfPirServer instance with the given PirConfig and
  // Database, acting as a Helper server. `decrypter` should wrap around an
  // implementation of crypto::tink::HybridDecrypt::Decrypt for which the client
  // has the public key that is used to encrypt the helper's request.
  // See DpfPirServer documentation for more details.
  //
  // Returns INVALID_ARGUMENT if `decrypter` or `database` is NULL, or if
  // `config` is invalid.
  static absl::StatusOr<std::unique_ptr<DenseDpfPirServer>> CreateHelper(
      const PirConfig& config, std::unique_ptr<Database> database,
      DecryptHelperRequestFn decrypter);

  // Creates a new DenseDpfPirServer instance with the given PirConfig and
  // Database, acting as a plain server.
  //
  // Returns INVALID_ARGUMENT if `database` is NULL, or if `config` is invalid.
  static absl::StatusOr<std::unique_ptr<DenseDpfPirServer>> CreatePlain(
      const PirConfig& config, std::unique_ptr<Database> database);

  // Returns a reference to the server's database.
  const Database& database() const { return *database_; }

  virtual ~DenseDpfPirServer() = default;

  // Returns an empty PirServerPublicParams proto. DenseDpfPirServer does not
  // have any public parameters.
  const PirServerPublicParams& GetPublicParams() const override;

 protected:
  // Computes the response to the client's `request`. Should not be called by
  // users, but only from DpfPirServer::HandleRequest.
  absl::StatusOr<PirResponse> HandlePlainRequest(
      const PirRequest& request) const override;

 private:
  static constexpr int kDpfBlockSize = 8 * sizeof(absl::uint128);

  DenseDpfPirServer(std::unique_ptr<DistributedPointFunction> dpf,
                    std::unique_ptr<Database> database);

  std::unique_ptr<DistributedPointFunction> dpf_;
  std::unique_ptr<Database> database_;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_DENSE_DPF_PIR_SERVER_H_
