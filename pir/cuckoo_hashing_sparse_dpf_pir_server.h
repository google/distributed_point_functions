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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_SPARSE_DPF_PIR_SERVER_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_SPARSE_DPF_PIR_SERVER_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "dpf/distributed_point_function.h"
#include "dpf/xor_wrapper.h"
#include "pir/dpf_pir_server.h"
#include "pir/pir_database_interface.h"
#include "pir/private_information_retrieval.pb.h"

namespace distributed_point_functions {

// Implements sparse two-server PIR with DPFs. Works by first applying Cuckoo
// Hashing on the database to densify it, and subsequently run queries on the
// dense database. Cuckoo hashing guarantees that every key gets mapped to one
// out of 3 locations. The client simply queries all three locations.
class CuckooHashingSparseDpfPirServer : public DpfPirServer {
 public:
  using Database = PirDatabaseInterface<XorWrapper<absl::uint128>,
                                        std::pair<std::string, std::string>>;

  // Function type for the `sender` argument passed to CreateLeader. See
  // DpfPirServer documentation for details.
  using DpfPirServer::ForwardHelperRequestFn;

  // Function type for the `decrypter` argument passed to CreateHelper. See
  // DpfPirServer documentation for details.
  using DpfPirServer::DecryptHelperRequestFn;

  // Context Info passed to the decrypter when created as Helper. Should be the
  // same as used on the client for encryption.
  static inline constexpr absl::string_view kEncryptionContextInfo =
      "CuckooHashingSparseDpfPirServer";

  // Generates parameters to be used by the client and for constructing the
  // database.
  static absl::StatusOr<CuckooHashingParams> GenerateParams(
      const PirConfig& config);

  // Creates a new CuckooHashingSparseDpfPirServer instance with the given
  // CuckooHashingParams and Database, acting as a Leader server. `sender`
  // should be a function that forwards the EncryptedHelperRequest to the
  // Helper, and executes its callback while waiting for the response (which
  // will in turn compute the Leader's response). For correctness, `params` must
  // match the parameters used to construct `database`.
  //
  // Returns INVALID_ARGUMENT if `sender` or `database` is NULL, or if `params`
  // is invalid.
  static absl::StatusOr<std::unique_ptr<CuckooHashingSparseDpfPirServer>>
  CreateLeader(CuckooHashingParams params, std::unique_ptr<Database> database,
               ForwardHelperRequestFn sender);

  // Creates a new DenseDpfPirServer instance with the given CuckooHashingParams
  // and Database, acting as a Helper server. `decrypter` should wrap around an
  // implementation of crypto::tink::HybridDecrypt::Decrypt for which the client
  // has the public key that is used to encrypt the helper's request.
  // For correctness, `params` must match the parameters used to construct
  // `database`.
  // See DpfPirServer documentation for more details.
  //
  // Returns INVALID_ARGUMENT if `decrypter` or `database` is NULL, or if
  // `params` is invalid.
  static absl::StatusOr<std::unique_ptr<CuckooHashingSparseDpfPirServer>>
  CreateHelper(CuckooHashingParams params, std::unique_ptr<Database> database,
               DecryptHelperRequestFn decrypter);

  // Creates a new DenseDpfPirServer instance with the given CuckooHashingParams
  // and Database, acting as a plain server. For correctness, `params` must
  // match the parameters used to construct `database`.
  //
  // Returns INVALID_ARGUMENT if `database` is NULL, or if `params` is invalid.
  static absl::StatusOr<std::unique_ptr<CuckooHashingSparseDpfPirServer>>
  CreatePlain(CuckooHashingParams params, std::unique_ptr<Database> database);

  // Returns this server's public parameters to be used at the Client.
  const PirServerPublicParams& GetPublicParams() const override {
    return params_;
  }

 protected:
  // Computes the response to the client's `request`. Should not be called
  // by users, but only from DpfPirServer::HandleRequest.
  absl::StatusOr<PirResponse> HandlePlainRequest(
      const PirRequest& request) const override;

 private:
  static constexpr int kHashFunctionSeedLengthBytes = 16;
  static constexpr int kDpfBlockSizeBits = 8 * sizeof(absl::uint128);

  CuckooHashingSparseDpfPirServer(PirServerPublicParams params,
                                  std::unique_ptr<DistributedPointFunction> dpf,
                                  std::unique_ptr<Database> database);

  PirServerPublicParams params_;
  std::unique_ptr<DistributedPointFunction> dpf_;
  std::unique_ptr<Database> database_;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_SPARSE_DPF_PIR_SERVER_H_
