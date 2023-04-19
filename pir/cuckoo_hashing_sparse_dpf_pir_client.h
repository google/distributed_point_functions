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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_CUCKOO_HASHING_SPARSE_DPF_PIR_CLIENT_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_CUCKOO_HASHING_SPARSE_DPF_PIR_CLIENT_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "pir/cuckoo_hashing_sparse_dpf_pir_server.h"
#include "pir/dense_dpf_pir_client.h"
#include "pir/dpf_pir_client.h"
#include "pir/hashing/hash_family.h"
#include "pir/private_information_retrieval.pb.h"

namespace distributed_point_functions {

class CuckooHashingSparseDpfPirClient
    : public DpfPirClient<absl::Span<const std::string>,
                          std::vector<absl::optional<std::string>>> {
 public:
  // Creates a new CuckooHashingSparseDpfPirClient with the given `params` and
  // an `encrypter` function that should wrap around an implementation of
  // `crypto::tink::HybridEncrypt::Encrypt()`. See the documentation of
  // DpfPirClient for more details about the type of `encrypter`.
  //
  // Returns INVALID_ARGUMENT if `params` is invalid, or if `encrypter` is NULL.
  static absl::StatusOr<std::unique_ptr<CuckooHashingSparseDpfPirClient>>
  Create(const PirServerPublicParams& params, EncryptHelperRequestFn encrypter,
         absl::string_view encryption_context_info =
             CuckooHashingSparseDpfPirServer::kEncryptionContextInfo);

  // Creates a new PIR request for the given `query`. If successful, returns the
  // request together with the private key needed to decrypt the server's
  // response.
  absl::StatusOr<std::pair<PirRequest, PirRequestClientState>> CreateRequest(
      absl::Span<const std::string> query) const override;

  // Handles the server's `pir_response`. `request_client_state` is the
  // per-request client state corresponding to the request sent to the server.
  //
  // For each query key passed to the corresponding `CreateRequest` call,
  // returns the database value at that key. The returned values will be padded
  // with null bytes to the size of the largest database entry. Returns
  // INVALID_ARGUMENT if either the response or the client state is invalid.
  absl::StatusOr<std::vector<absl::optional<std::string>>> HandleResponse(
      const PirResponse& pir_response,
      const PirRequestClientState& request_client_state) const override;

 private:
  CuckooHashingSparseDpfPirClient(
      std::unique_ptr<DenseDpfPirClient> wrapped_client,
      std::vector<HashFunction> hash_functions, int num_buckets);

  std::unique_ptr<DenseDpfPirClient> wrapped_client_;
  std::vector<HashFunction> hash_functions_;
  int num_buckets_;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_CUCKOO_HASHING_SPARSE_DPF_PIR_CLIENT_H_
