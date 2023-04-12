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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_DENSE_DPF_PIR_CLIENT_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_DENSE_DPF_PIR_CLIENT_H_

#include <memory>

#include "absl/numeric/int128.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dpf/distributed_point_function.h"
#include "pir/pir_client.h"
#include "tink/hybrid_encrypt.h"

namespace distributed_point_functions {

class DenseDpfPirClient
    : public PirClient<absl::Span<const int>, std::vector<std::string>> {
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
  using EncryptHelperRequestFn =
      std::function<crypto::tink::util::StatusOr<std::string>(
          absl::string_view plain_helper_request,
          absl::string_view encryption_context_info)>;

  // Creates a new DenseDpfPirClient instance with the given PirConfig and
  // an `encrypter` function that should wrap around an implementation of
  // `crypto::tink::HybridEncrypt::Encrypt()`. See above for more details about
  // the type of `encrypter`.
  //
  // Returns INVALID_ARGUMENT if `config` is invalid, or if `encrypter` is NULL.
  static absl::StatusOr<std::unique_ptr<DenseDpfPirClient>> Create(
      const PirConfig& config, EncryptHelperRequestFn encrypter);

  virtual ~DenseDpfPirClient() = default;

  // Creates a new PIR request for the given `query_indices`. If successful,
  // returns the request together with the private key needed to decrypt the
  // server's response.
  virtual absl::StatusOr<std::pair<PirRequest, PirRequestPrivateKey>>
  CreateRequest(absl::Span<const int> query_indices) const override;

  // Handles the server's `pir_response`. `decryption_key` is the per-request
  // key corresponding to the request sent to the server.
  virtual absl::StatusOr<std::vector<std::string>> HandleResponse(
      const PirResponse& pir_response,
      const PirRequestPrivateKey& decryption_key) const override;

 private:
  static constexpr int kBitsPerBlock = 8 * sizeof(absl::uint128);

  DenseDpfPirClient(std::unique_ptr<DistributedPointFunction> dpf,
                    EncryptHelperRequestFn encrypter, int database_size);

  std::unique_ptr<DistributedPointFunction> dpf_;
  EncryptHelperRequestFn encrypter_;
  int database_size_;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_DENSE_DPF_PIR_CLIENT_H_
