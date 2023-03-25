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
  static absl::StatusOr<std::unique_ptr<DenseDpfPirClient>> Create(
      const PirConfig& config,
      std::unique_ptr<crypto::tink::HybridEncrypt> encrypter);

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
                    std::unique_ptr<crypto::tink::HybridEncrypt> encrypter,
                    int database_size);

  std::unique_ptr<DistributedPointFunction> dpf_;
  std::unique_ptr<crypto::tink::HybridEncrypt> encrypter_;
  int database_size_;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_DENSE_DPF_PIR_CLIENT_H_
