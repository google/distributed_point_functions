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
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "dpf/distributed_point_function.h"
#include "pir/dense_dpf_pir_server.h"
#include "pir/dpf_pir_client.h"
#include "pir/private_information_retrieval.pb.h"

namespace distributed_point_functions {

class DenseDpfPirClient
    : public DpfPirClient<absl::Span<const int>, std::vector<std::string>> {
 public:
  // Creates a new DenseDpfPirClient instance with the given PirConfig and
  // an `encrypter` function that should wrap around an implementation of
  // `crypto::tink::HybridEncrypt::Encrypt()`. See the documentation of
  // DpfPirClient for more details about the type of `encrypter`.
  //
  // Returns INVALID_ARGUMENT if `config` is invalid, or if `encrypter` is NULL.
  static absl::StatusOr<std::unique_ptr<DenseDpfPirClient>> Create(
      const PirConfig& config, EncryptHelperRequestFn encrypter,
      absl::string_view encryption_context_info =
          DenseDpfPirServer::kEncryptionContextInfo);

  virtual ~DenseDpfPirClient() = default;

  // Creates a pair of plain PIR requests for the given `query`. If successful,
  // returns the requests together with the private state needed to decrypt the
  // server's response.
  //
  // Returns INVALID_ARGUMENT if any element of `query_indices` is negative, or
  // out of bounds for the database specified in the config passed at
  // construction.
  virtual absl::StatusOr<
      std::tuple<DpfPirRequest::PlainRequest, DpfPirRequest::HelperRequest,
                 PirRequestClientState>>
  CreatePlainRequests(absl::Span<const int> query_indices) const override;

  // Handles the server's `pir_response`. `request_client_state` is the
  // per-request client state corresponding to the request sent to the server.
  //
  // For each query index passed to the corresponding `CreateRequest` call,
  // returns the database value at that index. The returned values will be
  // padded with null bytes to the size of the largest database entry. Returns
  // INVALID_ARGUMENT if either the response or the client state is invalid.
  virtual absl::StatusOr<std::vector<std::string>> HandleResponse(
      const PirResponse& pir_response,
      const PirRequestClientState& request_client_state) const override;

 private:
  static constexpr int kBitsPerBlock = 8 * sizeof(absl::uint128);

  DenseDpfPirClient(std::unique_ptr<DistributedPointFunction> dpf,
                    EncryptHelperRequestFn encrypter,
                    std::string encryption_context_info, int database_size);

  std::unique_ptr<DistributedPointFunction> dpf_;
  int database_size_;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_DENSE_DPF_PIR_CLIENT_H_
