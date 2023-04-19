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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_PIR_CLIENT_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_PIR_CLIENT_H_

#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "pir/private_information_retrieval.pb.h"

namespace distributed_point_functions {

template <typename QueryType, typename ResponseType>
class PirClient {
 public:
  virtual ~PirClient() = default;

  // Creates a new PIR request for the given `query`. If successful, returns the
  // request together with the private key needed to decrypt the server's
  // response.
  virtual absl::StatusOr<std::pair<PirRequest, PirRequestClientState>>
  CreateRequest(QueryType query) const = 0;

  // Handles the server's `pir_response`. `request_client_state` is the
  // per-request client state corresponding to the request sent to the server.
  virtual absl::StatusOr<ResponseType> HandleResponse(
      const PirResponse& pir_response,
      const PirRequestClientState& request_client_state) const = 0;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_PIR_CLIENT_H_
