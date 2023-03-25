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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_PIR_SERVER_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_PIR_SERVER_H_

#include "absl/status/statusor.h"
#include "pir/private_information_retrieval.pb.h"

namespace distributed_point_functions {

class PirServer {
 public:
  virtual ~PirServer() = default;

  // Returns public parameters to be used by the client for creating requests
  // in some PIR schemes.
  virtual const PirServerPublicParams& GetPublicParams() const = 0;

  // Computes the response to the client's `request`.
  virtual absl::StatusOr<PirResponse> HandleRequest(
      const PirRequest& request) const = 0;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_PIR_SERVER_H_
