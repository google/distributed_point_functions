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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_TESTING_REQUEST_GENERATOR_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_TESTING_REQUEST_GENERATOR_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dpf/distributed_point_function.h"
#include "pir/private_information_retrieval.pb.h"

namespace distributed_point_functions {
namespace pir_testing {

// Generates syntactically correct PirRequests for testing PIR implementations.
class RequestGenerator {
 public:
  static absl::StatusOr<std::unique_ptr<RequestGenerator>> Create(
      int database_size, absl::string_view encryption_context_info);

  // Creates a pair of DpfPirRequest::PlainRequests for the given indices.
  absl::StatusOr<
      std::pair<DpfPirRequest::PlainRequest, DpfPirRequest::PlainRequest>>
  CreateDpfPirPlainRequests(absl::Span<const int> indices) const;

  // Creates a pair of DpfPirRequest::LeaderRequest for the given indices.
  absl::StatusOr<DpfPirRequest::LeaderRequest> CreateDpfPirLeaderRequest(
      absl::Span<const int> indices) const;

  // Returns the one-time-pad seed used for the HelperRequest in
  // CreateDpfPirLeaderRequest.
  absl::string_view otp_seed() const { return otp_seed_; }

 private:
  explicit RequestGenerator(std::unique_ptr<DistributedPointFunction> dpf,
                            std::string otp_seed,
                            std::string encryption_context_info,
                            int database_size);

  std::unique_ptr<DistributedPointFunction> dpf_;
  std::string otp_seed_;
  std::string encryption_context_info_;
  int database_size_;
};

}  // namespace pir_testing
}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_TESTING_REQUEST_GENERATOR_H_
