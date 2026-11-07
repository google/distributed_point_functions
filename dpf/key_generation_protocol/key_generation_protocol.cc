// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dpf/key_generation_protocol/key_generation_protocol.h"

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "dpf/distributed_point_function.h"
#include "dpf/status_macros.h"

namespace distributed_point_functions {

KeyGenerationProtocol::KeyGenerationProtocol(
    std::unique_ptr<DistributedPointFunction> dpf, int party)
    : dpf_(std::move(dpf)), party_(party) {}

absl::StatusOr<std::unique_ptr<KeyGenerationProtocol>>
KeyGenerationProtocol::Create(absl::Span<const DpfParameters> parameters,
                              int party) {
  if (party != 0 && party != 1) {
    return absl::InvalidArgumentError("`party` must be 0 or 1");
  }
  DPF_ASSIGN_OR_RETURN(auto dpf,
                       DistributedPointFunction::CreateIncremental(parameters));
  return absl::WrapUnique(new KeyGenerationProtocol(std::move(dpf), party));
}

}  // namespace distributed_point_functions