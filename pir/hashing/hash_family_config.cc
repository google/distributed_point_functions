// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pir/hashing/hash_family_config.h"

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "pir/hashing/hash_family.h"
#include "pir/hashing/sha256_hash_family.h"

namespace distributed_point_functions {

absl::StatusOr<HashFamily> CreateHashFamilyFromConfig(
    const HashFamilyConfig& config) {
  // Not strictly required but forces users to initialize their parameter
  // protos.
  if (config.seed().empty()) {
    return absl::InvalidArgumentError("`seed` must not be empty");
  }
  HashFamily family;
  switch (config.hash_family()) {
    case HashFamilyConfig::HASH_FAMILY_SHA256:
      family = SHA256HashFamily();
      break;
    case HashFamilyConfig::HASH_FAMILY_UNSPECIFIED:
      return absl::InvalidArgumentError("Hash family unspecified");
    default:
      return absl::InvalidArgumentError("Unknown hash family specified");
  }
  return WrapWithSeed(std::move(family), config.seed());
}

}  // namespace distributed_point_functions
