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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_HASHING_HASH_FAMILY_CONFIG_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_HASHING_HASH_FAMILY_CONFIG_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "pir/hashing/hash_family.h"
#include "pir/hashing/hash_family_config.pb.h"

namespace distributed_point_functions {

// Creates a HashFamily from the given `config` and `seed`. Returns
// INVALID_ARGUMENT if `config` is invalid or `seed.empty()`.
absl::StatusOr<HashFamily> CreateHashFamilyFromConfig(
    const HashFamilyConfig& config);

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_HASHING_HASH_FAMILY_CONFIG_H_
