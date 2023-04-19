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

#include "pir/hashing/hash_family.h"

#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace distributed_point_functions {

absl::StatusOr<std::vector<HashFunction>> CreateHashFunctions(
    HashFamily hash_family, int num_hash_functions) {
  if (num_hash_functions < 0) {
    return absl::InvalidArgumentError(
        "num_hash_functions must not be negative");
  }
  std::vector<HashFunction> result;
  result.reserve(num_hash_functions);
  for (int i = 0; i < num_hash_functions; i++) {
    result.push_back(hash_family(absl::StrCat(i)));
  }
  return result;
}

}  // namespace distributed_point_functions
