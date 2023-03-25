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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_HASHING_SHA256_HASH_FAMILY_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_HASHING_SHA256_HASH_FAMILY_H_

#include "openssl/sha.h"
#include "pir/hashing/hash_family.h"

namespace distributed_point_functions {

// SHA256HashFunction is a HashFunction that uses SHA256.
class SHA256HashFunction {
 public:
  // Initializes the SHA256 state with a seed.
  explicit SHA256HashFunction(absl::string_view seed);

  // Clears the SHA256 state.
  ~SHA256HashFunction();

  // Computes the hash in `input` as SHA256(seed || input), and reduces the hash
  // to the range [0,upper_bound).
  int operator()(absl::string_view input, int upper_bound) const;

 private:
  SHA256_CTX ctx_;
};

// SHA256HashFamily is a HashFamily that creates SHA256HashFunctions.
struct SHA256HashFamily {
  HashFunction operator()(absl::string_view seed) const {
    return SHA256HashFunction(seed);
  }
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_HASHING_SHA256_HASH_FAMILY_H_
