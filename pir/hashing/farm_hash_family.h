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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_HASHING_FARM_HASH_FAMILY_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_HASHING_FARM_HASH_FAMILY_H_

#include "absl/strings/string_view.h"
#include "farmhash/farmhash.h"
#include "pir/hashing/hash_family.h"

namespace distributed_point_functions {

// FarmHashFunction is a HashFunction that uses farmhash.
class FarmHashFunction {
 public:
  explicit FarmHashFunction(absl::string_view seed)
      : seed_(util::Hash128(seed)) {}
  int operator()(absl::string_view input, int upper_bound) const;

 private:
  util::uint128_t seed_;
};

// FarmHashFamily is a HashFamily that creates FarmHashFunctions.
struct FarmHashFamily {
  HashFunction operator()(absl::string_view seed) const {
    return FarmHashFunction(seed);
  }
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_HASHING_FARM_HASH_FAMILY_H_
