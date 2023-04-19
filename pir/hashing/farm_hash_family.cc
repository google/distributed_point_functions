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

#include "pir/hashing/farm_hash_family.h"

#include <utility>

#include "absl/numeric/int128.h"
#include "absl/strings/string_view.h"

namespace distributed_point_functions {

int FarmHashFunction::operator()(absl::string_view input,
                                 int upper_bound) const {
  auto hash = util::Hash128WithSeed(input.data(), input.length(), seed_);
  absl::uint128 absl_hash = absl::MakeUint128(hash.second, hash.first);
  return static_cast<int>(absl_hash % absl::uint128(upper_bound));
}

}  // namespace distributed_point_functions
