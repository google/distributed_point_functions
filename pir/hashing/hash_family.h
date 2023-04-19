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

// This header file defines HashFunction and HashFamily types. A HashFunction in
// this context takes a string_view and hashes it to an integer range. A
// HashFamily returns a HashFunction given a seed as a string_view.

#ifndef PRIVACY_PRIVATE_MEMBERSHIP_INTERNAL_HASHING_HASH_FAMILY_H_
#define PRIVACY_PRIVATE_MEMBERSHIP_INTERNAL_HASHING_HASH_FAMILY_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace distributed_point_functions {

// A HashFunction is any function object that hashes a string to a value between
// 0 and an upper bound.
using HashFunction = absl::AnyInvocable<int(absl::string_view, int) const>;

// A HashFamily is a function that returns a HashFunction given a seed.
using HashFamily = absl::AnyInvocable<HashFunction(absl::string_view) const>;

// Wraps a HashFamily with a given seed value. This allows obtaining multiple
// randomized hash families from a single one. The seed is prepended to each
// seed used in subsequent calls to operator().
struct WrapWithSeed {
  WrapWithSeed(HashFamily family, absl::string_view family_seed)
      : family(std::move(family)), family_seed(family_seed) {}
  HashFunction operator()(absl::string_view seed) const {
    return family(absl::StrCat(family_seed, seed));
  }
  HashFamily family;
  std::string family_seed;
};

// Helper function to generate a vector of HashFunctions from a HashFamily.
//
// Returns INVALID_ARGUMENT if num_hash_functions is negative.
absl::StatusOr<std::vector<HashFunction>> CreateHashFunctions(
    HashFamily hash_family, int num_hash_functions);

}  // namespace distributed_point_functions

#endif  // PRIVACY_PRIVATE_MEMBERSHIP_INTERNAL_HASHING_HASH_FAMILY_H_
