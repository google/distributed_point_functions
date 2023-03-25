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

// A CuckooHashTable hashes elements to indexes into a single dense vector using
// multiple hash functions. Thus, each element has a constant number of possible
// locations. Hash collisions are resolved by evicting one of the colliding
// elements and re-inserting it recursively at a different position.

#ifndef PRIVACY_PRIVATE_MEMBERSHIP_INTERNAL_HASHING_CUCKOO_HASH_TABLE_H_
#define PRIVACY_PRIVATE_MEMBERSHIP_INTERNAL_HASHING_CUCKOO_HASH_TABLE_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "dpf/status_macros.h"
#include "pir/hashing/hash_family.h"

namespace distributed_point_functions {

class CuckooHashTable {
 public:
  // Constructs a CuckooHashTable with the given hash functions, number of
  // buckets num_buckets, and limit max_relocations on the depth of the
  // recursion during Insert. If set and positive, max_stash_size limits the
  // size of the stash, otherwise the stash size is unlimited.
  static absl::StatusOr<std::unique_ptr<CuckooHashTable>> Create(
      std::vector<HashFunction> hash_functions, int num_buckets,
      int max_relocations,
      absl::optional<int> max_stash_size = absl::optional<int>());

  // Overload that creates num_hash_functions hash functions from the given
  // HashFamily.
  static inline absl::StatusOr<std::unique_ptr<CuckooHashTable>> Create(
      HashFamily hash_family, int num_buckets, int num_hash_functions,
      int max_relocations,
      absl::optional<int> max_stash_size = absl::optional<int>()) {
    DPF_ASSIGN_OR_RETURN(
        std::vector<HashFunction> hash_functions,
        CreateHashFunctions(std::move(hash_family), num_hash_functions));
    return Create(std::move(hash_functions), num_buckets, max_relocations,
                  max_stash_size);
  }

  // CuckooHashTable is neither copyable nor movable.
  CuckooHashTable(const CuckooHashTable&) = delete;
  CuckooHashTable& operator=(const CuckooHashTable&) = delete;

  // Inserts an element into the table by hashing to indices between 0 and
  // num_buckets_ - 1, using each of the num_hash_functions_ hash functions.
  // If one of the resulting indices is not occupied, the input is stored there.
  // Otherwise, the element at a random index is evicted and replaced by the
  // input and subsequently re-inserted recursively. After max_relocations_ re-
  // insertions the element is put on the stash.
  //
  // Returns INTERNAL max_stash_size_ is non-negative and the stash exceeds it,
  // OK otherwise.
  absl::Status Insert(absl::string_view input);

  const std::vector<absl::optional<std::string>>& GetTable() const {
    return table_;
  }

  const std::vector<std::string>& GetStash() const { return stash_; }

  // Returns a reference to the hash functions used in this table.
  //
  // Currently being used in experiments under experimental/blinders.
  absl::Span<const HashFunction> GetHashFunctions() const {
    return hash_functions_;
  }

 private:
  CuckooHashTable(std::vector<HashFunction> hash_functions, int num_buckets,
                  int max_relocations, absl::optional<int> max_stash_size);

  const int num_buckets_;
  const int max_relocations_;
  const absl::optional<int> max_stash_size_;
  const std::vector<HashFunction> hash_functions_;

  std::vector<absl::optional<std::string>> table_;
  std::vector<std::string> stash_;
  // Random number generator used to deterministically choose element to evict
  // on collisions.
  std::mt19937_64 rng_;
  absl::uniform_int_distribution<int> random_hash_function_;
};

}  // namespace distributed_point_functions

#endif  // PRIVACY_PRIVATE_MEMBERSHIP_INTERNAL_HASHING_CUCKOO_HASH_TABLE_H_
