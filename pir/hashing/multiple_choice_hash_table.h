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

// Hashes elements using the multiple-choice method. That is, each inserted
// element is hashed to multiple locations using a fixed number of hash
// functions, and the least occupied bucket is chosen.
#ifndef PRIVACY_PRIVATE_MEMBERSHIP_INTERNAL_HASHING_MULTIPLE_CHOICE_HASH_TABLE_H_
#define PRIVACY_PRIVATE_MEMBERSHIP_INTERNAL_HASHING_MULTIPLE_CHOICE_HASH_TABLE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "dpf/status_macros.h"
#include "pir/hashing/hash_family.h"

namespace distributed_point_functions {

class MultipleChoiceHashTable {
 public:
  // Creates a MultipleChoiceHashTable with the given hash function, number of
  // buckets, and optional maximum bucket size.
  //
  // Returns INVALID_ARGUMENT if hash_functions.size() < 2, or if num_buckets or
  // max_bucket_size are negative.
  static absl::StatusOr<std::unique_ptr<MultipleChoiceHashTable>> Create(
      std::vector<HashFunction> hash_functions, int num_buckets,
      absl::optional<int> max_bucket_size = absl::optional<int>());

  // Overload that creates num_hash_functions hash functions from the given
  // HashFamily.
  static inline absl::StatusOr<std::unique_ptr<MultipleChoiceHashTable>> Create(
      HashFamily hash_family, int num_buckets, int num_hash_functions = 2,
      absl::optional<int> max_bucket_size = absl::optional<int>()) {
    DPF_ASSIGN_OR_RETURN(
        std::vector<HashFunction> hash_functions,
        CreateHashFunctions(std::move(hash_family), num_hash_functions));
    return Create(std::move(hash_functions), num_buckets, max_bucket_size);
  }

  // SimpleHashTable is neither copyable nor movable.
  MultipleChoiceHashTable(const MultipleChoiceHashTable&) = delete;
  MultipleChoiceHashTable& operator=(const MultipleChoiceHashTable&) = delete;

  // Inserts an element by hashing it using each hash function and inserting it
  // into the bucket with the least elements.
  //
  // Returns OK if insertion was successful and INTERNAL if all buckets reached
  // their maximum size.
  absl::Status Insert(absl::string_view input);

  const std::vector<std::vector<std::string>>& GetTable() const {
    return table_;
  }

 private:
  MultipleChoiceHashTable(std::vector<HashFunction> hash_functions,
                          int num_buckets, absl::optional<int> max_bucket_size);

  const int num_buckets_;
  const absl::optional<int> max_bucket_size_;
  const std::vector<HashFunction> hash_functions_;

  std::vector<std::vector<std::string>> table_;
};

}  // namespace distributed_point_functions

#endif  // PRIVACY_PRIVATE_MEMBERSHIP_INTERNAL_HASHING_MULTIPLE_CHOICE_HASH_TABLE_H_
