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

// Stores elements in a vector of buckets, which grow dynamically as elements
// are inserted. The number of hash functions determines the number of copies
// that are stored of each element.

#ifndef PRIVACY_PRIVATE_MEMBERSHIP_INTERNAL_HASHING_SIMPLE_HASH_TABLE_H_
#define PRIVACY_PRIVATE_MEMBERSHIP_INTERNAL_HASHING_SIMPLE_HASH_TABLE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "dpf/status_macros.h"
#include "pir/hashing/hash_family.h"

namespace distributed_point_functions {

class SimpleHashTable {
 public:
  // Creates a SimpleHashTable with the given hash functions, number of buckets,
  // and maximum bucket size.
  //
  // Returns INVALID_ARGUMENT if hash_functions is empty, or if num_buckets or
  // max_bucket_size are negative.
  static absl::StatusOr<std::unique_ptr<SimpleHashTable>> Create(
      std::vector<HashFunction> hash_functions, int num_buckets,
      absl::optional<int> max_bucket_size = absl::optional<int>());

  // Overload that creates num_hash_functions hash functions from the given
  // HashFamily.
  static inline absl::StatusOr<std::unique_ptr<SimpleHashTable>> Create(
      HashFamily hash_family, int num_buckets, int num_hash_functions = 1,
      absl::optional<int> max_bucket_size = absl::optional<int>()) {
    DPF_ASSIGN_OR_RETURN(
        std::vector<HashFunction> hash_functions,
        CreateHashFunctions(std::move(hash_family), num_hash_functions));
    return Create(std::move(hash_functions), num_buckets, max_bucket_size);
  }

  // SimpleHashTable is neither copyable nor movable.
  SimpleHashTable(const SimpleHashTable&) = delete;
  SimpleHashTable& operator=(const SimpleHashTable&) = delete;

  // Inserts an element by hashing it to an index in table_ and appending it to
  // the corresponding bucket.
  //
  // Returns OK if insertion was successful and INTERNAL if one of the buckets
  // reached its maximum size.
  absl::Status Insert(absl::string_view input);

  const std::vector<std::vector<std::string>>& GetTable() const {
    return table_;
  }

  // Returns a copy of the hash functions used in this table.
  absl::Span<const HashFunction> GetHashFunctions() const {
    return hash_functions_;
  }

 private:
  SimpleHashTable(std::vector<HashFunction> hash_functions, int num_buckets,
                  absl::optional<int> max_bucket_size);

  const int num_buckets_;
  const absl::optional<int> max_bucket_size_;
  const std::vector<HashFunction> hash_functions_;

  std::vector<std::vector<std::string>> table_;
};

}  // namespace distributed_point_functions

#endif  // PRIVACY_PRIVATE_MEMBERSHIP_INTERNAL_HASHING_SIMPLE_HASH_TABLE_H_
