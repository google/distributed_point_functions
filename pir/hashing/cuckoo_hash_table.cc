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

#include "pir/hashing/cuckoo_hash_table.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"

namespace distributed_point_functions {

CuckooHashTable::CuckooHashTable(std::vector<HashFunction> hash_functions,
                                 int num_buckets, int max_relocations,
                                 absl::optional<int> max_stash_size)
    : num_buckets_(num_buckets),
      max_relocations_(max_relocations),
      max_stash_size_(max_stash_size),
      hash_functions_(std::move(hash_functions)),
      table_(num_buckets),
      random_hash_function_(0, hash_functions_.size() - 1) {
  if (max_stash_size) {
    stash_.reserve(*max_stash_size);
  }
}

absl::StatusOr<std::unique_ptr<CuckooHashTable>> CuckooHashTable::Create(
    std::vector<HashFunction> hash_functions, int num_buckets,
    int max_relocations, absl::optional<int> max_stash_size) {
  if (num_buckets <= 0) {
    return absl::InvalidArgumentError("num_buckets must be positive");
  }
  if (hash_functions.size() < 2) {
    return absl::InvalidArgumentError(
        "hash_functions.size() must be at least 2");
  }
  if (max_relocations < 0) {
    return absl::InvalidArgumentError("max_relocations must be non-negative");
  }
  if (max_stash_size && *max_stash_size < 0) {
    return absl::InvalidArgumentError("max_stash_size must be non-negative");
  }
  return absl::WrapUnique(new CuckooHashTable(
      std::move(hash_functions), num_buckets, max_relocations, max_stash_size));
}

absl::Status CuckooHashTable::Insert(absl::string_view input) {
  std::string current_element(input);
  for (int i = 0; i < max_relocations_; i++) {
    // Choose a random hash function and hash the current element.
    int hash = hash_functions_[random_hash_function_(rng_)](current_element,
                                                            num_buckets_);
    if (table_[hash]) {
      // If bucket is full, evict element and re-insert it recursively.
      std::swap(current_element, *table_[hash]);
    } else {
      // Otherwise just insert our current element and return.
      table_[hash] = std::move(current_element);
      return absl::OkStatus();
    }
  }
  // If we're still here after max_relocations_, put current_element on the
  // stash.
  if (max_stash_size_ && stash_.size() >= *max_stash_size_) {
    return absl::InternalError("Cannot insert element: stash is full");
  } else {
    stash_.push_back(std::move(current_element));
    return absl::OkStatus();
  }
}

}  // namespace distributed_point_functions
