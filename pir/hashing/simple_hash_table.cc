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

#include "pir/hashing/simple_hash_table.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"

namespace distributed_point_functions {

absl::StatusOr<std::unique_ptr<SimpleHashTable>> SimpleHashTable::Create(
    std::vector<HashFunction> hash_functions, int num_buckets,
    absl::optional<int> max_bucket_size) {
  if (num_buckets <= 0) {
    return absl::InvalidArgumentError("num_buckets must be positive");
  }
  if (hash_functions.empty()) {
    return absl::InvalidArgumentError("hash_functions must not be empty");
  }
  if (max_bucket_size && *max_bucket_size <= 0) {
    return absl::InvalidArgumentError("max_bucket_size must be positive");
  }
  return absl::WrapUnique(new SimpleHashTable(std::move(hash_functions),
                                              num_buckets, max_bucket_size));
}

SimpleHashTable::SimpleHashTable(std::vector<HashFunction> hash_functions,
                                 int num_buckets,
                                 absl::optional<int> max_bucket_size)
    : num_buckets_(num_buckets),
      max_bucket_size_(max_bucket_size),
      hash_functions_(std::move(hash_functions)),
      table_(num_buckets) {}

absl::Status SimpleHashTable::Insert(absl::string_view input) {
  std::vector<int> hashes(hash_functions_.size());
  for (int i = 0; i < hash_functions_.size(); i++) {
    hashes[i] = hash_functions_[i](input, num_buckets_);
    if (max_bucket_size_ && table_[hashes[i]].size() >= *max_bucket_size_) {
      return absl::InternalError(
          "Cannot insert element: maximum bucket size reached");
    }
  }
  // Second loop to ensure that an element is either inserted into all buckets,
  // or none in case of an error.
  for (int i = 0; i < hash_functions_.size(); i++) {
    table_[hashes[i]].push_back(std::string(input));
  }
  return absl::OkStatus();
}

}  // namespace distributed_point_functions
