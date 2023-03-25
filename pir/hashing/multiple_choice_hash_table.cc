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

#include "pir/hashing/multiple_choice_hash_table.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"

namespace distributed_point_functions {

absl::StatusOr<std::unique_ptr<MultipleChoiceHashTable>>
MultipleChoiceHashTable::Create(std::vector<HashFunction> hash_functions,
                                int num_buckets,
                                absl::optional<int> max_bucket_size) {
  if (num_buckets <= 0) {
    return absl::InvalidArgumentError("num_buckets must be positive");
  }
  if (hash_functions.size() < 2) {
    return absl::InvalidArgumentError(
        "hash_functions.size() must be at least 2");
  }
  if (max_bucket_size && *max_bucket_size <= 0) {
    return absl::InvalidArgumentError("max_bucket_size must be positive");
  }
  return absl::WrapUnique(new MultipleChoiceHashTable(
      std::move(hash_functions), num_buckets, max_bucket_size));
}

MultipleChoiceHashTable::MultipleChoiceHashTable(
    std::vector<HashFunction> hash_functions, int num_buckets,
    absl::optional<int> max_bucket_size)
    : num_buckets_(num_buckets),
      max_bucket_size_(max_bucket_size),
      hash_functions_(std::move(hash_functions)),
      table_(num_buckets) {}

absl::Status MultipleChoiceHashTable::Insert(absl::string_view input) {
  std::vector<int> hashes(hash_functions_.size());
  int smallest_bucket = 0;
  for (int i = 0; i < hash_functions_.size(); i++) {
    hashes[i] = hash_functions_[i](input, num_buckets_);
    if (i == 0 || table_[hashes[i]].size() < table_[smallest_bucket].size()) {
      smallest_bucket = hashes[i];
    }
  }
  if (max_bucket_size_ && table_[smallest_bucket].size() >= *max_bucket_size_) {
    return absl::InternalError(
        "Cannot insert element: maximum bucket size reached");
  }
  table_[smallest_bucket].push_back(std::string(input));
  return absl::OkStatus();
}

}  // namespace distributed_point_functions
