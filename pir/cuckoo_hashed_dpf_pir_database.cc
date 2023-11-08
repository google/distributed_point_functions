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

#include "pir/cuckoo_hashed_dpf_pir_database.h"

#include <stdint.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/memory/memory.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "dpf/status_macros.h"
#include "pir/dense_dpf_pir_database.h"
#include "pir/hashing/cuckoo_hash_table.h"
#include "pir/hashing/hash_family.h"
#include "pir/hashing/hash_family_config.h"

namespace distributed_point_functions {

namespace {
absl::Status CheckHasNotBeenBuilt(bool has_been_built) {
  if (has_been_built) {
    return absl::FailedPreconditionError("Database already built");
  }
  return absl::OkStatus();
}
}  // namespace

CuckooHashedDpfPirDatabase::Builder::Builder()
    : params_(),
      key_database_builder_(nullptr),
      value_database_builder_(nullptr),
      records_(),
      has_been_built_(false) {}

std::unique_ptr<CuckooHashedDpfPirDatabase::Interface::Builder>
CuckooHashedDpfPirDatabase::Builder::Clone() const {
  auto result = std::make_unique<Builder>();
  result->params_ = params_;
  if (key_database_builder_ != nullptr) {
    result->key_database_builder_ = key_database_builder_->Clone();
  }
  if (value_database_builder_ != nullptr) {
    result->value_database_builder_ = value_database_builder_->Clone();
  }
  result->records_ = records_;
  result->has_been_built_ = has_been_built_;
  return result;
}

CuckooHashedDpfPirDatabase::Builder&
CuckooHashedDpfPirDatabase::Builder::Insert(RecordType key_value) {
  records_.insert(std::move(key_value));
  return *this;
}

CuckooHashedDpfPirDatabase::Builder&
CuckooHashedDpfPirDatabase::Builder::Clear() {
  if (key_database_builder_ != nullptr) {
    key_database_builder_->Clear();
  }
  if (value_database_builder_ != nullptr) {
    value_database_builder_->Clear();
  }
  records_.clear();
  has_been_built_ = false;
  return *this;
}

CuckooHashedDpfPirDatabase::Builder&
CuckooHashedDpfPirDatabase::Builder::SetParams(CuckooHashingParams params) {
  params_ = std::move(params);
  return *this;
}

CuckooHashedDpfPirDatabase::Builder&
CuckooHashedDpfPirDatabase::Builder::SetKeyDatabaseBuilder(
    std::unique_ptr<DenseDatabase::Builder> builder) {
  if (builder != nullptr) {
    builder->Clear();
  }
  key_database_builder_ = std::move(builder);
  return *this;
}

CuckooHashedDpfPirDatabase::Builder&
CuckooHashedDpfPirDatabase::Builder::SetValueDatabaseBuilder(
    std::unique_ptr<DenseDatabase::Builder> builder) {
  if (builder != nullptr) {
    builder->Clear();
  }
  value_database_builder_ = std::move(builder);
  return *this;
}

absl::StatusOr<std::unique_ptr<CuckooHashedDpfPirDatabase::Interface>>
CuckooHashedDpfPirDatabase::Builder::Build() {
  DPF_RETURN_IF_ERROR(CheckHasNotBeenBuilt(has_been_built_));
  has_been_built_ = true;

  if (params_.num_buckets() <= 0) {
    return absl::InvalidArgumentError("`num_buckets` must be positive");
  }
  if (params_.num_hash_functions() <= 0) {
    return absl::InvalidArgumentError("`num_hash_functions` must be positive");
  }
  DPF_ASSIGN_OR_RETURN(
      HashFamily hash_family,
      CreateHashFamilyFromConfig(params_.hash_family_config()));

  // Create dense database builders if not set already.
  if (key_database_builder_ == nullptr) {
    key_database_builder_ = std::make_unique<DenseDpfPirDatabase::Builder>();
  }
  if (value_database_builder_ == nullptr) {
    value_database_builder_ = std::make_unique<DenseDpfPirDatabase::Builder>();
  }

  // Cuckoo hash all the keys.
  int64_t num_records = records_.size();
  DPF_ASSIGN_OR_RETURN(
      auto cuckoo_hasher,
      CuckooHashTable::Create(std::move(hash_family), params_.num_buckets(),
                              params_.num_hash_functions(), num_records));
  for (const auto& [key, _] : records_) {
    if (key.empty()) {
      return absl::InvalidArgumentError("Key cannot be empty");
    }
    DPF_RETURN_IF_ERROR(cuckoo_hasher->Insert(key));
  }

  // For each key in the cuckoo hash table, insert it into key_database_ and
  // the corresponding value into value_database_.
  absl::Span<const absl::optional<std::string>> cuckoo_table =
      cuckoo_hasher->GetTable();
  for (int i = 0; i < cuckoo_table.size(); ++i) {
    if (cuckoo_table[i].has_value()) {
      const std::string& key = cuckoo_table[i].value();
      key_database_builder_->Insert(key);
      value_database_builder_->Insert(
          std::move(records_.extract(key).mapped()));
    } else {  // Insert dummy strings.
      key_database_builder_->Insert("");
      value_database_builder_->Insert("");
    }
  }

  DPF_ASSIGN_OR_RETURN(auto key_database, key_database_builder_->Build());
  DPF_ASSIGN_OR_RETURN(auto value_database, value_database_builder_->Build());

  size_t num_selection_bits = key_database->num_selection_bits();
  if (num_selection_bits != value_database->num_selection_bits() ||
      num_selection_bits != params_.num_buckets()) {
    return absl::InternalError(
        "Number of selection bits in underlying databases doesn't match");
  }

  return absl::WrapUnique(new CuckooHashedDpfPirDatabase(
      std::move(key_database), std::move(value_database), num_records,
      num_selection_bits));
}

absl::StatusOr<std::vector<CuckooHashedDpfPirDatabase::RecordType>>
CuckooHashedDpfPirDatabase::InnerProductWith(
    absl::Span<const std::vector<BlockType>> selections) const {
  DPF_ASSIGN_OR_RETURN(std::vector<std::string> keys,
                       key_database_->InnerProductWith(selections));
  DPF_ASSIGN_OR_RETURN(std::vector<std::string> values,
                       value_database_->InnerProductWith(selections));
  if (keys.size() != values.size() || keys.size() != selections.size()) {
    return absl::InternalError(
        "Result sizes do not match. This should not happen.");
  }

  std::vector<RecordType> result;
  result.reserve(keys.size());
  for (auto kit = keys.begin(), vit = values.begin();
       kit != keys.end() && vit != values.end(); ++kit, ++vit) {
    result.push_back({std::move(*kit), std::move(*vit)});
  }
  return result;
}

CuckooHashedDpfPirDatabase::CuckooHashedDpfPirDatabase(
    std::unique_ptr<DenseDatabase> key_database,
    std::unique_ptr<DenseDatabase> value_database, size_t size,
    size_t num_selection_bits)
    : key_database_(std::move(key_database)),
      value_database_(std::move(value_database)),
      size_(size),
      num_selection_bits_(num_selection_bits) {}

}  // namespace distributed_point_functions
