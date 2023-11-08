// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pir/simple_hashed_dpf_pir_database.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dpf/status_macros.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "pir/dense_dpf_pir_database.h"
#include "pir/hashing/hash_family.h"
#include "pir/hashing/hash_family_config.h"
#include "pir/private_information_retrieval.pb.h"

namespace distributed_point_functions {

namespace {
absl::Status CheckHasNotBeenBuilt(bool has_been_built) {
  if (has_been_built) {
    return absl::FailedPreconditionError("Database already built");
  }
  return absl::OkStatus();
}
}  // namespace

SimpleHashedDpfPirDatabase::Builder::Builder()
    : params_(), dense_database_builder_(nullptr), has_been_built_(false) {}

std::unique_ptr<SimpleHashedDpfPirDatabase::Interface::Builder>
SimpleHashedDpfPirDatabase::Builder::Clone() const {
  auto result = std::make_unique<Builder>();
  result->params_ = params_;
  if (dense_database_builder_ != nullptr) {
    result->dense_database_builder_ = dense_database_builder_->Clone();
  }
  result->records_ = records_;
  result->has_been_built_ = has_been_built_;
  return result;
}

SimpleHashedDpfPirDatabase::Builder&
SimpleHashedDpfPirDatabase::Builder::Clear() {
  if (dense_database_builder_ != nullptr) {
    dense_database_builder_->Clear();
  }
  records_.clear();
  has_been_built_ = false;
  return *this;
}

SimpleHashedDpfPirDatabase::Builder&
SimpleHashedDpfPirDatabase::Builder::SetParams(SimpleHashingParams params) {
  params_ = std::move(params);
  return *this;
}

SimpleHashedDpfPirDatabase::Builder&
SimpleHashedDpfPirDatabase::Builder::SetDenseDatabaseBuilder(
    std::unique_ptr<DenseDatabase::Builder> builder) {
  if (builder != nullptr) {
    builder->Clear();
  }
  dense_database_builder_ = std::move(builder);
  return *this;
}

SimpleHashedDpfPirDatabase::Builder&
SimpleHashedDpfPirDatabase::Builder::Insert(RecordType key_value) {
  records_.push_back(std::move(key_value));
  return *this;
}

absl::StatusOr<std::unique_ptr<SimpleHashedDpfPirDatabase::Interface>>
SimpleHashedDpfPirDatabase::Builder::Build() {
  DPF_RETURN_IF_ERROR(CheckHasNotBeenBuilt(has_been_built_));
  has_been_built_ = true;
  int num_buckets = params_.num_buckets();
  int num_records = records_.size();

  if (num_buckets <= 0) {
    return absl::InvalidArgumentError("`num_buckets` must be positive");
  }
  DPF_ASSIGN_OR_RETURN(
      HashFamily hash_family,
      CreateHashFamilyFromConfig(params_.hash_family_config()));
  DPF_ASSIGN_OR_RETURN(auto hash_functions,
                       CreateHashFunctions(std::move(hash_family), 1));
  HashFunction& hash_function = hash_functions[0];

  // Create dense database builder if not set already.
  if (dense_database_builder_ == nullptr) {
    dense_database_builder_ = std::make_unique<DenseDpfPirDatabase::Builder>();
  }

  // Allocate protos for buckets. The final bucket will consist of a single
  // serialized string containing all key-value pairs that hashed to it.
  std::vector<HashedPirDatabaseBucket> bucket_protos(num_buckets);

  // Hash all keys, and insert the key-value pairs into the resulting bucket.
  for (auto& kv : records_) {
    if (kv.first.empty()) {
      return absl::InvalidArgumentError("Key cannot be empty");
    }
    int bucket_index = hash_function(kv.first, num_buckets);
    *(bucket_protos[bucket_index].add_keys()) = std::move(kv.first);
    *(bucket_protos[bucket_index].add_values()) = std::move(kv.second);
  }

  // Serialize all Key-Value pairs deterministically, and insert the resulting
  // strings into the dense database builder.
  for (int i = 0; i < num_buckets; ++i) {
    std::string serialized_bucket;
    {  // Start new block so that stream destructors are run before moving the
       // string.
      ::google::protobuf::io::StringOutputStream string_stream(
          &serialized_bucket);
      ::google::protobuf::io::CodedOutputStream coded_stream(&string_stream);
      coded_stream.SetSerializationDeterministic(true);
      if (!bucket_protos[i].SerializeToCodedStream(&coded_stream)) {
        return absl::InternalError("Serializing bucket to string failed");
      }
    }
    dense_database_builder_->Insert(std::move(serialized_bucket));
  }
  records_.clear();

  DPF_ASSIGN_OR_RETURN(std::unique_ptr<DenseDatabase> dense_database,
                       dense_database_builder_->Build());
  return absl::WrapUnique(new SimpleHashedDpfPirDatabase(
      std::move(dense_database), num_records, num_buckets));
}

SimpleHashedDpfPirDatabase::SimpleHashedDpfPirDatabase(
    std::unique_ptr<DenseDatabase> dense_database, size_t size,
    size_t num_selection_bits)
    : dense_database_(std::move(dense_database)),
      size_(size),
      num_selection_bits_(num_selection_bits) {}

absl::StatusOr<std::vector<SimpleHashedDpfPirDatabase::ResponseType>>
SimpleHashedDpfPirDatabase::InnerProductWith(
    absl::Span<const std::vector<BlockType>> selections) const {
  return dense_database_->InnerProductWith(selections);
}

}  // namespace distributed_point_functions
