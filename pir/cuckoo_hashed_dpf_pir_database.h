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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_SPARSE_DPF_PIR_DATABASE_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_SPARSE_DPF_PIR_DATABASE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/numeric/int128.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dpf/xor_wrapper.h"
#include "pir/pir_database_interface.h"
#include "pir/private_information_retrieval.pb.h"

namespace distributed_point_functions {

// Cuckoo Hashing based database for key-value pairs, to be used in sparse
// two-server PIR. Wraps a dense database. Keys are first inserted into a cuckoo
// hash table, assigning each key to a unique bucket. Then, two dense databases
// are used to to store keys and values separately. When computing the inner
// product with a selection vector, this class computes the two inner products
// with the key and value databases, and combines the results into a std;:pair.
class CuckooHashedDpfPirDatabase
    : public PirDatabaseInterface<XorWrapper<absl::uint128>,
                                  std::pair<std::string, std::string>> {
 public:
  using Interface = PirDatabaseInterface;
  // Type of the underlying database implementation that stores keys and values
  // separately.
  using DenseDatabase =
      PirDatabaseInterface<XorWrapper<absl::uint128>, std::string>;

  // The concrete Builder for CuckooHashedDpfPirDatabase.
  class Builder : public PirDatabaseInterface::Builder {
   public:
    Builder();
    // Inserts the given key-value pair into the database once Build() is
    // called.
    Builder& Insert(std::pair<std::string, std::string>) override;
    // Sets the cuckoo hashing parameters used for this database. Must be called
    // before calling `Build`.
    Builder& SetParams(CuckooHashingParams params);
    // Uses `builder` to build the key database. Defaults to a newly constructed
    // DenseDpfPirDatabase::Builder.
    Builder& SetKeyDatabaseBuilder(
        std::unique_ptr<DenseDatabase::Builder> builder);
    // Uses `builder` to build the value database. Defaults to a newly
    // constructed DenseDpfPirDatabase::Builder.
    Builder& SetValueDatabaseBuilder(
        std::unique_ptr<DenseDatabase::Builder> builder);
    // Returns a copy of this builder.
    std::unique_ptr<PirDatabaseInterface::Builder> Clone() const override;
    // Builds the database and invalidated the builder. All subsequent calls to
    // Build() will fail with FAILED_PRECONDITION.
    absl::StatusOr<std::unique_ptr<PirDatabaseInterface>> Build() override;

   private:
    CuckooHashingParams params_;
    std::unique_ptr<DenseDatabase::Builder> key_database_builder_,
        value_database_builder_;
    absl::btree_map<std::string, std::string> records_;
    bool has_been_built_;
  };

  // Returns the number of elements contained in the database.
  size_t size() const override { return size_; }

  // The number of selection bits is the same as in the underlying databases.
  // We check in Builder::Build() that this is the same for both.
  size_t num_selection_bits() const override { return num_selection_bits_; }

  // Returns the inner product of the `selections` bit vector with the database
  // elements. Called by the PirServer implementation.
  absl::StatusOr<std::vector<RecordType>> InnerProductWith(
      absl::Span<const std::vector<BlockType>> selections) const override;

 private:
  CuckooHashedDpfPirDatabase(std::unique_ptr<DenseDatabase> key_database,
                             std::unique_ptr<DenseDatabase> value_database,
                             size_t size, size_t num_selection_bits);
  // We store keys and values separately in a dense database, and
  // combine them after doing the inner products.
  std::unique_ptr<DenseDatabase> key_database_, value_database_;
  // Number of elements in the database.
  size_t size_;
  // Number of selection bits required for InnerProductWith.
  size_t num_selection_bits_;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_SPARSE_DPF_PIR_DATABASE_H_
