/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_SIMPLE_HASHED_DPF_PIR_DATABASE_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_SIMPLE_HASHED_DPF_PIR_DATABASE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dpf/xor_wrapper.h"
#include "pir/pir_database_interface.h"
#include "pir/private_information_retrieval.pb.h"

namespace distributed_point_functions {

class SimpleHashedDpfPirDatabase
    : public PirDatabaseInterface<XorWrapper<absl::uint128>,
                                  std::pair<std::string, std::string>,
                                  std::string> {
 public:
  using Interface = PirDatabaseInterface;
  // Type of the underlying database implementation that stores keys and values
  // in a combined serialized proto.
  using DenseDatabase =
      PirDatabaseInterface<XorWrapper<absl::uint128>, std::string>;

  // The concrete Builder for SimpleHashedSparseDpfPirDatabase.
  class Builder : public PirDatabaseInterface::Builder {
   public:
    Builder();
    // Inserts the given key-value pair into the database once Build() is
    // called.
    Builder& Insert(std::pair<std::string, std::string>) override;
    // Clears all elements inserted into this builder, but leaves any other
    // configuration intact.
    Builder& Clear() override;
    // Sets the hashing parameters used for this database. Must be called before
    // calling `Build`.
    Builder& SetParams(SimpleHashingParams params);
    // Uses `builder` to build the dense database that stores each simple-hashed
    // bucket. Defaults to a newly constructed DenseDpfPirDatabase::Builder.
    Builder& SetDenseDatabaseBuilder(
        std::unique_ptr<DenseDatabase::Builder> builder);
    // Returns a copy of this builder.
    std::unique_ptr<PirDatabaseInterface::Builder> Clone() const override;
    // Builds the database and invalidated the builder. All subsequent calls to
    // Build() will fail with FAILED_PRECONDITION.
    absl::StatusOr<std::unique_ptr<PirDatabaseInterface>> Build() override;

   private:
    SimpleHashingParams params_;
    std::unique_ptr<DenseDatabase::Builder> dense_database_builder_;
    std::vector<std::pair<std::string, std::string>> records_;
    bool has_been_built_;
  };

  // Returns the number of elements contained in the database.
  size_t size() const override { return size_; }

  // The number of selection bits is the same as in the underlying databases.
  // We check in Builder::Build() that this is the same for both.
  size_t num_selection_bits() const override { return num_selection_bits_; }

  // Returns the inner product of the `selections` bit vector with the database
  // elements. Called by the PirServer implementation.
  absl::StatusOr<std::vector<ResponseType>> InnerProductWith(
      absl::Span<const std::vector<BlockType>> selections) const override;

 private:
  SimpleHashedDpfPirDatabase(std::unique_ptr<DenseDatabase> dense_database,
                             size_t size, size_t num_selection_bits);
  // We store keys and values separately in a dense database, and
  // combine them after doing the inner products.
  std::unique_ptr<DenseDatabase> dense_database_;
  // Number of elements in the database.
  size_t size_;
  // Number of selection bits required for InnerProductWith.
  size_t num_selection_bits_;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_SIMPLE_HASHED_DPF_PIR_DATABASE_H_
