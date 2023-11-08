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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_DENSE_DPF_PIR_DATABASE_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_DENSE_DPF_PIR_DATABASE_H_

#include <stddef.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "dpf/xor_wrapper.h"
#include "pir/pir_database_interface.h"

namespace distributed_point_functions {

// This database class is intended to be used with DPF PIR with a dense key
// space, where keys are integers in the space [0, .size()) and each key has an
// associated value in the database. This class implements the database
// interface with BlockType being XorWrapper<absl::uint128>, same as the block
// type for the DPF evaluation result.
//
//
class DenseDpfPirDatabase
    : public PirDatabaseInterface<XorWrapper<absl::uint128>, std::string> {
 public:
  using Interface = PirDatabaseInterface;

  // The concrete Builder for DenseDpfPirDatabase.
  class Builder : public PirDatabaseInterface::Builder {
   public:
    Builder();
    // Appends a record `value` at the end of the database.
    Builder& Insert(std::string) override;
    // Clears all elements inserted into this builder, but leaves any other
    // configuration intact.
    Builder& Clear() override;
    // Returns a copy of this builder.
    std::unique_ptr<PirDatabaseInterface::Builder> Clone() const override;
    // Builds the database and invalidated the builder. All subsequent calls to
    // Build() will fail with FAILED_PRECONDITION.
    absl::StatusOr<std::unique_ptr<PirDatabaseInterface>> Build() override;
    // Returns the total number of bytes in the database, including padding.
    // Used for testing.
    int64_t total_database_bytes() const { return total_database_bytes_; }

   private:
    std::vector<std::string> values_;
    int64_t total_database_bytes_;
    bool has_been_built_;
  };

  // Returns the number of records contained in the database.
  size_t size() const override { return content_views_.size(); }

  // The number of selection bits for dense PIR is equal to the number of
  // elements.
  size_t num_selection_bits() const override { return size(); }

  // Returns the inner product between the database values and a bit vector
  // (packed in blocks).
  absl::StatusOr<std::vector<RecordType>> InnerProductWith(
      absl::Span<const std::vector<BlockType>> selections) const override;

  // Returns a flat array holding all values of the database. Used for testing.
  absl::Span<const absl::string_view> content() const { return content_views_; }

  // Returns the maximal size of values in the database. Used for testing.
  size_t max_value_size_in_bytes() const { return max_value_size_; }

 private:
  static constexpr int kBitsPerBlock = 8 * sizeof(absl::uint128);

  // Constructs a DenseDpfPirDatabase object.
  DenseDpfPirDatabase(int64_t num_values, int64_t total_database_bytes);

  // Appends a record `value` at the current end of the database. Used by
  // Builder::Build() to construct the database.
  absl::Status Append(std::string value);

  // Maximal size (in bytes) of values in the database
  size_t max_value_size_;

  // Stores all the values of the database. For better memory access performance
  // when computing the inner product, the beginning address of each value will
  // be aligned to 128-bit boundary.
  std::vector<BlockType> buffer_;

  // Stores the offset and size of each value in the database.
  std::vector<std::pair<size_t, size_t>> value_offsets_;

  // Stores the absl::string_view pointers of all values in the database.
  std::vector<absl::string_view> content_views_;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_DENSE_DPF_PIR_DATABASE_H_
