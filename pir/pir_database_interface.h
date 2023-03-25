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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_PIR_DATABASE_INTERFACE_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_PIR_DATABASE_INTERFACE_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace distributed_point_functions {

// This class defines the basic database interfaces used by a PIR server.
template <typename BlockTypeT, typename RecordTypeT>
class PirDatabaseInterface {
 public:
  using BlockType = BlockTypeT;
  using RecordType = RecordTypeT;

  // Builder interface for constructing concrete databases. Allows inserting
  // records one-by-one before constructing the underlying in-memory database in
  // one shot. May be used co compose different database implementations.
  class Builder {
   public:
    Builder() = default;
    virtual ~Builder() = default;
    // Disable copy operations. Copies should be obtained explicitly using
    // Clone().
    Builder(Builder&) = delete;
    Builder& operator=(Builder&) = delete;

    // Inserts an element into the database.
    virtual Builder& Insert(RecordType) = 0;
    // Returns a copy of this Builder.
    virtual std::unique_ptr<Builder> Clone() const = 0;
    // Builds the database after all elements have been inserted and invalidated
    // the builder. All subsequent calls to Build() should fail with
    // FAILED_PRECONDITION.
    virtual absl::StatusOr<std::unique_ptr<PirDatabaseInterface>> Build() = 0;
  };

  virtual ~PirDatabaseInterface() {}

  // Returns the inner product between the database records and a bit-vector.
  // For compactness and efficiency, the binary selection vector is packed in
  // blocks, represented as a vector of `BlockType`.
  // The result of the inner product is held in a `ResponseType` value.
  virtual absl::StatusOr<std::vector<RecordType>> InnerProductWith(
      absl::Span<const std::vector<BlockType>> selections) const = 0;

  // Returns the number of elements contained in the database.
  virtual size_t size() const = 0;

  // Returns the number of bits needed in the selection vector used in
  // `InnerProductWith`. May or may not be equal to `size()`.
  virtual size_t num_selection_bits() const = 0;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_PIR_DATABASE_INTERFACE_H_
