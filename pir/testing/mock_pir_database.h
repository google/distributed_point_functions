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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_TESTING_MOCK_PIR_DATABASE_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_TESTING_MOCK_PIR_DATABASE_H_

#include <stddef.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "pir/pir_database_interface.h"

namespace distributed_point_functions {
namespace pir_testing {

template <typename BlockType, typename RecordType>
class MockPirDatabase : public PirDatabaseInterface<BlockType, RecordType> {
 public:
  using Base = PirDatabaseInterface<BlockType, RecordType>;

  class Builder : public Base::Builder {
   public:
    Builder() = default;
    MOCK_METHOD(Builder&, Insert, (RecordType), (override));
    MOCK_METHOD(Builder&, Clear, (), (override));
    MOCK_METHOD(absl::StatusOr<std::unique_ptr<Base>>, Build, (), (override));
    MOCK_METHOD(std::unique_ptr<typename Base::Builder>, Clone, (),
                (const override));
  };

  MOCK_METHOD(size_t, size, (), (const));
  MOCK_METHOD(size_t, num_selection_bits, (), (const));
  MOCK_METHOD(absl::StatusOr<std::vector<RecordType>>, InnerProductWith,
              (absl::Span<const std::vector<BlockType>> selections), (const));
};

// Creates `num_elements` strings to be used as database elements, with the i-th
// string being absl::StrCat(prefix, i).
absl::StatusOr<std::vector<std::string>> GenerateCountingStrings(
    int num_elements, absl::string_view prefix);

// Creates random strings to be used as database elements, where the elements'
// sizes are given in `element_sizes`.
absl::StatusOr<std::vector<std::string>> GenerateRandomStrings(
    absl::Span<const int> element_sizes);

// Creates `num_elements` random strings to be used as database elements, where
// all have size `element_size`.
absl::StatusOr<std::vector<std::string>> GenerateRandomStringsEqualSize(
    int num_elements, int element_size);

// Creates `num_elements` random strings to be used as database elements, where
// elements have variable sizes in the range [avg_element_size_bytes +/-
// max_size_diff].
absl::StatusOr<std::vector<std::string>> GenerateRandomStringsVariableSize(
    int num_elements, int avg_element_size, int max_size_diff);

// Creates a Database containing the given `elements`.
template <typename Database>
absl::StatusOr<std::unique_ptr<typename Database::Interface>>
CreateFakeDatabase(absl::Span<const typename Database::RecordType> elements,
                   typename Database::Builder* builder = nullptr) {
  std::unique_ptr<typename Database::Builder> owned_builder;
  if (builder == nullptr) {
    owned_builder = std::make_unique<typename Database::Builder>();
    builder = owned_builder.get();
  }
  for (const auto& element : elements) {
    builder->Insert(element);
  }
  return builder->Build();
}

}  // namespace pir_testing
}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_TESTING_MOCK_PIR_DATABASE_H_
