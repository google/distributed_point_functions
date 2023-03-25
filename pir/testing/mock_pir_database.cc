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

#include "pir/testing/mock_pir_database.h"

#include <vector>

#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "openssl/rand.h"

namespace distributed_point_functions {
namespace pir_testing {

// Creates `num_elements` strings to be used as database elements, with the i-th
// string being absl::StrCat(prefix, i).
absl::StatusOr<std::vector<std::string>> GenerateCountingStrings(
    int num_elements, absl::string_view prefix) {
  if (num_elements < 0) {
    return absl::InvalidArgumentError("num_elements must be non-negative");
  }
  std::vector<std::string> elements;
  elements.reserve(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    elements.push_back(absl::StrCat(prefix, i));
  }
  return elements;
}

// Creates random strings to be used as database elements, where the elements'
// sizes are given in `element_sizes`.
absl::StatusOr<std::vector<std::string>> GenerateRandomStrings(
    absl::Span<const int> element_sizes) {
  std::vector<std::string> elements;
  elements.reserve(element_sizes.size());
  for (const int element_size : element_sizes) {
    if (element_size < 0) {
      return absl::InvalidArgumentError("element_size must be non-negative");
    }
    elements.emplace_back(element_size, '\0');
    if (element_size > 0) {
      RAND_bytes(reinterpret_cast<uint8_t*>(&elements.back()[0]), element_size);
    }
  }
  return elements;
}

// Creates `num_elements` random strings to be used as database elements, where
// all have size `element_size`.
absl::StatusOr<std::vector<std::string>> GenerateRandomStringsEqualSize(
    int num_elements, int element_size) {
  if (num_elements < 0) {
    return absl::InvalidArgumentError("num_elements must be non-negative");
  }
  if (element_size < 0) {
    return absl::InvalidArgumentError("element_size must be non-negative");
  }
  std::vector<int> element_sizes(num_elements, element_size);
  return GenerateRandomStrings(element_sizes);
}

// Creates `num_elements` random strings to be used as database elements, where
// elements have variable sizes in the range [avg_element_size_bytes +/-
// max_size_diff].
absl::StatusOr<std::vector<std::string>> GenerateRandomStringsVariableSize(
    int num_elements, int avg_element_size, int max_size_diff) {
  if (num_elements < 0) {
    return absl::InvalidArgumentError("num_elements must be non-negative");
  }
  if (avg_element_size < 0) {
    return absl::InvalidArgumentError("avg_element_size must be non-negative");
  }
  if (max_size_diff < 0) {
    return absl::InvalidArgumentError("max_size_diff must be non-negative");
  }

  absl::BitGen bitgen;
  std::vector<int> element_sizes(num_elements, avg_element_size);
  for (auto& element_size : element_sizes) {
    element_size += absl::Uniform(bitgen, -max_size_diff, max_size_diff);
  }
  return GenerateRandomStrings(element_sizes);
}

}  // namespace pir_testing
}  // namespace distributed_point_functions
