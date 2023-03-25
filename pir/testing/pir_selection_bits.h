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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_TESTING_PIR_SELECTION_BITS_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_TESTING_PIR_SELECTION_BITS_H_

#include <limits>

#include "absl/random/random.h"
#include "pir/dense_dpf_pir_database.h"

namespace distributed_point_functions {
namespace pir_testing {

// Pack the selections bits to a vector of 128-bit blocks
// The template parameter `BlockType` must be an instance of XorWrapper.
template <typename BlockType>
std::vector<BlockType> PackSelectionBits(const std::vector<bool>& selections) {
  constexpr int kBitsPerBlock = 8 * sizeof(BlockType);

  int num_blocks = (selections.size() + kBitsPerBlock - 1) / kBitsPerBlock;
  std::vector<BlockType> blocks;
  blocks.reserve(num_blocks);
  for (int i = 0; i < num_blocks; ++i) {
    typename BlockType::WrappedType block{0};
    int base = i * kBitsPerBlock;
    for (int j = 0; j < kBitsPerBlock; ++j) {
      if (base + j >= selections.size()) {
        break;  // reached the last partial block
      }
      if (selections[base + j]) {
        block |= typename BlockType::WrappedType{1} << j;
      }
    }
    blocks.push_back(BlockType(block));
  }
  return blocks;
}

// Sample a vector of packed random selections bits.
// The template parameter `BlockType` must be an instance of XorWrapper.
template <typename BlockType>
std::vector<BlockType> GenerateRandomPackedSelectionBits(int num_bits) {
  using WrappedType = typename BlockType::WrappedType;
  constexpr int kBitsPerBlock = 8 * sizeof(BlockType);

  int num_blocks = (num_bits + kBitsPerBlock - 1) / kBitsPerBlock;
  std::vector<BlockType> blocks;
  blocks.reserve(num_blocks);

  absl::BitGen bitgen;
  for (int i = 0; i < num_blocks; ++i) {
    auto bits = absl::Uniform<WrappedType>(bitgen);
    blocks.push_back(BlockType(bits));
  }
  return blocks;
}

// Returns the inner product between `values` and `selections` for testing
// the implementation with packed selection bits.
absl::StatusOr<std::string> InnerProductWithUnpacked(
    const std::vector<bool>& selections, absl::Span<const std::string> values);

}  // namespace pir_testing
}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_TESTING_PIR_SELECTION_BITS_H_
