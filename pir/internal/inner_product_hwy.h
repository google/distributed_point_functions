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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_INTERNAL_INNER_PRODUCT_HWY_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_INTERNAL_INNER_PRODUCT_HWY_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "dpf/xor_wrapper.h"

namespace distributed_point_functions {
namespace pir_internal {

using BlockType = distributed_point_functions::XorWrapper<absl::uint128>;

// Returns the inner product the between `values` and the selection bits,
// where the selection bits are packed in entries of `selections`.
absl::StatusOr<std::vector<std::string>> InnerProduct(
    absl::Span<const absl::string_view> values,
    absl::Span<const std::vector<BlockType>> selections,
    int64_t max_value_size);

// As `InnerProduct`, but does not provide explicit SIMD implementation.
absl::StatusOr<std::vector<std::string>> InnerProductNoHwy(
    absl::Span<const absl::string_view> values,
    absl::Span<const std::vector<BlockType>> selections,
    int64_t max_value_size);

}  // namespace pir_internal
}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_INTERNAL_INNER_PRODUCT_HWY_H_
