// Copyright 2021 Google LLC
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

#include "dpf/internal/value_type_helpers.h"

#include "dpf/status_macros.h"

namespace distributed_point_functions {
namespace dpf_internal {

absl::StatusOr<int> GetTotalBitsize(const ValueType& value_type) {
  if (value_type.type_case() == ValueType::kInteger) {
    return value_type.integer().bitsize();
  } else if (value_type.type_case() == ValueType::kTuple) {
    int bitsize = 0;
    for (const ValueType& el : value_type.tuple().elements()) {
      DPF_ASSIGN_OR_RETURN(int el_bitsize, GetTotalBitsize(el));
      bitsize += el_bitsize;
    }
    return bitsize;
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported value_type:\n", value_type.DebugString()));
  }
}

bool ValueTypesAreEqual(const ValueType& lhs, const ValueType& rhs) {
  if (lhs.type_case() == ValueType::kInteger &&
      rhs.type_case() == ValueType::kInteger) {
    return lhs.integer().bitsize() == rhs.integer().bitsize();
  } else if (lhs.type_case() == ValueType::kTuple &&
             rhs.type_case() == ValueType::kTuple &&
             lhs.tuple().elements_size() == rhs.tuple().elements_size()) {
    bool result = true;
    for (int i = 0; i < static_cast<int>(lhs.tuple().elements_size()); ++i) {
      result &=
          ValueTypesAreEqual(lhs.tuple().elements(i), rhs.tuple().elements(i));
    }
    return result;
  }
  return false;
}

Value ToValue(absl::uint128 input) {
  Value result;
  Block& block = *(result.mutable_integer()->mutable_value_uint128());
  block.set_high(absl::Uint128High64(input));
  block.set_low(absl::Uint128Low64(input));
  return result;
}

absl::StatusOr<int> BlocksNeeded(const ValueType& value_type) {
  DPF_ASSIGN_OR_RETURN(int value_type_bitsize, GetTotalBitsize(value_type));
  return (value_type_bitsize + 127) / 128;
}

}  // namespace dpf_internal

}  // namespace distributed_point_functions
