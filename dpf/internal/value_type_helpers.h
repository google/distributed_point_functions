/*
 * Copyright 2021 Google LLC
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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_DPF_INTERNAL_VALUE_TYPE_HELPERS_H_
#define DISTRIBUTED_POINT_FUNCTIONS_DPF_INTERNAL_VALUE_TYPE_HELPERS_H_

#include <glog/logging.h>

#include <array>
#include <type_traits>

#include "absl/base/casts.h"
#include "absl/numeric/int128.h"
#include "absl/status/statusor.h"
#include "absl/types/any.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/tuple.h"

namespace distributed_point_functions {
namespace dpf_internal {

// Helper type for overloading by T. C++ only allows overloading by input types,
// not return type, and doesn't support partial function template
// specialization. Therefore the template type is artificially turned into a
// parameter to enable overloading.
template <typename T>
struct type_helper {};

// Helper function to compute the combined size of an element of type T. For
// tuples, this is the sum of the sizes of its elements, ignoring alignment.
template <typename T>
constexpr size_t GetTotalSize() {
  return GetTotalSizeImpl(type_helper<T>());
}

// Overload for integers.
template <typename T>
constexpr size_t GetTotalSizeImpl(type_helper<T>) {
  return sizeof(T);
}

// Overload for tuples.
template <typename... T>
constexpr size_t GetTotalSizeImpl(type_helper<Tuple<T...>>) {
  size_t total_size = 0;
  ((total_size += GetTotalSizeImpl(type_helper<T>())), ...);
  return total_size;
}

// Computes the number of values of type T that fit into an absl::uint128. For
// all types except unsigned integers, this returns 1.
template <typename T>
constexpr size_t ElementsPerBlock() {
  //  return ElementsPerBlockImpl(type_helper<T>{})
  static_assert(GetTotalSize<T>() <= sizeof(T));
  return sizeof(absl::uint128) / GetTotalSize<T>();
}

// Converts a given absl::any to a numerical type T. Tries converting directly
// first. If that doesn't work, tries converting to absl::uin128 and then doing
// a static_cast to T. Returns INVALID_ARGUMENT if both approaches fail.
template <typename T>
absl::StatusOr<T> ConvertAnyTo(const absl::any& in) {
  // Try casting directly first.
  const T* in_T = absl::any_cast<T>(&in);
  if (in_T) {
    return *in_T;
  }
  // Then try casting to absl::uint128 and then using a static_cast.
  if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>) {
    const absl::uint128* in_128 = absl::any_cast<absl::uint128>(&in);
    if (in_128) {
      return static_cast<T>(*in_128);
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Conversion of absl::any (typeid: ", in.type().name(),
                   ") to type T (typeid: ", typeid(T).name(),
                   ", size: ", sizeof(T), ") failed"));
}

// GetValueTypeProtoFor<T> Returns a `ValueType` message describing T.
template <typename T>
ValueType GetValueTypeProtoFor() {
  return GetValueTypeProtoForImpl(type_helper<T>());
}

// GetValueTypeProtoForImpl should be overloaded for any new types supported in
// the `ValueType` proto.
template <typename T,
          typename = std::enable_if_t<
              std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
              std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
              std::is_same_v<T, absl::uint128>>>
ValueType GetValueTypeProtoForImpl(type_helper<T>) {
  ValueType result;
  result.mutable_integer()->set_bitsize(8 * sizeof(T));
  return result;
}

// Overload for tuples.
template <typename... ElementType>
ValueType GetValueTypeProtoForImpl(type_helper<Tuple<ElementType...>>) {
  ValueType result;
  ValueType::Tuple* tuple = result.mutable_tuple();
  // Append the type of each ElementType. We use a C++17 fold expression to
  // guarantee the order is well-defined. See
  // https://stackoverflow.com/a/54053084.
  ((*(tuple->add_elements()) = GetValueTypeProtoFor<ElementType>()), ...);
  return result;
}

// Checks that `value_type` is well-formed.
//
// Returns the total bit size of types with `value_type` if it is valid, and
// INVALID_ARGUMENT otherwise.
absl::StatusOr<int> ValidateValueTypeAndGetBitSize(const ValueType& value_type);

// Returns `true` if `lhs` and `rhs` describe the same types, and `false`
// otherwise.
bool ValueTypesAreEqual(const ValueType& lhs, const ValueType& rhs);

// Converts a given Value to the template parameter T.
//
// Returns INVALID_ARGUMENT if the conversion fails.
template <typename T>
absl::StatusOr<T> ConvertValueTo(const Value& value) {
  return ConvertValueToImpl(value, type_helper<T>());
}

// Implementation of ConvertValueTo<T> for native integer types T.
template <typename T,
          typename = std::enable_if_t<
              std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
              std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>>>
absl::StatusOr<T> ConvertValueToImpl(const Value& value, type_helper<T>) {
  if (value.value_case() != Value::kInteger) {
    return absl::InvalidArgumentError("The given Value is not an integer");
  }
  if (value.integer().value_case() != Value::Integer::kValueUint64) {
    return absl::InvalidArgumentError(
        "The given Value does not have value_uint64 set");
  }
  uint64_t value_uint64 = value.integer().value_uint64();
  if (value_uint64 > static_cast<uint64_t>(std::numeric_limits<T>::max())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "value_uint64 (= ", value_uint64, ") too large for the given type T"));
  }
  return static_cast<T>(value_uint64);
}

// Implementation of ConvertValueTo<T> for absl::uint128.
absl::StatusOr<absl::uint128> ConvertValueToImpl(const Value& value,
                                                 type_helper<absl::uint128>);

// Implementation of ConvertValueTo<T> for Tuple<ElementType...>.
// We need two templates here: One to select the right overload of
// `ConvertValueToImpl`, The second to resolve the integers in the corresponding
// std::index_sequence.
template <typename... ElementType>
absl::StatusOr<Tuple<ElementType...>> ConvertValueToImpl(
    const Value& value, type_helper<Tuple<ElementType...>> helper) {
  return ConvertValueToImpl2(
      value, helper,
      std::make_index_sequence<std::tuple_size_v<Tuple<ElementType...>>>());
}
template <typename TupleType, size_t... Index>
absl::StatusOr<TupleType> ConvertValueToImpl2(const Value& value,
                                              type_helper<TupleType>,
                                              std::index_sequence<Index...>) {
  if (value.value_case() != Value::kTuple) {
    return absl::InvalidArgumentError("The given Value is not a tuple");
  }
  constexpr auto tuple_size = static_cast<int>(std::tuple_size_v<TupleType>);
  if (value.tuple().elements_size() != tuple_size) {
    return absl::InvalidArgumentError(
        "The tuple in the given Value has the wrong number of elements");
  }

  // Create a Tuple by unpacking value.tuple().elements(). If we encounter an
  // error, return it at the end.
  absl::Status status = absl::OkStatus();
  TupleType result = {[&value, &status] {
    using CurrentElementType = std::tuple_element_t<Index, TupleType>;
    if (status.ok()) {
      absl::StatusOr<CurrentElementType> element =
          ConvertValueTo<CurrentElementType>(value.tuple().elements(Index));
      if (element.ok()) {
        return *element;
      } else {
        status = element.status();
      }
    }
    return CurrentElementType{};
  }()...};
  if (status.ok()) {
    return result;
  } else {
    return status;
  }
}

// Converts a `repeated Value` proto field to a std::array with element type T.
//
// Returns INVALID_ARGUMENT in case the input has the wrong size, or if the
// conversion fails.
template <typename T>
absl::StatusOr<std::array<T, ElementsPerBlock<T>()>> ValuesToArray(
    const ::google::protobuf::RepeatedPtrField<Value>& values) {
  if (values.size() != ElementsPerBlock<T>()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "values.size() (= ", values.size(),
        ") does not match ElementsPerBlock<T>() (= ", ElementsPerBlock<T>(),
        ")"));
  }
  std::array<T, ElementsPerBlock<T>()> result;
  for (int i = 0; i < ElementsPerBlock<T>(); ++i) {
    absl::StatusOr<T> element = ConvertValueTo<T>(values[i]);
    if (element.ok()) {
      result[i] = std::move(*element);
    } else {
      return element.status();
    }
  }
  return result;
}

// ToValue Converts the argument to a Value.
//
// Overload for native integers.
template <typename T,
          typename = std::enable_if_t<
              std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
              std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>>>
Value ToValue(T input) {
  Value result;
  result.mutable_integer()->set_value_uint64(input);
  return result;
}

// Overload for absl::uint128.
Value ToValue(absl::uint128 input);

// Overload for Tuple<ElementType...>.
template <typename... ElementType>
Value ToValue(Tuple<ElementType...> input) {
  Value result;
  std::apply(
      [&result](const ElementType&... element) {
        ((*(result.mutable_tuple()->add_elements()) = ToValue(element)), ...);
      },
      input);
  return result;
}

// Converts all inputs to Values and returns the result as a std::vector.
//
// Returns INVALID_ARGUMENT if any conversion fails.
template <typename T>
std::vector<Value> ToValues(absl::Span<const T> inputs) {
  std::vector<Value> result;
  result.reserve(inputs.size());
  for (const T& element : inputs) {
    result.push_back(ToValue(element));
  }
  return result;
}

// Creates a value of type T from the given `bytes`.
//
// Crashes if `bytes.size()` is too small for the output type.
template <typename T>
T ConvertBytesTo(absl::string_view bytes) {
  return ConvertBytesToImpl(bytes, type_helper<T>{});
}

// Overload for integer types.
template <typename T,
          typename = std::enable_if_t<
              std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
              std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
              std::is_same_v<T, absl::uint128>>>
T ConvertBytesToImpl(absl::string_view bytes, type_helper<T>) {
  CHECK(ABSL_PREDICT_FALSE(bytes.size() == sizeof(T)));
  T out{0};
#ifdef ABSL_IS_LITTLE_ENDIAN
  std::copy_n(bytes.begin(), sizeof(T), reinterpret_cast<char*>(&out));
#else
  for (int i = sizeof(T) - 1; i >= 0; --i) {
    out |= absl::bit_cast<uint8_t>(bytes[i]);
    out <<= 8;
  }
#endif
  return out;
}

// Overload for tuples.
template <typename... ElementType>
Tuple<ElementType...> ConvertBytesToImpl(absl::string_view bytes,
                                         type_helper<Tuple<ElementType...>>) {
  using TupleType = Tuple<ElementType...>;
  CHECK(bytes.size() >= GetTotalSize<TupleType>());
  int offset = 0;
  absl::Status status = absl::OkStatus();
  return TupleType{[&bytes, &offset, &status] {
    ElementType element = ConvertBytesTo<ElementType>(
        bytes.substr(offset, GetTotalSize<ElementType>()));
    offset += GetTotalSize<ElementType>();
    return element;
  }()...};
}

// Converts a given string to an array of elements of type T.
template <typename T>
std::array<T, ElementsPerBlock<T>()> ConvertBytesToArrayOf(
    absl::string_view bytes) {
  CHECK(bytes.size() >= ElementsPerBlock<T>() * GetTotalSize<T>());
  std::array<T, ElementsPerBlock<T>()> out;
  for (int i = 0; i < ElementsPerBlock<T>(); ++i) {
    out[i] = ConvertBytesTo<T>(
        bytes.substr(i * GetTotalSize<T>(), GetTotalSize<T>()));
  }
  return out;
}

// Computes the value correction word given two seeds `seed_a`, `seed_b` for
// parties a and b, such that the element at `block_index` is equal to `beta`.
// If `invert` is true, the result is multiplied element-wise by -1. Templated
// to use the correct integer type without needing modular reduction.
//
// Returns multiple values in case of packing, and a single value otherwise.
template <typename T>
absl::StatusOr<std::vector<Value>> ComputeValueCorrectionFor(
    absl::Span<const absl::uint128> seeds, int block_index,
    const absl::any& beta, bool invert) {
  absl::StatusOr<T> beta_T = ConvertAnyTo<T>(beta);
  if (!beta_T.ok()) {
    return beta_T.status();
  }

  constexpr int elements_per_block = ElementsPerBlock<T>();

  // Split up seeds into individual integers.
  std::array<T, elements_per_block>
      ints_a = dpf_internal::ConvertBytesToArrayOf<T>(absl::string_view(
          reinterpret_cast<const char*>(&seeds[0]), sizeof(absl::uint128))),
      ints_b = dpf_internal::ConvertBytesToArrayOf<T>(absl::string_view(
          reinterpret_cast<const char*>(&seeds[1]), sizeof(absl::uint128)));

  // Add beta to the right position.
  ints_b[block_index] += *beta_T;

  // Add up shares, invert if needed.
  for (int i = 0; i < elements_per_block; i++) {
    ints_b[i] = ints_b[i] - ints_a[i];
    if (invert) {
      ints_b[i] = -ints_b[i];
    }
  }

  // Convert to std::vector and return.
  return ToValues(absl::MakeConstSpan(ints_b));
}

}  // namespace dpf_internal
}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DPF_INTERNAL_VALUE_TYPE_HELPERS_H_
