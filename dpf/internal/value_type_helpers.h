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
#include "dpf/distributed_point_function.pb.h"
#include "dpf/int_mod_n.h"
#include "dpf/tuple.h"

namespace distributed_point_functions {

namespace dpf_internal {

// Helper type for overloading by T. C++ only allows overloading by input types,
// not return type, and doesn't support partial function template
// specialization. Therefore the template type is artificially turned into a
// parameter to enable overloading.
template <typename T>
struct type_helper {};

// Type trait for all integer types we support, i.e., 8 to 128 bit types.
template <typename T>
struct is_unsigned_integer {
  static constexpr bool value =
      std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
      std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
      std::is_same_v<T, absl::uint128>;
};
template <typename T>
inline constexpr bool is_unsigned_integer_v = is_unsigned_integer<T>::value;

// Type trait for all supported types. Used to provide meaningful error messages
// in std::enable_if template guards.

// Integers: true.
template <typename T, bool = is_unsigned_integer_v<T>>
struct is_supported_type : std::true_type {};

// Default case: false.
template <typename T>
struct is_supported_type<T, false> : std::false_type {};

// Tuples of supported types: true.
template <typename... ElementType>
struct is_supported_type<Tuple<ElementType...>, false> {
  static constexpr bool value = (is_supported_type<ElementType>::value && ...);
};

// Modular integers with supported base types: true.
template <typename BaseInteger, typename ModulusType, ModulusType kModulus>
struct is_supported_type<IntModNImpl<BaseInteger, ModulusType, kModulus>,
                         false> {
  static constexpr bool value = is_unsigned_integer_v<BaseInteger>;
};

template <typename T>
inline constexpr bool is_supported_type_v = is_supported_type<T>::value;

// Checks if the template parameter can be converted directly from a string of
// bytes.

// Default case: True.
template <typename T, typename = void>
struct can_be_converted_directly : std::true_type {};

// Tuples: True, if all elements can be converted directly.
template <typename... ElementType>
struct can_be_converted_directly<Tuple<ElementType...>, void> {
  static constexpr bool value =
      (can_be_converted_directly<ElementType>::value && ...);
};

// IntModN: False.
template <typename BaseInteger, typename ModulusType, ModulusType kModulus>
struct can_be_converted_directly<
    IntModNImpl<BaseInteger, ModulusType, kModulus>, void> {
  static constexpr bool value = false;
};

template <typename T>
inline constexpr bool can_be_converted_directly_v =
    can_be_converted_directly<T>::value;

// Helper function to compute the combined size of an element of type T. For
// tuples, this is the sum of the sizes of its elements, ignoring alignment.
// Used to convert strings to elements of type T.
template <typename T>
constexpr int GetTotalBitsize() {
  return GetTotalBitsizeImpl(type_helper<T>());
}

// Overload for integers.
template <typename T, typename = std::enable_if_t<is_unsigned_integer_v<T>>>
constexpr int GetTotalBitsizeImpl(type_helper<T>) {
  return static_cast<int>(8 * sizeof(T));
}

// Overload for tuples.
template <typename... T>
constexpr int GetTotalBitsizeImpl(type_helper<Tuple<T...>>) {
  int bitsize = 0;
  ((bitsize += GetTotalBitsize<T>()), ...);
  return bitsize;
}

// Type trait to test if a type supports batching. True if T can be converted
// directly from bytes and has a size of at most 128 bits.

// Directly convertible types.
template <typename T, bool = can_be_converted_directly_v<T>>
struct supports_batching {
  static constexpr bool value = GetTotalBitsize<T>() <= 128;
};

// Default: false.
template <typename T>
struct supports_batching<T, false> : std::false_type {};

template <typename T>
inline constexpr bool supports_batching_v = supports_batching<T>::value;

// Computes the number of values of type T that fit into an absl::uint128.
// Returns a value >= 1 if supports_batching_v<T> is true, and 1 otherwise.
template <typename T>
constexpr int ElementsPerBlock() {
  if constexpr (supports_batching_v<T>) {
    return static_cast<int>(8 * sizeof(absl::uint128)) / GetTotalBitsize<T>();
  } else {
    return 1;
  }
}

// Computes the number of pseudorandom bits needed to get a uniform element of
// the given `ValueType`. For types whose elements can be bijectively mapped to
// strings (e.g., unsigned integers and tuples of integers), this is equivalent
// to the bit size of the value type. For all other types, returns the number of
// bits needed so that converting a uniform string with the given number of bits
// to an element of `value_type` results in a distribution with total variation
// distance < 2^(-`security_parameter`) from uniform.
//
// Returns INVALID_ARGUMENT in case value_type does not represent a known type.
absl::StatusOr<int> BitsNeeded(const ValueType& value_type,
                               double security_parameter);

// Converts the given Value::Integer to an absl::uint128. Used as a helper
// function in `ConvertValueTo` and `ValueTypesAreEqual`.
//
// Returns INVALID_ARGUMENT if `in` is not a simple integer or IntModN.
absl::StatusOr<absl::uint128> ValueIntegerToUint128(const Value::Integer& in);

// Returns `true` if `lhs` and `rhs` describe the same types, and `false`
// otherwise.
//
// Returns INVALID_ARGUMENT if an error occurs while parsing either argument.
absl::StatusOr<bool> ValueTypesAreEqual(const ValueType& lhs,
                                        const ValueType& rhs);

// Converts a given Value to the template parameter T.
//
// Returns INVALID_ARGUMENT if the conversion fails.
template <typename T>
absl::StatusOr<T> ConvertValueTo(const Value& value) {
  return ConvertValueToImpl(value, type_helper<T>());
}

// Implementation of ConvertValueTo<T> for native integer types T.
template <typename T, typename = std::enable_if_t<is_unsigned_integer_v<T>>>
absl::StatusOr<T> ConvertValueToImpl(const Value& value, type_helper<T>) {
  if (value.value_case() != Value::kInteger) {
    return absl::InvalidArgumentError("The given Value is not an integer");
  }
  // We first parse the value into an absl::uint128, then check its range if it
  // is supposed to be smaller than 128 bits.
  absl::StatusOr<absl::uint128> value_128 =
      ValueIntegerToUint128(value.integer());
  if (!value_128.ok()) {
    return value_128.status();
  }
  // Check whether value is in range if it's smaller than 128 bits.
  if (!std::is_same_v<T, absl::uint128> &&
      absl::Uint128Low64(*value_128) >
          static_cast<uint64_t>(std::numeric_limits<T>::max())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Value (= ", absl::Uint128Low64(*value_128),
        ") too large for the given type T (size ", sizeof(T), ")"));
  }
  return static_cast<T>(*value_128);
}

// Implementation of ConvertValueTo<T> for Tuple<ElementType...>.
// We need two templates here: One to select the right overload of
// `ConvertValueToImpl`, The second to resolve the integers in the corresponding
// std::index_sequence.
template <typename... ElementType>
absl::StatusOr<Tuple<ElementType...>> ConvertValueToImpl(
    const Value& value, type_helper<Tuple<ElementType...>> helper) {
  return ConvertValueToImpl2(
      value, helper,
      std::make_index_sequence<
          std::tuple_size_v<std::tuple<ElementType...>>>());
}
template <typename TupleType, size_t... Index>
absl::StatusOr<TupleType> ConvertValueToImpl2(const Value& value,
                                              type_helper<TupleType>,
                                              std::index_sequence<Index...>) {
  if (value.value_case() != Value::kTuple) {
    return absl::InvalidArgumentError("The given Value is not a tuple");
  }
  constexpr auto tuple_size =
      static_cast<int>(std::tuple_size_v<typename TupleType::Base>);
  if (value.tuple().elements_size() != tuple_size) {
    return absl::InvalidArgumentError(
        "The tuple in the given Value has the wrong number of elements");
  }

  // Create a Tuple by unpacking value.tuple().elements(). If we encounter an
  // error, return it at the end.
  absl::Status status = absl::OkStatus();
  TupleType result = {[&value, &status] {
    using CurrentElementType =
        std::tuple_element_t<Index, typename TupleType::Base>;
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

// Overload for IntModN.
template <typename BaseInteger, typename ModulusType, ModulusType kModulus>
absl::StatusOr<IntModNImpl<BaseInteger, ModulusType, kModulus>>
ConvertValueToImpl(
    const Value& value,
    type_helper<IntModNImpl<BaseInteger, ModulusType, kModulus>>) {
  if (value.value_case() != Value::kIntModN) {
    return absl::InvalidArgumentError("The given Value is not an IntModN");
  }
  absl::StatusOr<absl::uint128> value_128 =
      ValueIntegerToUint128(value.int_mod_n());
  if (!value_128.ok()) {
    return value_128.status();
  }
  if (*value_128 >= absl::uint128{kModulus}) {
    return absl::InvalidArgumentError(
        absl::StrFormat("The given value (= %d) is larger than kModulus (= %d)",
                        *value_128, absl::uint128{kModulus}));
  }
  return IntModNImpl<BaseInteger, ModulusType, kModulus>(
      static_cast<BaseInteger>(*value_128));
}

// Converts a `repeated Value` proto field to a std::array with element type T.
//
// Returns INVALID_ARGUMENT in case the input has the wrong size, or if the
// conversion fails.
template <typename T, typename = std::enable_if_t<is_supported_type_v<T>>>
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

// Converts an absl::uint128 to a Value::Integer. Used as a helper function in
// ToValue.
Value::Integer Uint128ToValueInteger(absl::uint128 input);

// ToValue Converts the argument to a Value.
//
// Overload for native integers.
Value ToValue(absl::uint128 input);

// Overload for IntModN.
template <typename BaseInteger, typename ModulusType, ModulusType kModulus>
Value ToValue(const IntModNImpl<BaseInteger, ModulusType, kModulus>& input) {
  Value result;
  *(result.mutable_int_mod_n()) = Uint128ToValueInteger(input.value());
  return result;
}

// Overload for Tuple<ElementType...>.
template <typename... ElementType>
Value ToValue(const Tuple<ElementType...>& input) {
  Value result;
  std::apply(
      [&result](const ElementType&... element) {
        ((*(result.mutable_tuple()->add_elements()) = ToValue(element)), ...);
      },
      input.value());
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

// ToValueType<T> Returns a `ValueType` message describing T.
template <typename T>
ValueType ToValueType() {
  return ToValueTypeImpl(type_helper<T>());
}

// ToValueTypeImpl should be overloaded for any new types supported in the
// `ValueType` proto.
template <typename T, typename = std::enable_if_t<is_unsigned_integer_v<T>>>
ValueType ToValueTypeImpl(type_helper<T>) {
  ValueType result;
  result.mutable_integer()->set_bitsize(8 * sizeof(T));
  return result;
}

// Overload for tuples.
template <typename... ElementType>
ValueType ToValueTypeImpl(type_helper<Tuple<ElementType...>>) {
  ValueType result;
  ValueType::Tuple* tuple = result.mutable_tuple();
  // Append the type of each ElementType. We use a C++17 fold expression to
  // guarantee the order is well-defined. See
  // https://stackoverflow.com/a/54053084.
  ((*(tuple->add_elements()) = ToValueType<ElementType>()), ...);
  return result;
}

// Overload for IntModN.
template <typename BaseInteger, typename ModulusType, ModulusType kModulus>
ValueType ToValueTypeImpl(
    type_helper<IntModNImpl<BaseInteger, ModulusType, kModulus>>) {
  ValueType result;
  *(result.mutable_int_mod_n()->mutable_base_integer()) =
      ToValueType<BaseInteger>().integer();
  *(result.mutable_int_mod_n()->mutable_modulus()) =
      ToValue(kModulus).integer();
  return result;
}

// Creates a value of type T from the given `bytes`. If possible, converts bytes
// directly using ConvertBytesDirectlyTo. Otherwise, uses SampleFromBytes.
//
// Crashes if `bytes.size()` is too small for the output type.
template <typename T>
T ConvertBytesTo(absl::string_view bytes) {
  if constexpr (can_be_converted_directly_v<T>) {
    return ConvertBytesDirectlyTo(bytes, type_helper<T>{});
  } else {
    return SampleFromBytes(bytes, type_helper<T>{});
  }
}

// Overload for integer types.
template <typename T, typename = std::enable_if_t<is_unsigned_integer_v<T>>>
T ConvertBytesDirectlyTo(absl::string_view bytes, type_helper<T>) {
  CHECK(bytes.size() == sizeof(T));
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

// Overload of ConvertBytesDirectlyTo for tuples.
// TOOD(b/193007723): Move this to tuple.h.
template <typename... ElementType>
Tuple<ElementType...> ConvertBytesDirectlyTo(
    absl::string_view bytes, type_helper<Tuple<ElementType...>>) {
  using TupleType = Tuple<ElementType...>;
  CHECK(8 * bytes.size() >= GetTotalBitsize<TupleType>());
  int offset = 0;
  absl::Status status = absl::OkStatus();
  // Braced-init-list ensures the elements are constructed in-order.
  return TupleType{[&bytes, &offset, &status] {
    const int element_size_bytes = (GetTotalBitsize<ElementType>() + 7) / 8;
    ElementType element =
        ConvertBytesTo<ElementType>(bytes.substr(offset, element_size_bytes));
    offset += element_size_bytes;
    return element;
  }()...};
}

// Converts `block` to type T. Then, if `update == true`, fills up `block` from
// `bytes` and advances `bytes` by the amount of bytes read.
template <typename T>
T SampleAndUpdateBytes(bool update, absl::uint128& block,
                       absl::string_view& bytes) {
  return SampleAndUpdateBytesImpl(update, block, bytes, type_helper<T>{});
}

// Overload for simple integers.
template <typename T, typename = std::enable_if_t<is_unsigned_integer_v<T>>>
T SampleAndUpdateBytesImpl(bool update, absl::uint128& block,
                           absl::string_view& bytes, type_helper<T>) {
  T result = static_cast<T>(block);

  if (update) {
    // Set sizeof(T) least significant bytes to 0.
    if constexpr (std::is_same_v<T, absl::uint128>) {
      block = 0;
    } else {
      constexpr absl::uint128 mask =
          ~absl::uint128{std::numeric_limits<T>::max()};
      block &= mask;
    }

    // Fill up with `bytes` and advance `bytes` by sizeof(T).
    DCHECK(bytes.size() >= sizeof(T));
    block |= ConvertBytesTo<T>(bytes.substr(0, sizeof(T)));
    bytes = bytes.substr(sizeof(T));
  }

  return result;
}

// Overload for IntModN.
template <typename BaseInteger, typename ModulusType, ModulusType kModulus>
IntModNImpl<BaseInteger, ModulusType, kModulus> SampleAndUpdateBytesImpl(
    bool update, absl::uint128& block, absl::string_view& bytes,
    type_helper<IntModNImpl<BaseInteger, ModulusType, kModulus>>) {
  // Optimization for native uint128. This is equivalent to what's done in
  // int128.cc, but since division is not defined in the header, the compiler
  // cannot optimize the division and modulus into a single operation.
#ifdef ABSL_HAVE_INTRINSIC_INT128
  absl::uint128 quotient = static_cast<unsigned __int128>(block) / kModulus,
                remainder = static_cast<unsigned __int128>(block) / kModulus;
#else
  absl::uint128 quotient = block / kModulus, remainder = block % kModulus;
#endif
  IntModNImpl<BaseInteger, ModulusType, kModulus> result(
      static_cast<BaseInteger>(remainder));

  if (update) {
    block = quotient << (sizeof(BaseInteger) * 8);
    block |= ConvertBytesTo<BaseInteger>(bytes.substr(0, sizeof(BaseInteger)));
    bytes = bytes.substr(sizeof(BaseInteger));
  }

  return result;
}

template <typename... ElementType>
Tuple<ElementType...> SampleAndUpdateBytesImpl(
    bool update, absl::uint128& block, absl::string_view& bytes,
    type_helper<Tuple<ElementType...>>) {
  using TupleType = Tuple<ElementType...>;

  int element_counter = 0;
  // Braced-init-list ensures the elements are constructed in-order.
  return TupleType{[update, &element_counter, &block, &bytes]() -> ElementType {
    // If `update` is true, update after all elements. Otherwise, don't update
    // after the last one.
    constexpr int num_elements = std::tuple_size_v<typename TupleType::Base>;
    bool update2 = update || (++element_counter < num_elements);
    return SampleAndUpdateBytes<ElementType>(update2, block, bytes);
  }()...};
}

// Implementation of SampleFromBytes for single IntModNs.
template <typename BaseInteger, typename ModulusType, ModulusType kModulus>
IntModNImpl<BaseInteger, ModulusType, kModulus> SampleFromBytes(
    absl::string_view bytes,
    type_helper<IntModNImpl<BaseInteger, ModulusType, kModulus>>) {
  DCHECK(bytes.size() >= sizeof(absl::uint128));
  absl::uint128 block =
      ConvertBytesTo<absl::uint128>(bytes.substr(0, sizeof(absl::uint128)));
  return SampleAndUpdateBytes<IntModNImpl<BaseInteger, ModulusType, kModulus>>(
      false, block, bytes);
}

// Implementation of SampleFromBytes for tuples.
template <typename... ElementType>
Tuple<ElementType...> SampleFromBytes(absl::string_view bytes,
                                      type_helper<Tuple<ElementType...>>) {
  using TupleType = Tuple<ElementType...>;
  using FirstElementType = std::tuple_element_t<0, typename TupleType::Base>;
  static_assert(sizeof(FirstElementType) <= sizeof(absl::uint128));
  DCHECK(bytes.size() >= sizeof(absl::uint128) - sizeof(FirstElementType) +
                             (sizeof(ElementType) + ...));
  absl::uint128 block =
      ConvertBytesTo<absl::uint128>(bytes.substr(0, sizeof(absl::uint128)));
  bytes = bytes.substr(sizeof(absl::uint128));

  return SampleAndUpdateBytes<TupleType>(false, block, bytes);
}

// Converts a given string to an array of exactly ElementsPerBlock<T>() elements
// of type T.
//
// Crashes if `bytes.size()` is too small for the output type.
template <typename T>
std::array<T, ElementsPerBlock<T>()> ConvertBytesToArrayOf(
    absl::string_view bytes) {
  std::array<T, ElementsPerBlock<T>()> out;
  if constexpr (supports_batching_v<T>) {
    const int element_size_bytes = (GetTotalBitsize<T>() + 7) / 8;
    CHECK(bytes.size() >= ElementsPerBlock<T>() * element_size_bytes);
    for (int i = 0; i < ElementsPerBlock<T>(); ++i) {
      out[i] = ConvertBytesTo<T>(
          bytes.substr(i * element_size_bytes, element_size_bytes));
    }
  } else {
    static_assert(out.size() == 1,
                  "T does not support batching, but ElementsPerBlock<T> != 1");
    out[0] = ConvertBytesTo<T>(bytes);
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
    absl::string_view seed_a, absl::string_view seed_b, int block_index,
    const Value& beta, bool invert) {
  absl::StatusOr<T> beta_T = ConvertValueTo<T>(beta);
  if (!beta_T.ok()) {
    return beta_T.status();
  }

  constexpr int elements_per_block = ElementsPerBlock<T>();

  // Compute values from seeds. Both arrays will have multiple elements if T
  // supports batching, and a single one otherwise.
  std::array<T, elements_per_block> ints_a =
                                        dpf_internal::ConvertBytesToArrayOf<T>(
                                            seed_a),
                                    ints_b =
                                        dpf_internal::ConvertBytesToArrayOf<T>(
                                            seed_b);

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
