#ifndef DISTRIBUTED_POINT_FUNCTIONS_DPF_TUPLE_H_
#define DISTRIBUTED_POINT_FUNCTIONS_DPF_TUPLE_H_

#include <tuple>

namespace distributed_point_functions {

// A Tuple alias with added element-wise addition, subtraction, and negation
// operators.
template <typename... T>
using Tuple = std::tuple<T...>;

namespace dpf_internal {

// Implementation of addition and negation. See
// https://stackoverflow.com/a/50815143.
template <typename... T, std::size_t... I>
constexpr Tuple<T...> add(const Tuple<T...>& lhs, const Tuple<T...>& rhs,
                          std::index_sequence<I...>) {
  return Tuple<T...>{std::get<I>(lhs) + std::get<I>(rhs)...};
}

template <typename... T, std::size_t... I>
constexpr Tuple<T...> negate(const Tuple<T...>& t, std::index_sequence<I...>) {
  return Tuple<T...>{-std::get<I>(t)...};
}

}  // namespace dpf_internal

template <typename... T>
constexpr Tuple<T...> operator+(const Tuple<T...>& lhs,
                                const Tuple<T...>& rhs) {
  return dpf_internal::add(lhs, rhs, std::make_index_sequence<sizeof...(T)>{});
}

template <typename... T>
constexpr Tuple<T...>& operator+=(Tuple<T...>& lhs, const Tuple<T...>& rhs) {
  lhs = lhs + rhs;
  return lhs;
}

template <typename... T>
constexpr Tuple<T...> operator-(const Tuple<T...>& t) {
  return dpf_internal::negate(t, std::make_index_sequence<sizeof...(T)>{});
}

template <typename... T>
constexpr Tuple<T...> operator-(const Tuple<T...>& lhs,
                                const Tuple<T...>& rhs) {
  return lhs + (-rhs);
}

template <typename... T>
constexpr Tuple<T...>& operator-=(Tuple<T...>& lhs, const Tuple<T...>& rhs) {
  lhs = lhs - rhs;
  return lhs;
}

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DPF_TUPLE_H_
