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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_DCF_DISTRIBUTED_COMPARISON_FUNCTION_H_
#define DISTRIBUTED_POINT_FUNCTIONS_DCF_DISTRIBUTED_COMPARISON_FUNCTION_H_

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/meta/type_traits.h"
#include "absl/numeric/int128.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dcf/distributed_comparison_function.pb.h"
#include "dpf/distributed_point_function.h"
#include "dpf/distributed_point_function.pb.h"
#include "hwy/base.h"

namespace distributed_point_functions {

class DistributedComparisonFunction {
 public:
  static absl::StatusOr<std::unique_ptr<DistributedComparisonFunction>> Create(
      const DcfParameters& parameters);

  // Creates keys for a DCF that evaluates to shares of `beta` on any input x <
  // `alpha`, and shares of 0 otherwise.
  //
  // Returns INVALID_ARGUMENT if `alpha` or `beta` do not match the
  // DcfParameters passed at construction.
  //
  // Overload for explicit Value proto.
  absl::StatusOr<std::pair<DcfKey, DcfKey>> GenerateKeys(absl::uint128 alpha,
                                                         const Value& beta);

  // Template for automatic conversion to Value proto. Disabled if the argument
  // is convertible to `absl::uint128` or `Value` to make overloading
  // unambiguous.
  template <typename T, typename = absl::enable_if_t<
                            !std::is_convertible<T, Value>::value &&
                            is_supported_type_v<T>>>
  absl::StatusOr<std::pair<DcfKey, DcfKey>> GenerateKeys(absl::uint128 alpha,
                                                         const T& beta) {
    absl::StatusOr<Value> value = dpf_->ToValue(beta);
    if (!value.ok()) {
      return value.status();
    }
    return GenerateKeys(alpha, *value);
  }

  // Evaluates a DcfKey at the given point `x`.
  //
  // Returns INVALID_ARGUMENT if `key` or `x` do not match the parameters passed
  // at construction.
  template <typename T>
  inline absl::StatusOr<T> Evaluate(const DcfKey& key, absl::uint128 x) {
    T result{};
    absl::Status status = BatchEvaluate<T>(absl::MakeConstSpan(&key, 1),
                                           absl::MakeConstSpan(&x, 1),
                                           absl::MakeSpan(&result, 1));
    if (!status.ok()) {
      return status;
    }
    return result;
  }

  // Evaluates `keys[i]` at `evaluation_points[i]` for all i.
  //
  // Returns INVALID_ARGUMENT if `keys` and `evaluation_points` have different
  // sizes, or if any element of `keys` is invalid or any element of
  // `evaluation_points` is out of scope.
  template <typename T>
  inline absl::StatusOr<std::vector<T>> BatchEvaluate(
      absl::Span<const DcfKey> keys,
      absl::Span<const absl::uint128> evaluation_points) {
    std::vector<T> result(keys.size());
    absl::Status status =
        BatchEvaluate<T>(keys, evaluation_points, absl::MakeSpan(result));
    if (!status.ok()) {
      return status;
    }
    return result;
  }

  // As the two-argument version above, but writes to `output` instead of
  // allocating a std::vector.
  //
  // Returns INVALID_ARGUMENT `output`, `keys`, and `evaluation_points` don't
  // have the same size.
  template <typename T>
  absl::Status BatchEvaluate(absl::Span<const DcfKey> keys,
                             absl::Span<const absl::uint128> evaluation_points,
                             absl::Span<T> output);

  // DistributedComparisonFunction is neither copyable nor movable.
  DistributedComparisonFunction(const DistributedComparisonFunction&) = delete;
  DistributedComparisonFunction& operator=(
      const DistributedComparisonFunction&) = delete;

 private:
  DistributedComparisonFunction(DcfParameters parameters,
                                std::unique_ptr<DistributedPointFunction> dpf);

  const DcfParameters parameters_;
  const std::unique_ptr<DistributedPointFunction> dpf_;
};

// Implementation details.

namespace dpf_internal {

// Returns the level at which it's worth saving the evaluation context.
// Default: always save it.
template <typename T>
struct EvaluationContextCutoff {
  static constexpr int kValue = -1;
};
// Integer types: Informed by benchmarks.
template <>
struct EvaluationContextCutoff<uint8_t> {
  static constexpr int kValue = 50;
};
template <>
struct EvaluationContextCutoff<uint16_t> {
  static constexpr int kValue = 34;
};
template <>
struct EvaluationContextCutoff<uint32_t> {
  static constexpr int kValue = 28;
};
template <>
struct EvaluationContextCutoff<uint64_t> {
  static constexpr int kValue = 24;
};
template <>
struct EvaluationContextCutoff<absl::uint128> {
  static constexpr int kValue = 22;
};
// XorWrapper: Same as the underlying type.
template <typename T>
struct EvaluationContextCutoff<XorWrapper<T>> {
  static constexpr int kValue = EvaluationContextCutoff<T>::kValue;
};

}  // namespace dpf_internal

template <typename T>
absl::Status DistributedComparisonFunction::BatchEvaluate(
    absl::Span<const DcfKey> keys,
    absl::Span<const absl::uint128> evaluation_points, absl::Span<T> output) {
  if (keys.size() != evaluation_points.size()) {
    // Different error message for the two-argument version.
    return absl::InvalidArgumentError(
        "`keys` and `evaluation_points` must have the same size");
  }
  if (output.size() != keys.size()) {
    return absl::InvalidArgumentError(
        "`keys`, `evaluation_points`, and `output` must have the same size");
  }

  const int log_domain_size = parameters_.parameters().log_domain_size();

  absl::StatusOr<EvaluationContext> ctx;
  for (int j = 0; j < keys.size(); ++j) {
    output[j] = T{};
    const DcfKey& key = keys[j];
    const absl::uint128& x = evaluation_points[j];

    if (log_domain_size >= dpf_internal::EvaluationContextCutoff<T>::kValue) {
      ctx = dpf_->CreateEvaluationContext(key.key());
      if (!ctx.ok()) {
        return ctx.status();
      }
    }
    absl::StatusOr<std::vector<T>> dpf_evaluation;
    for (int i = 0; i < log_domain_size; ++i) {
      int current_bit = static_cast<int>(
          (x & (absl::uint128{1} << (log_domain_size - i - 1))) != 0);
      // Only evaluate the DPF when we actually need it. This leaks information
      // about x through a timing side-channel. However, we don't protect
      // against this anyway, and in many cases x is public.
      if (current_bit == 0) {
        HWY_ALIGN_MAX absl::uint128 prefix = 0;
        if (log_domain_size < 128) {
          prefix = x >> (log_domain_size - i);
        }
        if (log_domain_size >=
            dpf_internal::EvaluationContextCutoff<T>::kValue) {
          dpf_evaluation =
              dpf_->EvaluateAt<T>(i, absl::MakeConstSpan(&prefix, 1), *ctx);
        } else {
          dpf_evaluation = dpf_->EvaluateAt<T>(key.key(), i,
                                               absl::MakeConstSpan(&prefix, 1));
        }
        if (!dpf_evaluation.ok()) {
          return dpf_evaluation.status();
        }
        output[j] += (*dpf_evaluation)[0];
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DCF_DISTRIBUTED_COMPARISON_FUNCTION_H_
