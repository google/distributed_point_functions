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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dcf/distributed_comparison_function.pb.h"
#include "dpf/distributed_point_function.h"
#include "dpf/distributed_point_function.pb.h"
#include "hwy/aligned_allocator.h"
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
  absl::StatusOr<T> Evaluate(const DcfKey& key, absl::uint128 x) {
    T result{};
    absl::Status status = Evaluate(key, {x}, absl::MakeSpan(&result, 1));
    if (!status.ok()) {
      return status;
    }
    return result;
  }

  // Evaluates `key` at multiple `evaluation_points`, and writes the results to
  // `output`.
  //
  // Returns INVALID_ARGUMENT if `key` is invalid, if any element of
  // `evaluation_points` is out of bounds, or if the size of `output` does not
  // match `evaluation_points`.
  template <typename T>
  absl::Status Evaluate(const DcfKey& key,
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
absl::Status DistributedComparisonFunction::Evaluate(
    const DcfKey& key, absl::Span<const absl::uint128> evaluation_points,
    absl::Span<T> output) {
  if (evaluation_points.size() != output.size()) {
    return absl::InvalidArgumentError(
        "`evaluation_points` and `output` must have the same size");
  }
  const int log_domain_size = parameters_.parameters().log_domain_size();
  auto prefixes = hwy::AllocateAligned<absl::uint128>(evaluation_points.size());

  absl::StatusOr<EvaluationContext> ctx;
  if (log_domain_size > dpf_internal::EvaluationContextCutoff<T>::kValue) {
    ctx = dpf_->CreateEvaluationContext(key.key());
    if (!ctx.ok()) {
      return ctx.status();
    }
  }

  absl::StatusOr<std::vector<T>> dpf_evaluation;
  std::fill_n(&output[0], output.size(), T{});
  for (int i = 0; i < log_domain_size; ++i) {
    std::fill_n(&prefixes[0], evaluation_points.size(), 0);
    if ((log_domain_size - i) < 128) {
      for (int j = 0; j < evaluation_points.size(); ++j) {
        prefixes[j] = evaluation_points[j] >> (log_domain_size - i);
      }
    }
    if (log_domain_size >= dpf_internal::EvaluationContextCutoff<T>::kValue) {
      dpf_evaluation = dpf_->EvaluateAt<T>(
          i, absl::MakeConstSpan(&prefixes[0], evaluation_points.size()), *ctx);
    } else {
      dpf_evaluation = dpf_->EvaluateAt<T>(
          key.key(), i,
          absl::MakeConstSpan(&prefixes[0], evaluation_points.size()));
    }
    if (!dpf_evaluation.ok()) {
      return dpf_evaluation.status();
    }

    for (int j = 0; j < evaluation_points.size(); ++j) {
      // Check that the bit at position i of evaluation j is 0, and if so add
      // the DPF result to the corresponding output.
      if ((log_domain_size - i - 1) >= 128 ||
          (evaluation_points[j] &
           (absl::uint128{1} << (log_domain_size - i - 1))) == 0) {
        output[j] += (*dpf_evaluation)[j];
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DCF_DISTRIBUTED_COMPARISON_FUNCTION_H_
