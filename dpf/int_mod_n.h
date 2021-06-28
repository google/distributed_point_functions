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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_DPF_INTERNAL_INT_MOD_N_H_
#define DISTRIBUTED_POINT_FUNCTIONS_DPF_INTERNAL_INT_MOD_N_H_

#include "absl/numeric/int128.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "dpf/internal/value_type_helpers.h"

namespace distributed_point_functions {

template <typename BaseInteger, BaseInteger modulus>
class IntModN {
 public:
  IntModN() : value_(0) {}
  explicit IntModN(BaseInteger value) { value_ = value % modulus; }

  BaseInteger value() { return value_; }

  void sub(const IntModN& a) { sub(a.value_); }

  void add(const IntModN& a) { sub(modulus - a.value_); }

  //  Returns the number of (pseudo)random bytes required to extract
  // `num_samples` samples r1, ..., rn
  //  so that the stream r1, ..., rn is close to a truly (pseudo) random
  //  sequence up to total variation distance < 2^(-`security_parameter`)
  static absl::StatusOr<int> GetNumBytesRequired(int num_samples,
                                                 double security_parameter) {
    // Compute the level of security that we will get, and fail if it is
    // insufficient.
    double sigma = 128 + 3 -
                   (std::log2(static_cast<double>(modulus)) +
                    std::log2(static_cast<double>(num_samples)) +
                    std::log2(static_cast<double>(num_samples + 1)));
    if (security_parameter > sigma) {
      return absl::InvalidArgumentError(absl::StrCat(
          "For num_samples = ", num_samples, " and modulus = ", modulus,
          " this approach can only provide ", sigma,
          " bits of statistical security. You can try calling this function "
          "several times with smaller values of num_samples."));
    }
    // We start the sampling by requiring a 128-bit (16 bytes) block, see
    // function `sample`.
    return 16 + sizeof(BaseInteger) * (num_samples - 1);
  }
  //  Checks that length(`bytes`) is enough to extract
  // `samples.size()` samples r1, ..., rn
  //  so that the stream r1, ..., rn is close to a truly (pseudo) random
  //  sequence up to total variation distance < 2^(-`security_parameter`) and
  //  fails if that is not the case.
  //  Otherwise returns r1, ..., rn in `samples`.
  static absl::Status SampleFromBytes(
      absl::string_view bytes, double security_parameter,
      absl::Span<IntModN<BaseInteger, modulus>> samples) {
    if (samples.empty()) {
      return absl::InvalidArgumentError(
          "The number of samples required must be > 0");
    }
    absl::StatusOr<int> num_bytes_lower_bound =
        GetNumBytesRequired(samples.size(), security_parameter);
    if (!num_bytes_lower_bound.ok()) {
      return num_bytes_lower_bound.status();
    }
    if (*num_bytes_lower_bound > bytes.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("The number of bytes provided (", bytes.size(),
                       ") is insufficient for the required "
                       "statistical security and number of samples."));
    }
    absl::uint128 r =
        dpf_internal::ConvertBytesTo<absl::uint128>(bytes.substr(0, 16));
    std::vector<BaseInteger> randomness =
        std::vector<BaseInteger>(samples.size() - 1);
    for (int i = 0; i < randomness.size(); ++i) {
      randomness[i] = dpf_internal::ConvertBytesTo<BaseInteger>(
          bytes.substr(16 + i * sizeof(BaseInteger), sizeof(BaseInteger)));
    }
    int index = 0;
    for (int i = 0; i < samples.size(); ++i) {
      samples[i] =
          IntModN<BaseInteger, modulus>(static_cast<BaseInteger>(r % modulus));
      absl::uint128 r_div = r / modulus;
      if (i < samples.size() - 1) {
        r = r_div << (sizeof(BaseInteger) * 8);
        r |= randomness[index];
        index++;
      }
    }
    return absl::OkStatus();
  }

  // Copyable.
  constexpr IntModN<BaseInteger, modulus>(
      const IntModN<BaseInteger, modulus>& a) = default;

  IntModN<BaseInteger, modulus>& operator=(
      const IntModN<BaseInteger, modulus>& a) = default;

  // Assignment operators.
  IntModN<BaseInteger, modulus>& operator=(BaseInteger a) {
    value_ = a % modulus;
    return *this;
  }

  IntModN<BaseInteger, modulus>& operator+=(IntModN<BaseInteger, modulus> a) {
    add(a);
    return *this;
  }

  IntModN<BaseInteger, modulus>& operator-=(IntModN<BaseInteger, modulus> a) {
    sub(a);
    return *this;
  }

 private:
  void sub(BaseInteger a) {
    if (value_ >= a) {
      value_ -= a;
    } else {
      value_ = modulus - a + value_;
    }
  }

  BaseInteger value_;
};

template <typename BaseInteger, BaseInteger modulus>
constexpr IntModN<BaseInteger, modulus> operator+(
    IntModN<BaseInteger, modulus> a, const IntModN<BaseInteger, modulus>& b) {
  a += b;
  return a;
}

template <typename BaseInteger, BaseInteger modulus>
constexpr IntModN<BaseInteger, modulus> operator-(
    IntModN<BaseInteger, modulus> a, const IntModN<BaseInteger, modulus>& b) {
  a -= b;
  return a;
}

}  // namespace distributed_point_functions
#endif  // DISTRIBUTED_POINT_FUNCTIONS_DPF_INTERNAL_INT_MOD_N_H_
