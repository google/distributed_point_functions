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

#include "dcf/fss_gates/multiple_interval_containment.h"

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "dcf/distributed_comparison_function.h"
#include "dcf/fss_gates/multiple_interval_containment.pb.h"
#include "dcf/fss_gates/prng/basic_rng.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/status_macros.h"

namespace distributed_point_functions {
namespace fss_gates {

absl::StatusOr<std::unique_ptr<MultipleIntervalContainmentGate>>
MultipleIntervalContainmentGate::Create(const MicParameters& mic_parameters) {
  // Declaring the parameters for a Distributed Comparison Function (DCF).
  DcfParameters dcf_parameters;

  // Return error if log_group_size is not between 0 and 127.
  if (mic_parameters.log_group_size() < 0 ||
      mic_parameters.log_group_size() > 127) {
    return absl::InvalidArgumentError(
        "log_group_size should be in > 0 and < 128");
  }

  // Setting N = 2 ^ log_group_size.
  absl::uint128 N = absl::uint128(1) << mic_parameters.log_group_size();

  for (int i = 0; i < mic_parameters.intervals_size(); i++) {
    // Return error if some interval is empty.
    if (!mic_parameters.intervals(i).has_lower_bound() ||
        !mic_parameters.intervals(i).has_upper_bound()) {
      return absl::InvalidArgumentError("Intervals should be non-empty");
    }

    absl::uint128 p = absl::MakeUint128(
        mic_parameters.intervals(i).lower_bound().value_uint128().high(),
        mic_parameters.intervals(i).lower_bound().value_uint128().low());

    absl::uint128 q = absl::MakeUint128(
        mic_parameters.intervals(i).upper_bound().value_uint128().high(),
        mic_parameters.intervals(i).upper_bound().value_uint128().low());

    // Return error if the intervals are not valid group elements.
    if (p < 0 || p >= N) {
      return absl::InvalidArgumentError(
          "Interval bounds should be between 0 and 2^log_group_size");
    }

    // Return error if the intervals are not valid group elements.
    if (q < 0 || q >= N) {
      return absl::InvalidArgumentError(
          "Interval bounds should be between 0 and 2^log_group_size");
    }

    // Return error if lower bound of an interval is less that its
    // upper bound.
    if (p > q) {
      return absl::InvalidArgumentError(
          "Interval upper bounds should be >= lower bound");
    }
  }

  // Setting the `log_domain_size` of the DCF to be same as the
  // `log_group_size` of the Multiple Interval Containment Gate.
  dcf_parameters.mutable_parameters()->set_log_domain_size(
      mic_parameters.log_group_size());

  // Setting the output ValueType of the DCF so that it can store 128 bit
  // integers.
  *(dcf_parameters.mutable_parameters()->mutable_value_type()) =
      ToValueType<absl::uint128>();

  // Creating a DCF with appropriate parameters.
  DPF_ASSIGN_OR_RETURN(std::unique_ptr<DistributedComparisonFunction> dcf,
                       DistributedComparisonFunction::Create(dcf_parameters));

  return absl::WrapUnique(
      new MultipleIntervalContainmentGate(mic_parameters, std::move(dcf)));
}

MultipleIntervalContainmentGate::MultipleIntervalContainmentGate(
    MicParameters mic_parameters,
    std::unique_ptr<DistributedComparisonFunction> dcf)
    : mic_parameters_(std::move(mic_parameters)), dcf_(std::move(dcf)) {}

absl::StatusOr<std::pair<MicKey, MicKey>> MultipleIntervalContainmentGate::Gen(
    absl::uint128 r_in, std::vector<absl::uint128> r_out) {
  if (r_out.size() != mic_parameters_.intervals_size()) {
    return absl::InvalidArgumentError(
        "Count of output masks should be equal to the number of intervals");
  }

  // Setting N = 2 ^ log_group_size.
  absl::uint128 N = absl::uint128(1) << mic_parameters_.log_group_size();

  // Checking whether r_in is a group element.
  if (r_in < 0 || r_in >= N) {
    return absl::InvalidArgumentError(
        "Input mask should be between 0 and 2^log_group_size");
  }

  // Checking whether each element of r_out is a group element.
  for (int i = 0; i < r_out.size(); i++) {
    if (r_out[i] < 0 || r_out[i] >= N) {
      return absl::InvalidArgumentError(
          "Output mask should be between 0 and 2^log_group_size");
    }
  }

  // The following code is commented using their Line numbering in
  // https://eprint.iacr.org/2020/1392 Fig. 14 Gen procedure.

  // Line 1
  absl::uint128 gamma = (N - 1 + r_in) % N;

  // Line 2
  DcfKey key_0, key_1;

  absl::uint128 alpha = gamma;
  absl::uint128 beta = 1;

  DPF_ASSIGN_OR_RETURN(std::tie(key_0, key_1),
                       this->dcf_->GenerateKeys(alpha, beta));

  MicKey k0, k1;

  // Part of Line 7
  *(k0.mutable_dcfkey()) = key_0;
  *(k1.mutable_dcfkey()) = key_1;

  // Line 3
  for (int i = 0; i < mic_parameters_.intervals_size(); i++) {
    absl::uint128 p = absl::MakeUint128(
        mic_parameters_.intervals(i).lower_bound().value_uint128().high(),
        mic_parameters_.intervals(i).lower_bound().value_uint128().low());

    absl::uint128 q = absl::MakeUint128(
        mic_parameters_.intervals(i).upper_bound().value_uint128().high(),
        mic_parameters_.intervals(i).upper_bound().value_uint128().low());

    // Line 4
    absl::uint128 q_prime = (q + 1) % N;

    absl::uint128 alpha_p = (p + r_in) % N;

    absl::uint128 alpha_q = (q + r_in) % N;

    absl::uint128 alpha_q_prime = (q + 1 + r_in) % N;

    // Line 5 - This computes the correction term for the output of the gate,
    // and the logic and proof of its correctness is described in
    // https://eprint.iacr.org/2020/1392 Lemma 1, Lemma 2 and Theorem 3.
    absl::uint128 z =
        r_out[i] + (alpha_p > alpha_q ? 1 : 0) + (alpha_p > p ? -1 : 0) +
        (alpha_q_prime > q_prime ? 1 : 0) + (alpha_q == (N - 1) ? 1 : 0);
    z = z % N;

    const absl::string_view kSampleSeed = absl::string_view();
    DPF_ASSIGN_OR_RETURN(
        auto rng, distributed_point_functions::BasicRng::Create(kSampleSeed));
    DPF_ASSIGN_OR_RETURN(absl::uint128 z_0, rng->Rand128());

    z_0 = z_0 % N;

    absl::uint128 z_1 = (z - z_0) % N;

    // Part of Line 7
    Value_Integer* k0_output_mask_share = k0.add_output_mask_share();

    k0_output_mask_share->mutable_value_uint128()->set_high(
        absl::Uint128High64(z_0));
    k0_output_mask_share->mutable_value_uint128()->set_low(
        absl::Uint128Low64(z_0));

    Value_Integer* k1_output_mask_share = k1.add_output_mask_share();

    k1_output_mask_share->mutable_value_uint128()->set_high(
        absl::Uint128High64(z_1));
    k1_output_mask_share->mutable_value_uint128()->set_low(
        absl::Uint128Low64(z_1));
  }

  // Line 8
  return std::pair<MicKey, MicKey>(k0, k1);
}

absl::StatusOr<std::vector<absl::uint128>>
MultipleIntervalContainmentGate::BatchEval(
    absl::Span<const MicKey> keys,
    absl::Span<const absl::uint128> evaluation_points) {
  if (keys.size() != evaluation_points.size()) {
    return absl::InvalidArgumentError(
        "`keys` and `evaluations_points` must have the same size");
  }

  // Setting N = 2 ^ log_group_size
  absl::uint128 N = absl::uint128(1) << mic_parameters_.log_group_size();

  // Checking whether evaluation_points are all group elements
  for (int i = 0; i < evaluation_points.size(); i++) {
    if (evaluation_points[i] < 0 || evaluation_points[i] >= N) {
      return absl::InvalidArgumentError(
          "Masked input should be between 0 and 2^log_group_size");
    }
  }

  int num_intervals = mic_parameters_.intervals_size();
  std::vector<absl::uint128> p(num_intervals), q(num_intervals),
      q_prime(num_intervals), x_p(keys.size() * num_intervals),
      x_q_prime(keys.size() * num_intervals), s_p(keys.size() * num_intervals),
      s_q_prime(keys.size() * num_intervals);

  // The following code is commented using their Line numbering in
  // https://eprint.iacr.org/2020/1392 Fig. 14 Eval procedure.
  for (int i = 0; i < num_intervals; ++i) {
    // Line 2
    p[i] = absl::MakeUint128(
        mic_parameters_.intervals(i).lower_bound().value_uint128().high(),
        mic_parameters_.intervals(i).lower_bound().value_uint128().low());

    q[i] = absl::MakeUint128(
        mic_parameters_.intervals(i).upper_bound().value_uint128().high(),
        mic_parameters_.intervals(i).upper_bound().value_uint128().low());

    // Line 3
    q_prime[i] = (q[i] + 1) % N;
  }

  std::vector<const DcfKey*> dcf_keys;
  dcf_keys.reserve(keys.size() * num_intervals);
  for (int i = 0; i < keys.size(); ++i) {
    const absl::uint128& x = evaluation_points[i];
    for (int j = 0; j < num_intervals; ++j) {
      int index = i * num_intervals + j;
      // Line 4
      x_p[index] = (x + N - 1 - p[j]) % N;
      x_q_prime[index] = (x + N - 1 - q_prime[j]) % N;

      // Extract DCF keys from MIC keys.
      dcf_keys.push_back(&(keys[i].dcfkey()));
    }
  }

  // Line 5
  DPF_RETURN_IF_ERROR(dcf_->BatchEvaluate<absl::uint128>(
      absl::MakeConstSpan(dcf_keys), x_p, absl::MakeSpan(s_p)));

  // Line 6
  DPF_RETURN_IF_ERROR(dcf_->BatchEvaluate<absl::uint128>(
      absl::MakeConstSpan(dcf_keys), x_q_prime, absl::MakeSpan(s_q_prime)));

  std::vector<absl::uint128> res;
  res.reserve(keys.size() * num_intervals);
  for (int i = 0; i < keys.size(); ++i) {
    const absl::uint128& x = evaluation_points[i];
    const MicKey& k = keys[i];
    for (int j = 0; j < num_intervals; ++j) {
      int index = i * num_intervals + j;

      s_p[index] = s_p[index] % N;
      s_q_prime[index] = s_q_prime[index] % N;

      // Line 7
      absl::uint128 z =
          absl::MakeUint128(k.output_mask_share(j).value_uint128().high(),
                            k.output_mask_share(j).value_uint128().low());
      absl::uint128 y = (k.dcfkey().key().party()
                             ? ((x > p[j] ? 1 : 0) - (x > q_prime[j] ? 1 : 0))
                             : 0) -
                        s_p[index] + s_q_prime[index] + z;
      res.push_back(y % N);
    }
  }
  // Line 9
  return res;
}

}  // namespace fss_gates
}  // namespace distributed_point_functions
