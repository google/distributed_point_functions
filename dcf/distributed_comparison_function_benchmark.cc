// Copyright 2022 Google LLC
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

#include <memory>
#include <tuple>

#include "absl/numeric/int128.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "benchmark/benchmark.h"
#include "dcf/distributed_comparison_function.h"
#include "dcf/distributed_comparison_function.pb.h"
#include "dpf/distributed_point_function.h"

namespace distributed_point_functions {

template <typename T>
void BM_EvaluateDcf(benchmark::State& state) {
  int log_domain_size = state.range(0);
  DcfParameters parameters;
  *(parameters.mutable_parameters()->mutable_value_type()) = ToValueType<T>();
  parameters.mutable_parameters()->set_log_domain_size(log_domain_size);
  std::unique_ptr<DistributedComparisonFunction> dcf =
      DistributedComparisonFunction::Create(parameters).value();

  absl::BitGen rng;
  absl::uint128 domain_mask = absl::Uint128Max();
  absl::uint128 alpha = absl::Uniform<absl::uint128>(rng);
  if (log_domain_size < 128) {
    domain_mask = (absl::uint128{1} << log_domain_size) - 1;
  }
  alpha &= domain_mask;
  T beta(42);
  DcfKey key;
  std::tie(key, std::ignore) = dcf->GenerateKeys(alpha, beta).value();
  absl::uint128 x = 0;
  for (auto s : state) {
    T evaluation = dcf->Evaluate<T>(key, x).value();
    x = (x + 1) & domain_mask;
    benchmark::DoNotOptimize(evaluation);
  }
}
BENCHMARK_TEMPLATE(BM_EvaluateDcf, uint8_t)->RangeMultiplier(2)->Range(2, 64);
BENCHMARK_TEMPLATE(BM_EvaluateDcf, uint16_t)->RangeMultiplier(2)->Range(2, 64);
BENCHMARK_TEMPLATE(BM_EvaluateDcf, uint32_t)->RangeMultiplier(2)->Range(2, 64);
BENCHMARK_TEMPLATE(BM_EvaluateDcf, uint64_t)->RangeMultiplier(2)->Range(2, 64);
BENCHMARK_TEMPLATE(BM_EvaluateDcf, absl::uint128)
    ->RangeMultiplier(2)
    ->Range(2, 64);

}  // namespace distributed_point_functions
