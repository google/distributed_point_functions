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
#include <string>
#include <tuple>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "benchmark/benchmark.h"  // third_party/benchmark
#include "dcf/distributed_comparison_function.h"
#include "dcf/distributed_comparison_function.pb.h"
#include "dpf/distributed_point_function.h"
#include "dpf/internal/status_matchers.h"

namespace distributed_point_functions {

template <typename T>
void BM_EvaluateDcf(benchmark::State& state) {
  int log_domain_size = state.range(0);
  int batch_size = state.range(1);
  DcfParameters parameters;
  *(parameters.mutable_parameters()->mutable_value_type()) = ToValueType<T>();
  parameters.mutable_parameters()->set_log_domain_size(log_domain_size);
  std::unique_ptr<DistributedComparisonFunction> dcf =
      DistributedComparisonFunction::Create(parameters).value();

  std::vector<DcfKey> keys(batch_size);
  std::vector<absl::uint128> evaluation_points(batch_size);
  absl::BitGen rng;
  absl::uint128 domain_mask = absl::Uint128Max();
  if (log_domain_size < 128) {
    domain_mask = (absl::uint128{1} << log_domain_size) - 1;
  }
  for (int i = 0; i < batch_size; ++i) {
    absl::uint128 alpha = absl::Uniform<absl::uint128>(rng);
    alpha &= domain_mask;
    T beta = absl::Uniform<T>(rng);
    std::tie(keys[i], std::ignore) = dcf->GenerateKeys(alpha, beta).value();
    evaluation_points[i] = absl::Uniform<absl::uint128>(rng);
    evaluation_points[i] &= domain_mask;
  }

  for (auto s : state) {
    DPF_ASSERT_OK_AND_ASSIGN(std::vector<T> evaluation,
                             dcf->BatchEvaluate<T>(keys, evaluation_points));
    benchmark::DoNotOptimize(evaluation);
  }
}
BENCHMARK_TEMPLATE(BM_EvaluateDcf, uint8_t)
    ->RangeMultiplier(2)
    ->RangePair(2, 128, 1, 1024);
BENCHMARK_TEMPLATE(BM_EvaluateDcf, uint16_t)
    ->RangeMultiplier(2)
    ->RangePair(2, 128, 1, 1024);
BENCHMARK_TEMPLATE(BM_EvaluateDcf, uint32_t)
    ->RangeMultiplier(2)
    ->RangePair(2, 128, 1, 1024);
BENCHMARK_TEMPLATE(BM_EvaluateDcf, uint64_t)
    ->RangeMultiplier(2)
    ->RangePair(2, 128, 1, 1024);
BENCHMARK_TEMPLATE(BM_EvaluateDcf, absl::uint128)->RangePair(2, 128, 1, 1024);

}  // namespace distributed_point_functions

// Declare benchmark_filter flag, which will be defined by benchmark library.
// Use it to check if any benchmarks were specified explicitly.
//
namespace benchmark {
extern std::string FLAGS_benchmark_filter;
}
using benchmark::FLAGS_benchmark_filter;

int main(int argc, char* argv[]) {
  FLAGS_benchmark_filter = "";
  benchmark::Initialize(&argc, argv);
  if (!FLAGS_benchmark_filter.empty()) {
    benchmark::RunSpecifiedBenchmarks();
  }
  benchmark::Shutdown();
  return 0;
}