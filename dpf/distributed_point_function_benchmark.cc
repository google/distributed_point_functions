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

#include "absl/random/random.h"
#include "benchmark/benchmark.h"
#include "dpf/distributed_point_function.h"

namespace private_statistics {
namespace dpf {
namespace {

// Benchmarks a regular DPF evaluation. Expects the first range argument to
// specify the output log domain size.
template <typename T>
void BM_EvaluateRegularDpf(benchmark::State& state) {
  DpfParameters parameters;
  parameters.set_log_domain_size(state.range(0));
  parameters.set_element_bitsize(sizeof(T) * 8);
  std::unique_ptr<DistributedPointFunction> dpf =
      DistributedPointFunction::Create(parameters).value();
  absl::uint128 alpha = 0, beta = 1;
  std::pair<DpfKey, DpfKey> keys = dpf->GenerateKeys(alpha, beta).value();
  EvaluationContext ctx_0 = dpf->CreateEvaluationContext(keys.first).value();
  for (auto s : state) {
    EvaluationContext ctx = ctx_0;
    std::vector<T> result = dpf->EvaluateNext<T>({}, ctx).value();
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK_TEMPLATE(BM_EvaluateRegularDpf, uint8_t)->DenseRange(12, 24, 2);
BENCHMARK_TEMPLATE(BM_EvaluateRegularDpf, uint16_t)->DenseRange(12, 24, 2);
BENCHMARK_TEMPLATE(BM_EvaluateRegularDpf, uint32_t)->DenseRange(12, 24, 2);
BENCHMARK_TEMPLATE(BM_EvaluateRegularDpf, uint64_t)->DenseRange(12, 24, 2);
BENCHMARK_TEMPLATE(BM_EvaluateRegularDpf, absl::uint128)->DenseRange(12, 24, 2);

// Benchmarks full evaluation of all hierarchy levels. Expects the first range
// argument to specify the number of iterations. The output domain size is fixed
// to 2**20.
template <typename T>
void BM_EvaluateHierarchicalFull(benchmark::State& state) {
  // Set up DPF with the given parameters.
  const int kMaxLogDomainSize = 20;
  int num_hierarchy_levels = state.range(0);
  std::vector<DpfParameters> parameters(num_hierarchy_levels);
  for (int i = 0; i < num_hierarchy_levels; ++i) {
    parameters[i].set_log_domain_size(static_cast<int>(
        static_cast<double>(i + 1) / num_hierarchy_levels * kMaxLogDomainSize));
    parameters[i].set_element_bitsize(sizeof(T) * 8);
  }
  std::unique_ptr<DistributedPointFunction> dpf =
      DistributedPointFunction::CreateIncremental(parameters).value();

  // Generate keys.
  absl::uint128 alpha = 12345;
  std::vector<absl::uint128> beta(num_hierarchy_levels);
  for (int i = 0; i < num_hierarchy_levels; ++i) {
    beta[i] = i;
  }
  std::pair<DpfKey, DpfKey> keys =
      dpf->GenerateKeysIncremental(alpha, beta).value();

  // Set up evaluation context and evaluation prefixes for each level.
  EvaluationContext ctx_0 = dpf->CreateEvaluationContext(keys.first).value();
  std::vector<std::vector<absl::uint128>> prefixes(num_hierarchy_levels);
  for (int i = 1; i < num_hierarchy_levels; ++i) {
    prefixes[i].resize(1 << parameters[i - 1].log_domain_size());
    std::iota(prefixes[i].begin(), prefixes[i].end(), absl::uint128{0});
  }

  // Measure evaluation time.
  for (auto s : state) {
    EvaluationContext ctx = ctx_0;
    for (int i = 0; i < num_hierarchy_levels; ++i) {
      std::vector<T> result = dpf->EvaluateNext<T>(prefixes[i], ctx).value();
      benchmark::DoNotOptimize(result);
    }
    benchmark::DoNotOptimize(ctx);
  }
}
BENCHMARK_TEMPLATE(BM_EvaluateHierarchicalFull, uint8_t)->DenseRange(1, 16, 2);
BENCHMARK_TEMPLATE(BM_EvaluateHierarchicalFull, uint16_t)->DenseRange(1, 16, 2);
BENCHMARK_TEMPLATE(BM_EvaluateHierarchicalFull, uint32_t)->DenseRange(1, 16, 2);
BENCHMARK_TEMPLATE(BM_EvaluateHierarchicalFull, uint64_t)->DenseRange(1, 16, 2);
BENCHMARK_TEMPLATE(BM_EvaluateHierarchicalFull, absl::uint128)
    ->DenseRange(1, 16, 2);

}  // namespace
}  // namespace dpf
}  // namespace private_statistics
