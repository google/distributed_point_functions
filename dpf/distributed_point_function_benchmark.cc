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

#include "benchmark/benchmark.h"
#include "dpf/distributed_point_function.h"

namespace private_statistics {
namespace dpf {

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

}  // namespace dpf
}  // namespace private_statistics
