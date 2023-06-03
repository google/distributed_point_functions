// Copyright 2023 Google LLC
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

#include <cstdint>
#include <string>
#include <utility>

#include "absl/numeric/int128.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "benchmark/benchmark.h"  // third_party/benchmark
#include "dcf/fss_gates/multiple_interval_containment.h"
#include "dcf/fss_gates/multiple_interval_containment.pb.h"
#include "dpf/internal/status_matchers.h"

namespace distributed_point_functions {
namespace fss_gates {
namespace {

void BM_BatchedMicEvaluation(benchmark::State& state) {
  int num_keys = state.range(0);
  int num_intervals = state.range(1);
  std::vector<MicKey> keys(num_keys);
  std::vector<absl::uint128> evaluation_points(num_keys);
  MicParameters params;
  params.set_log_group_size(64);
  std::vector<absl::uint128> r_out(num_intervals);
  absl::BitGen gen;
  for (int i = 0; i < num_intervals; ++i) {
    uint64_t lower = absl::Uniform<uint64_t>(gen);
    uint64_t upper = absl::Uniform<uint64_t>(gen);
    if (lower > upper) std::swap(lower, upper);
    Interval* interval = params.add_intervals();
    interval->mutable_lower_bound()->set_value_uint64(lower);
    interval->mutable_upper_bound()->set_value_uint64(upper);
    r_out[i] = absl::Uniform<uint64_t>(gen);
  }
  DPF_ASSERT_OK_AND_ASSIGN(auto mic_gate,
                           MultipleIntervalContainmentGate::Create(params));
  for (int i = 0; i < num_keys; ++i) {
    DPF_ASSERT_OK_AND_ASSIGN(
        std::tie(keys[i], std::ignore),
        mic_gate->Gen(absl::Uniform<uint64_t>(gen), r_out));
  }

  std::vector<absl::uint128> evaluations(num_intervals);
  for (auto s : state) {
    for (int i = 0; i < num_keys; ++i) {
      DPF_ASSERT_OK_AND_ASSIGN(evaluations,
                               mic_gate->Eval(keys[i], evaluation_points[i]));
      benchmark::DoNotOptimize(evaluations);
    }
  }
}

BENCHMARK(BM_BatchedMicEvaluation)
    ->RangePair(1, 128, 1, 128)
    ->ArgPair(1000, 6)
    ->ArgPair(1000, 10);

}  // namespace
}  // namespace fss_gates
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
