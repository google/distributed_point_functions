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

#include <stdint.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"  // third_party/benchmark
#include "dpf/internal/status_matchers.h"
#include "pir/dense_dpf_pir_database.h"
#include "pir/testing/mock_pir_database.h"
#include "pir/testing/pir_selection_bits.h"

namespace distributed_point_functions {
namespace {

using BlockType = DenseDpfPirDatabase::BlockType;

constexpr int kMaxSizeDiff = 8;

void BM_InnerProductOnEqualSizeValues(benchmark::State& state) {
  int num_values = state.range(0);
  int num_bytes_per_value = state.range(1);

  // Insert random values to the database.
  DPF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> values,
                           pir_testing::GenerateRandomStringsEqualSize(
                               num_values, num_bytes_per_value));
  DenseDpfPirDatabase::Builder builder;
  for (auto& value : values) {
    builder.Insert(std::move(value));
  }
  DPF_ASSERT_OK_AND_ASSIGN(auto database, builder.Build());

  // Random selection bits packed in blocks.
  std::vector<BlockType> selections =
      pir_testing::GenerateRandomPackedSelectionBits<BlockType>(num_values);

  // Compute the inner product
  for (auto _ : state) {
    benchmark::DoNotOptimize(database->InnerProductWith({selections}));
  }
}

BENCHMARK(BM_InnerProductOnEqualSizeValues)
    ->Args({1 << 16, 80})
    ->Args({1 << 20, 80})
    ->Args({1 << 16, 256})
    ->Args({1 << 20, 256});

void BM_InnerProductOnVariableSizeValues(benchmark::State& state) {
  int num_values = state.range(0);
  int avg_num_bytes_per_value = state.range(1);

  // Insert random values to the database.
  DPF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::string> values,
      pir_testing::GenerateRandomStringsVariableSize(
          num_values, avg_num_bytes_per_value, kMaxSizeDiff));
  DenseDpfPirDatabase::Builder builder;
  for (auto& value : values) {
    builder.Insert(std::move(value));
  }
  DPF_ASSERT_OK_AND_ASSIGN(auto database, builder.Build());

  // Random selection bits packed in blocks.
  std::vector<BlockType> selections =
      pir_testing::GenerateRandomPackedSelectionBits<BlockType>(num_values);

  // Compute the inner product
  for (auto _ : state) {
    benchmark::DoNotOptimize(database->InnerProductWith({selections}));
  }
}

BENCHMARK(BM_InnerProductOnVariableSizeValues)
    ->Args({1 << 16, 80})
    ->Args({1 << 20, 80})
    ->Args({1 << 16, 256})
    ->Args({1 << 20, 256});

void BM_BatchedInnerProductOnVariableSizeValues(benchmark::State& state) {
  int num_values = state.range(0);
  int avg_num_bytes_per_value = state.range(1);
  int batch_size = state.range(2);

  // Insert random values to the database.
  DPF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::string> values,
      pir_testing::GenerateRandomStringsVariableSize(
          num_values, avg_num_bytes_per_value, kMaxSizeDiff));
  DenseDpfPirDatabase::Builder builder;
  for (auto& value : values) {
    builder.Insert(std::move(value));
  }
  DPF_ASSERT_OK_AND_ASSIGN(auto database, builder.Build());

  // Random selection bits packed in blocks.
  std::vector<std::vector<BlockType>> selections(
      batch_size,
      pir_testing::GenerateRandomPackedSelectionBits<BlockType>(num_values));

  // Compute the inner product
  for (auto _ : state) {
    benchmark::DoNotOptimize(database->InnerProductWith(selections));
  }
}

BENCHMARK(BM_BatchedInnerProductOnVariableSizeValues)
    ->Args({1 << 16, 32, 1})
    ->Args({1 << 20, 32, 1})
    ->Args({1 << 16, 256, 1})
    ->Args({1 << 20, 256, 1})
    ->Args({1 << 16, 2048, 1})
    ->Args({1 << 20, 2048, 1})
    ->Args({1 << 16, 16384, 1})
    ->Args({1 << 20, 16384, 1})
    ->Args({1 << 16, 32, 2})
    ->Args({1 << 20, 32, 2})
    ->Args({1 << 16, 256, 2})
    ->Args({1 << 20, 256, 2})
    ->Args({1 << 16, 2048, 2})
    ->Args({1 << 20, 2048, 2})
    ->Args({1 << 16, 16384, 2})
    ->Args({1 << 20, 16384, 2})
    ->Args({1 << 16, 32, 10})
    ->Args({1 << 20, 32, 10})
    ->Args({1 << 16, 256, 10})
    ->Args({1 << 20, 256, 10})
    ->Args({1 << 16, 2048, 10})
    ->Args({1 << 20, 2048, 10})
    ->Args({1 << 16, 16384, 10})
    ->Args({1 << 20, 16384, 10})
    ->Args({1 << 16, 32, 100})
    ->Args({1 << 20, 32, 100})
    ->Args({1 << 16, 256, 100})
    ->Args({1 << 20, 256, 100})
    ->Args({1 << 16, 2048, 100})
    ->Args({1 << 20, 2048, 100})
    ->Args({1 << 16, 16384, 100})
    ->Args({1 << 20, 16384, 100});

}  // namespace
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
