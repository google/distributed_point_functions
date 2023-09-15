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

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"  // third_party/benchmark
#include "dpf/internal/status_matchers.h"
#include "pir/dense_dpf_pir_database.h"
#include "pir/dense_dpf_pir_server.h"
#include "pir/testing/mock_pir_database.h"
#include "pir/testing/request_generator.h"

// We use the following flags instead of benchmark arguments to set the database
// dimension and query size for all the benchmarks to avoid recompilation.
ABSL_FLAG(int, num_records, 1 << 16,
          "The number of records in the dense database.");
ABSL_FLAG(int, num_bytes_per_record, 128,
          "The number of bytes in each record.");
ABSL_FLAG(int, num_indices_per_request, 1,
          "The number of query indices in each PIR request.");

namespace distributed_point_functions {
namespace {

// Benchmarks `HandlePlainRequest()` which is the core part of `HandleRequest()`
// on both the main and the helper server.
void BM_HandlePlainRequestWithEqualSizeRecords(benchmark::State& state) {
  int num_records = absl::GetFlag(FLAGS_num_records);
  int num_bytes_per_record = absl::GetFlag(FLAGS_num_bytes_per_record);
  int num_indices_per_request = absl::GetFlag(FLAGS_num_indices_per_request);

  // Setup PIR parameters.
  PirConfig config;
  config.mutable_dense_dpf_pir_config()->set_num_elements(num_records);

  // Build a dense database with random records.
  DPF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> values,
                           pir_testing::GenerateRandomStringsEqualSize(
                               num_records, num_bytes_per_record));
  DenseDpfPirDatabase::Builder builder;
  for (auto& value : values) {
    builder.Insert(std::move(value));
  }
  DPF_ASSERT_OK_AND_ASSIGN(auto database, builder.Build());

  // Create the server.
  DPF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<DenseDpfPirServer> server,
      DenseDpfPirServer::CreatePlain(config, std::move(database)));

  DPF_ASSERT_OK_AND_ASSIGN(
      auto request_generator,
      pir_testing::RequestGenerator::Create(
          num_records, DenseDpfPirServer::kEncryptionContextInfo));
  absl::BitGen bitgen;

  for (auto _ : state) {
    state.PauseTiming();

    // Generate dense PIR queries for random indices.
    std::vector<int> indices;
    indices.reserve(num_indices_per_request);
    for (int i = 0; i < num_indices_per_request; ++i) {
      indices.push_back(absl::Uniform<int>(bitgen, 0, num_records));
    }

    PirRequest request1, request2;
    DPF_ASSERT_OK_AND_ASSIGN(
        std::tie(*request1.mutable_dpf_pir_request()->mutable_plain_request(),
                 *request2.mutable_dpf_pir_request()->mutable_plain_request()),
        request_generator->CreateDpfPirPlainRequests(indices));

    // Record the time to handle the request on a single server.
    state.ResumeTiming();
    benchmark::DoNotOptimize(server->HandleRequest(request1));
  }
}
BENCHMARK(BM_HandlePlainRequestWithEqualSizeRecords);

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
  absl::ParseCommandLine(argc, argv);
  if (!FLAGS_benchmark_filter.empty()) {
    benchmark::RunSpecifiedBenchmarks();
  }
  benchmark::Shutdown();
  return 0;
}
