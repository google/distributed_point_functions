// Copyright 2024 Google LLC
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
#include "pir/hashing/hash_family.h"
#include "pir/hashing/hash_family_config.h"
#include "pir/hashing/hash_family_config.pb.h"
#include "pir/simple_hashed_dpf_pir_database.h"
#include "pir/simple_hashing_sparse_dpf_pir_server.h"
#include "pir/testing/mock_pir_database.h"
#include "pir/testing/request_generator.h"

// Set the database parameters to be benchmarked using the following flags.
ABSL_FLAG(int, num_records, 1 << 16,
          "The number of records in the sparse database.");
ABSL_FLAG(int, num_buckets, 1 << 12, "The number of buckets.");
ABSL_FLAG(int, num_bytes_per_key, 6, "The number of bytes in each key.");
ABSL_FLAG(int, num_bytes_per_value, 8, "The number of bytes in each value.");
ABSL_FLAG(int, num_keys_per_request, 1,
          "The number of query keys in each PIR request.");

namespace distributed_point_functions {
namespace {

using Database = SimpleHashingSparseDpfPirServer::Database;

constexpr HashFamilyConfig::HashFamily kHashFamily =
    HashFamilyConfig::HASH_FAMILY_SHA256;

void BM_HandlePlainRequest(benchmark::State& state) {
  int num_records = absl::GetFlag(FLAGS_num_records);
  int num_buckets = absl::GetFlag(FLAGS_num_buckets);
  int num_bytes_per_key = absl::GetFlag(FLAGS_num_bytes_per_key);
  int num_bytes_per_value = absl::GetFlag(FLAGS_num_bytes_per_value);
  int num_keys_per_request = absl::GetFlag(FLAGS_num_keys_per_request);

  // Set up the hashing parameters.
  PirConfig config;
  config.mutable_simple_hashing_sparse_dpf_pir_config()->set_num_buckets(
      num_buckets);
  config.mutable_simple_hashing_sparse_dpf_pir_config()->set_hash_family(
      kHashFamily);

  DPF_ASSERT_OK_AND_ASSIGN(
      SimpleHashingParams params,
      SimpleHashingSparseDpfPirServer::GenerateParams(config));

  // Generate the database of (key, value) pairs, where all keys have the
  // same size `num_bytes_per_key`, and similarly for all the values.
  // We generate random keys, and assume that there is no collision.
  DPF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> keys,
                           pir_testing::GenerateRandomStringsEqualSize(
                               num_records, num_bytes_per_key));
  DPF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> values,
                           pir_testing::GenerateRandomStringsEqualSize(
                               num_records, num_bytes_per_value));

  std::vector<Database::RecordType> pairs(num_records);
  for (int i = 0; i < num_records; ++i) {
    pairs[i] = {keys[i], values[i]};
  }

  SimpleHashedDpfPirDatabase::Builder builder;
  builder.SetParams(params);
  DPF_ASSERT_OK_AND_ASSIGN(
      auto database,
      pir_testing::CreateFakeDatabase<SimpleHashedDpfPirDatabase>(pairs,
                                                                  &builder));

  // Setup server.
  DPF_ASSERT_OK_AND_ASSIGN(auto server,
                           SimpleHashingSparseDpfPirServer::CreatePlain(
                               params, std::move(database)));

  // Create hash functions to hash the client's query index.
  DPF_ASSERT_OK_AND_ASSIGN(
      HashFamily hash_family,
      CreateHashFamilyFromConfig(params.hash_family_config()));
  DPF_ASSERT_OK_AND_ASSIGN(std::vector<HashFunction> hash_functions,
                           CreateHashFunctions(std::move(hash_family), 1));

  absl::BitGen bitgen;
  for (auto _ : state) {
    state.PauseTiming();

    std::vector<int> indices;
    indices.reserve(num_keys_per_request);
    for (int i = 0; i < num_keys_per_request; ++i) {
      int query_index = absl::Uniform<int>(bitgen, 0, num_records);
      int index = hash_functions[0](keys[query_index], params.num_buckets());
      indices.push_back(index);
    }

    // Generate plain requests for `indices`.
    DPF_ASSERT_OK_AND_ASSIGN(
        auto request_generator,
        pir_testing::RequestGenerator::Create(
            params.num_buckets(),
            SimpleHashingSparseDpfPirServer::kEncryptionContextInfo));
    PirRequest request1, request2;
    DPF_ASSERT_OK_AND_ASSIGN(
        std::tie(*request1.mutable_dpf_pir_request()->mutable_plain_request(),
                 *request2.mutable_dpf_pir_request()->mutable_plain_request()),
        request_generator->CreateDpfPirPlainRequests(indices));

    // Record the time to handle the request on a single server.
    state.ResumeTiming();
    PirResponse response1, response2;
    std::vector<std::string> response_keys;
    DPF_ASSERT_OK_AND_ASSIGN(response1, server->HandleRequest(request1));
  }
}
BENCHMARK(BM_HandlePlainRequest);

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
