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
#include "pir/cuckoo_hashed_dpf_pir_database.h"
#include "pir/cuckoo_hashing_sparse_dpf_pir_server.h"
#include "pir/hashing/hash_family.h"
#include "pir/hashing/hash_family_config.h"
#include "pir/hashing/hash_family_config.pb.h"
#include "pir/testing/mock_pir_database.h"
#include "pir/testing/request_generator.h"

// We use the following flags instead of benchmark arguments to set the database
// dimension and query size for all the benchmarks to avoid recompilation.
ABSL_FLAG(int, num_records, 1 << 16,
          "The number of records in the sparse database.");
ABSL_FLAG(int, num_bytes_per_key, 6, "The number of bytes in each key.");
ABSL_FLAG(int, num_bytes_per_value, 8, "The number of bytes in each value.");
ABSL_FLAG(int, num_keys_per_request, 1,
          "The number of query keys in each PIR request.");

namespace distributed_point_functions {
namespace {

using Database = CuckooHashingSparseDpfPirServer::Database;

constexpr HashFamilyConfig::HashFamily kHashFamily =
    HashFamilyConfig::HASH_FAMILY_SHA256;

// Benchmarks `HandlePlainRequest()` which is the core part of `HandleRequest()`
// on both the main and the helper server.
void BM_HandlePlainRequest(benchmark::State& state) {
  int num_records = absl::GetFlag(FLAGS_num_records);
  int num_bytes_per_key = absl::GetFlag(FLAGS_num_bytes_per_key);
  int num_bytes_per_value = absl::GetFlag(FLAGS_num_bytes_per_value);
  int num_keys_per_request = absl::GetFlag(FLAGS_num_keys_per_request);

  // Setup cuckoo hashing parameters.
  PirConfig config;
  config.mutable_cuckoo_hashing_sparse_dpf_pir_config()->set_num_elements(
      num_records);
  config.mutable_cuckoo_hashing_sparse_dpf_pir_config()->set_hash_family(
      kHashFamily);
  DPF_ASSERT_OK_AND_ASSIGN(
      CuckooHashingParams params,
      CuckooHashingSparseDpfPirServer::GenerateParams(config));

  // Generate random (key, value) pairs.
  DPF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> keys,
                           pir_testing::GenerateRandomStringsEqualSize(
                               num_records, num_bytes_per_key));
  DPF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> values,
                           pir_testing::GenerateRandomStringsEqualSize(
                               num_records, num_bytes_per_value));
  std::vector<Database::RecordType> pairs(num_records);
  for (int i = 0; i < num_records; ++i) {
    pairs[i] = {keys[i], std::move(values[i])};
  }

  // Build the database using the random (key, value) pairs.
  CuckooHashedDpfPirDatabase::Builder builder;
  builder.SetParams(params);
  for (auto& element : pairs) {
    builder.Insert(std::move(element));
  }
  DPF_ASSERT_OK_AND_ASSIGN(auto database, builder.Build());

  DPF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CuckooHashingSparseDpfPirServer> server,
      CuckooHashingSparseDpfPirServer::CreatePlain(params,
                                                   std::move(database)));

  // Instantiate hash functions to create the client request.
  DPF_ASSERT_OK_AND_ASSIGN(
      HashFamily hash_family,
      CreateHashFamilyFromConfig(params.hash_family_config()));
  DPF_ASSERT_OK_AND_ASSIGN(
      std::vector<HashFunction> hash_functions,
      CreateHashFunctions(std::move(hash_family), params.num_hash_functions()));

  DPF_ASSERT_OK_AND_ASSIGN(
      auto request_generator,
      pir_testing::RequestGenerator::Create(
          params.num_buckets(),
          CuckooHashingSparseDpfPirServer::kEncryptionContextInfo));

  absl::BitGen bitgen;
  for (auto _ : state) {
    state.PauseTiming();

    // Generate `num_keys_per_request` many queries with random keys. Every
    // query key is hashed into indices using all the hash functions. So we will
    // have `num_keys_per_request * hash_functions.size()` many indices.
    std::vector<int> indices;
    indices.reserve(num_keys_per_request * hash_functions.size());
    for (int i = 0; i < num_keys_per_request; ++i) {
      int query_index = absl::Uniform<int>(bitgen, 0, keys.size());
      absl::string_view query_key = keys[query_index];
      for (const HashFunction& hash_function : hash_functions) {
        indices.push_back(hash_function(query_key, params.num_buckets()));
      }
    }
    // Generate plain requests for `indices`.
    PirRequest request1, request2;
    DPF_ASSERT_OK_AND_ASSIGN(
        std::tie(*request1.mutable_dpf_pir_request()->mutable_plain_request(),
                 *request2.mutable_dpf_pir_request()->mutable_plain_request()),
        request_generator->CreateDpfPirPlainRequests(indices));

    // Record the time to handle the request on a single server.
    state.ResumeTiming();
    PirResponse response1;
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
