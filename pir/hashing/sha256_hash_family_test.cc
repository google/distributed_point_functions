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

#include "pir/hashing/sha256_hash_family.h"

#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "shell_encryption/int256.h"

namespace distributed_point_functions {
namespace {

TEST(Sha256HashFunction, IsAHashFunction) {
  HashFunction hash_function = SHA256HashFunction("");
  ::benchmark::DoNotOptimize(hash_function);
}

TEST(Sha256HashFamily, IsAHashFamily) {
  HashFamily hash_family = SHA256HashFamily{};
  ::benchmark::DoNotOptimize(hash_family);
}

TEST(Sha256HashFamily, HashesCorrectly) {
  // The following constants (seed, input) and the hash output are from
  // http://csrc.nist.gov/groups/STM/cavp/index.html#03
  // Specifically, we have SHA256(StrCat(seed, input)) = output
  constexpr absl::string_view kHashFunctionSeedHex =
      "5a86b737eaea8ee976a0a24da63e7ed7";
  constexpr absl::string_view kHashInputHex =
      "eefad18a101c1211e2b3650c5187c2a8a650547208251f6d4237e661c7bf4c77f3353903"
      "94c37fa1a9f9be836ac28509";
  constexpr absl::string_view kHashOutputHex =
      "42e61e174fbb3897d6dd6cef3dd2802fe67b331953b06114a65c772859dfc1aa";

  // convert hex representations to byte strings
  const std::string kHashFunctionSeed =
      absl::HexStringToBytes(kHashFunctionSeedHex);
  const std::string kHashInput = absl::HexStringToBytes(kHashInputHex);
  const std::string kFullHashOutput = absl::HexStringToBytes(kHashOutputHex);

  // convert hash to a 256-bit integer to check modulo reduction
  rlwe::uint256 wanted_hash_value{0};
  for (int i = 0; i < kFullHashOutput.size(); ++i) {
    rlwe::uint256 x{static_cast<uint8_t>(kFullHashOutput[i])};
    wanted_hash_value |= x << (8 * i);
  }

  HashFunction hash = SHA256HashFamily{}(kHashFunctionSeed);

  for (int i = 1; i < 1000; ++i) {
    int wanted = static_cast<int>(wanted_hash_value % rlwe::uint256{i});
    EXPECT_EQ(hash(kHashInput, i), wanted);
  }
}

void BM_Hash(benchmark::State& state) {
  constexpr absl::string_view kHashFunctionSeed = "kHashFunctionSeed";
  int num_values = state.range(0);
  int upper_bound = state.range(1);

  for (auto _ : state) {
    HashFunction hash = SHA256HashFamily{}(kHashFunctionSeed);

    for (int i = 0; i < num_values; i++) {
      ::benchmark::DoNotOptimize(hash(absl::StrCat(i), upper_bound));
    }
  }
}
BENCHMARK(BM_Hash)
    ->Args({1 << 20, 256})
    ->Args({1 << 20, 1 << 20})
    ->Args({1 << 20, 1 << 30});

}  // namespace
}  // namespace distributed_point_functions
