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

#include "pir/hashing/farm_hash_family.h"

#include "absl/functional/any_invocable.h"
#include "absl/numeric/int128.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"  // third_party/benchmark
#include "farmhash/farmhash.h"
#include "gtest/gtest.h"

namespace distributed_point_functions {
namespace {

inline absl::uint128 ToAbslU128(util::uint128_t x) {
  return absl::MakeUint128(x.second, x.first);
}

TEST(FarmHashFunction, IsAHashFunction) {
  HashFunction hash_function = FarmHashFunction("");
  ::benchmark::DoNotOptimize(hash_function);
}

TEST(FarmHashFamily, IsAHashFamily) {
  HashFamily hash_family = FarmHashFamily{};
  ::benchmark::DoNotOptimize(hash_family);
}

TEST(FarmHashFamily, HashesCorrectly) {
  constexpr absl::string_view kHashFunctionSeed = "kHashFunctionSeed";
  constexpr absl::string_view kHashInput = "kHashInput";
  absl::uint128 hash128 = ToAbslU128(util::Hash128WithSeed(
      kHashInput.data(), kHashInput.size(), util::Hash128(kHashFunctionSeed)));

  HashFunction hasher = FarmHashFamily{}(kHashFunctionSeed);

  for (int i = 1; i < 1000; ++i) {
    int wanted = static_cast<int>(hash128 % i);
    EXPECT_EQ(hasher(kHashInput, i), wanted);
  }
}

}  // namespace
}  // namespace distributed_point_functions
