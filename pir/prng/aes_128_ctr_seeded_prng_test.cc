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

#include "pir/prng/aes_128_ctr_seeded_prng.h"

#include <stddef.h>

#include <iostream>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "benchmark/benchmark.h"  // third_party/benchmark
#include "dpf/internal/status_matchers.h"
#include "gtest/gtest.h"
#include "openssl/rand.h"

namespace distributed_point_functions {
namespace {

class SeededPrngTest : public ::testing::Test {
 protected:
  void SetUp() override {
    DPF_ASSERT_OK_AND_ASSIGN(seed_, Aes128CtrSeededPrng::GenerateSeed());
    DPF_ASSERT_OK_AND_ASSIGN(prng_, Aes128CtrSeededPrng::Create(seed_));
  }

  std::string seed_;
  std::unique_ptr<Aes128CtrSeededPrng> prng_;
};

TEST(SeededPrng, SeedHasCorrectSize) {
  DPF_ASSERT_OK_AND_ASSIGN(std::string seed,
                           Aes128CtrSeededPrng::GenerateSeed());
  EXPECT_EQ(seed.size(), Aes128CtrSeededPrng::SeedSize());
}

TEST(SeededPrng, DifferentSeedsAreGenerated) {
  DPF_ASSERT_OK_AND_ASSIGN(std::string seed1,
                           Aes128CtrSeededPrng::GenerateSeed());
  DPF_ASSERT_OK_AND_ASSIGN(std::string seed2,
                           Aes128CtrSeededPrng::GenerateSeed());

  EXPECT_NE(seed1, seed2);
}

TEST(SeededPrng, CreateFailsOnWrongSizeKey) {
  EXPECT_FALSE(Aes128CtrSeededPrng::Create("wrong_seed_size").ok());
}

TEST(SeededPrng, CreateFailsOnWrongSizeNonce) {
  DPF_ASSERT_OK_AND_ASSIGN(std::string seed,
                           Aes128CtrSeededPrng::GenerateSeed());

  EXPECT_FALSE(Aes128CtrSeededPrng::CreateWithNonce(seed, "wrong_size").ok());
}

TEST_F(SeededPrngTest, OutputIsDeterministic) {
  DPF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Aes128CtrSeededPrng> prng2,
                           Aes128CtrSeededPrng::Create(seed_));
  size_t num_elements = 100;

  std::string output1 = prng_->GetRandomBytes(num_elements);
  std::string output2 = prng2->GetRandomBytes(num_elements);

  EXPECT_EQ(output1, output2);
}

TEST_F(SeededPrngTest, MultipleCallsGiveSameSequenceOfBytes) {
  DPF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Aes128CtrSeededPrng> prng2,
                           Aes128CtrSeededPrng::Create(seed_));
  size_t num_elements = 100;

  std::string output1a = prng_->GetRandomBytes(num_elements);
  std::string output1b = prng_->GetRandomBytes(num_elements);
  std::string output2 = prng2->GetRandomBytes(2 * num_elements);

  EXPECT_EQ(absl::StrCat(output1a, output1b), output2);
}

TEST_F(SeededPrngTest, DifferentSeedsGiveDifferentOutputs) {
  DPF_ASSERT_OK_AND_ASSIGN(std::string seed2,
                           Aes128CtrSeededPrng::GenerateSeed());
  DPF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Aes128CtrSeededPrng> prng2,
                           Aes128CtrSeededPrng::Create(seed2));
  size_t num_elements = 100;

  std::string output1 = prng_->GetRandomBytes(num_elements);
  std::string output2 = prng2->GetRandomBytes(num_elements);

  EXPECT_NE(output1, output2);
}

TEST_F(SeededPrngTest, DifferentNoncesGiveDifferentOutputs) {
  DPF_ASSERT_OK_AND_ASSIGN(std::string nonce1,
                           Aes128CtrSeededPrng::GenerateSeed());
  DPF_ASSERT_OK_AND_ASSIGN(std::string nonce2,
                           Aes128CtrSeededPrng::GenerateSeed());
  DPF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Aes128CtrSeededPrng> prng1,
                           Aes128CtrSeededPrng::CreateWithNonce(seed_, nonce1));
  DPF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Aes128CtrSeededPrng> prng2,
                           Aes128CtrSeededPrng::CreateWithNonce(seed_, nonce2));
  size_t num_elements = 100;

  std::string output1 = prng_->GetRandomBytes(num_elements);
  std::string output2 = prng2->GetRandomBytes(num_elements);

  EXPECT_NE(output1, output2);
}

TEST_F(SeededPrngTest, SucceedsOnLength0) {
  std::string output = prng_->GetRandomBytes(0);

  EXPECT_EQ(output, std::string({}));
}

TEST_F(SeededPrngTest, FixedSeed) {
  const size_t num_elements = 32;
  const std::string fixed_seed = {0, 1, 2,  3,  4,  5,  6,  7,
                                  8, 9, 10, 11, 12, 13, 14, 15};
  const std::string reference = {
      '\xc6', '\xa1', '\x3b', '\x37', '\x87', '\x8f', '\x5b', '\x82',
      '\x6f', '\x4f', '\x81', '\x62', '\xa1', '\xc8', '\xd8', '\x79',
      '\x73', '\x46', '\x13', '\x95', '\x95', '\xc0', '\xb4', '\x1e',
      '\x49', '\x7b', '\xbd', '\xe3', '\x65', '\xf4', '\x2d', '\x0a',
  };
  DPF_ASSERT_OK_AND_ASSIGN(prng_, Aes128CtrSeededPrng::Create(fixed_seed));

  std::string output = prng_->GetRandomBytes(num_elements);

  EXPECT_EQ(output.size(), num_elements);
  for (size_t i = 0; i < num_elements; i++) {
    EXPECT_EQ(output[i], reference[i]);
  }
  std::cout << std::endl;
}

void BM_GetRandomBytesFromSeed(benchmark::State& state) {
  DPF_ASSERT_OK_AND_ASSIGN(std::string seed,
                           Aes128CtrSeededPrng::GenerateSeed());
  DPF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Aes128CtrSeededPrng> prng,
                           Aes128CtrSeededPrng::Create(seed));
  size_t num_elements = state.range(0);

  for (auto s : state) {
    std::string output = prng->GetRandomBytes(num_elements);
    benchmark::DoNotOptimize(output);
  }
}

BENCHMARK(BM_GetRandomBytesFromSeed)
    ->Range(128, /* blocklist bloom filter has about 800 entries */
            16384 /* interest group bloom filter has about 10k entries */);
}  // namespace
}  // namespace distributed_point_functions
