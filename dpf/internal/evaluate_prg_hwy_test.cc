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

#include "dpf/internal/evaluate_prg_hwy.h"

#include <gtest/gtest.h>

#include "absl/numeric/int128.h"
#include "dpf/aes_128_fixed_key_hash.h"
#include "dpf/internal/status_matchers.h"
#include "hwy/aligned_allocator.h"

// clang-format off
#define HWY_IS_TEST 1;
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "dpf/internal/evaluate_prg_hwy_test.cc"  // NOLINT
#include "hwy/foreach_target.h"
// clang-format on
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace distributed_point_functions {
namespace dpf_internal {
namespace HWY_NAMESPACE {

constexpr absl::uint128 kKey0 =
    absl::MakeUint128(0x0000000000000000, 0x0000000000000000);
constexpr absl::uint128 kKey1 =
    absl::MakeUint128(0x1111111111111111, 0x1111111111111111);
constexpr int kNumBlocks = 123;

void TestOutputMatchesNoHwyVersion(int num_levels) {
  // Generate seeds.
  auto seeds_in = hwy::AllocateAligned<absl::uint128>(kNumBlocks);
  auto control_bits_in = hwy::AllocateAligned<bool>(kNumBlocks);
  auto paths = hwy::AllocateAligned<absl::uint128>(kNumBlocks);
  for (int i = 0; i < kNumBlocks; ++i) {
    // All of these are arbitrary.
    seeds_in[i] = absl::MakeUint128(i, i + 1);
    paths[i] = absl::MakeUint128(23 * i + 42, 42 * i + 23);
    control_bits_in[i] = (i % 7 == 0);
  }
  auto seeds_out = hwy::AllocateAligned<absl::uint128>(kNumBlocks);
  auto control_bits_out = hwy::AllocateAligned<bool>(kNumBlocks);

  // Generate correction words.
  auto correction_seeds = hwy::AllocateAligned<absl::uint128>(num_levels);
  auto correction_controls_left = hwy::AllocateAligned<bool>(num_levels);
  auto correction_controls_right = hwy::AllocateAligned<bool>(num_levels);
  for (int i = 0; i < num_levels; ++i) {
    correction_seeds[i] = absl::MakeUint128(i + 1, i);
    correction_controls_left[i] = (i % 23 == 0);
    correction_controls_right[i] = (i % 42 != 0);
  }

  // Set up PRGs.
  DPF_ASSERT_OK_AND_ASSIGN(
      auto prg_left,
      distributed_point_functions::Aes128FixedKeyHash::Create(kKey0));
  DPF_ASSERT_OK_AND_ASSIGN(
      auto prg_right,
      distributed_point_functions::Aes128FixedKeyHash::Create(kKey1));

  // Evaluate with Highway enabled.
  DPF_ASSERT_OK(EvaluateSeeds(
      kNumBlocks, num_levels, seeds_in.get(), control_bits_in.get(),
      paths.get(), correction_seeds.get(), correction_controls_left.get(),
      correction_controls_right.get(), prg_left, prg_right, seeds_out.get(),
      control_bits_out.get()));

  // Evaluate without highway.
  auto seeds_out_wanted = hwy::AllocateAligned<absl::uint128>(kNumBlocks);
  auto control_bits_out_wanted = hwy::AllocateAligned<bool>(kNumBlocks);
  DPF_ASSERT_OK(EvaluateSeedsNoHwy(
      kNumBlocks, num_levels, seeds_in.get(), control_bits_in.get(),
      paths.get(), correction_seeds.get(), correction_controls_left.get(),
      correction_controls_right.get(), prg_left, prg_right,
      seeds_out_wanted.get(), control_bits_out_wanted.get()));

  // Check that both evaluations are equal.
  for (int i = 0; i < kNumBlocks; ++i) {
    EXPECT_EQ(seeds_out[i], seeds_out_wanted[i]);
    EXPECT_EQ(control_bits_out[i], control_bits_out_wanted[i]);
  }
}

void TestAll() {
  constexpr int kMaxLevels = 3;
  for (int num_levels = 1; num_levels < kMaxLevels; ++num_levels) {
    TestOutputMatchesNoHwyVersion(num_levels);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace dpf_internal
}  // namespace distributed_point_functions
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace distributed_point_functions {
namespace dpf_internal {
HWY_BEFORE_TEST(EvaluatePrgHwyTest);
HWY_EXPORT_AND_TEST_P(EvaluatePrgHwyTest, TestAll);
}  // namespace dpf_internal
}  // namespace distributed_point_functions

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif