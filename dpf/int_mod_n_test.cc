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

#include "dpf/int_mod_n.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>

#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "dpf/internal/status_matchers.h"

namespace distributed_point_functions {
namespace {

using BaseInteger = uint32_t;
constexpr BaseInteger kModulus = 4294967295;  // pow(2, 32)-1
using MyIntModN = IntModN<BaseInteger, kModulus>;

TEST(IntModNTest, DefaultValueIsZero) {
  MyIntModN a;
  EXPECT_EQ(a.value(), 0);
}

TEST(IntModNTest, SetValueWorks) {
  MyIntModN a;
  EXPECT_EQ(a.value(), 0);
  a = 23;
  EXPECT_EQ(a.value(), 23);
}

TEST(IntModNTest, AdditionWithoutWrapAroundWorks) {
  MyIntModN a;
  MyIntModN b;
  a += b;
  EXPECT_EQ(a.value(), 0);
  b = 23;
  a += b;
  EXPECT_EQ(a.value(), 23);
  b = 4294967200;
  a += b;
  EXPECT_EQ(a.value(), 4294967223);
}

TEST(IntModNTest, AdditionWithWrapAroundWorks) {
  MyIntModN a;
  MyIntModN b;
  a += b;
  EXPECT_EQ(a.value(), 0);
  b = 23;
  a += b;
  EXPECT_EQ(a.value(), 23);
  b = kModulus - 10;
  a += b;
  EXPECT_EQ(a.value(), 13);
}

TEST(IntModNTest, NegationWorks) {
  MyIntModN a(10);
  MyIntModN b = -a;
  EXPECT_EQ(a + b, MyIntModN(0));
}

TEST(IntModNTest, GetNumBytesRequiredFailsIfUnfeasible) {
  absl::StatusOr<int> result = MyIntModN::GetNumBytesRequired(5, 95);
  EXPECT_THAT(result,
              dpf_internal::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::StartsWith(absl::StrCat(
                      "For num_samples = 5 and kModulus = ", kModulus))));
}

TEST(IntModNTest, GetNumBytesRequiredSucceedsIfFeasible) {
  absl::StatusOr<int> result = MyIntModN::GetNumBytesRequired(5, 32);
  EXPECT_EQ(result.ok(), true);
}

TEST(IntModNTest, SampleFailsIfUnfeasible) {
  absl::StatusOr<int> r_getnum = MyIntModN::GetNumBytesRequired(5, 94);
  EXPECT_EQ(r_getnum.ok(), true);

  std::string bytes = std::string(16, '#');
  EXPECT_GT(r_getnum.value(), bytes.size());
  std::vector<MyIntModN> samples(5);
  absl::Status r_sample =
      MyIntModN::SampleFromBytes(bytes, 94, absl::MakeSpan(samples));
  EXPECT_EQ(r_sample.ok(), false);
  EXPECT_THAT(
      r_sample,
      dpf_internal::StatusIs(
          absl::StatusCode::kInvalidArgument,
          "The number of bytes provided (16) is insufficient for the required "
          "statistical security and number of samples."));
}

TEST(IntModNTest, SampleSucceedsIfFeasible) {
  absl::StatusOr<int> r_getnum = MyIntModN::GetNumBytesRequired(5, 94);
  EXPECT_EQ(r_getnum.ok(), true);

  std::string bytes = std::string(r_getnum.value(), '#');
  std::vector<MyIntModN> samples(5);
  absl::Status r_sample =
      MyIntModN::SampleFromBytes(bytes, 94, absl::MakeSpan(samples));
  EXPECT_EQ(r_sample.ok(), true);
}

TEST(IntModNTest, FirstEntryOfSamplesIsAsExpected) {
  absl::StatusOr<int> r_getnum = MyIntModN::GetNumBytesRequired(5, 94);
  EXPECT_EQ(r_getnum.ok(), true);

  std::string bytes = std::string(r_getnum.value(), '#');
  std::vector<MyIntModN> samples(5);
  absl::Status r_sample =
      MyIntModN::SampleFromBytes(bytes, 94, absl::MakeSpan(samples));
  EXPECT_EQ(r_sample.ok(), true);
  EXPECT_EQ(
      samples[0].value(),
      MyIntModN::ConvertBytesTo<absl::uint128>(bytes.substr(0, 16)) % kModulus);
}

TEST(IntModNTest, SamplesIsZeroIfBytesIsZero) {
  absl::StatusOr<int> r_getnum = MyIntModN::GetNumBytesRequired(5, 94);
  EXPECT_EQ(r_getnum.ok(), true);

  std::string bytes = std::string(r_getnum.value(), '\0');
  std::vector<MyIntModN> samples(5);
  absl::Status r_sample =
      MyIntModN::SampleFromBytes(bytes, 94, absl::MakeSpan(samples));
  EXPECT_EQ(r_sample.ok(), true);
  EXPECT_EQ(samples[0].value(), 0);
  EXPECT_EQ(samples[1].value(), 0);
  EXPECT_EQ(samples[2].value(), 0);
  EXPECT_EQ(samples[3].value(), 0);
  EXPECT_EQ(samples[4].value(), 0);
}

TEST(IntModNTest, SampleFromBytesWorksInConcreteExample) {
  absl::StatusOr<int> r_getnum = MyIntModN::GetNumBytesRequired(5, 94);
  EXPECT_EQ(r_getnum.ok(), true);
  EXPECT_EQ(*r_getnum, 32);
  std::string bytes = "this is a length 32 test string.";
  EXPECT_EQ(bytes.size(), 32);

  std::vector<MyIntModN> samples(5);
  absl::Status r_sample =
      MyIntModN::SampleFromBytes(bytes, 94, absl::MakeSpan(samples));
  EXPECT_EQ(r_sample.ok(), true);
  absl::uint128 r =
      MyIntModN::ConvertBytesTo<absl::uint128>("this is a length");
  EXPECT_EQ(samples[0].value(), r % kModulus);
  r /= kModulus;
  r <<= (sizeof(BaseInteger) * 8);
  r |= MyIntModN::ConvertBytesTo<BaseInteger>(" 32 ");
  EXPECT_EQ(samples[1].value(), r % kModulus);
  r /= kModulus;
  r <<= (sizeof(BaseInteger) * 8);
  r |= MyIntModN::ConvertBytesTo<BaseInteger>("test");
  EXPECT_EQ(samples[2].value(), r % kModulus);
  r /= kModulus;
  r <<= (sizeof(BaseInteger) * 8);
  r |= MyIntModN::ConvertBytesTo<BaseInteger>(" str");
  EXPECT_EQ(samples[3].value(), r % kModulus);
  r /= kModulus;
  r <<= (sizeof(BaseInteger) * 8);
  r |= MyIntModN::ConvertBytesTo<BaseInteger>("ing.");
  EXPECT_EQ(samples[4].value(), r % kModulus);
}

TEST(IntModNTest, SampleFromBytesFailsAsExpectedInConcreteExample) {
  absl::StatusOr<int> r_getnum = MyIntModN::GetNumBytesRequired(5, 94);
  EXPECT_EQ(r_getnum.ok(), true);
  EXPECT_EQ(*r_getnum, 32);
  std::string bytes = "this is a length 32 test string.";
  EXPECT_EQ(bytes.size(), 32);

  std::vector<MyIntModN> samples(5);
  absl::Status r_sample =
      MyIntModN::SampleFromBytes(bytes, 94, absl::MakeSpan(samples));
  EXPECT_EQ(r_sample.ok(), true);
  absl::uint128 r =
      MyIntModN::ConvertBytesTo<absl::uint128>("this is a length");
  EXPECT_EQ(samples[0].value(), r % kModulus);
  r /= kModulus;
  r <<= (sizeof(BaseInteger) * 8);
  r |= MyIntModN::ConvertBytesTo<BaseInteger>(" 32 ");
  EXPECT_EQ(samples[1].value(), r % kModulus);
  r /= kModulus;
  r <<= (sizeof(BaseInteger) * 8);
  r |= MyIntModN::ConvertBytesTo<BaseInteger>("test");
  EXPECT_EQ(samples[2].value(), r % kModulus);
  r /= kModulus;
  r <<= (sizeof(BaseInteger) * 8);
  r |= MyIntModN::ConvertBytesTo<BaseInteger>(" str");
  EXPECT_EQ(samples[3].value(), r % kModulus);
  r /= kModulus;
  r <<= (sizeof(BaseInteger) * 8);
  r |= MyIntModN::ConvertBytesTo<BaseInteger>("ing#");  // # instead of .
  EXPECT_NE(samples[4].value(), r % kModulus);
}

// Test if IntModN operators are in fact constexpr. This will fail to compile
// otherwise.
constexpr MyIntModN TestAddition() { return MyIntModN(2) + MyIntModN(5); }
static_assert(TestAddition().value() == 7,
              "constexpr addition of IntModNs incorrect");

constexpr MyIntModN TestSubtraction() { return MyIntModN(5) - MyIntModN(2); }
static_assert(TestSubtraction().value() == 3,
              "constexpr subtraction of IntModNs incorrect");

constexpr MyIntModN TestAssignment() {
  MyIntModN x(0);
  x = 5;
  return x;
}
static_assert(TestAssignment().value() == 5,
              "constexpr assignment to IntModN incorrect");

#ifdef ABSL_HAVE_INTRINSIC_INT128
constexpr unsigned __int128 kModulus128 =
    std::numeric_limits<unsigned __int128>::max() - 158;  // 2^128 - 159
using MyIntModN128 = IntModN<unsigned __int128, kModulus128>;
constexpr MyIntModN128 TestAddition128() {
  return MyIntModN128(2) + MyIntModN128(5);
}
static_assert(TestAddition128().value() == 7,
              "constexpr addition of IntModNs incorrect");
#endif

}  // namespace
}  // namespace distributed_point_functions
