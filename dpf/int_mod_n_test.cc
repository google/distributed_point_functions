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

TEST(IntModNTest, DefaultValueIsZero) {
  IntModN<BaseInteger, kModulus> a;
  EXPECT_EQ(a.value(), 0);
}

TEST(IntModNTest, SetValueWorks) {
  IntModN<BaseInteger, kModulus> a;
  EXPECT_EQ(a.value(), 0);
  a = 23;
  EXPECT_EQ(a.value(), 23);
}

TEST(IntModNTest, AdditionWithoutWrapAroundWorks) {
  IntModN<BaseInteger, kModulus> a;
  IntModN<BaseInteger, kModulus> b;
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
  IntModN<BaseInteger, kModulus> a;
  IntModN<BaseInteger, kModulus> b;
  a += b;
  EXPECT_EQ(a.value(), 0);
  b = 23;
  a += b;
  EXPECT_EQ(a.value(), 23);
  b = kModulus - 10;
  a += b;
  EXPECT_EQ(a.value(), 13);
}

TEST(IntModNTest, GetNumBytesRequiredFailsIfUnfeasible) {
  absl::StatusOr<int> result =
      IntModN<BaseInteger, kModulus>::GetNumBytesRequired(5, 95);
  EXPECT_THAT(
      result,
      dpf_internal::StatusIs(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("For num_samples = 5 and modulus = ", kModulus,
                       " this approach can only provide 94.0931"
                       " bits of statistical security. You can try calling "
                       "this function "
                       "several times with smaller values of num_samples.")));
}

TEST(IntModNTest, GetNumBytesRequiredSucceedsIfFeasible) {
  absl::StatusOr<int> result =
      IntModN<BaseInteger, kModulus>::GetNumBytesRequired(5, 32);
  EXPECT_EQ(result.ok(), true);
}

TEST(IntModNTest, SampleFailsIfUnfeasible) {
  absl::StatusOr<int> r_getnum =
      IntModN<BaseInteger, kModulus>::GetNumBytesRequired(5, 94);
  EXPECT_EQ(r_getnum.ok(), true);

  std::string bytes = std::string(16, '#');
  EXPECT_GT(r_getnum.value(), bytes.size());
  std::vector<IntModN<BaseInteger, kModulus>> samples =
      std::vector<IntModN<BaseInteger, kModulus>>(5);
  absl::Status r_sample = IntModN<BaseInteger, kModulus>::SampleFromBytes(
      bytes, 94, absl::MakeSpan(samples));
  EXPECT_EQ(r_sample.ok(), false);
  EXPECT_THAT(
      r_sample,
      dpf_internal::StatusIs(
          absl::StatusCode::kInvalidArgument,
          "The number of bytes provided (16) is insufficient for the required "
          "statistical security and number of samples."));
}

TEST(IntModNTest, SampleSucceedsIfFeasible) {
  absl::StatusOr<int> r_getnum =
      IntModN<BaseInteger, kModulus>::GetNumBytesRequired(5, 94);
  EXPECT_EQ(r_getnum.ok(), true);

  std::string bytes = std::string(r_getnum.value(), '#');
  std::vector<IntModN<BaseInteger, kModulus>> samples =
      std::vector<IntModN<BaseInteger, kModulus>>(5);
  absl::Status r_sample = IntModN<BaseInteger, kModulus>::SampleFromBytes(
      bytes, 94, absl::MakeSpan(samples));
  EXPECT_EQ(r_sample.ok(), true);
}

TEST(IntModNTest, FirstEntryOfSamplesIsAsExpected) {
  absl::StatusOr<int> r_getnum =
      IntModN<BaseInteger, kModulus>::GetNumBytesRequired(5, 94);
  EXPECT_EQ(r_getnum.ok(), true);

  std::string bytes = std::string(r_getnum.value(), '#');
  std::vector<IntModN<BaseInteger, kModulus>> samples =
      std::vector<IntModN<BaseInteger, kModulus>>(5);
  absl::Status r_sample = IntModN<BaseInteger, kModulus>::SampleFromBytes(
      bytes, 94, absl::MakeSpan(samples));
  EXPECT_EQ(r_sample.ok(), true);
  EXPECT_EQ(samples[0].value(),
            dpf_internal::ConvertBytesTo<absl::uint128>(bytes.substr(0, 16)) %
                kModulus);
}

TEST(IntModNTest, SamplesIsZeroIfBytesIsZero) {
  absl::StatusOr<int> r_getnum =
      IntModN<BaseInteger, kModulus>::GetNumBytesRequired(5, 94);
  EXPECT_EQ(r_getnum.ok(), true);

  std::string bytes = std::string(r_getnum.value(), '\0');
  std::vector<IntModN<BaseInteger, kModulus>> samples =
      std::vector<IntModN<BaseInteger, kModulus>>(5);
  absl::Status r_sample = IntModN<BaseInteger, kModulus>::SampleFromBytes(
      bytes, 94, absl::MakeSpan(samples));
  EXPECT_EQ(r_sample.ok(), true);
  EXPECT_EQ(samples[0].value(), 0);
  EXPECT_EQ(samples[1].value(), 0);
  EXPECT_EQ(samples[2].value(), 0);
  EXPECT_EQ(samples[3].value(), 0);
  EXPECT_EQ(samples[4].value(), 0);
}

TEST(IntModNTest, SampleFromBytesWorksInConcreteExample) {
  absl::StatusOr<int> r_getnum =
      IntModN<BaseInteger, kModulus>::GetNumBytesRequired(5, 94);
  EXPECT_EQ(r_getnum.ok(), true);
  EXPECT_EQ(*r_getnum, 32);
  std::string bytes = "this is a length 32 test string.";
  EXPECT_EQ(bytes.size(), 32);

  std::vector<IntModN<BaseInteger, kModulus>> samples =
      std::vector<IntModN<BaseInteger, kModulus>>(5);
  absl::Status r_sample = IntModN<BaseInteger, kModulus>::SampleFromBytes(
      bytes, 94, absl::MakeSpan(samples));
  EXPECT_EQ(r_sample.ok(), true);
  absl::uint128 r =
      dpf_internal::ConvertBytesTo<absl::uint128>("this is a length");
  EXPECT_EQ(samples[0].value(), r % kModulus);
  r /= kModulus;
  r <<= (sizeof(BaseInteger) * 8);
  r |= dpf_internal::ConvertBytesTo<BaseInteger>(" 32 ");
  EXPECT_EQ(samples[1].value(), r % kModulus);
  r /= kModulus;
  r <<= (sizeof(BaseInteger) * 8);
  r |= dpf_internal::ConvertBytesTo<BaseInteger>("test");
  EXPECT_EQ(samples[2].value(), r % kModulus);
  r /= kModulus;
  r <<= (sizeof(BaseInteger) * 8);
  r |= dpf_internal::ConvertBytesTo<BaseInteger>(" str");
  EXPECT_EQ(samples[3].value(), r % kModulus);
  r /= kModulus;
  r <<= (sizeof(BaseInteger) * 8);
  r |= dpf_internal::ConvertBytesTo<BaseInteger>("ing.");
  EXPECT_EQ(samples[4].value(), r % kModulus);
}

TEST(IntModNTest, SampleFromBytesFailsAsExpectedInConcreteExample) {
  absl::StatusOr<int> r_getnum =
      IntModN<BaseInteger, kModulus>::GetNumBytesRequired(5, 94);
  EXPECT_EQ(r_getnum.ok(), true);
  EXPECT_EQ(*r_getnum, 32);
  std::string bytes = "this is a length 32 test string.";
  EXPECT_EQ(bytes.size(), 32);

  std::vector<IntModN<BaseInteger, kModulus>> samples =
      std::vector<IntModN<BaseInteger, kModulus>>(5);
  absl::Status r_sample = IntModN<BaseInteger, kModulus>::SampleFromBytes(
      bytes, 94, absl::MakeSpan(samples));
  EXPECT_EQ(r_sample.ok(), true);
  absl::uint128 r =
      dpf_internal::ConvertBytesTo<absl::uint128>("this is a length");
  EXPECT_EQ(samples[0].value(), r % kModulus);
  r /= kModulus;
  r <<= (sizeof(BaseInteger) * 8);
  r |= dpf_internal::ConvertBytesTo<BaseInteger>(" 32 ");
  EXPECT_EQ(samples[1].value(), r % kModulus);
  r /= kModulus;
  r <<= (sizeof(BaseInteger) * 8);
  r |= dpf_internal::ConvertBytesTo<BaseInteger>("test");
  EXPECT_EQ(samples[2].value(), r % kModulus);
  r /= kModulus;
  r <<= (sizeof(BaseInteger) * 8);
  r |= dpf_internal::ConvertBytesTo<BaseInteger>(" str");
  EXPECT_EQ(samples[3].value(), r % kModulus);
  r /= kModulus;
  r <<= (sizeof(BaseInteger) * 8);
  r |= dpf_internal::ConvertBytesTo<BaseInteger>("ing#");  // # instead of .
  EXPECT_NE(samples[4].value(), r % kModulus);
}

}  // namespace
}  // namespace distributed_point_functions
