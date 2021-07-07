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

#include "dpf/tuple.h"

#include <gtest/gtest.h>

#include "absl/numeric/int128.h"

namespace distributed_point_functions {

namespace {

using T = Tuple<int, double, absl::uint128>;

TEST(TupleTest, TestAddition) {
  T a = std::make_tuple(1, 2, 3);
  T b = std::make_tuple(4, 5, 6);

  T c = a + b;

  EXPECT_EQ(std::get<0>(c), std::get<0>(a) + std::get<0>(b));
  EXPECT_EQ(std::get<1>(c), std::get<1>(a) + std::get<1>(b));
  EXPECT_EQ(std::get<2>(c), std::get<2>(a) + std::get<2>(b));
}

TEST(TupleTest, TestAdditionInplace) {
  T a = std::make_tuple(1, 2, 3);
  T b = std::make_tuple(4, 5, 6);

  T a2 = a;
  a += b;

  EXPECT_EQ(std::get<0>(a), std::get<0>(a2) + std::get<0>(b));
  EXPECT_EQ(std::get<1>(a), std::get<1>(a2) + std::get<1>(b));
  EXPECT_EQ(std::get<2>(a), std::get<2>(a2) + std::get<2>(b));
}
TEST(TupleTest, TestSubtraction) {
  T a = std::make_tuple(1, 2, 3);
  T b = std::make_tuple(4, 5, 6);

  T c = a - b;

  EXPECT_EQ(std::get<0>(c), std::get<0>(a) - std::get<0>(b));
  EXPECT_EQ(std::get<1>(c), std::get<1>(a) - std::get<1>(b));
  EXPECT_EQ(std::get<2>(c), std::get<2>(a) - std::get<2>(b));
}

TEST(TupleTest, TestSubtractionInplace) {
  T a = std::make_tuple(1, 2, 3);
  T b = std::make_tuple(4, 5, 6);

  T a2 = a;
  a -= b;

  EXPECT_EQ(std::get<0>(a), std::get<0>(a2) - std::get<0>(b));
  EXPECT_EQ(std::get<1>(a), std::get<1>(a2) - std::get<1>(b));
  EXPECT_EQ(std::get<2>(a), std::get<2>(a2) - std::get<2>(b));
}

TEST(TupleTest, TestNegation) {
  T a = std::make_tuple(1, 2, 3);

  T a2 = -a;

  EXPECT_EQ(std::get<0>(a2), -std::get<0>(a));
  EXPECT_EQ(std::get<1>(a2), -std::get<1>(a));
  EXPECT_EQ(std::get<2>(a2), -std::get<2>(a));
}

}  // namespace

}  // namespace distributed_point_functions
