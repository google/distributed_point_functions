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

#include "pir/hashing/hash_family_config.h"

#include "dpf/internal/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "pir/hashing/hash_family.h"
#include "pir/hashing/hash_family_config.pb.h"
#include "pir/hashing/sha256_hash_family.h"

namespace distributed_point_functions {
namespace {

using dpf_internal::StatusIs;
using ::testing::HasSubstr;

inline constexpr char kHashFamilySeed[] = "kHashFamilySeed";

TEST(CreateHashFamilyFromConfig, FailsOnUnspecifiedHashFunction) {
  HashFamilyConfig config;
  config.set_hash_family(HashFamilyConfig::HASH_FAMILY_UNSPECIFIED);
  config.set_seed(kHashFamilySeed);

  EXPECT_THAT(
      CreateHashFamilyFromConfig(config),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("unspecified")));
}

TEST(CreateHashFamilyFromConfig, FailsOnUnknownHashFunction) {
  HashFamilyConfig config;
  config.set_hash_family(static_cast<HashFamilyConfig::HashFamily>(0xBAD));
  config.set_seed(kHashFamilySeed);

  EXPECT_THAT(
      CreateHashFamilyFromConfig(config),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Unknown")));
}

TEST(CreateHashFamilyFromConfig, FailsOnEmptySeed) {
  HashFamilyConfig config;
  config.set_hash_family(HashFamilyConfig::HASH_FAMILY_SHA256);
  config.set_seed("");

  EXPECT_THAT(CreateHashFamilyFromConfig(config),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("empty")));
}

TEST(CreateHashFamilyFromConfig, ReturnsSha256HashFunction) {
  HashFamilyConfig config;
  config.set_hash_family(HashFamilyConfig::HASH_FAMILY_SHA256);
  config.set_seed(kHashFamilySeed);

  DPF_ASSERT_OK_AND_ASSIGN(auto hash_family,
                           CreateHashFamilyFromConfig(config));

  constexpr absl::string_view hash_function_seed = "hash_function_seed";
  HashFunction hash_function = hash_family(hash_function_seed);
  constexpr absl::string_view input = "input";
  constexpr int bound = 1 << 20;

  EXPECT_EQ(hash_function(input, bound),
            WrapWithSeed(SHA256HashFamily{},
                         kHashFamilySeed)(hash_function_seed)(input, bound));
}
}  // namespace
}  // namespace distributed_point_functions
