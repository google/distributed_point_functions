// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dpf/key_generation_protocol/key_generation_protocol.h"

#include "dpf/distributed_point_function.pb.h"
#include "dpf/internal/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace distributed_point_functions {
namespace {

using dpf_internal::IsOkAndHolds;
using dpf_internal::StatusIs;
using ::testing::HasSubstr;
using ::testing::NotNull;

class KeyGenerationProtocolTest : public testing::Test {
 protected:
  void SetUp() override {
    parameters_.resize(2);
    parameters_[0].set_log_domain_size(5);
    parameters_[0].mutable_value_type()->mutable_integer()->set_bitsize(64);
    parameters_[1].set_log_domain_size(10);
    parameters_[1].mutable_value_type()->mutable_integer()->set_bitsize(64);
  }
  std::vector<DpfParameters> parameters_;
};

TEST_F(KeyGenerationProtocolTest, CreateSucceeds) {
  constexpr int party = 0;

  EXPECT_THAT(KeyGenerationProtocol::Create(parameters_, party),
              IsOkAndHolds(NotNull()));
}

TEST_F(KeyGenerationProtocolTest, CreateFailsIfPartyIsNot0Or1) {
  constexpr int party = 2;

  EXPECT_THAT(KeyGenerationProtocol::Create(parameters_, party),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("party")));
}
}  // namespace
}  // namespace distributed_point_functions