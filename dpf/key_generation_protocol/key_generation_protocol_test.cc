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

    levels = 20;
    // There will be 2^levels number of leaves in the DPF tree

    parameters_.resize(levels);

    for (int i = 0; i < levels; i++){
        parameters_[i].set_log_domain_size(i + 1);
        parameters_[i].mutable_value_type()->mutable_integer()->set_bitsize(64);
    }
//    parameters_[0].set_log_domain_size(5);
//    parameters_[0].mutable_value_type()->mutable_integer()->set_bitsize(64);
//    parameters_[1].set_log_domain_size(10);
//    parameters_[1].mutable_value_type()->mutable_integer()->set_bitsize(64);
  }
  std::vector<DpfParameters> parameters_;
  int levels;
  using T = uint64_t;

  absl::uint128 BlockToUint128(Block x){
      absl::uint128 y = absl::MakeUint128(x.high(),x.low());
      return y;
  }
  void displayDpfKey(DpfKey key, int levels){
    std::cout << "Party : " << key.party() << std::endl;
    std::cout << "Root seed : " << BlockToUint128(key.seed()) << std::endl;

    for(int i = 0; i < levels; i++){
        std::cout << "Level : " << i << std::endl;
        std::cout << "seed correction : " << BlockToUint128(key.correction_words(i).seed()) << std::endl;
        std::cout << "left control correction : " << key.correction_words(i).control_left() << std::endl;
        std::cout << "right control correction : " << key.correction_words(i).control_right() << std::endl;
        T v = *(FromValue<T>(key.correction_words(i).value_correction(0)));
        std::cout << "Value correction : " << v << std::endl;
    }
  }
};


//
//TEST_F(KeyGenerationProtocolTest, CreateSucceeds) {
//
//  EXPECT_THAT(KeyGenerationProtocol::Create(parameters_),
//              IsOkAndHolds(NotNull()));
//}

//TEST_F(KeyGenerationProtocolTest, CreateFailsIfPartyIsNot0Or1) {
//  constexpr int party = 2;
//
//  EXPECT_THAT(KeyGenerationProtocol::Create(parameters_, party),
//              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("party")));
//}

TEST_F(KeyGenerationProtocolTest, EndToEndSucceeds) {

    std::unique_ptr<KeyGenerationProtocol> keygen;

    DPF_ASSERT_OK_AND_ASSIGN(keygen,
            KeyGenerationProtocol::Create(parameters_));

    std::pair<KeyGenerationPreprocessing, KeyGenerationPreprocessing> preproc;

    DPF_ASSERT_OK_AND_ASSIGN(preproc,
                             keygen->PerformKeyGenerationPrecomputation());

    absl::uint128 alpha = 23;


    // Generating shares of alpha for Party 0 and Party 1

    absl::uint128 alpha_share_party0, alpha_share_party1;

    const absl::string_view kSampleSeed = absl::string_view();
    DPF_ASSERT_OK_AND_ASSIGN(
            auto rng, distributed_point_functions::BasicRng::Create(kSampleSeed));

    DPF_ASSERT_OK_AND_ASSIGN(alpha_share_party0, rng->Rand128());

    alpha_share_party1 = alpha ^ alpha_share_party0;

    // Generating shares of beta for Party 0 and Party 1
    std::vector<Value> beta;

    for(int i = 0; i < levels; i++){
        Value beta_i;
//        beta_i.mutable_tuple()->add_elements()->mutable_integer()->set_value_uint64(42);
        beta_i.mutable_integer()->set_value_uint64(42);
        beta.push_back(beta_i);
    }

    std::vector<Value> beta_shares_party0, beta_shares_party1;

    for (int i = 0; i < beta.size(); i++){
        DPF_ASSERT_OK_AND_ASSIGN(absl::uint128 beta_share_party0_seed, rng->Rand128());

        Value value0 = ToValue<T>(static_cast<T>(beta_share_party0_seed));

        DPF_ASSERT_OK_AND_ASSIGN(Value value1,
                                 keygen->ValueSub<T>(beta[i], value0));

        beta_shares_party0.push_back(value0);

        beta_shares_party1.push_back(value1);

        // TODO: Look at ValueCorrection implementation in iDPF
//        keygen->dpf_->ValueCorrectionFunction func;
//
//        DPF_ASSERT_OK_AND_ASSIGN(
//                ValueCorrectionFunction func,
//        GetValueCorrectionFunction(parameters_[hierarchy_level]));
    }

        // Running KeyGen Initialization

        ProtocolState state_party0, state_party1;

        DPF_ASSERT_OK_AND_ASSIGN(state_party0,
                keygen->Initialize(0,
                                   alpha_share_party0,
                                   beta_shares_party0,
                                   preproc.first));

        DPF_ASSERT_OK_AND_ASSIGN(state_party1,
                keygen->Initialize(1,
                                   alpha_share_party1,
                                   beta_shares_party1,
                                   preproc.second));

        // Running KeyGen 2PC offline phase for each level

        for(int i = 0; i < levels; i++) {

            SeedCorrectionOtReceiverMessage round1_party0, round1_party1;

            DPF_ASSERT_OK_AND_ASSIGN(round1_party0,
                    keygen->ComputeSeedCorrectionOtReceiverMessage(
                            0,
                            state_party0));

            DPF_ASSERT_OK_AND_ASSIGN(round1_party1,
                    keygen->ComputeSeedCorrectionOtReceiverMessage(
                            1,
                            state_party1));

            SeedCorrectionOtSenderMessage round2_party0, round2_party1;

            DPF_ASSERT_OK_AND_ASSIGN(round2_party0,
                    keygen->ComputeSeedCorrectionOtSenderMessage(
                            0,
                            round1_party1,
                            state_party0));

            DPF_ASSERT_OK_AND_ASSIGN(round2_party1,
                    keygen->ComputeSeedCorrectionOtSenderMessage(
                            1,
                            round1_party0,
                            state_party1));

            SeedCorrectionShare round3_party0, round3_party1;

            DPF_ASSERT_OK_AND_ASSIGN(round3_party0,
                    keygen->ComputeSeedCorrectionOpening(
                            0,
                            round2_party1,
                            state_party0));


            DPF_ASSERT_OK_AND_ASSIGN(round3_party1,
                    keygen->ComputeSeedCorrectionOpening(
                            1,
                            round2_party0,
                            state_party1));

            MaskedTau round4_party0, round4_party1;

            DPF_ASSERT_OK_AND_ASSIGN(round4_party0,
                    keygen->ApplySeedCorrectionShare(
                            0,
                            round3_party1,
                            state_party0));

            DPF_ASSERT_OK_AND_ASSIGN(round4_party1,
                    keygen->ApplySeedCorrectionShare(
                            1,
                            round3_party0,
                            state_party1));

            ValueCorrectionOtReceiverMessage round5_party0, round5_party1;

            DPF_ASSERT_OK_AND_ASSIGN(round5_party0,
                    keygen->ComputeValueCorrectionOtReceiverMessage(
                            0,
                            round4_party1,
                            state_party0));

            DPF_ASSERT_OK_AND_ASSIGN(round5_party1,
                    keygen->ComputeValueCorrectionOtReceiverMessage(
                            1,
                            round4_party0,
                            state_party1));

            ValueCorrectionOtSenderMessage round6_party0, round6_party1;

            DPF_ASSERT_OK_AND_ASSIGN(round6_party0,
                    keygen->ComputeValueCorrectionOtSenderMessage(
                            0,
                            round5_party1,
                            state_party0));

            DPF_ASSERT_OK_AND_ASSIGN(round6_party1,
                    keygen->ComputeValueCorrectionOtSenderMessage(
                            0,
                            round5_party0,
                            state_party1));


            ValueCorrectionShare round7_party0, round7_party1;

            DPF_ASSERT_OK_AND_ASSIGN(round7_party0,
                    keygen->ComputeValueCorrectionOtShare(
                            0,
                            round6_party1,
                            state_party0));

            DPF_ASSERT_OK_AND_ASSIGN(round7_party1,
                    keygen->ComputeValueCorrectionOtShare(
                            1,
                            round6_party0,
                            state_party1));

            int x;

            DPF_ASSERT_OK_AND_ASSIGN(x,
            keygen->ApplyValueCorrectionShare(
                            0,
                            round7_party1,
                            state_party0));

            DPF_ASSERT_OK_AND_ASSIGN(x,
                                     keygen->ApplyValueCorrectionShare(
                                        1,
                                        round7_party0,
                                        state_party1));
            }

        std::cout << "\n\n";

        displayDpfKey(state_party0.key, levels);

        std::cout << "\n\n";

        displayDpfKey(state_party1.key, levels);


    }
}  // namespace
}  // namespace distributed_point_functions