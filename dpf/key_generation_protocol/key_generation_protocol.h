/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DISTRIBUTED_POINT_FUNCTIONS_DPF_KEY_GENERATION_PROTOCOL_KEY_GENERATION_PROTOCOL_H_
#define DISTRIBUTED_POINT_FUNCTIONS_DPF_KEY_GENERATION_PROTOCOL_KEY_GENERATION_PROTOCOL_H_

#include <memory>

#include "absl/numeric/int128.h"
#include "absl/status/statusor.h"
#include "dpf/distributed_point_function.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/key_generation_protocol/key_generation_protocol.pb.h"
#include "dcf/fss_gates/prng/basic_rng.h"

namespace distributed_point_functions {

// A two-party protocol for generating a DPF key.
// For each level of the DPF evaluation tree, the following messages are
// exchanged between the parties. We refer to the corresponding lines in
// Algorithm 8 of https://eprint.iacr.org/2022/866.pdf.
//
// 1. Perform two parallel OTs to obtain shares of s_{CW} (Step 5)
// 2. Exchange shares of s_{CW}, t^L_{CW}, and t^R_{CW} (Step 5)
// 3. Perform two parallel OTs to obtain shares of W_{CW} (Step 11)
// 4. Exchange shares of W_{CW}.
//
// These steps correspond to the following functions in this class:
//
// 1a. ComputeSeedCorrectionOtReceiverMessage
// 1b. ComputeSeedCorrectionOtSenderMessage
// 2. ComputeSeedCorrectionShare
// 3a. ComputeValueCorrectionOtReceiverMessage
// 3b. ComputeValueCorrectionOtSenderMessage
// 4. ComputeValueCorrectionShare
//
// Each of these methods takes the other party's message from the previous
// round, as well as a ProtocolState message containing the party's local state.
// It updates the state and returns the computed message or a Status indicating
// any errors.
//
// NOTE: We may want to compute the value correction first, as done in
// DistributedPointFunction::GenerateIncremental.

struct BitBeaverTriple {
    bool mask;
    bool a;
    bool b;
    bool c;
};

// Mux involves 2 parallel OTs (let's call OT_A and OT_B)
// Each party acts as OT sender in one OT
// and OT receiver in the other OT.

struct MuxCorrelation{

    // One OT
    absl::uint128 rot_sender_first_string, rot_sender_second_string;

    // Other OT
    bool rot_receiver_choice_bit;
    absl::uint128 rot_receiver_string;
};

struct IdpfLevelCorrelation{
    MuxCorrelation mux_1, mux_2;
    BitBeaverTriple bit_triple;
};


struct KeyGenerationPreprocessing{
    // i^th element of this vector will contain the correlation needed to
    // perform i^th level Doerner Shelat.
    std::vector<IdpfLevelCorrelation> level_corr;
};

struct ProtocolState{


    // Round 2 state

        // Uncorrected seeds at the next level (left to right)
        // - twice the length of seeds
        std::vector<absl::uint128> uncorrected_seeds;

        // Uncorrected control bits at the next level (left to right)
        // - twice the length of shares_of_control_bits
        std::vector<bool> shares_of_uncorrected_control_bits;

        // Mux 1 randomness mask
        absl::uint128 mux_1_randomness;


        // Cumulative left seed, right seed, left control bit,
        // and right control bit [Obtained in Step 4]
        absl::uint128 seed_left_cumulative, seed_right_cumulative;
        bool control_left_cumulative, control_right_cumulative;

    // Round 3 state

    absl::uint128 mux_1_output;

    bool control_left_correction, control_right_correction;


    // Round 4 state

    bool masked_tau_zero;

    bool tau_zero, tau_one;

    Value cumulative_word;


    // Round 5 state

    bool share_of_t_star;


    // Round 6 state

        // Mux 2 randomness mask
        absl::uint128 mux_2_randomness;



    // global  state variables

    // DPF key
    DpfKey key;

    absl::uint128 alpha_shares;
    std::vector<Value> beta_shares;
    KeyGenerationPreprocessing keygen_preproc;

    uint64_t tree_level;
    // Add more local state variables here.

    // Seeds at the current level (left to right)
    std::vector<absl::uint128> seeds;

    // Control bits at the current level (left to right)
    std::vector<bool> shares_of_control_bits;
};


class KeyGenerationProtocol {
 public:

    uint64_t levels;

  // Creates a new instance of the key generation protocol for a DPF with the
  // given parameters. Party must be 0 or 1.
  static absl::StatusOr<std::unique_ptr<KeyGenerationProtocol>> Create(
      absl::Span<const DpfParameters> parameters);

    // Performs precomputation stage of Key Generation protocol and returns a pair of
    // KeyGenerationPreprocessing - one for each party.
    absl::StatusOr<std::pair<KeyGenerationPreprocessing, KeyGenerationPreprocessing>>
    PerformKeyGenerationPrecomputation();

  // Create ProtocolState given shares of alphas and betas.
  absl::StatusOr<ProtocolState> Initialize(int partyid,
      const absl::uint128 alpha_shares,
      const std::vector<Value> beta_shares,
      KeyGenerationPreprocessing keygen_preproc);

  // Receiver OT message for the MUX in Step 5. Just takes the state as input.
  absl::StatusOr<SeedCorrectionOtReceiverMessage>
  ComputeSeedCorrectionOtReceiverMessage(int partyid, ProtocolState& state) const;

  // Computes the sender OT message given the receiver message and the state.
  absl::StatusOr<SeedCorrectionOtSenderMessage>
  ComputeSeedCorrectionOtSenderMessage(int partyid,
                                       const SeedCorrectionOtReceiverMessage& seed_ot_receiver_message,
      ProtocolState& state) const;

  // Computes the share of the seed correction word given the sender OT message
  // and the state.
  absl::StatusOr<SeedCorrectionShare> ComputeSeedCorrectionOpening(int partyid,
                                                                   const SeedCorrectionOtSenderMessage& seed_ot_sender_message,
      ProtocolState& state) const;

  // Updates the state with the other party's seed correction share
  // and generate tau mult msg
  absl::StatusOr<MaskedTau> ApplySeedCorrectionShare(int partyid,
                                                     const SeedCorrectionShare& seed_correction_share,
      ProtocolState& state) const;


  // Computes the OT receiver message for the MUX gate in Step 11 given the
  // state.
  absl::StatusOr<ValueCorrectionOtReceiverMessage>
  ComputeValueCorrectionOtReceiverMessage(int partyid,
                                          const MaskedTau& masked_tau,
          ProtocolState& state) const;

  // Computes the OT sender message in Step 11 given the receiver message and
  // the state.
  absl::StatusOr<ValueCorrectionOtSenderMessage>
  ComputeValueCorrectionOtSenderMessage(int partyid,
                                        const ValueCorrectionOtReceiverMessage& value_ot_receiver_message,
      ProtocolState& state) const;

  // Computes the value correction share given the OT sender message and the
  // state.
  absl::StatusOr<ValueCorrectionShare> ComputeValueCorrectionOtShare(int partyid,
                                                                     const ValueCorrectionOtSenderMessage& value_ot_sender_message,
      ProtocolState& state) const;

  // Updates the state with the other party's value correction share.
  absl::StatusOr<int> ApplyValueCorrectionShare(int partyid,
                                         const ValueCorrectionShare& value_correction_share,
      ProtocolState& state) const;

  // Finalizes the protocol after all tree levels have been computed and returns
  // the generated DpfKey.
  absl::StatusOr<DpfKey> Finalize(int partyid, ProtocolState& state) const;

    template<typename T>
    Value ValueZero() const{
        T zero = 0;
        Value value = ToValue<T>(zero);
        return value;
    }

    template<typename T>
    absl::StatusOr<Value> ValueAdd(const Value& value1, const Value& value2) const{
        DPF_ASSIGN_OR_RETURN(T v1, FromValue<T>(value1));
        DPF_ASSIGN_OR_RETURN(T v2, FromValue<T>(value2));
        T v3 = v1 + v2;
        Value value3 = ToValue<T>(v3);
        return value3;
    }

    template<typename T>
    absl::StatusOr<Value> ValueNegate(const Value& value) const{
        DPF_ASSIGN_OR_RETURN(T v, FromValue<T>(value));
        T v_neg = -v;
        Value value_neg = ToValue<T>(v_neg);
        return value_neg;
    }

    template<typename T>
    absl::StatusOr<Value> ValueSub(const Value& value1, const Value& value2) const{
        DPF_ASSIGN_OR_RETURN(T v1, FromValue<T>(value1));
        DPF_ASSIGN_OR_RETURN(T v2, FromValue<T>(value2));
        T v3 = v1 - v2;
        Value value3 = ToValue<T>(v3);
        return value3;
    }

    // Expands seed s into a new seed and Value
//    template<typename T>
//    absl::StatusOr<std::pair<absl::uint128, Value>> Convert(const absl::uint128 s) const{
//
//        std::vector<absl::uint128> in_seed, out_seed, out_value;
//        in_seed.push_back(s);
//
//        out_seed.resize(1);
//        out_value.resize(1);
//
//        DPF_RETURN_IF_ERROR(
//                dpf_->prg_left_.Evaluate(in_seed,
//                                         absl::MakeSpan(out_seed)));
//
//
//        DPF_RETURN_IF_ERROR(
//                dpf_->prg_value_.Evaluate(in_seed,
//                                         absl::MakeSpan(out_value)));
//
//        // Temporary hack for converting absl::uint128 into
//        // required integer type (e.g. uint64_t)
//        T out_value_temp = static_cast<T>(out_value[0]);
//
//        Value value = ToValue<T>(out_value_temp);
//
//        return std::make_pair(out_seed[0], value);
//    }


    // Helper method for converting randomness into Value type
    template<typename T>
    absl::StatusOr<Value> ConvertRandToVal(const absl::uint128 s) const{

        std::vector<absl::uint128> in_seed, out_value;
        in_seed.push_back(s);

        out_value.resize(1);

        DPF_RETURN_IF_ERROR(
                dpf_->prg_value_.Evaluate(in_seed,
                                          absl::MakeSpan(out_value)));

        // Temporary hack for converting absl::uint128 into
        // required integer type (e.g. uint64_t)
        T out_value_temp = static_cast<T>(out_value[0]);

        Value value = ToValue<T>(out_value_temp);

        return value;
    }



private:
  explicit KeyGenerationProtocol(std::unique_ptr<DistributedPointFunction> dpf);

  std::unique_ptr<DistributedPointFunction> dpf_;

  // Number of leaves = 2 ^ levels.


  absl::StatusOr<std::pair<MuxCorrelation, MuxCorrelation>> genMuxCorrelation(){

        MuxCorrelation mux_corr_party0, mux_corr_party1;

        const absl::string_view kSampleSeed = absl::string_view();
        DPF_ASSIGN_OR_RETURN(
                auto rng, distributed_point_functions::BasicRng::Create(kSampleSeed));

        absl::uint128 r_0, r_1, b, r_b;
        DPF_ASSIGN_OR_RETURN(r_0, rng->Rand128());
        DPF_ASSIGN_OR_RETURN(r_1, rng->Rand128());
        DPF_ASSIGN_OR_RETURN(b, rng->Rand128());
        b = b & 1;
        if (b == 0) r_b = r_0;
        else r_b = r_1;

        absl::uint128 s_0, s_1, c, s_c;
        DPF_ASSIGN_OR_RETURN(s_0, rng->Rand128());
        DPF_ASSIGN_OR_RETURN(s_1, rng->Rand128());
        DPF_ASSIGN_OR_RETURN(c, rng->Rand128());
        c = c & 1;
        if (c == 0) s_c = s_0;
        else s_c = s_1;

        mux_corr_party0.rot_sender_first_string = r_0;
        mux_corr_party0.rot_sender_second_string = r_1;
        mux_corr_party0.rot_receiver_choice_bit = (c ? 1 : 0);
        mux_corr_party0.rot_receiver_string = s_c;

        mux_corr_party1.rot_sender_first_string = s_0;
        mux_corr_party1.rot_sender_second_string = s_1;
        mux_corr_party1.rot_receiver_choice_bit = (b ? 1 : 0);
        mux_corr_party1.rot_receiver_string = r_b;


        return std::make_pair(mux_corr_party0, mux_corr_party1);

    }

};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DPF_KEY_GENERATION_PROTOCOL_KEY_GENERATION_PROTOCOL_H_