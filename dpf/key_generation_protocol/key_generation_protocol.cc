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

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "dpf/distributed_point_function.h"
#include "dpf/status_macros.h"
#include "dcf/fss_gates/prng/basic_rng.h"
#include "dpf/internal/evaluate_prg_hwy.h"
#include "dpf/internal/get_hwy_mode.h"
#include "dpf/internal/proto_validator.h"
#include "dpf/internal/value_type_helpers.h"
#include "dpf/status_macros.h"


namespace distributed_point_functions {

KeyGenerationProtocol::KeyGenerationProtocol(
    std::unique_ptr<DistributedPointFunction> dpf)
    : dpf_(std::move(dpf)) {}

absl::StatusOr<std::unique_ptr<KeyGenerationProtocol>>
KeyGenerationProtocol::Create(absl::Span<const DpfParameters> parameters) {
//  if (party != 0 && party != 1) {
//    return absl::InvalidArgumentError("`party` must be 0 or 1");
//  }
  DPF_ASSIGN_OR_RETURN(auto dpf,
                       DistributedPointFunction::CreateIncremental(parameters));


//  uint64_t levels = parameters.back().log_domain_size();
//  uint64_t levels = 63;

  return absl::WrapUnique(new KeyGenerationProtocol(std::move(dpf)));
}

    absl::StatusOr<std::pair<KeyGenerationPreprocessing, KeyGenerationPreprocessing>>
    KeyGenerationProtocol::PerformKeyGenerationPrecomputation(){

        KeyGenerationPreprocessing preproc_party0, preproc_party1;


        int n = dpf_->parameters_.size();

        for(int i = 0; i < n; i++){
            IdpfLevelCorrelation ipdfcorr_party0, ipdfcorr_party1;

            // Generating correlations for first mux
            std::pair<MuxCorrelation, MuxCorrelation> mux_1;

            DPF_ASSIGN_OR_RETURN(mux_1,
                                 KeyGenerationProtocol::genMuxCorrelation());

            ipdfcorr_party0.mux_1 = mux_1.first;
            ipdfcorr_party1.mux_1 = mux_1.second;


            // Generating correlations for second mux
            std::pair<MuxCorrelation, MuxCorrelation> mux_2;

            DPF_ASSIGN_OR_RETURN(mux_2,
                                 KeyGenerationProtocol::genMuxCorrelation());

            ipdfcorr_party0.mux_2 = mux_2.first;
            ipdfcorr_party1.mux_2 = mux_2.second;

            // Generating bit beaver triples

            const absl::string_view kSampleSeed = absl::string_view();
            DPF_ASSIGN_OR_RETURN(
                    auto rng, distributed_point_functions::BasicRng::Create(kSampleSeed));

            absl::uint128 a_temp, b_temp, a_0_temp, b_0_temp, c_0_temp;
            bool a, b, c, a_0, b_0, c_0, a_1, b_1, c_1;
            DPF_ASSIGN_OR_RETURN(a_temp, rng->Rand128());
            DPF_ASSIGN_OR_RETURN(b_temp, rng->Rand128());
            DPF_ASSIGN_OR_RETURN(a_0_temp, rng->Rand128());
            DPF_ASSIGN_OR_RETURN(b_0_temp, rng->Rand128());
            DPF_ASSIGN_OR_RETURN(c_0_temp, rng->Rand128());
            a = (a_temp & 1) ? 1 : 0;
            b = (b_temp & 1) ? 1 : 0;
            c = a & b;
            a_0 = (a_0_temp & 1) ? 1 : 0;
            b_0 = (b_0_temp & 1) ? 1 : 0;
            c_0 = (c_0_temp & 1) ? 1 : 0;
            a_1 = a ^ a_0;
            b_1 = b ^ b_0;
            c_1 = c ^ c_0;

            ipdfcorr_party0.bit_triple = {a, a_0, b_0, c_0};
            ipdfcorr_party1.bit_triple = {b, a_1, b_1, c_1};

            // Populating the idpf correlation for this level

            preproc_party0.level_corr.push_back(ipdfcorr_party0);
            preproc_party1.level_corr.push_back(ipdfcorr_party1);
        }

        return std::make_pair(std::move(preproc_party0), std::move(preproc_party1));

//    return absl::UnimplementedError("");
    }

    absl::StatusOr<ProtocolState> KeyGenerationProtocol::Initialize(
            int partyid,
            const absl::uint128 alpha_shares,
            const std::vector<Value> beta_shares,
            KeyGenerationPreprocessing keygen_preproc){

        // We are assuming that number of parameters = number of levels
        levels = dpf_->parameters_.size();


        // Check validity of beta.
        if (beta_shares.size() != dpf_->parameters_.size()) {
            return absl::InvalidArgumentError(
                    "`beta` has to have the same size as `parameters` passed at "
                    "construction");
        }
        for (int i = 0; i < static_cast<int>(dpf_->parameters_.size()); ++i) {
            absl::Status status = dpf_->proto_validator_->ValidateValue(beta_shares[i], i);
            if (!status.ok()) {
                return status;
            }
        }

        // Sampling root seed

        std::vector<absl::uint128> seeds;

        const absl::string_view kSampleSeed = absl::string_view();
        DPF_ASSIGN_OR_RETURN(
                auto rng, distributed_point_functions::BasicRng::Create(kSampleSeed));

        absl::uint128 seed;
        DPF_ASSIGN_OR_RETURN(seed, rng->Rand128());

        seeds.push_back(seed);


        // Sampling root control bit share

        std::vector<bool> shares_of_control_bits;

        shares_of_control_bits.push_back(partyid);


        // Populating the root seed in DPF key
        DpfKey key;

//        RAND_bytes(reinterpret_cast<uint8_t*>(&seeds), sizeof(absl::uint128));
        key.mutable_seed()->set_high(absl::Uint128High64(seed));
        key.mutable_seed()->set_low(absl::Uint128Low64(seed));

        ProtocolState state;

        state.tree_level = 0;
        state.seeds = seeds;
        state.shares_of_control_bits = shares_of_control_bits;
        state.key = key;
        state.alpha_shares = alpha_shares;
        state.beta_shares = beta_shares;
        state.keygen_preproc = keygen_preproc;

        return state;

//         return absl::UnimplementedError("");
    }


    absl::StatusOr<SeedCorrectionOtReceiverMessage>
    KeyGenerationProtocol::ComputeSeedCorrectionOtReceiverMessage(
            int partyid,
            ProtocolState& state) const{

        // Prepare OT receiver message using state.alpha_shares
        // and mux_1 correlation

        SeedCorrectionOtReceiverMessage msg_ot_recv;

        bool alpha_level_share = state.alpha_shares & (1 << (levels - state.tree_level)) ? 1 : 0;
        bool rot_masked_alpha_level_share = alpha_level_share ^
                state.keygen_preproc.level_corr[state.tree_level].mux_1.rot_receiver_choice_bit;
//        msg_ot_recv.set_choice_bit_mask(alpha_level_share ^ state.ke);


        return msg_ot_recv;
    }

    absl::StatusOr<SeedCorrectionOtSenderMessage>
    KeyGenerationProtocol::ComputeSeedCorrectionOtSenderMessage(int partyid,
                                         const SeedCorrectionOtReceiverMessage& seed_ot_receiver_message,
                                         ProtocolState& state) const{

        absl::uint128 seed_left_cumulative_xor = 0, seed_right_cumulative_xor = 0;
        bool control_left_cumulative_xor = 0, control_right_cumulative_xor = 0;

        std::vector<absl::uint128> expanded_seeds_left;
        expanded_seeds_left.resize(state.seeds.size());

        std::vector<absl::uint128> expanded_seeds_right;
        expanded_seeds_right.resize(state.seeds.size());

        // Line 3: Expanding all the left children at next level.
        DPF_RETURN_IF_ERROR(
                dpf_->prg_left_.Evaluate(state.seeds,
                                         absl::MakeSpan(expanded_seeds_left)));

        // Line 3: Expanding all the right children at next level.
        DPF_RETURN_IF_ERROR(
                dpf_->prg_right_.Evaluate(state.seeds,
                                          absl::MakeSpan(expanded_seeds_right)));

        state.uncorrected_seeds.clear();
        state.shares_of_uncorrected_control_bits.clear();

        // Line 4 : Cumulative XOR of all the left seeds, left control bits,
        // right seeds, right control bits at the next level.
        for(int i = 0; i < state.seeds.size(); i++){
            bool control_left = dpf_internal::ExtractAndClearLowestBit(
                    expanded_seeds_left[i]);
            bool control_right = dpf_internal::ExtractAndClearLowestBit(
                    expanded_seeds_right[i]);
            seed_left_cumulative_xor ^= expanded_seeds_left[i];
            seed_right_cumulative_xor ^= expanded_seeds_right[i];
            control_left_cumulative_xor ^= control_left;
            control_right_cumulative_xor ^= control_right;

            // Populating new (uncorrected) seeds.
            state.uncorrected_seeds.push_back(expanded_seeds_left[i]);
            state.uncorrected_seeds.push_back(expanded_seeds_right[i]);
            state.shares_of_uncorrected_control_bits.push_back(control_left);
            state.shares_of_uncorrected_control_bits.push_back(control_right);
        }

        // Storing cumulative seeds and control bits in the state.
        state.seed_left_cumulative = seed_left_cumulative_xor;
        state.seed_right_cumulative = seed_right_cumulative_xor;
        state.control_left_cumulative = control_left_cumulative_xor;
        state.control_right_cumulative = control_right_cumulative_xor;


        // Sampling randomness for Mux 1 mask
        const absl::string_view kSampleSeed = absl::string_view();
        DPF_ASSIGN_OR_RETURN(
                auto rng, distributed_point_functions::BasicRng::Create(kSampleSeed));

        DPF_ASSIGN_OR_RETURN(absl::uint128 r, rng->Rand128());

        state.mux_1_randomness = r;
        // Preparing OT sender messages using r, seed_left_cumulative_xor,
        // seed_right_cumulative_xor and mux_1 correlation

        SeedCorrectionOtSenderMessage mux_sender_msg;

        absl::uint128 mux_sender_first_msg_without_rot_mask,
        mux_sender_second_msg_without_rot_mask;

        // We want to perform MUX2((s^R, s^L), \alpha_l) meaning that the output
        // should be s^R be \alpha_L = 0 and s^L otherwise.

        // Preparing the unmasked OT sender messages.
        bool alpha_level_share =
                state.alpha_shares & (1 << (levels - state.tree_level)) ? 1 : 0;

        if(alpha_level_share == false){
            mux_sender_first_msg_without_rot_mask = r ^ seed_right_cumulative_xor;
            mux_sender_second_msg_without_rot_mask = r ^ seed_left_cumulative_xor;
        }
        else{
            mux_sender_first_msg_without_rot_mask = r ^ seed_left_cumulative_xor;
            mux_sender_second_msg_without_rot_mask = r ^ seed_right_cumulative_xor;
        }

        // Swap the ROT masks if seed_ot_receiver_message = 1
        absl::uint128 rot_sender_first_msg, rot_sender_second_msg;
        if(seed_ot_receiver_message.choice_bit_mask()){
            rot_sender_first_msg =
                    state.keygen_preproc.level_corr[state.tree_level].mux_1.rot_sender_second_string;
            rot_sender_second_msg =
                    state.keygen_preproc.level_corr[state.tree_level].mux_1.rot_sender_first_string;
        }
        else{
            rot_sender_first_msg =
                    state.keygen_preproc.level_corr[state.tree_level].mux_1.rot_sender_first_string;
            rot_sender_second_msg =
                    state.keygen_preproc.level_corr[state.tree_level].mux_1.rot_sender_second_string;
        }

        // Preparing the masked OT sender msgs.
        absl::uint128 mux_sender_first_msg_with_rot_mask,
                mux_sender_second_msg_with_rot_mask;

        mux_sender_first_msg_with_rot_mask = mux_sender_first_msg_without_rot_mask ^
                rot_sender_first_msg;

        mux_sender_second_msg_with_rot_mask = mux_sender_second_msg_without_rot_mask ^
                                             rot_sender_second_msg;

        mux_sender_msg.mutable_masked_message_one()->set_high(
                absl::Uint128High64(mux_sender_first_msg_with_rot_mask));

        mux_sender_msg.mutable_masked_message_one()->set_low(
                absl::Uint128Low64(mux_sender_first_msg_with_rot_mask));

        mux_sender_msg.mutable_masked_message_two()->set_high(
                absl::Uint128High64(mux_sender_second_msg_with_rot_mask));

        mux_sender_msg.mutable_masked_message_two()->set_low(
                absl::Uint128Low64(mux_sender_second_msg_with_rot_mask));

        return mux_sender_msg;
    }

    absl::StatusOr<SeedCorrectionShare>
            KeyGenerationProtocol::ComputeSeedCorrectionOpening(
                    int partyid,
                    const SeedCorrectionOtSenderMessage& seed_ot_sender_message,
                    ProtocolState& state) const{

        SeedCorrectionShare opening_msg;

        bool alpha_level_share =
                state.alpha_shares & (1 << (levels - state.tree_level)) ? 1 : 0;

        // Compute mux output

        absl::uint128 mux_output;

            // Retrieve the correct OT msg

            absl::uint128 ot_output;

            absl::uint128 sender_string_one, sender_string_two;

            sender_string_one = absl::MakeUint128(
                    seed_ot_sender_message.masked_message_one().high(),
                    seed_ot_sender_message.masked_message_one().low());

            sender_string_two = absl::MakeUint128(
                    seed_ot_sender_message.masked_message_two().high(),
                    seed_ot_sender_message.masked_message_two().low());


            if(alpha_level_share == false){
                ot_output = sender_string_one ^ state.keygen_preproc.
                        level_corr[state.tree_level].mux_1.rot_receiver_string;
            }
            else{
                ot_output = sender_string_two ^ state.keygen_preproc.
                        level_corr[state.tree_level].mux_1.rot_receiver_string;
            }

            // Add the randomness of mux to compute the output : Step 6
            // in cryptflow2 mux protocol

            mux_output = ot_output ^ state.mux_1_randomness;

        opening_msg.mutable_seed()->set_high(
                absl::Uint128High64(mux_output));

        opening_msg.mutable_seed()->set_low(
                absl::Uint128Low64(mux_output));

        state.mux_1_output = mux_output;

        // TODO : Store mux_output in state

        // Computing shares of left and right contol bit correction.
        bool control_left_correction, control_right_correction;


        control_left_correction = state.control_left_cumulative ^ alpha_level_share ^ partyid;

        control_right_correction = state.control_right_cumulative ^ alpha_level_share;

        state.control_left_correction = control_left_correction;
        state.control_right_correction = control_right_correction;

        opening_msg.set_control_bit_left(control_left_correction);
        opening_msg.set_control_bit_right(control_right_correction);


        return opening_msg;
    }

    absl::StatusOr<MaskedTau> KeyGenerationProtocol::ApplySeedCorrectionShare
    (int partyid,
     const SeedCorrectionShare& seed_correction_share,
     ProtocolState& state) const{

        absl::uint128 reconstructed_seed_correction;
        bool reconstructed_control_left_correction,
        reconstructed_control_right_correction;

        absl::uint128 seed_correction_other_party_share =
                absl::MakeUint128(
                        seed_correction_share.seed().high(),
                        seed_correction_share.seed().low());

        reconstructed_seed_correction =
                state.mux_1_output ^ seed_correction_other_party_share;

        reconstructed_control_left_correction =
                state.control_left_correction ^ seed_correction_share.control_bit_left();

        reconstructed_control_right_correction =
                state.control_right_correction ^ seed_correction_share.control_bit_right();

        // Adding reconstructed_seed_correction, reconstructed_control_left_correction
        // reconstructed_control_right_correction to the DPF key
//        CorrectionWord* correction_word = state.keys.add_correction_words();

//        state.key.


        // TODO: Implement steps 6 - 10

        uint64_t n = state.seeds.size();


            for(int i = 0; i < n; i++) {
                uint64_t left_index = i << 1;
                uint64_t right_index = left_index + 1;

                bool control_bit_parent = state.shares_of_control_bits[i];

                // Perform correction of left and right seeds and control bits - Step 6 and Step 7
                if (control_bit_parent) {
                    state.uncorrected_seeds[left_index] ^= reconstructed_seed_correction;
                    state.uncorrected_seeds[right_index] ^= reconstructed_seed_correction;
                    state.shares_of_uncorrected_control_bits[left_index] =
                            (state.shares_of_uncorrected_control_bits[left_index] ^ reconstructed_control_left_correction);
                    state.shares_of_uncorrected_control_bits[right_index] = (
                            state.shares_of_uncorrected_control_bits[right_index] ^ reconstructed_control_right_correction);
                }

            }



                // TODO : Remove the template hardcoding to uint64_t

                // Issue: this initialization will depend on the Value type
                Value cumulative_word = ValueZero<uint64_t>();

                absl::uint128 cumulative_control_sum = partyid;



                // TODO : Instead of calling Convert() one by one,
                // call it over the entire set of seeds at once

                std::vector<absl::uint128> seed_after_convert, value_seed_after_convert;

                seed_after_convert.resize(state.uncorrected_seeds.size());
                value_seed_after_convert.resize(state.uncorrected_seeds.size());

                DPF_RETURN_IF_ERROR(
                        dpf_->prg_left_.Evaluate(state.uncorrected_seeds,
                                                 absl::MakeSpan(seed_after_convert)));

                DPF_RETURN_IF_ERROR(
                        dpf_->prg_value_.Evaluate(state.uncorrected_seeds,
                                                  absl::MakeSpan(value_seed_after_convert)));

                state.uncorrected_seeds = seed_after_convert;


                for(int i = 0; i < value_seed_after_convert.size(); i++){

                    // Temporary hack for converting absl::uint128 into
                    // required integer type (e.g. uint64_t)
                    uint64_t out_value_temp = static_cast<uint64_t>(value_seed_after_convert[i]);

                    Value value = ToValue<uint64_t>(out_value_temp);

                    // Line 9 : Adding words
                    DPF_ASSIGN_OR_RETURN(cumulative_word,
                                         ValueAdd<uint64_t>(cumulative_word, value));

                    // Line 10: Adding control bits
                    if(partyid == 0){
                        cumulative_control_sum +=
                                (state.shares_of_uncorrected_control_bits[i]) ? 1 : 0;
                    }
                    else{
                        // TODO : Check the -1 operation because we are operating over uint
                        cumulative_control_sum +=
                                (state.shares_of_uncorrected_control_bits[i]) ? -1 : 0;
                    }

                }


            state.cumulative_word = cumulative_word;

            // Line 10 : Setting tau_zero and tau_one to be the LSB and second LSB respectively.
            state.tau_zero = cumulative_control_sum & (1) ? 1 : 0;
            state.tau_one = cumulative_control_sum & (1 << 1) ? 1 : 0;

            bool masked_tau_msg =
                    state.tau_zero ^ state.keygen_preproc.level_corr[state.tree_level].bit_triple.mask;

            state.masked_tau_zero = masked_tau_msg;

            MaskedTau round4msg;

            round4msg.set_masked_tau_zero(masked_tau_msg);

            return round4msg;
        }

    absl::StatusOr<ValueCorrectionOtReceiverMessage>
    KeyGenerationProtocol::ComputeValueCorrectionOtReceiverMessage(
            int partyid,
            const MaskedTau& masked_tau,
            ProtocolState& state) const{

        // Compute share of t* using masked tau msg

        bool masked_tau_zero_party_0, masked_tau_zero_party_1;

        if(partyid = 0){
            masked_tau_zero_party_0 = state.masked_tau_zero;
            masked_tau_zero_party_1 = masked_tau.masked_tau_zero();
        }
        else{
            masked_tau_zero_party_1 = state.masked_tau_zero;
            masked_tau_zero_party_0 = masked_tau.masked_tau_zero();
        }


        bool share_of_product;


        if(partyid == 0){
            share_of_product = (masked_tau_zero_party_0 & state.keygen_preproc.level_corr[state.tree_level].bit_triple.b)
            ^ (masked_tau_zero_party_1 & state.keygen_preproc.level_corr[state.tree_level].bit_triple.a)
            ^ state.keygen_preproc.level_corr[state.tree_level].bit_triple.c;
        }
        else{
            share_of_product = (masked_tau_zero_party_0 & masked_tau_zero_party_1)
                               ^ (masked_tau_zero_party_0 & state.keygen_preproc.level_corr[state.tree_level].bit_triple.b)
                               ^ (masked_tau_zero_party_1 & state.keygen_preproc.level_corr[state.tree_level].bit_triple.a)
                               ^ state.keygen_preproc.level_corr[state.tree_level].bit_triple.c;
        }



        bool share_of_t_star;

        if(partyid == 0){
            share_of_t_star = state.tau_one ^ share_of_product;
        }
        else{
            share_of_t_star = 1 ^ state.tau_one ^ share_of_product;
        }

        state.share_of_t_star = share_of_t_star;

        // Compute MUX 2 round 1 msg

        ValueCorrectionOtReceiverMessage round5msg;

        bool masked_choice_bit = state.share_of_t_star ^
                state.keygen_preproc.level_corr[state.tree_level].mux_2.rot_receiver_choice_bit;

        round5msg.set_choice_bit_mask(masked_choice_bit);

        return round5msg;
    }

    absl::StatusOr<ValueCorrectionOtSenderMessage>
    KeyGenerationProtocol::ComputeValueCorrectionOtSenderMessage(
            int partyid,
            const ValueCorrectionOtReceiverMessage& value_ot_receiver_message,
            ProtocolState& state) const{

        // Construct OT sender msg

        Value share_of_W0_CW, share_of_W1_CW;

        Value beta_share = state.beta_shares[state.tree_level];

        if(partyid == 0){
            DPF_ASSIGN_OR_RETURN(share_of_W0_CW,
                                 ValueSub<uint64_t>(
                                         beta_share,
                                         state.cumulative_word));
        }
        else{
            DPF_ASSIGN_OR_RETURN(share_of_W0_CW,
                                 ValueAdd<uint64_t>(
                                         beta_share,
                                         state.cumulative_word));
        }

        if(partyid == 0){
            DPF_ASSIGN_OR_RETURN(share_of_W1_CW,
                                 ValueSub<uint64_t>(
                                         state.cumulative_word,
                                         beta_share));
        }
        else{
            DPF_ASSIGN_OR_RETURN(Value share_of_W1_CW_temp,
                                 ValueAdd<uint64_t>(
                                         beta_share,
                                         state.cumulative_word));

            DPF_ASSIGN_OR_RETURN(Value share_of_W1_CW,
                    ValueNegate<uint64_t>(share_of_W1_CW_temp));
        }


        // Swap ROT mask depending on the receiver msg
        absl::uint128 rot_sender_mask_first, rot_sender_mask_second;

        if(value_ot_receiver_message.choice_bit_mask()){
            rot_sender_mask_first = state.keygen_preproc.level_corr[state.tree_level].mux_2.rot_sender_second_string;
            rot_sender_mask_second = state.keygen_preproc.level_corr[state.tree_level].mux_2.rot_sender_first_string;
        }
        else{
            rot_sender_mask_first = state.keygen_preproc.level_corr[state.tree_level].mux_2.rot_sender_first_string;
            rot_sender_mask_second = state.keygen_preproc.level_corr[state.tree_level].mux_2.rot_sender_second_string;
        }

        // Convert ROT masks into Value type
        Value rot_sender_mask_first_value, rot_sender_mask_second_value;

        DPF_ASSIGN_OR_RETURN(rot_sender_mask_first_value,
                ConvertRandToVal<uint64_t>(rot_sender_mask_first));

        DPF_ASSIGN_OR_RETURN(rot_sender_mask_second_value,
                             ConvertRandToVal<uint64_t>(rot_sender_mask_second));

        // Sample mux 2 randomness
        const absl::string_view kSampleSeed = absl::string_view();
        DPF_ASSIGN_OR_RETURN(
                auto rng, distributed_point_functions::BasicRng::Create(kSampleSeed));

        DPF_ASSIGN_OR_RETURN(state.mux_2_randomness, rng->Rand128());

        DPF_ASSIGN_OR_RETURN(Value random_value_mask,
                             ConvertRandToVal<uint64_t>(state.mux_2_randomness));
//
        // Generate mux sender msg depending on own share of t*

        Value masked_ot_1, masked_ot_2, masked_ot_1_tmp, masked_ot_2_tmp;

        if(state.share_of_t_star == false){
            DPF_ASSIGN_OR_RETURN(masked_ot_1_tmp,
                                 ValueAdd<uint64_t>(share_of_W0_CW,
                                                    rot_sender_mask_first_value));

            DPF_ASSIGN_OR_RETURN(masked_ot_2_tmp,
                    ValueAdd<uint64_t>(share_of_W1_CW,
                                       rot_sender_mask_second_value));
        }
        else{
            DPF_ASSIGN_OR_RETURN(masked_ot_1_tmp,
                    ValueAdd<uint64_t>(share_of_W1_CW,
                                       rot_sender_mask_first_value));

            DPF_ASSIGN_OR_RETURN(masked_ot_2_tmp,
                    ValueAdd<uint64_t>(share_of_W0_CW,
                                       rot_sender_mask_second_value));
        }

        DPF_ASSIGN_OR_RETURN(masked_ot_1,
                ValueSub<uint64_t>(masked_ot_1_tmp,
                                   random_value_mask));
//
        DPF_ASSIGN_OR_RETURN(masked_ot_2,
                ValueSub<uint64_t>(masked_ot_2_tmp,
                                   random_value_mask));

        ValueCorrectionOtSenderMessage round6msg;

        *(round6msg.mutable_masked_message_one()) = masked_ot_1;

        *(round6msg.mutable_masked_message_two()) = masked_ot_2;

        return round6msg;
    }


    absl::StatusOr<ValueCorrectionShare>
    KeyGenerationProtocol::ComputeValueCorrectionOtShare(
            int partyid,
            const ValueCorrectionOtSenderMessage& value_ot_sender_message,
            ProtocolState& state) const{

        // Decode mux message
        ValueCorrectionShare value_corr_share;

//        return absl::UnimplementedError("");

        return value_corr_share;
    }

    absl::StatusOr<int> KeyGenerationProtocol::ApplyValueCorrectionShare(
            int partyid,
            const ValueCorrectionShare& value_correction_share,
            ProtocolState& state) const{

        // TODO: Update seeds and control bits
        state.seeds = state.uncorrected_seeds;
        state.uncorrected_seeds.clear();

        state.shares_of_control_bits = state.shares_of_uncorrected_control_bits;
        state.shares_of_uncorrected_control_bits.clear();

        // TODO: Update the correction word in  DPF key

        // TODO : Update tree_level
        state.tree_level += 1;

        // Todo : Clear aux state variables
        return 0;
    }

} // namespace distributed_point_functions