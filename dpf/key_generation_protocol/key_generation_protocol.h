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
class KeyGenerationProtocol {
 public:
  struct ProtocolState {
    int tree_level;
    // Add more local state variables here.
  };

  // Creates a new instance of the key generation protocol for a DPF with the
  // given parameters. Party must be 0 or 1.
  static absl::StatusOr<std::unique_ptr<KeyGenerationProtocol>> Create(
      absl::Span<const DpfParameters> parameters, int party);

  // Create ProtocolState given shares of alphas and betas. Arguments are given
  // as Spans to allow batching.
  absl::StatusOr<ProtocolState> Initialize(
      absl::Span<const absl::uint128> alpha_shares,
      absl::Span<const std::vector<Value>> beta_shares) const;

  // Receiver OT message for the MUX in Step 5. Just takes the state as input.
  absl::StatusOr<SeedCorrectionOtReceiverMessage>
  ComputeSeedCorrectionOtReceiverMessage(ProtocolState& state) const;

  // Computes the sender OT message given the receiver message and the state.
  absl::StatusOr<SeedCorrectionOtSenderMessage>
  ComputeSeedCorrectionOtSenderMessage(
      const SeedCorrectionOtReceiverMessage& seed_ot_receiver_message,
      ProtocolState& state) const;

  // Computes the share of the seed correction word given the sender OT message
  // and the state.
  absl::StatusOr<SeedCorrectionShare> ComputeSeedCorrectionOtShare(
      const SeedCorrectionOtSenderMessage& seed_ot_sender_message,
      ProtocolState& state) const;

  // Updates the state with the other party's seed correction share.
  absl::Status ApplySeedCorrectionShare(
      const SeedCorrectionShare& seed_correction_share,
      ProtocolState& state) const;

  // Computes the OT receiver message for the MUX gate in Step 11 given the
  // state.
  absl::StatusOr<ValueCorrectionOtReceiverMessage>
  ComputeValueCorrectionOtReceiverMessage(ProtocolState& state) const;

  // Computes the OT sender message in Step 11 given the receiver message and
  // the state.
  absl::StatusOr<ValueCorrectionOtSenderMessage>
  ComputeValueCorrectionOtSenderMessage(
      const ValueCorrectionOtReceiverMessage& value_ot_receiver_message,
      ProtocolState& state) const;

  // Computes the value correction share given the OT sender message and the
  // state.
  absl::StatusOr<ValueCorrectionShare> ComputeValueCorrectionOtShare(
      const ValueCorrectionOtSenderMessage& value_ot_sender_message,
      ProtocolState& state) const;

  // Updates the state with the other party's value correction share.
  absl::Status ApplyValueCorrectionShare(
      const ValueCorrectionShare& value_correction_share,
      ProtocolState& state) const;

  // Finalizes the protocol after all tree levels have been computed and returns
  // the generated DpfKey.
  absl::StatusOr<DpfKey> Finalize(ProtocolState& state) const;

 private:
  explicit KeyGenerationProtocol(std::unique_ptr<DistributedPointFunction> dpf,
                                 int party);

  std::unique_ptr<DistributedPointFunction> dpf_;
  int party_;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DPF_KEY_GENERATION_PROTOCOL_KEY_GENERATION_PROTOCOL_H_