
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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_DPF_KEY_GENERATION_PROTOCOL_KEY_GENERATION_PROTOCOL_RPC_IMPL_H_
#define DISTRIBUTED_POINT_FUNCTIONS_DPF_KEY_GENERATION_PROTOCOL_KEY_GENERATION_PROTOCOL_RPC_IMPL_H_

#include "include/grpcpp/grpcpp.h"
#include "include/grpcpp/server_context.h"
#include "include/grpcpp/support/status.h"
#include "dpf/key_generation_protocol/key_generation_protocol.h"
#include "dpf/key_generation_protocol/key_generation_protocol.pb.h"
#include "dpf/key_generation_protocol/key_generation_protocol_rpc.grpc.pb.h"
#include "dpf/key_generation_protocol/key_generation_protocol_rpc.pb.h"
#include "absl/strings/string_view.h"
#include <random>
#include <chrono>

namespace distributed_point_functions {

// Implements the Gradient Descent RPC-handling Server.
class KeyGenerationProtocolRpcImpl : public KeyGenerationProtocolRpc::Service {
 public:
  KeyGenerationProtocolRpcImpl(  
		std::unique_ptr<KeyGenerationProtocol> keygen, size_t num_levels,
		absl::uint128 alpha_share_party_1,
	 	std::vector<Value> beta_shares_party_1,
	 	KeyGenerationPreprocessing preproc_party_1
		): keygen_(std::move(keygen)), current_level_(0), num_levels_(num_levels), 
		alpha_share_party_1_(std::move(alpha_share_party_1)), 
		beta_shares_party_1_(std::move(beta_shares_party_1)), 
		preproc_party_1_(std::move(preproc_party_1)) {
			protocol_state_party_1_ = keygen_->Initialize(1,
			alpha_share_party_1_,
			beta_shares_party_1_,
			preproc_party_1_).value();

		} 
  
  // Executes a round of the protocol.
  ::grpc::Status Handle(::grpc::ServerContext* context,
                        const KeyGenerationProtocolClientMessage* request,
                        KeyGenerationProtocolServerMessage* response) override;
	
	size_t current_level() {
		return current_level_;
	  }

 private:
	 // Internal version of Handle, that returns a non-grpc Status.
	 absl::Status HandleInternal(::grpc::ServerContext* context,
	 	const KeyGenerationProtocolClientMessage* request,
		KeyGenerationProtocolServerMessage* response);
	 
	 std::unique_ptr<KeyGenerationProtocol> keygen_;
	 
	 volatile size_t current_level_ = 0;
	 const size_t num_levels_;
	 
	 absl::uint128 alpha_share_party_1_;
	 std::vector<Value> beta_shares_party_1_;
	 KeyGenerationPreprocessing preproc_party_1_;
	 ProtocolState protocol_state_party_1_;

	 
	 size_t total_client_message_size_ = 0;
	 size_t total_server_message_size_ = 0;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DPF_KEY_GENERATION_PROTOCOL_KEY_GENERATION_PROTOCOL_RPC_IMPL_H_
