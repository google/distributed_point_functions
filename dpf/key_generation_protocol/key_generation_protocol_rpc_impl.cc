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
#include "dpf/key_generation_protocol/key_generation_protocol_rpc_impl.h"
#include "absl/status/status.h"
#include "dpf/status_macros.h"

namespace distributed_point_functions {
namespace {
	// Translates Status to grpc::Status
	::grpc::Status ConvertStatus(const absl::Status& status) {
	  if (status.ok()) {
	    return ::grpc::Status::OK;
	  }
	  if (absl::IsInvalidArgument(status)) {
	    return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
			          std::string(status.message()));
	  }
	  if (absl::IsInternal(status)) {
	    return ::grpc::Status(::grpc::StatusCode::INTERNAL,
			          std::string(status.message()));
	  }
	  return ::grpc::Status(::grpc::StatusCode::UNKNOWN,
			        std::string(status.message()));
	}
} // namespace

::grpc::Status KeyGenerationProtocolRpcImpl::Handle(::grpc::ServerContext* context,
                      const KeyGenerationProtocolClientMessage* request,
                      KeyGenerationProtocolServerMessage* response) {
	return ConvertStatus(HandleInternal(context, request, response));
}
absl::Status KeyGenerationProtocolRpcImpl::HandleInternal(::grpc::ServerContext* context,
	const KeyGenerationProtocolClientMessage* request,
	KeyGenerationProtocolServerMessage* response)	{
	if(request->has_start_message()) {
		DPF_ASSIGN_OR_RETURN(*response->mutable_server_round_1_message(),
                    keygen_->ComputeSeedCorrectionOtReceiverMessage(
                            1,
                            protocol_state_party_1_));
  	} else if(request->has_client_round_1_message()) {
  		DPF_ASSIGN_OR_RETURN(*response->mutable_server_round_2_message(),
                    keygen_->ComputeSeedCorrectionOtSenderMessage(
                            1,
                            request->client_round_1_message(),
                            protocol_state_party_1_));
  	} else if(request->has_client_round_2_message()) {
  		DPF_ASSIGN_OR_RETURN(*response->mutable_server_round_3_message(),
                    keygen_->ComputeSeedCorrectionOpening(
                            1,
                            request->client_round_2_message(),
                            protocol_state_party_1_));
	} else if(request->has_client_round_3_message()) {
		DPF_ASSIGN_OR_RETURN(*response->mutable_server_round_4_message(),
                    keygen_->ApplySeedCorrectionShare(
                            1,
                            request->client_round_3_message(),
                            protocol_state_party_1_));
	} else if(request->has_client_round_4_message()) {
		DPF_ASSIGN_OR_RETURN(*response->mutable_server_round_5_message(),
                    keygen_->ComputeValueCorrectionOtReceiverMessage(
                            1,
                            request->client_round_4_message(),
                            protocol_state_party_1_));
	} else if(request->has_client_round_5_message()) {
		DPF_ASSIGN_OR_RETURN(*response->mutable_server_round_6_message(),
                    keygen_->ComputeValueCorrectionOtSenderMessage(
                            1,
                            request->client_round_5_message(),
                            protocol_state_party_1_));
	} else if(request->has_client_round_6_message()) {
		DPF_ASSIGN_OR_RETURN(*response->mutable_server_round_7_message(),
                    keygen_->ComputeValueCorrectionOtShare(
                            1,
                            request->client_round_6_message(),
                            protocol_state_party_1_));
	} else if(request->has_client_round_7_message()) {
		
		int x;
		DPF_ASSIGN_OR_RETURN(x,
                     keygen_->ApplyValueCorrectionShare(
                        1,
                        request->client_round_7_message(),
                        protocol_state_party_1_));
                *response->mutable_end_message() = EndMessage();
		
		// Last message needs to update the current level
		std::cout << "Server: completed iteration " << current_level_+1
		 	 << std::endl;
		current_level_++;
	} else {
		return absl::InvalidArgumentError(absl::StrCat("KeyGenerationProtocolServer server"
				" received an unrecognized message, with case ",
				request->client_message_oneof_case()));
	}
	
	total_client_message_size_ += request->ByteSizeLong();
	total_server_message_size_ += response->ByteSizeLong();
		
	if(current_level_ == num_levels_) {
		std::cout << "Server completed." <<std::endl
			<< "Total client message size = "<<  total_client_message_size_
			<< " bytes" <<std::endl
 			<< "Total server message size = " << total_server_message_size_
 			<< " bytes" <<std::endl
 			<< "Grand total message size = " 
			<< total_server_message_size_ + total_client_message_size_
 			<< " bytes" <<std::endl;
  	} 
		
	return absl::OkStatus();
}

}  // namespace distributed_point_functions
