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

#include <iostream>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "include/grpc/grpc_security_constants.h"
#include "include/grpcpp/grpcpp.h"
#include "dpf/key_generation_protocol/key_generation_protocol.h"
#include "dpf/key_generation_protocol/key_generation_protocol.pb.h"
#include "dpf/key_generation_protocol/key_generation_protocol_rpc.grpc.pb.h"
#include "dpf/key_generation_protocol/key_generation_protocol_rpc.pb.h"
#include "include/grpcpp/security/server_credentials.h"
#include "include/grpcpp/server_builder.h"
#include "include/grpcpp/server_context.h"
#include "include/grpcpp/support/status.h"
#include "absl/status/status.h"
#include "dpf/status_macros.h"
#include "absl/strings/string_view.h"

ABSL_FLAG(std::string, port, "0.0.0.0:10501",
          "Port on which to contact server");
ABSL_FLAG(size_t, num_levels, 20,
          "The number of levels for the DPF, also how many iterations to execute.");

namespace distributed_point_functions {

absl::Status ExecuteProtocol() {
  	// Setup
 	size_t levels = absl::GetFlag(FLAGS_num_levels);
	
	// Generate parameters for KeyGenProtocol.
  	std::vector<DpfParameters> parameters;
  	parameters.reserve(levels);

	for (size_t i = 0; i < levels; i++){
		parameters.push_back(DpfParameters());
        	parameters[i].set_log_domain_size(i + 1);
        	parameters[i].mutable_value_type()->mutable_integer()->set_bitsize(64);
    	}

  	std::unique_ptr<KeyGenerationProtocol> keygen;

    	DPF_ASSIGN_OR_RETURN(keygen,
    		KeyGenerationProtocol::Create(parameters));

    	std::pair<KeyGenerationPreprocessing, KeyGenerationPreprocessing> preproc;
	
	DPF_ASSIGN_OR_RETURN(preproc,
                             keygen->PerformKeyGenerationPrecomputation());
	
	
	absl::uint128 alpha = 23;
	
	// Generating shares of alpha for Party 0 and Party 1
	absl::uint128 alpha_share_party0, alpha_share_party1;
	const absl::string_view kSampleSeed = absl::string_view("abcdefg");
	DPF_ASSIGN_OR_RETURN(
		auto rng, distributed_point_functions::BasicRng::Create(kSampleSeed));
		

	DPF_ASSIGN_OR_RETURN(alpha_share_party0, rng->Rand128());
	alpha_share_party1 = alpha ^ alpha_share_party0;

    	// Generating shares of beta for Party 0 and Party 1
    	std::vector<Value> beta;


    	for(size_t i = 0; i < levels; i++){
        	Value beta_i;
        	beta_i.mutable_integer()->set_value_uint64(42);
        	beta.push_back(beta_i);
    	}

   	 std::vector<Value> beta_shares_party0, beta_shares_party1;

    	for (size_t i = 0; i < beta.size(); i++){
        	DPF_ASSIGN_OR_RETURN(absl::uint128 beta_share_party0_seed, rng->Rand128());

        	Value value0 = ToValue<uint64_t>(static_cast<uint64_t>(beta_share_party0_seed));

        	DPF_ASSIGN_OR_RETURN(Value value1,
                                 keygen->ValueSub<uint64_t>(beta[i], value0));

       	 	beta_shares_party0.push_back(value0);
        	beta_shares_party1.push_back(value1);
        }
  
	// Consider grpc::SslServerCredentials if not running locally.
	std::cout << "Client: Creating server stub..." << std::endl;
	 	grpc::ChannelArguments ch_args;
	  ch_args.SetMaxReceiveMessageSize(-1); // consider limiting max message size
 	std::unique_ptr<KeyGenerationProtocolRpc::Stub> stub =
      	KeyGenerationProtocolRpc::NewStub(::grpc::CreateCustomChannel(
      		absl::GetFlag(FLAGS_port), grpc::InsecureChannelCredentials(), ch_args));
	std::cout << "Client: Starting KeyGenerationProtocol "	<< std::endl;
  	double pzero_time = 0;
  	double pone_time_incl_comm = 0;
  	double end_to_end_time = 0;
	auto start = std::chrono::high_resolution_clock::now();
	auto client_start = start;
	auto client_end = start;
	auto server_start = start;
	auto server_end = start;
	
	::grpc::Status grpc_status;
	DPF_ASSIGN_OR_RETURN(ProtocolState protocol_state_party_0,
		keygen->Initialize(1,
			alpha_share_party0,
			beta_shares_party0,
			preproc.first));
	grpc::CompletionQueue cq;
	
	// Initiate server work.
	std::cout << "Client: Starting protocol" << std::endl;
	
	uint64_t cq_index=1;
	void* got_tag;
	bool ok = false;
	
	for(size_t i = 0; i < levels; i++) {
		std::cout << "Client: Starting iteration " << i + 1 << std::endl;
		
		// Run Round 1
		client_start = std::chrono::high_resolution_clock::now();
		::grpc::ClientContext client_context0;
		KeyGenerationProtocolClientMessage client_message_0;
		*client_message_0.mutable_start_message() = StartMessage();
		KeyGenerationProtocolServerMessage server_message_1;
		std::unique_ptr<grpc::ClientAsyncResponseReader<KeyGenerationProtocolServerMessage> > rpc(stub->AsyncHandle(&client_context0, client_message_0, &cq));
		rpc->Finish(&server_message_1, &grpc_status, (void*)cq_index);
		
		KeyGenerationProtocolClientMessage client_message_1;
            	DPF_ASSIGN_OR_RETURN(*client_message_1.mutable_client_round_1_message(),
                    keygen->ComputeSeedCorrectionOtReceiverMessage(
                            0,
                            protocol_state_party_0));
                            
                client_end = std::chrono::high_resolution_clock::now();
		pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
	       		client_end - client_start).count())/ 1e6;
		
		server_start = std::chrono::high_resolution_clock::now();
		GPR_ASSERT(cq.Next(&got_tag, &ok));
		GPR_ASSERT(got_tag == (void*) cq_index);
		GPR_ASSERT(ok);
	
		if (!grpc_status.ok()) {
			std::cerr << "Client: Failed on message round 1 of level " << i+1 <<  " with status " <<
			grpc_status.error_code() << " error_message: " <<
	 		 grpc_status.error_message() << std::endl;
			return absl::UnknownError("");
		}
		server_end = std::chrono::high_resolution_clock::now();
		pone_time_incl_comm +=
			(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;
		
		// Run Round 2
		client_start = std::chrono::high_resolution_clock::now();
		::grpc::ClientContext client_context1;
		cq_index++;
		ok=false;
		KeyGenerationProtocolServerMessage server_message_2;
	 	rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<KeyGenerationProtocolServerMessage>>(stub->AsyncHandle(&client_context1, client_message_1, &cq));
		rpc->Finish(&server_message_2, &grpc_status, (void*)cq_index);
	
		KeyGenerationProtocolClientMessage client_message_2;
		
		DPF_ASSIGN_OR_RETURN(*client_message_2.mutable_client_round_2_message(),
			keygen->ComputeSeedCorrectionOtSenderMessage( 
				0,
				server_message_1.server_round_1_message(), 
				protocol_state_party_0));
		
		client_end = std::chrono::high_resolution_clock::now();
		pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            	client_end - client_start).count())/ 1e6;
		server_start = std::chrono::high_resolution_clock::now();
	
		GPR_ASSERT(cq.Next(&got_tag, &ok));
		GPR_ASSERT(got_tag == (void*) cq_index);
		GPR_ASSERT(ok);
                 
                if (!grpc_status.ok()) {
			std::cerr << "Client: Failed on message round 2 of level " << i+1 <<  " with status " <<
			grpc_status.error_code() << " error_message: " <<
	 		 grpc_status.error_message() << std::endl;
			return absl::UnknownError("");
		}
		server_end = std::chrono::high_resolution_clock::now();
		pone_time_incl_comm +=
			(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;

		// Run Round 3
		client_start = std::chrono::high_resolution_clock::now();
		::grpc::ClientContext client_context2;
		cq_index++;
		ok=false;
		KeyGenerationProtocolServerMessage server_message_3;
	 	rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<KeyGenerationProtocolServerMessage>>(stub->AsyncHandle(&client_context2, client_message_2, &cq));
		rpc->Finish(&server_message_3, &grpc_status, (void*)cq_index);
	
		KeyGenerationProtocolClientMessage client_message_3;
		
		DPF_ASSIGN_OR_RETURN(*client_message_3.mutable_client_round_3_message(),
			keygen->ComputeSeedCorrectionOpening( 
				0,
				server_message_2.server_round_2_message(), 
				protocol_state_party_0));
		
		client_end = std::chrono::high_resolution_clock::now();
		pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            	client_end - client_start).count())/ 1e6;
		server_start = std::chrono::high_resolution_clock::now();
	
		GPR_ASSERT(cq.Next(&got_tag, &ok));
		GPR_ASSERT(got_tag == (void*) cq_index);
		GPR_ASSERT(ok);
                 
                if (!grpc_status.ok()) {
			std::cerr << "Client: Failed on message round 3 of level " << i+1 <<  " with status " <<
			grpc_status.error_code() << " error_message: " <<
	 		 grpc_status.error_message() << std::endl;
			return absl::UnknownError("");
		}
		server_end = std::chrono::high_resolution_clock::now();
		pone_time_incl_comm +=
			(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;


		// Run Round 4
		client_start = std::chrono::high_resolution_clock::now();
		::grpc::ClientContext client_context3;
		cq_index++;
		ok=false;
		KeyGenerationProtocolServerMessage server_message_4;
	 	rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<KeyGenerationProtocolServerMessage>>(stub->AsyncHandle(&client_context3, client_message_3, &cq));
		rpc->Finish(&server_message_4, &grpc_status, (void*)cq_index);
	
		KeyGenerationProtocolClientMessage client_message_4;
		
		DPF_ASSIGN_OR_RETURN(*client_message_4.mutable_client_round_4_message(),
			keygen->ApplySeedCorrectionShare( 
				0,
				server_message_3.server_round_3_message(), 
				protocol_state_party_0));
		
		client_end = std::chrono::high_resolution_clock::now();
		pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            	client_end - client_start).count())/ 1e6;
		server_start = std::chrono::high_resolution_clock::now();
	
		GPR_ASSERT(cq.Next(&got_tag, &ok));
		GPR_ASSERT(got_tag == (void*) cq_index);
		GPR_ASSERT(ok);
                 
                if (!grpc_status.ok()) {
			std::cerr << "Client: Failed on message round 4 of level " << i+1 <<  " with status " <<
			grpc_status.error_code() << " error_message: " <<
	 		 grpc_status.error_message() << std::endl;
			return absl::UnknownError("");
		}
		server_end = std::chrono::high_resolution_clock::now();
		pone_time_incl_comm +=
			(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;

		// Run Round 5
		
		client_start = std::chrono::high_resolution_clock::now();
		::grpc::ClientContext client_context4;
		cq_index++;
		ok=false;
		KeyGenerationProtocolServerMessage server_message_5;
	 	rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<KeyGenerationProtocolServerMessage>>(stub->AsyncHandle(&client_context4, client_message_4, &cq));
		rpc->Finish(&server_message_5, &grpc_status, (void*)cq_index);
	
		KeyGenerationProtocolClientMessage client_message_5;
		
		DPF_ASSIGN_OR_RETURN(*client_message_5.mutable_client_round_5_message(),
			keygen->ComputeValueCorrectionOtReceiverMessage( 
				0,
				server_message_4.server_round_4_message(), 
				protocol_state_party_0));
		
		client_end = std::chrono::high_resolution_clock::now();
		pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            	client_end - client_start).count())/ 1e6;
		server_start = std::chrono::high_resolution_clock::now();
	
		GPR_ASSERT(cq.Next(&got_tag, &ok));
		GPR_ASSERT(got_tag == (void*) cq_index);
		GPR_ASSERT(ok);
                 
                if (!grpc_status.ok()) {
			std::cerr << "Client: Failed on message round 5 of level " << i+1 <<  " with status " <<
			grpc_status.error_code() << " error_message: " <<
	 		 grpc_status.error_message() << std::endl;
			return absl::UnknownError("");
		}
		server_end = std::chrono::high_resolution_clock::now();
		pone_time_incl_comm +=
			(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;


		// Run Round 6
		
		client_start = std::chrono::high_resolution_clock::now();
		::grpc::ClientContext client_context5;
		cq_index++;
		ok=false;
		KeyGenerationProtocolServerMessage server_message_6;
	 	rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<KeyGenerationProtocolServerMessage>>(stub->AsyncHandle(&client_context5, client_message_5, &cq));
		rpc->Finish(&server_message_6, &grpc_status, (void*)cq_index);
	
		KeyGenerationProtocolClientMessage client_message_6;
		
		DPF_ASSIGN_OR_RETURN(*client_message_6.mutable_client_round_6_message(),
			keygen->ComputeValueCorrectionOtSenderMessage( 
				0,
				server_message_5.server_round_5_message(), 
				protocol_state_party_0));
		
		client_end = std::chrono::high_resolution_clock::now();
		pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            	client_end - client_start).count())/ 1e6;
		server_start = std::chrono::high_resolution_clock::now();
	
		GPR_ASSERT(cq.Next(&got_tag, &ok));
		GPR_ASSERT(got_tag == (void*) cq_index);
		GPR_ASSERT(ok);
                 
                if (!grpc_status.ok()) {
			std::cerr << "Client: Failed on message round 6 of level " << i+1 <<  " with status " <<
			grpc_status.error_code() << " error_message: " <<
	 		 grpc_status.error_message() << std::endl;
			return absl::UnknownError("");
		}
		server_end = std::chrono::high_resolution_clock::now();
		pone_time_incl_comm +=
			(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;

		// Run Round 7
		
		client_start = std::chrono::high_resolution_clock::now();
		::grpc::ClientContext client_context6;
		cq_index++;
		ok=false;
		KeyGenerationProtocolServerMessage server_message_7;
	 	rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<KeyGenerationProtocolServerMessage>>(stub->AsyncHandle(&client_context6, client_message_6, &cq));
		rpc->Finish(&server_message_7, &grpc_status, (void*)cq_index);
	
		KeyGenerationProtocolClientMessage client_message_7;
		
		
		DPF_ASSIGN_OR_RETURN(*client_message_7.mutable_client_round_7_message(),
			keygen->ComputeValueCorrectionOtShare( 
				0,
				server_message_6.server_round_6_message(), 
				protocol_state_party_0));
		
		client_end = std::chrono::high_resolution_clock::now();
		pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            	client_end - client_start).count())/ 1e6;
		server_start = std::chrono::high_resolution_clock::now();
	
		GPR_ASSERT(cq.Next(&got_tag, &ok));
		GPR_ASSERT(got_tag == (void*) cq_index);
		GPR_ASSERT(ok);
                 
                if (!grpc_status.ok()) {
			std::cerr << "Client: Failed on message round 7 of level " << i+1 <<  " with status " <<
			grpc_status.error_code() << " error_message: " <<
	 		 grpc_status.error_message() << std::endl;
			return absl::UnknownError("");
		}
		server_end = std::chrono::high_resolution_clock::now();
		pone_time_incl_comm +=
			(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;

		// Compute result
		client_start = std::chrono::high_resolution_clock::now();
		::grpc::ClientContext client_context7;
		cq_index++;
		ok=false;
		KeyGenerationProtocolServerMessage server_message_end;
	 	rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<KeyGenerationProtocolServerMessage>>(stub->AsyncHandle(&client_context7, client_message_7, &cq));
		rpc->Finish(&server_message_end, &grpc_status, (void*)cq_index);
		
            	DPF_ASSIGN_OR_RETURN(int x,
           		keygen->ApplyValueCorrectionShare(
                        	0,
                            	server_message_7.server_round_7_message(),
                           	protocol_state_party_0));
		
		client_end = std::chrono::high_resolution_clock::now();
		pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            	client_end - client_start).count())/ 1e6;
		server_start = std::chrono::high_resolution_clock::now();
	
		GPR_ASSERT(cq.Next(&got_tag, &ok));
		GPR_ASSERT(got_tag == (void*) cq_index);
		GPR_ASSERT(ok);
                 
                if (!grpc_status.ok()) {
			std::cerr << "Client: Failed on end message of level " << i+1 <<  " with status " <<
			grpc_status.error_code() << " error_message: " <<
	 		 grpc_status.error_message() << std::endl;
			return absl::UnknownError("");
		}
		server_end = std::chrono::high_resolution_clock::now();
		pone_time_incl_comm +=
			(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;

	}
	
  	auto end = std::chrono::high_resolution_clock::now();
  	
  	// Add in preprocessing phase. For the online phase, since the initial round for client and server can be done at the same time
 	 end_to_end_time = (std::chrono::duration_cast<std::chrono::microseconds>(
          	end-start).count())
      		/ 1e6;
	// Print results
	std::cout << "Completed run" << std::endl << "num_levels="
		<< levels << std::endl
	  << "Client time total (s) =" << pzero_time <<std::endl
	  << "Server time (incl. comm) total (s) = " << pone_time_incl_comm <<std::endl
	      << "End to End time (excluding preprocessing) total (s) = " << end_to_end_time <<std::endl;
  	
  	return absl::OkStatus();
}
}  // namespace distributed_point_functions

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  auto status = ::distributed_point_functions::ExecuteProtocol();
  if(!status.ok()){
  	std::cerr << "Client failed: " << status;
  	return 1;
  }
  return 0;
}
