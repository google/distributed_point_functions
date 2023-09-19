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
#include "dpf/key_generation_protocol/key_generation_protocol_rpc_impl.h"
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

ABSL_FLAG(std::string, port, "0.0.0.0:10501", "Port on which to listen");
ABSL_FLAG(size_t, num_levels, 20,
          "The number of levels for the DPF, also how many iterations to execute.");

namespace distributed_point_functions {

absl::Status RunServer() {
	std::cout << "Server: starting... " << std::endl;
	
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
        
	
	// Initialize the service
  	std::unique_ptr<KeyGenerationProtocolRpcImpl> service = std::make_unique<KeyGenerationProtocolRpcImpl>(
  		std::move(keygen), levels, std::move(alpha_share_party1),
  		std::move(beta_shares_party1), std::move(preproc.second)
  	);
	::grpc::ServerBuilder builder;
	// Consider grpc::SslServerCredentials if not running locally.
	builder.AddListeningPort(absl::GetFlag(FLAGS_port),
		grpc::InsecureServerCredentials());
	builder.SetMaxReceiveMessageSize(INT_MAX); // consider limiting max message size
	builder.RegisterService(service.get());
	std::unique_ptr<::grpc::Server> grpc_server(builder.BuildAndStart());
	// Run the server on a background thread.
	
	std::thread grpc_server_thread(
	[](::grpc::Server* grpc_server_ptr) {
        	std::cout << "Server: listening on " << absl::GetFlag(FLAGS_port)
                  << std::endl;
		grpc_server_ptr->Wait();
      	},
      	grpc_server.get());
  	while (service->current_level() < levels) {
  	}
  	// Shut down server.
  	grpc_server->Shutdown();
  	grpc_server_thread.join();
  	std::cout << "Server completed protocol and shut down." << std::endl;
  	
  	return absl::OkStatus();
}

}  // namespace distributed_point_functions

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  auto status = ::distributed_point_functions::RunServer();
  if(!status.ok()){
  	std::cerr << "Server failed: " << status.message();
  	return 1;
  }
  return 0;
}
