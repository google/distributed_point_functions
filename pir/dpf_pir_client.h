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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_DPF_PIR_CLIENT_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_DPF_PIR_CLIENT_H_

#include <string>

#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "pir/pir_client.h"

namespace distributed_point_functions {

template <typename QueryType, typename ResponseType>
class DpfPirClient : public PirClient<QueryType, ResponseType> {
 public:
  // Function type for the client to encrypt a PIR request to the helper.
  // This function has the same parameter and return types as
  // `crypto::tink::HybridEncrypt::Encrypt()`: it takes `plain_helper_request`
  // storing the PIR request and `encryption_context_info` to be passed to the
  // helper to correctly decrypt the encrypted PIR request, and it returns the
  // result of the encryption.
  //
  // The client stores a function object of this type because in some cases a
  // HybridEncrypt object may have to be refreshed before being invoked on a
  // PIR request.
  // Using this wrapper allows the underlying HybridEncrypt to change between
  // creation of a client and a call to `CreateRequest`.
  using EncryptHelperRequestFn = absl::AnyInvocable<absl::StatusOr<std::string>(
      absl::string_view plain_helper_request,
      absl::string_view encryption_context_info) const>;

  virtual ~DpfPirClient() = default;
};

}  // namespace distributed_point_functions
#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_DPF_PIR_CLIENT_H_
