// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pir/testing/encrypt_decrypt.h"

#include <memory>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "dpf/status_macros.h"
#include "pir/testing/data/embedded_private_key.h"
#include "pir/testing/data/embedded_public_key.h"
#include "tink/cleartext_keyset_handle.h"
#include "tink/config/hybrid_defaults.h"
#include "tink/hybrid/hybrid_config.h"
#include "tink/json_keyset_reader.h"
#include "tink/keyset_handle.h"
#include "tink/keyset_reader.h"

namespace distributed_point_functions {
namespace pir_testing {

using ::crypto::tink::CleartextKeysetHandle;
using ::crypto::tink::HybridDecrypt;
using ::crypto::tink::HybridEncrypt;
using ::crypto::tink::JsonKeysetReader;
using ::crypto::tink::KeysetHandle;
using ::crypto::tink::KeysetReader;

absl::once_flag register_tink_once ABSL_ATTRIBUTE_UNUSED;

absl::StatusOr<std::unique_ptr<HybridDecrypt>> CreateFakeHybridDecrypt() {
  absl::call_once(register_tink_once, []() {
    ABSL_CHECK_OK(crypto::tink::HybridConfig::Register());
  });

  const auto* const toc = embedded_private_key_create();
  absl::string_view private_key_json(toc->data, toc->size);

  DPF_ASSIGN_OR_RETURN(std::unique_ptr<KeysetReader> private_key_reader,
                       JsonKeysetReader::New(private_key_json));

  DPF_ASSIGN_OR_RETURN(
      std::unique_ptr<KeysetHandle> private_key_handle,
      CleartextKeysetHandle::Read(std::move(private_key_reader)));

  return private_key_handle->GetPrimitive<HybridDecrypt>(
      crypto::tink::HybridDefaults());
}

absl::StatusOr<std::unique_ptr<HybridEncrypt>> CreateFakeHybridEncrypt() {
  absl::call_once(register_tink_once, []() {
    ABSL_CHECK_OK(crypto::tink::HybridConfig::Register());
  });

  const auto* const toc = embedded_public_key_create();
  absl::string_view public_key_json(toc->data, toc->size);

  DPF_ASSIGN_OR_RETURN(std::unique_ptr<KeysetReader> public_key_reader,
                       JsonKeysetReader::New(public_key_json));

  DPF_ASSIGN_OR_RETURN(
      std::unique_ptr<KeysetHandle> public_key_handle,
      CleartextKeysetHandle::Read(std::move(public_key_reader)));

  return public_key_handle->GetPrimitive<HybridEncrypt>(
      crypto::tink::HybridDefaults());
}

}  // namespace pir_testing
}  // namespace distributed_point_functions
