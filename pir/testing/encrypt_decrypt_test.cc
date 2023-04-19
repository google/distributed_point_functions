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
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "dpf/internal/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace distributed_point_functions {
namespace pir_testing {
namespace {

using dpf_internal::IsOkAndHolds;

TEST(EncryptDecrypt, CreateFakeHybridDecryptSucceeds) {
  DPF_EXPECT_OK(CreateFakeHybridDecrypt());
}

TEST(EncryptDecrypt, CreateFakeHybridEncryptSucceeds) {
  DPF_EXPECT_OK(CreateFakeHybridEncrypt());
}

TEST(EncryptDecrypt, CanDecryptRealCiphertext) {
  DPF_ASSERT_OK_AND_ASSIGN(auto encrypter, CreateFakeHybridEncrypt());
  DPF_ASSERT_OK_AND_ASSIGN(auto decrypter, CreateFakeHybridDecrypt());
  const absl::string_view plaintext = "Some Message";

  DPF_ASSERT_OK_AND_ASSIGN(std::string ciphertext,
                           encrypter->Encrypt(plaintext, ""));

  EXPECT_THAT(decrypter->Decrypt(ciphertext, ""), IsOkAndHolds(plaintext));
}

TEST(EncryptDecrypt, CannotDecryptFakeCiphertext) {
  DPF_ASSERT_OK_AND_ASSIGN(auto decrypter, CreateFakeHybridDecrypt());

  const absl::string_view fake_ciphertext = "Not a real ciphertext";

  EXPECT_FALSE(decrypter->Decrypt(fake_ciphertext, "").ok());
}

}  // namespace
}  // namespace pir_testing
}  // namespace distributed_point_functions
