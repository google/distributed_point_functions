// Copyright 2021 Google LLC
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

#include "pir/prng/aes_128_ctr_seeded_prng.h"

#include <cstdint>
#include <iterator>
#include <string>

#include "absl/base/config.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "openssl/aes.h"
#include "openssl/err.h"
#include "openssl/rand.h"

namespace distributed_point_functions {
namespace {

std::string OpenSSLErrorString() {
  char buf[256];
  ERR_error_string_n(ERR_get_error(), buf, sizeof(buf));
  return buf;
}

}  // namespace

absl::StatusOr<std::string> Aes128CtrSeededPrng::GenerateSeed() {
  std::string seed(SeedSize(), '\0');
  RAND_bytes(reinterpret_cast<uint8_t*>(&seed[0]), SeedSize());
  return seed;
}

Aes128CtrSeededPrng::Aes128CtrSeededPrng(AES_KEY aes_key,
                                         std::vector<uint8_t> ivec,
                                         std::vector<uint8_t> ecount_buf)
    : aes_key_(std::move(aes_key)),
      ivec_(std::move(ivec)),
      ecount_buf_(std::move(ecount_buf)),
      num_(0) {}

absl::StatusOr<std::unique_ptr<Aes128CtrSeededPrng>>
Aes128CtrSeededPrng::Create(absl::string_view seed) {
  std::string nonce(SeedSize(), '\0');
  return CreateWithNonce(seed, nonce);
}

absl::StatusOr<std::unique_ptr<Aes128CtrSeededPrng>>
Aes128CtrSeededPrng::CreateWithNonce(absl::string_view seed,
                                     absl::string_view nonce) {
  if (seed.size() != SeedSize()) {
    return absl::InvalidArgumentError(absl::StrCat("seed must be ", SeedSize(),
                                                   " bytes, supplied seed is ",
                                                   seed.size(), " bytes."));
  }
  if (seed.size() != nonce.size()) {
    return absl::InvalidArgumentError("seed and nonce must have the same size");
  }

  // Create an AES128 key from the supplied seed.
  AES_KEY aes_key;
  if (0 != AES_set_encrypt_key(reinterpret_cast<const uint8_t*>(seed.data()),
                               SeedSize() * 8, &aes_key)) {
    return absl::InternalError(
        absl::StrCat("AES_set_encrypt_key failed with error message: ",
                     OpenSSLErrorString()));
  }
  std::vector<uint8_t> ivec(AES_BLOCK_SIZE, 0);
  std::vector<uint8_t> ecount_buf(AES_BLOCK_SIZE, 0);
  std::copy_n(nonce.begin(), AES_BLOCK_SIZE, ivec.begin());
  return absl::WrapUnique(new Aes128CtrSeededPrng(
      std::move(aes_key), std::move(ivec), std::move(ecount_buf)));
}

std::string Aes128CtrSeededPrng::GetRandomBytes(size_t length) {
  std::string output(length, 0);
  char* output_buffer = &output[0];
  AES_ctr128_encrypt(reinterpret_cast<const uint8_t*>(output_buffer),
                     reinterpret_cast<uint8_t*>(output_buffer), length,
                     &aes_key_, ivec_.data(), ecount_buf_.data(), &num_);

  return output;
}

}  // namespace distributed_point_functions
