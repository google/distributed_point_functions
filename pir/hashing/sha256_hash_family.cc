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

#include "pir/hashing/sha256_hash_family.h"

#include <stdint.h>

#include <algorithm>
#include <cstring>

#include "absl/numeric/int128.h"
#include "absl/strings/string_view.h"
#include "openssl/sha.h"

namespace distributed_point_functions {

namespace {

// Update the hash state with new chunks of data.
//
void SHA256UpdateWithData(absl::string_view data, SHA256_CTX& ctx) {
  // truncate the input `data` into smaller chunks.
  constexpr static size_t chunk_size = 1 << 30;
  const char* ptr = data.data();
  size_t size = data.size();
  while (size > chunk_size) {
    SHA256_Update(&ctx, ptr, chunk_size);
    ptr += chunk_size;
    size -= chunk_size;
  }
  SHA256_Update(&ctx, ptr, size);
}

}  // namespace

SHA256HashFunction::SHA256HashFunction(absl::string_view seed) {
  // Initialize the OpenSSL SHA256 state.
  SHA256_Init(&ctx_);
  // Update the SHA256 state to compute the hash on the prefix `seed`.
  SHA256UpdateWithData(seed, ctx_);
}

SHA256HashFunction::~SHA256HashFunction() {
  // clear the OpenSSL state.
  memset(&ctx_, 0, sizeof(ctx_));
}

int SHA256HashFunction::operator()(absl::string_view input,
                                   int upper_bound) const {
  // Copy the default state on SHA256(seed).
  SHA256_CTX ctx = ctx_;
  // Compute the hash as SHA256(seed || input).
  SHA256UpdateWithData(input, ctx);
  // Finalize the SHA256 update and get the hash digest.
  char hash[SHA256_DIGEST_LENGTH];
  SHA256_Final(reinterpret_cast<unsigned char*>(hash), &ctx);

  // Get the lower and upper 128-bits of the hash.
  absl::uint128 hi, lo;
  constexpr size_t hi_offset = 16;
  std::copy(reinterpret_cast<unsigned char*>(&hash[hi_offset]),
            reinterpret_cast<unsigned char*>(&hash[hi_offset]) + hi_offset,
            reinterpret_cast<unsigned char*>(&hi));
  std::copy(reinterpret_cast<unsigned char*>(&hash[0]),
            reinterpret_cast<unsigned char*>(&hash[0]) + hi_offset,
            reinterpret_cast<unsigned char*>(&lo));
  // Long division using 64-bit "digits" and absl::uint128 builtin division.
  absl::uint128 dividend1 = hi;
  auto remainder1 = static_cast<uint64_t>(dividend1 % upper_bound);
  absl::uint128 dividend2 =
      absl::MakeUint128(remainder1, absl::Uint128High64(lo));
  auto remainder2 = static_cast<uint64_t>(dividend2 % upper_bound);
  absl::uint128 dividend3 =
      absl::MakeUint128(remainder2, absl::Uint128Low64(lo));
  return static_cast<int>(dividend3 % upper_bound);
}

}  // namespace distributed_point_functions
