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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PRNG_BASIC_RNG_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PRNG_BASIC_RNG_H_

#include <openssl/rand.h>

#include "absl/memory/memory.h"
#include "absl/numeric/int128.h"
#include "absl/strings/string_view.h"
#include "dcf/fss_gates/prng/prng.h"

namespace distributed_point_functions {

// Basic RNG class that uses RAND_bytes from OpenSSL to sample randomness.
// BasicRng does not require a seed internally.
class BasicRng : public SecurePrng {
 public:
  // Create a BasicRng object.
  // Returns an INTERNAL error code if the creation fails.
  static absl::StatusOr<std::unique_ptr<BasicRng>> Create(
      absl::string_view seed) {
    return absl::make_unique<BasicRng>();
  }

  // Sample 8 bits of randomness using OpenSSL RAND_bytes.
  // Returns an INTERNAL error code if the sampling fails.
  absl::StatusOr<uint8_t> Rand8() override {
    unsigned char rand[1];
    int success = RAND_bytes(rand, 1);
    if (!success) {
      return absl::InternalError(
          "BasicRng::Rand8() - Failed to create randomness");
    }
    return static_cast<uint8_t>(rand[0]);
  }

  // Sample 64 bits of randomness using OPENSSL RAND_bytes.
  // Returns an INTERNAL error code if the sampling fails.
  absl::StatusOr<uint64_t> Rand64() override {
    unsigned char rand[8];
    int success = RAND_bytes(rand, 8);
    if (!success) {
      return absl::InternalError(
          "BasicRng::Rand64() - Failed to create randomness");
    }
    uint64_t rand_uint64 = 0;
    for (int i = 0; i < 8; i++) {
      rand_uint64 += static_cast<uint64_t>(rand[8 - i]) << (8 * i);
    }
    return rand_uint64;
  }

  // Sample 128 bits of randomness using OPENSSL RAND_bytes.
  // Returns an INTERNAL error code if the sampling fails.
  absl::StatusOr<absl::uint128> Rand128() override {
    unsigned char rand[16];
    int success = RAND_bytes(rand, 16);
    if (!success) {
      return absl::InternalError(
          "BasicRng::Rand128() - Failed to create randomness");
    }
    absl::uint128 rand_uint128 = 0;
    for (int i = 0; i < 16; i++) {
      rand_uint128 += static_cast<absl::uint128>(rand[8 - i]) << (8 * i);
    }
    return rand_uint128;
  }

  // BasicRng does not use seeds.
  static absl::StatusOr<std::string> GenerateSeed() { return std::string(); }
  static int SeedLength() { return 0; }
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PRNG_BASIC_RNG_H_
