/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_PRNG_AES_128_CTR_SEEDED_PRNG_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_PRNG_AES_128_CTR_SEEDED_PRNG_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "openssl/aes.h"

namespace distributed_point_functions {

// A seeded PRNG based on AES128 using CTR mode.
class Aes128CtrSeededPrng {
 public:
  // Creates an AES128-CTR based PRNG seeded with the supplied `seed`. The
  // sequence of bytes generated is deterministic for any given `seed`. It is
  // the caller's responsibility to ensure it is never used to mask different
  // plaintexts.
  //
  // Fails with INVALID_ARGUMENT if `seed` is not SeedSize() bytes long.
  // Fails with INTERNAL if any crypto operations fail.
  static absl::StatusOr<std::unique_ptr<Aes128CtrSeededPrng>> Create(
      absl::string_view seed);

  // Creates an AES128-CTR based PRNG seeded with the supplied `seed` and
  // `nonce`. The sequence of bytes generated is deterministic for any given
  // pair of `seed` and `nonce`. This factory function may be used to obtain
  // multiple generators with the same `seed` (but different nonces), each
  // generating independent pseudorandom outputs.
  //
  // Note that the nonce is used as the initial counter in AES128-CTR (ivec_)
  // and increased after every 16 generated bytes. To ensure that counters never
  // take the same value twice, it is highly recommended to generate nonces
  // randomly.
  //
  // Fails with INVALID_ARGUMENT if `seed` and `nonce` are not both SeedSize()
  // bytes long.
  // Fails with INTERNAL if any crypto operations fail.
  static absl::StatusOr<std::unique_ptr<Aes128CtrSeededPrng>> CreateWithNonce(
      absl::string_view seed, absl::string_view nonce);

  // Size of an AES128 key, in bytes.
  static constexpr size_t SeedSize() { return kAes128KeySize; }

  // Returns a cryptographically random string of SeedSize() bytes.
  static absl::StatusOr<std::string> GenerateSeed();

  // Generates `length` pseudorandom bytes. The sequence of bytes is
  // deterministic for a PRNG created with a given seed, independent of how
  // often this function is called. In the following example, the strings s1 and
  // s2 are identical:
  //
  //   ASSIGN_OR_RETURN(std::string seed, Aes128CtrSeededPrng::GenerateSeed());
  //   ASSIGN_OR_RETURN(std::unique_ptr<Aes128CtrSeededPrng> prng_1,
  //                    Aes128CtrSeededPrng::Create(seed));
  //   std::string s1a = prng->GetRandomBytes(length);
  //   std::string s1b = prng->GetRandomBytes(length);
  //   std::string s1 = absl::StrCat(s1a, s1b);
  //
  //   ASSIGN_OR_RETURN(std::unique_ptr<Aes128CtrSeededPrng> prng_2,
  //                    Aes128CtrSeededPrng::Create(seed));
  //   std::string s2 = prng_2->GetRandomBytes(2 * length);
  //
  std::string GetRandomBytes(size_t length);

 private:
  // Called by `Create` and `CreateWithNonce`.
  Aes128CtrSeededPrng(AES_KEY aes_key, std::vector<uint8_t> ivec,
                      std::vector<uint8_t> ecount_buf);

  // The size of an AES-128 key.
  static constexpr size_t kAes128KeySize = 16;

  // The AES key used for encryption. Corresponds to the key expansion of `seed`
  // passed at construction.
  const AES_KEY aes_key_;
  // State of the stream cipher. Will be updated with every call to
  // `GetRandomBytes`.
  //
  std::vector<uint8_t> ivec_;
  std::vector<uint8_t> ecount_buf_;
  unsigned int num_;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_PRNG_AES_128_CTR_SEEDED_PRNG_H_
