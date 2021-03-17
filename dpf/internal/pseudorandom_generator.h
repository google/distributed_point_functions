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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_DPF_INTERNAL_PSEUDORANDOM_GENERATOR_H_
#define DISTRIBUTED_POINT_FUNCTIONS_DPF_INTERNAL_PSEUDORANDOM_GENERATOR_H_

#include <openssl/cipher.h>

#include "absl/numeric/int128.h"
#include "absl/status/statusor.h"

namespace private_statistics {
namespace dpf {
namespace dpf_internal {

// PseudorandomGenerator (PRG) based on AES. For key `key`, input `in` and
// output `out`, the PRG is defined as
//
//     out[i] = AES.Encrypt(key, sigma(in[i])) ^ sigma(in[i]),
//
// where sigma(x) = (x.high64 ^ x.low64, x.high64). This is the
// correlation-robust MMO construction from https://eprint.iacr.org/2019/074.pdf
// (pp. 18-19).
class PseudorandomGenerator {
 public:
  // Creates a new PseudorandomGenerator with the given `key`.
  //
  // Returns INTERNAL in case of allocation failures or OpenSSL errors.
  static absl::StatusOr<PseudorandomGenerator> Create(absl::uint128 key);

  // Computes pseudorandom values from `in` and writing the output to `out`.
  //
  // Returns INVALID_ARGUMENT if sizes of `in` and `out` don't match or their
  // sizes in bytes exceed an `int`, or INTERNAL in case of OpenSSL errors.
  absl::Status Evaluate(absl::Span<const absl::uint128> in,
                        absl::Span<absl::uint128> out) const;

  // PseudorandomGenerator is not copyable.
  PseudorandomGenerator(const PseudorandomGenerator&) = delete;
  PseudorandomGenerator& operator=(const PseudorandomGenerator&) = delete;

  // PseudorandomGenerator is movable (it just wraps a bssl::UniquePtr).
  PseudorandomGenerator(PseudorandomGenerator&&) = default;
  PseudorandomGenerator& operator=(PseudorandomGenerator&&) = default;

 private:
  // Called by `Create`.
  PseudorandomGenerator(bssl::UniquePtr<EVP_CIPHER_CTX> prg_ctx);

  // The OpenSSL encryption context used by `Evaluate`.
  bssl::UniquePtr<EVP_CIPHER_CTX> prg_ctx_;
};

}  // namespace dpf_internal
}  // namespace dpf
}  // namespace private_statistics

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DPF_INTERNAL_PSEUDORANDOM_GENERATOR_H_
