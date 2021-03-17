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

#include "dpf/internal/pseudorandom_generator.h"

#include <vector>

namespace private_statistics {
namespace dpf {
namespace dpf_internal {

PseudorandomGenerator::PseudorandomGenerator(
    bssl::UniquePtr<EVP_CIPHER_CTX> prg_ctx)
    : prg_ctx_(std::move(prg_ctx)) {}

absl::StatusOr<PseudorandomGenerator> PseudorandomGenerator::Create(
    absl::uint128 key) {
  bssl::UniquePtr<EVP_CIPHER_CTX> prg_ctx(EVP_CIPHER_CTX_new());
  if (!prg_ctx) {
    return absl::InternalError("Failed to allocate AES context");
  }
  // Set up the OpenSSL encryption context. We want to evaluate the PRG in
  // parallel on many seeds (see class comment in pseudorandom_generator.h), so
  // we're using ECB mode here to achieve that. This batched evaluation is not
  // to be confused with encryption of an array, for which ECB would be
  // insecure.
  int openssl_status =
      EVP_EncryptInit_ex(prg_ctx.get(), EVP_aes_128_ecb(), nullptr,
                         reinterpret_cast<const uint8_t*>(&key), nullptr);
  if (openssl_status != 1) {
    return absl::InternalError("Failed to set up AES context");
  }
  return PseudorandomGenerator(std::move(prg_ctx));
}

absl::Status PseudorandomGenerator::Evaluate(
    absl::Span<const absl::uint128> in, absl::Span<absl::uint128> out) const {
  if (in.size() != out.size()) {
    return absl::InvalidArgumentError("Input and output sizes don't match");
  }
  if (in.empty()) {
    // Nothing to do.
    return absl::OkStatus();
  }
  if (static_cast<int64_t>(in.size() * sizeof(absl::uint128)) >
      static_cast<int64_t>(std::numeric_limits<int>::max())) {
    return absl::InvalidArgumentError(
        "`in` is too large: OpenSSL needs the size (in bytes) to fit in an "
        "int");
  }
  // Compute orthomorphism sigma for each element in `in`.
  std::vector<absl::uint128> sigma_in(in.size());
  for (int64_t i = 0; i < static_cast<int64_t>(in.size()); ++i) {
    sigma_in[i] = absl::MakeUint128(
        absl::Uint128High64(in[i]) ^ absl::Uint128Low64(in[i]),
        absl::Uint128High64(in[i]));
  }
  int out_len;
  int openssl_status = EVP_EncryptUpdate(
      prg_ctx_.get(), reinterpret_cast<uint8_t*>(out.data()), &out_len,
      reinterpret_cast<const uint8_t*>(sigma_in.data()),
      static_cast<int>(in.size() * sizeof(absl::uint128)));
  if (openssl_status != 1) {
    return absl::InternalError("AES encryption failed");
  }
  if (static_cast<size_t>(out_len) != out.size() * sizeof(absl::uint128)) {
    return absl::InternalError("OpenSSL output size does not match");
  }
  for (int64_t i = 0; i < static_cast<int64_t>(in.size()); ++i) {
    out[i] ^= sigma_in[i];
  }
  return absl::OkStatus();
}

}  // namespace dpf_internal
}  // namespace dpf
}  // namespace private_statistics
