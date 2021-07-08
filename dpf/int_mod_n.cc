#include "dpf/int_mod_n.h"

namespace distributed_point_functions {

double IntModNBase::GetSecurityLevel(int num_samples, absl::uint128 modulus) {
  return 128 + 3 -
         (std::log2(static_cast<double>(modulus)) +
          std::log2(static_cast<double>(num_samples)) +
          std::log2(static_cast<double>(num_samples + 1)));
}

absl::Status IntModNBase::CheckParameters(int num_samples,
                                          int base_integer_bitsize,
                                          absl::uint128 modulus,
                                          double security_parameter) {
  if (num_samples <= 0) {
    return absl::InvalidArgumentError("num_samples must be positive");
  }
  if (base_integer_bitsize <= 0) {
    return absl::InvalidArgumentError("base_integer_bitsize must be positive");
  }
  if (base_integer_bitsize > 128) {
    return absl::InvalidArgumentError(
        "base_integer_bitsize must be at most 128");
  }
  if (base_integer_bitsize < 128 &&
      (absl::uint128{1} << base_integer_bitsize) < modulus) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "kModulus %d out of range for base_integer_bitsize = %d", modulus,
        base_integer_bitsize));
  }

  // Compute the level of security that we will get, and fail if it is
  // insufficient.
  const double sigma = GetSecurityLevel(num_samples, modulus);
  if (security_parameter > sigma) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "For num_samples = %d and kModulus = %d this approach can only "
        "provide "
        "%f bits of statistical security. You can try calling this function "
        "several times with smaller values of num_samples.",
        num_samples, modulus, sigma));
  }
  return absl::OkStatus();
}

absl::StatusOr<int> IntModNBase::GetNumBytesRequired(
    int num_samples, int base_integer_bitsize, absl::uint128 modulus,
    double security_parameter) {
  if (absl::Status status = CheckParameters(num_samples, base_integer_bitsize,
                                            modulus, security_parameter);
      !status.ok()) {
    return status;
  }

  const int base_integer_bytes = ((base_integer_bitsize + 7) / 8);
  // We start the sampling by requiring a 128-bit (16 bytes) block, see
  // function `SampleFromBytes`.
  return 16 + base_integer_bytes * (num_samples - 1);
}

}  // namespace distributed_point_functions
