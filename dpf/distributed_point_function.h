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

#ifndef DISTRIBUTED_POINT_FUNCTIONS_DPF_DISTRIBUTED_POINT_FUNCTION_H_
#define DISTRIBUTED_POINT_FUNCTIONS_DPF_DISTRIBUTED_POINT_FUNCTION_H_

#include <glog/logging.h>
#include <openssl/cipher.h>

#include <memory>
#include <type_traits>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/any.h"
#include "dpf/aes_128_fixed_key_hash.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/internal/proto_validator.h"
#include "dpf/internal/value_type_helpers.h"

namespace distributed_point_functions {

// Implements key generation and evaluation of distributed point functions.
// A distributed point function (DPF) is parameterized by an index `alpha` and a
// value `beta`. The key generation procedure produces two keys `k_a`, `k_b`.
// Evaluating each key on any point `x` in the DPF domain results in an additive
// secret share of `beta`, if `x == alpha`, and a share of 0 otherwise. This
// class also supports *incremental* DPFs that can additionally be evaluated on
// prefixes of points, resulting in different values `beta_i`for each prefix of
// `alpha`.
class DistributedPointFunction {
 public:
  // Creates a new instance of a distributed point function that can be
  // evaluated only at the output layer.
  //
  // Returns INVALID_ARGUMENT if the parameters are invalid.
  static absl::StatusOr<std::unique_ptr<DistributedPointFunction>> Create(
      const DpfParameters& parameters);

  // Creates a new instance of an *incremental* DPF that can be evaluated at
  // multiple layers. Each parameter set in `parameters` should specify the
  // domain size and element size at one of the layers to be evaluated, in
  // increasing domain size order. Element sizes must be non-decreasing.
  //
  // Returns INVALID_ARGUMENT if the parameters are invalid.
  static absl::StatusOr<std::unique_ptr<DistributedPointFunction>>
  CreateIncremental(absl::Span<const DpfParameters> parameters);

  // DistributedPointFunction is neither copyable nor movable.
  DistributedPointFunction(const DistributedPointFunction&) = delete;
  DistributedPointFunction& operator=(const DistributedPointFunction&) = delete;

  // Generates a pair of keys for a DPF that evaluates to `beta` when evaluated
  // `alpha`. `beta` may be any numeric type.
  //
  // Returns INVALID_ARGUMENT if used on an incremental DPF with more
  // than one set of parameters, if `alpha` is outside of the domain specified
  // at construction, or if `beta` does not match the element type passed at
  // construction.
  //
  absl::StatusOr<std::pair<DpfKey, DpfKey>> GenerateKeys(
      absl::uint128 alpha, absl::uint128 beta) const {
    return GenerateKeysIncremental(alpha, absl::MakeConstSpan(&beta, 1));
  }

  // Overload for types not convertible to absl::uint128.
  template <typename T, typename = std::enable_if_t<
                            !std::is_convertible_v<T, absl::uint128>>>
  absl::StatusOr<std::pair<DpfKey, DpfKey>> GenerateKeys(absl::uint128 alpha,
                                                         const T& beta) const {
    const absl::any beta_any = beta;
    return GenerateKeysIncremental(alpha, absl::MakeConstSpan(&beta_any, 1));
  }

  // Generates a pair of keys for an incremental DPF. For each parameter i
  // passed at construction, the DPF evaluates to `beta[i]` at the first
  // `parameters_[i].log_domain_size()` bits of `alpha`.
  //
  // `beta` must be a span of absl::uint128 values, or a span of unsigned
  // integers (wrapped in absl::any), where the integer at `beta[i]` matches the
  // size passed in `parameters[i]` at construction.
  //
  // Returns INVALID_ARGUMENT if `beta.size() != parameters_.size()`, if `alpha`
  // is outside of the domain specified at construction, or if `beta` does not
  // match the element type passed at construction.
  //
  absl::StatusOr<std::pair<DpfKey, DpfKey>> GenerateKeysIncremental(
      absl::uint128 alpha, absl::Span<const absl::any> beta) const;

  // Overload for absl::Span<const absl::uint128>.
  template <typename T,
            typename = std::enable_if_t<
                !std::is_convertible_v<T, absl::Span<const absl::any>> &&
                std::is_convertible_v<T, absl::Span<const absl::uint128>>>>
  absl::StatusOr<std::pair<DpfKey, DpfKey>> GenerateKeysIncremental(
      absl::uint128 alpha, const T& beta) const {
    absl::Span<const absl::uint128> beta_span(beta);
    std::vector<absl::any> beta_any(beta_span.begin(), beta_span.end());
    return GenerateKeysIncremental(alpha, beta_any);
  }

  // Returns an `EvaluationContext` for incrementally evaluating the given
  // DpfKey.
  //
  // Returns INVALID_ARGUMENT if `key` doesn't match the parameters given at
  // construction.
  absl::StatusOr<EvaluationContext> CreateEvaluationContext(DpfKey key);

  // Evaluates the given `hierarchy_level` of the DPF under all `prefixes`
  // passed to this function. If `prefixes` is empty, evaluation starts from the
  // seed of `ctx.key`. Otherwise, each element of `prefixes` must fit in the
  // domain size of `ctx.previous_hierarchy_level`. Further, `prefixes` may only
  // contain extensions of the prefixes passed in the previous call. For
  // example, in the following sequence of calls, for each element p2 of
  // `prefixes2`, there must be an element p1 of `prefixes1` such that p1 is a
  // prefix of p2:
  //
  //   DPF_ASSIGN_OR_RETURN(std::unique_ptr<EvaluationContext> ctx,
  //                        dpf->CreateEvaluationContext(key));
  //   using T0 = ...;
  //   DPF_ASSIGN_OR_RETURN(std::vector<T0> evaluations0,
  //                        dpf->EvaluateUntil(0, {}, *ctx));
  //
  //   std::vector<absl::uint128> prefixes1 = ...;
  //   using T1 = ...;
  //   DPF_ASSIGN_OR_RETURN(std::vector<T1> evaluations1,
  //                        dpf->EvaluateUntil(1, prefixes1, *ctx));
  //   ...
  //   std::vector<absl::uint128> prefixes2 = ...;
  //   using T2 = ...;
  //   DPF_ASSIGN_OR_RETURN(std::vector<T2> evaluations2,
  //                        dpf->EvaluateUntil(3, prefixes2, *ctx));
  //
  // The prefixes are read from the lowest-order bits of the corresponding
  // absl::uint128. The number of bits used for each prefix depends on the
  // output domain size of the previously evaluated hierarchy level. For
  // example, if `ctx` was last evaluated on a hierarchy level with output
  // domain size 2**20, then the 20 lowest-order bits of each element in
  // `prefixes` are used.
  //
  // Returns `INVALID_ARGUMENT` if
  //   - any element of `prefixes` is larger than the next hierarchy level's
  //     log_domain_size,
  //   - `prefixes` contains elements that are not extensions of previous
  //     prefixes, or
  //   - the bit-size of T doesn't match the next hierarchy level's
  //     element_bitsize.
  template <typename T>
  absl::StatusOr<std::vector<T>> EvaluateUntil(
      int hierarchy_level, absl::Span<const absl::uint128> prefixes,
      EvaluationContext& ctx) const;

  template <typename T>
  absl::StatusOr<std::vector<T>> EvaluateNext(
      absl::Span<const absl::uint128> prefixes, EvaluationContext& ctx) const {
    if (prefixes.empty()) {
      return EvaluateUntil<T>(0, prefixes, ctx);
    } else {
      return EvaluateUntil<T>(ctx.previous_hierarchy_level() + 1, prefixes,
                              ctx);
    }
  }

 private:
  // Private constructor, called by `CreateIncremental`.
  DistributedPointFunction(
      std::unique_ptr<dpf_internal::ProtoValidator> proto_validator,
      Aes128FixedKeyHash prg_left, Aes128FixedKeyHash prg_right,
      Aes128FixedKeyHash prg_value);

  // Registers the template parameter type with this DPF. Must be called before
  // generating any keys that use this value type.
  // For backwards compatibility, this function is called by `Create` and
  // `CreateIncremental` for all unsigned integer types, including
  // absl::uint128.
  //
  template <typename T>
  void RegisterValueType();

  // Computes the value correction for the given `hierarchy_level`, `seeds`,
  // index `alpha` and value `beta`. If `invert` is true, the individual values
  // in the returned block are multiplied element-wise by -1. Expands `seeds`
  // using `prg_ctx_value_`, then calls ComputeValueCorrectionFor<T> for the
  // right type depending on `parameters_[hierarchy_level].element_bitsize()`.
  //
  // Returns INTERNAL in case the PRG expansion fails, and UNIMPLEMENTED if
  // `element_bitsize` is not supported.
  absl::StatusOr<absl::uint128> ComputeValueCorrection(
      int hierarchy_level, absl::Span<const absl::uint128> seeds,
      absl::uint128 alpha, const absl::any& beta, bool invert) const;

  // Expands the PRG seeds at the next `tree_level` for an incremental DPF with
  // index `alpha` and values `beta`, updates `seeds` and `control_bits`, and
  // writes the next correction word to `keys`. Called from
  // `GenerateKeysIncremental`.
  absl::Status GenerateNext(int tree_level, absl::uint128 alpha,
                            absl::Span<const absl::any> beta,
                            absl::Span<absl::uint128> seeds,
                            absl::Span<bool> control_bits,
                            absl::Span<DpfKey> keys) const;

  // Checks if the parameters of `ctx` are compatible with this DPF. Returns OK
  // if that's the case, and INVALID_ARGUMENT otherwise.
  absl::Status CheckContextParameters(const EvaluationContext& ctx) const;

  // Computes the tree index (representing a path in the FSS tree) from the
  // given `domain_index` and `hierarchy_level`. Does NOT check whether the
  // given domain index fits in the domain at `hierarchy_level`.
  absl::uint128 DomainToTreeIndex(absl::uint128 domain_index,
                                  int hierarchy_level) const;

  // Computes the block index (pointing to an element in a batched 128-bit
  // block) from the given `domain_index` and `hierarchy_level`. Does NOT check
  // whether the given domain index fits in the domain at `hierarchy_level`.
  int DomainToBlockIndex(absl::uint128 domain_index, int hierarchy_level) const;

  // BitVector is a vector of bools. Allows for faster access times than
  // std::vector<bool>, as well as inlining if the size is small.
  using BitVector = absl::InlinedVector<bool, 8 / sizeof(bool)>;

  // Seeds and control bits resulting from a DPF expansion. This type is
  // returned by `ExpandSeeds` and `ExpandAndUpdateContext`.
  struct DpfExpansion {
    std::vector<absl::uint128> seeds;
    BitVector control_bits;
  };

  // Performs DPF evaluation of the given `partial_evaluations` using
  // prg_ctx_left_ or prg_ctx_right_, and the given `correction_words`. At each
  // level `l < correction_words.size()`, the evaluation for the i-th seed in
  // `partial_evaluations` continues along the left or right path depending on
  // the l-th most significant bit among the lowest `correction_words.size()`
  // bits of `paths[i]`.
  //
  // Returns INTERNAL in case of OpenSSL errors.
  absl::StatusOr<DpfExpansion> EvaluateSeeds(
      DpfExpansion partial_evaluations, absl::Span<const absl::uint128> paths,
      absl::Span<const CorrectionWord* const> correction_words) const;

  // Performs DPF expansion of the given `partial_evaluations` using
  // prg_ctx_left_ and prg_ctx_right_, and the given `correction_words`. In more
  // detail, each of the partial evaluations is subjected to a full subtree
  // expansion of `correction_words.size()` levels, and the concatenated result
  // is provided in the response. The result contains
  // `(partial_evaluations.size() * (2^correction_words.size())` evaluations in
  // a single `DpfExpansion`.
  //
  // Returns INTERNAL in case of OpenSSL errors.
  absl::StatusOr<DpfExpansion> ExpandSeeds(
      const DpfExpansion& partial_evaluations,
      absl::Span<const CorrectionWord* const> correction_words) const;

  // Computes partial evaluations of the paths to `prefixes` to be used as the
  // starting point of the expansion of `ctx`. If `update_ctx == true`, saves
  // the partial evaluations of `ctx.previous_hierarchy_level` to `ctx` and sets
  // `ctx.partial_evaluations_level` to `ctx.previous_hierarchy_level`.
  // Called by `ExpandAndUpdateContext`.
  //
  // Returns INVALID_ARGUMENT if any element of `prefixes` is not found in
  // `ctx.partial_evaluations()`, or `ctx.partial_evaluations()` contains
  // duplicate seeds.
  absl::StatusOr<DpfExpansion> ComputePartialEvaluations(
      absl::Span<const absl::uint128> prefixes, bool update_ctx,
      EvaluationContext& ctx) const;

  // Extracts the seeds for the given `prefixes` from `ctx` and expands them as
  // far as needed for the next hierarchy level. Returns the result as a
  // `DpfExpansion`. Called by `EvaluateUntil`, where the expanded seeds are
  // corrected to obtain output values.
  // After expansion, `ctx.hierarchy_level()` is increased. If this isn't the
  // last expansion, the expanded seeds are also saved in `ctx` for the next
  // expansion.
  //
  // Returns INVALID_ARGUMENT if any element of `prefixes` is not found in
  // `ctx.partial_evaluations()`, or `ctx.partial_evaluations()` contains
  // duplicate seeds. Returns INTERNAL in case of OpenSSL errors.
  absl::StatusOr<DpfExpansion> ExpandAndUpdateContext(
      int hierarchy_level, absl::Span<const absl::uint128> prefixes,
      EvaluationContext& ctx) const;

  // A function for computing value corrections. Used as return type in
  // `GetValueCorrectionFunction`.
  using ValueCorrectionFunction = std::function<absl::StatusOr<absl::uint128>(
      absl::Span<const absl::uint128>, int block_index, const absl::any&,
      bool)>;

  // Returns the value correction function for the given parameters.
  // For all value types except unsigned integers, these functions have to be
  // first registered using RegisterValueType<T>.
  //
  // Returns UNIMPLEMENTED if no matching function was registered.
  absl::StatusOr<ValueCorrectionFunction> GetValueCorrectionFunction(
      const DpfParameters& parameters) const;

  // Used to validate DpfKey and EvaluationContext protos.
  const std::unique_ptr<dpf_internal::ProtoValidator> proto_validator_;

  // DP parameters passed to the factory function. Contains the domain size and
  // element size for hierarchy level of the incremental DPF. Owned by
  // proto_validator_.
  const absl::Span<const DpfParameters> parameters_;

  // Number of levels in the evaluation tree. This is always less than or equal
  // to the largest log_domain_size in parameters_.
  const int tree_levels_needed_;

  // Maps levels of the FSS evaluation tree to hierarchy levels (i.e., elements
  // of parameters_).
  const absl::flat_hash_map<int, int>& tree_to_hierarchy_;

  // The inverse of tree_to_hierarchy_.
  const std::vector<int>& hierarchy_to_tree_;

  // Pseudorandom generator used for seed expansion (left and right), and value
  // correction. The PRG is defined by the concatenation of the following three
  // fixed-key hash functions
  const Aes128FixedKeyHash prg_left_;
  const Aes128FixedKeyHash prg_right_;
  const Aes128FixedKeyHash prg_value_;

  // Maps element bit sizes to functions computing a value correction word.
  // These functions be instantiations of
  // dpf_internal::ComputeValueCorrectionFor.
  absl::flat_hash_map<int, ValueCorrectionFunction> value_correction_functions_;
};

//========================//
// Implementation Details //
//========================//

template <typename T>
void DistributedPointFunction::RegisterValueType() {
  auto bit_size = static_cast<int>(sizeof(T)) * 8;
  value_correction_functions_[bit_size] =
      dpf_internal::ComputeValueCorrectionFor<T>;
}

template <typename T>
absl::StatusOr<std::vector<T>> DistributedPointFunction::EvaluateUntil(
    int hierarchy_level, absl::Span<const absl::uint128> prefixes,
    EvaluationContext& ctx) const {
  if (absl::Status status = proto_validator_->ValidateEvaluationContext(ctx);
      !status.ok()) {
    return status;
  }
  if (hierarchy_level < 0 ||
      hierarchy_level >= static_cast<int>(parameters_.size())) {
    return absl::InvalidArgumentError(
        "`hierarchy_level` must be non-negative and less than "
        "parameters_.size()");
  }
  if (sizeof(T) * 8 != parameters_[hierarchy_level].element_bitsize()) {
    return absl::InvalidArgumentError(
        "Size of template parameter T doesn't match the element size of "
        "`hierarchy_level`");
  }
  if (hierarchy_level <= ctx.previous_hierarchy_level()) {
    return absl::InvalidArgumentError(
        "`hierarchy_level` must be greater than "
        "`ctx.previous_hierarchy_level`");
  }
  if ((ctx.previous_hierarchy_level() < 0) != (prefixes.empty())) {
    return absl::InvalidArgumentError(
        "`prefixes` must be empty if and only if this is the first call with "
        "`ctx`.");
  }

  int previous_log_domain_size = 0;
  int previous_hierarchy_level = ctx.previous_hierarchy_level();
  if (!prefixes.empty()) {
    DCHECK(ctx.previous_hierarchy_level() >= 0);
    previous_log_domain_size =
        parameters_[previous_hierarchy_level].log_domain_size();
    for (absl::uint128 prefix : prefixes) {
      if (previous_log_domain_size < 128 &&
          prefix > (absl::uint128{1} << previous_log_domain_size)) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Index %d out of range for hierarchy level %d",
                            prefix, previous_hierarchy_level));
      }
    }
  }
  int64_t prefixes_size = static_cast<int64_t>(prefixes.size());

  // The `prefixes` passed in by the caller refer to the domain of the previous
  // hierarchy level. However, because we batch multiple elements of type T in a
  // single uint128 block, multiple prefixes can actually refer to the same
  // block in the FSS evaluation tree. On a high level, our approach is as
  // follows:
  //
  // 1. Split up each element of `prefixes` into a tree index, pointing to a
  //    block in the FSS tree, and a block index, pointing to an element of type
  //    T in that block.
  //
  // 2. Compute a list of unique `tree_indices`, and for each original prefix,
  //    remember the position of the corresponding tree index in `tree_indices`.
  //
  // 3. After expanding the unique `tree_indices`, use the positions saved in
  //    Step (2) together with the corresponding block index to retrieve the
  //    expanded values for each prefix, and return them in the same order as
  //    `prefixes`.
  //
  // `tree_indices` holds the unique tree indices from `prefixes`, to be passed
  // to `ExpandAndUpdateContext`.
  std::vector<absl::uint128> tree_indices;
  tree_indices.reserve(prefixes_size);
  // `tree_indices_inverse` is the inverse of `tree_indices`, used for
  // deduplicating and constructing `prefix_map`. Use a btree_map because we
  // expect `prefixes` (and thus `tree_indices`) to be sorted.
  absl::btree_map<absl::uint128, int64_t> tree_indices_inverse;
  // `prefix_map` maps each i < prefixes.size() to an element of `tree_indices`
  // and a block index. Used to select which elements to return after the
  // expansion, to ensure the result is ordered the same way as `prefixes`.
  std::vector<std::pair<int64_t, int>> prefix_map;
  prefix_map.reserve(prefixes_size);
  for (int64_t i = 0; i < prefixes_size; ++i) {
    absl::uint128 tree_index =
        DomainToTreeIndex(prefixes[i], previous_hierarchy_level);
    int block_index = DomainToBlockIndex(prefixes[i], previous_hierarchy_level);

    // Check if `tree_index` already exists in `tree_indices`.
    int64_t previous_size = tree_indices_inverse.size();
    auto it = tree_indices_inverse.try_emplace(tree_indices_inverse.end(),
                                               tree_index, tree_indices.size());
    if (tree_indices_inverse.size() > previous_size) {
      tree_indices.push_back(tree_index);
    }
    prefix_map.push_back(std::make_pair(it->second, block_index));
  }

  // Perform expansion of unique `tree_indices`.
  absl::StatusOr<DpfExpansion> expansion =
      ExpandAndUpdateContext(hierarchy_level, tree_indices, ctx);
  if (!expansion.ok()) {
    return expansion.status();
  }

  // Get output correction word from `ctx`.
  constexpr int elements_per_block = dpf_internal::ElementsPerBlock<T>();
  const Block* output_correction = nullptr;
  if (hierarchy_level < static_cast<int>(parameters_.size()) - 1) {
    output_correction =
        &(ctx.key()
              .correction_words(hierarchy_to_tree_[hierarchy_level])
              .output());
  } else {
    // Last level output correction is stored in an extra proto field, since we
    // have one less correction word than tree levels.
    output_correction = &(ctx.key().last_level_output_correction());
  }

  // Split output correction into elements of type T.
  std::array<T, elements_per_block> correction_ints =
      dpf_internal::Uint128ToArray<T>(absl::MakeUint128(
          output_correction->high(), output_correction->low()));

  // Compute output PRG value of expanded seeds using prg_ctx_value_.
  std::vector<absl::uint128> hashed_expansion(expansion->seeds.size());
  if (absl::Status status = prg_value_.Evaluate(
          expansion->seeds, absl::MakeSpan(hashed_expansion));
      !status.ok()) {
    return status;
  }

  // Compute value corrections for each block in `expanded_seeds`. We have to
  // account for the fact that blocks might not be full (i.e., have less than
  // elements_per_block elements).
  const int corrected_elements_per_block =
      1 << (parameters_[hierarchy_level].log_domain_size() -
            hierarchy_to_tree_[hierarchy_level]);
  std::vector<T> corrected_expansion(hashed_expansion.size() *
                                     corrected_elements_per_block);
  for (int64_t i = 0; i < static_cast<int64_t>(hashed_expansion.size()); ++i) {
    std::array<T, elements_per_block> current_elements =
        dpf_internal::Uint128ToArray<T>(hashed_expansion[i]);
    for (int j = 0; j < corrected_elements_per_block; ++j) {
      if (expansion->control_bits[i]) {
        current_elements[j] += correction_ints[j];
      }
      if (ctx.key().party() == 1) {
        current_elements[j] = -current_elements[j];
      }
      corrected_expansion[i * corrected_elements_per_block + j] =
          current_elements[j];
    }
  }

  // Compute the number of outputs we will have. For each prefix, we will have a
  // full expansion from the previous heirarchy level to the current heirarchy
  // level.
  int log_domain_size = parameters_[hierarchy_level].log_domain_size();
  DCHECK(log_domain_size - previous_log_domain_size < 63);
  int64_t outputs_per_prefix = int64_t{1}
                               << (log_domain_size - previous_log_domain_size);

  if (prefixes.empty()) {
    // If prefixes is empty (i.e., this is the first evaluation of `ctx`), just
    // return the expansion.
    DCHECK(static_cast<int>(corrected_expansion.size()) == outputs_per_prefix);
    return corrected_expansion;
  } else {
    // Otherwise, only return elements under `prefixes`.
    int blocks_per_tree_prefix = expansion->seeds.size() / tree_indices.size();
    std::vector<T> result(prefixes_size * outputs_per_prefix);
    for (int64_t i = 0; i < prefixes_size; ++i) {
      int64_t prefix_expansion_start =
          prefix_map[i].first * blocks_per_tree_prefix *
              corrected_elements_per_block +
          prefix_map[i].second * outputs_per_prefix;
      std::copy_n(&corrected_expansion[prefix_expansion_start],
                  outputs_per_prefix, &result[i * outputs_per_prefix]);
    }
    return result;
  }
}

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DPF_DISTRIBUTED_POINT_FUNCTION_H_
