#ifndef DISTRIBUTED_POINT_FUNCTIONS_DPF_INTERNAL_PROTO_VALIDATOR_H_
#define DISTRIBUTED_POINT_FUNCTIONS_DPF_INTERNAL_PROTO_VALIDATOR_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "dpf/distributed_point_function.pb.h"

namespace distributed_point_functions {
namespace dpf_internal {

// ProtoValidator is used to validate protos for DPF parameters, keys, and
// evaluation contexts. Also holds information computed from the DPF parameters,
// such as the mappings between hierarchy and tree levels.
class ProtoValidator {
 public:
  // Checks the validity of `parameters` and returns a ProtoValidator, which
  // will be used to validate DPF keys and evaluation contexts afterwards.
  //
  // Returns INVALID_ARGUMENT if `parameters` are invalid.
  static absl::StatusOr<std::unique_ptr<ProtoValidator>> Create(
      absl::Span<const DpfParameters> parameters);

  // Checks the validity of `parameters`.
  // Returns OK on success, and INVALID_ARGUMENT otherwise.
  static absl::Status ValidateParameters(
      absl::Span<const DpfParameters> parameters);

  // Checks that `key` is valid for the `parameters` passed at construction.
  // Returns OK on success, and INVALID_ARGUMENT otherwise.
  absl::Status ValidateDpfKey(const DpfKey& key) const;

  // Checks that `ctx` is valid for the `parameters` passed at construction.
  // Returns OK on success, and INVALID_ARGUMENT otherwise.
  absl::Status ValidateEvaluationContext(const EvaluationContext& ctx) const;

  // ProtoValidator is not copyable.
  ProtoValidator(const ProtoValidator&) = delete;
  ProtoValidator& operator=(const ProtoValidator&) = delete;

  // ProtoValidator is movable.
  ProtoValidator(ProtoValidator&&) = default;
  ProtoValidator& operator=(ProtoValidator&&) = default;

  // Getters.
  absl::Span<const DpfParameters> parameters() const { return parameters_; }
  int tree_levels_needed() const { return tree_levels_needed_; }
  const absl::flat_hash_map<int, int>& tree_to_hierarchy() const {
    return tree_to_hierarchy_;
  }
  const std::vector<int>& hierarchy_to_tree() const {
    return hierarchy_to_tree_;
  }

 private:
  ProtoValidator(std::vector<DpfParameters> parameters, int tree_levels_needed,
                 absl::flat_hash_map<int, int> tree_to_hierarchy,
                 std::vector<int> hierarchy_to_tree);

  // The DpfParameters passed at construction.
  std::vector<DpfParameters> parameters_;

  // Number of levels in the evaluation tree. This is always less than or equal
  // to the largest log_domain_size in parameters_.
  int tree_levels_needed_;

  // Maps levels of the FSS evaluation tree to hierarchy levels (i.e., elements
  // of parameters_).
  absl::flat_hash_map<int, int> tree_to_hierarchy_;

  // The inverse of tree_to_hierarchy_.
  std::vector<int> hierarchy_to_tree_;
};

}  // namespace dpf_internal
}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DPF_INTERNAL_PROTO_VALIDATOR_H_