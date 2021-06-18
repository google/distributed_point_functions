#include "dpf/internal/proto_validator.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/strings/str_format.h"
#include "dpf/internal/proto_validator_test_textproto_embed.h"
#include "dpf/internal/status_matchers.h"
#include "dpf/internal/value_type_helpers.h"
#include "dpf/tuple.h"
#include "google/protobuf/text_format.h"

namespace distributed_point_functions {
namespace dpf_internal {
namespace {

using ::testing::StartsWith;

class ProtoValidatorTest : public testing::Test {
 protected:
  void SetUp() override {
    const auto* const toc = proto_validator_test_textproto_embed_create();
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
        std::string(toc->data, toc->size), &ctx_));
    parameters_ = std::vector<DpfParameters>(ctx_.parameters().begin(),
                                             ctx_.parameters().end());
    dpf_key_ = ctx_.key();
    DPF_ASSERT_OK_AND_ASSIGN(proto_validator_,
                             ProtoValidator::Create(parameters_));
  }

  std::vector<DpfParameters> parameters_;
  DpfKey dpf_key_;
  EvaluationContext ctx_;
  std::unique_ptr<dpf_internal::ProtoValidator> proto_validator_;
};

TEST_F(ProtoValidatorTest, FailsWithoutParameters) {
  EXPECT_THAT(ProtoValidator::Create({}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`parameters` must not be empty"));
}

TEST_F(ProtoValidatorTest, FailsWhenParametersNotSorted) {
  parameters_.resize(2);
  parameters_[0].set_log_domain_size(10);
  parameters_[1].set_log_domain_size(8);

  EXPECT_THAT(ProtoValidator::Create(parameters_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`log_domain_size` fields must be in ascending order in "
                       "`parameters`"));
}

TEST_F(ProtoValidatorTest, FailsWhenDomainSizeNegative) {
  parameters_.resize(1);
  parameters_[0].set_log_domain_size(-1);

  EXPECT_THAT(ProtoValidator::Create(parameters_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`log_domain_size` must be non-negative"));
}

TEST_F(ProtoValidatorTest, FailsWhenElementBitsizeNegative) {
  parameters_.resize(1);
  parameters_[0].mutable_value_type()->mutable_integer()->set_bitsize(-1);

  EXPECT_THAT(ProtoValidator::Create(parameters_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`bitsize` must be positive"));
}

TEST_F(ProtoValidatorTest, FailsWhenElementBitsizeZero) {
  parameters_.resize(1);
  parameters_[0].mutable_value_type()->mutable_integer()->set_bitsize(0);

  EXPECT_THAT(ProtoValidator::Create(parameters_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`bitsize` must be positive"));
}

TEST_F(ProtoValidatorTest, FailsWhenElementBitsizeTooLarge) {
  parameters_.resize(1);
  parameters_[0].mutable_value_type()->mutable_integer()->set_bitsize(256);

  EXPECT_THAT(ProtoValidator::Create(parameters_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`bitsize` must be less than or equal to 128"));
}

TEST_F(ProtoValidatorTest, FailsWhenElementBitsizeNotAPowerOfTwo) {
  parameters_.resize(1);
  parameters_[0].mutable_value_type()->mutable_integer()->set_bitsize(23);

  EXPECT_THAT(ProtoValidator::Create(parameters_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`bitsize` must be a power of 2"));
}

TEST_F(ProtoValidatorTest, FailsWhenElementBitsizesDecrease) {
  parameters_.resize(2);
  parameters_[0].mutable_value_type()->mutable_integer()->set_bitsize(64);
  parameters_[1].mutable_value_type()->mutable_integer()->set_bitsize(32);

  EXPECT_THAT(ProtoValidator::Create(parameters_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`value_type` fields must be of non-decreasing size in "
                       "`parameters`"));
}

TEST_F(ProtoValidatorTest, FailsWhenHierarchiesAreTooFarApart) {
  parameters_.resize(2);
  parameters_[0].set_log_domain_size(10);
  parameters_[1].set_log_domain_size(73);

  EXPECT_THAT(ProtoValidator::Create(parameters_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Hierarchies may be at most 62 levels apart"));
}

TEST_F(ProtoValidatorTest, FailsIfNumberOfCorrectionWordsDoesntMatch) {
  dpf_key_.add_correction_words();

  EXPECT_THAT(proto_validator_->ValidateDpfKey(dpf_key_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       absl::StrCat("Malformed DpfKey: expected ",
                                    dpf_key_.correction_words_size() - 1,
                                    " correction words, but got ",
                                    dpf_key_.correction_words_size())));
}

TEST_F(ProtoValidatorTest, FailsIfSeedIsMissing) {
  dpf_key_.clear_seed();

  EXPECT_THAT(
      proto_validator_->ValidateDpfKey(dpf_key_),
      StatusIs(absl::StatusCode::kInvalidArgument, "key.seed must be present"));
}

TEST_F(ProtoValidatorTest, FailsIfLastLevelOutputCorrectionIsMissing) {
  dpf_key_.clear_last_level_value_correction();

  EXPECT_THAT(proto_validator_->ValidateDpfKey(dpf_key_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "key.last_level_value_correction must be present"));
}

TEST_F(ProtoValidatorTest, FailsIfOutputCorrectionIsMissing) {
  for (CorrectionWord& cw : *(dpf_key_.mutable_correction_words())) {
    cw.clear_value_correction();
  }

  EXPECT_THAT(
      proto_validator_->ValidateDpfKey(dpf_key_),
      StatusIs(absl::StatusCode::kInvalidArgument,
               StartsWith("Malformed DpfKey: expected correction_words")));
}

TEST_F(ProtoValidatorTest, FailsIfKeyIsMissing) {
  ctx_.clear_key();

  EXPECT_THAT(
      proto_validator_->ValidateEvaluationContext(ctx_),
      StatusIs(absl::StatusCode::kInvalidArgument, "ctx.key must be present"));
}

TEST_F(ProtoValidatorTest, FailsIfParameterSizeDoesntMatch) {
  ctx_.mutable_parameters()->erase(ctx_.parameters().end() - 1);

  EXPECT_THAT(proto_validator_->ValidateEvaluationContext(ctx_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Number of parameters in `ctx` doesn't match"));
}

TEST_F(ProtoValidatorTest, FailsIfParameterDoesntMatch) {
  ctx_.mutable_parameters(0)->set_log_domain_size(
      ctx_.parameters(0).log_domain_size() + 1);

  EXPECT_THAT(proto_validator_->ValidateEvaluationContext(ctx_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Parameter 0 in `ctx` doesn't match"));
}

TEST_F(ProtoValidatorTest, FailsIfContextFullyEvaluated) {
  ctx_.set_previous_hierarchy_level(parameters_.size() - 1);

  EXPECT_THAT(proto_validator_->ValidateEvaluationContext(ctx_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "This context has already been fully evaluated"));
}

TEST_F(ProtoValidatorTest, FailsIfPreviousHierarchyLevelEqualsHierarchyLevel) {
  ctx_.set_previous_hierarchy_level(ctx_.partial_evaluations_level());
  ctx_.add_partial_evaluations();

  EXPECT_THAT(proto_validator_->ValidateEvaluationContext(ctx_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "ctx.previous_hierarchy_level must be less than "
                       "ctx.partial_evaluations_level"));
}

TEST_F(ProtoValidatorTest, FailsIfTypeNotInteger) {
  ValueType type = GetValueTypeProtoFor<uint32_t>();
  Value value = ToValue(Tuple<uint32_t>{23});

  EXPECT_THAT(
      proto_validator_->ValidateValue(value, type),
      StatusIs(absl::StatusCode::kInvalidArgument, "Expected integer value"));
}

TEST_F(ProtoValidatorTest, FailsIfIntegerTooLarge) {
  ValueType type;
  Value value;

  int element_bitsize = 32;
  type.mutable_integer()->set_bitsize(element_bitsize);
  auto value_64 = uint64_t{1} << element_bitsize;
  value.mutable_integer()->set_value_uint64(value_64);

  EXPECT_THAT(
      proto_validator_->ValidateValue(value, type),
      StatusIs(absl::StatusCode::kInvalidArgument,
               absl::StrFormat(
                   "Value (= %d) too large for ValueType with bitsize = %d",
                   value_64, element_bitsize)));
}

TEST_F(ProtoValidatorTest, FailsIfTypeNotTuple) {
  ValueType type = GetValueTypeProtoFor<Tuple<uint32_t>>();
  Value value = ToValue(uint32_t{23});

  EXPECT_THAT(
      proto_validator_->ValidateValue(value, type),
      StatusIs(absl::StatusCode::kInvalidArgument, "Expected tuple value"));
}

TEST_F(ProtoValidatorTest, FailsIfTupleSizeDoesntMatch) {
  ValueType type = GetValueTypeProtoFor<Tuple<uint32_t>>();
  Value value = ToValue(Tuple<uint32_t, uint32_t>{23, 42});

  EXPECT_THAT(proto_validator_->ValidateValue(value, type),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Expected tuple value of size 1 but got size 2"));
}

}  // namespace
}  // namespace dpf_internal
}  // namespace distributed_point_functions
