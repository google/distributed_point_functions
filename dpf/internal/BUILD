load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")
load("@com_github_google_iree//build_tools/embed_data:build_defs.bzl", "cc_embed_data")

package(
    default_visibility = ["//dpf:__subpackages__"],
)

licenses(["notice"])

cc_library(
    name = "array_conversions",
    hdrs = ["array_conversions.h"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/numeric:int128",
    ],
)

cc_test(
    name = "array_conversions_test",
    srcs = ["array_conversions_test.cc"],
    deps = [
        ":array_conversions",
        "@com_github_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "status_matchers",
    testonly = 1,
    srcs = [
        "status_matchers.cc",
    ],
    hdrs = ["status_matchers.h"],
    deps = [
        "//dpf:status_macros",
        "@com_github_google_googletest//:gtest",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "proto_validator",
    srcs = [
        "proto_validator.cc",
    ],
    hdrs = [
        "proto_validator.h",
    ],
    deps = [
        "//dpf:distributed_point_function_cc_proto",
        "//dpf:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_embed_data(
    name = "proto_validator_test_textproto_embed",
    srcs = [
        "proto_validator_test.textproto",
    ],
    cc_file_output = "proto_validator_test_textproto_embed.cc",
    cpp_namespace = "distributed_point_functions::dpf_internal",
    h_file_output = "proto_validator_test_textproto_embed.h",
)

cc_test(
    name = "proto_validator_test",
    srcs = [
        "proto_validator_test.cc",
    ],
    data = [
        "proto_validator_test.textproto",
    ],
    deps = [
        ":proto_validator",
        ":proto_validator_test_textproto_embed",
        ":status_matchers",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)
