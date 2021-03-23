load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package_group(
    name = "allowlist",
    packages = [
        "//dpf/...",
    ],
)

package(
    default_visibility = [":allowlist"],
)

licenses(["notice"])

cc_library(
    name = "distributed_point_function",
    srcs = ["distributed_point_function.cc"],
    hdrs = ["distributed_point_function.h"],
    deps = [
        ":distributed_point_function_cc_proto",
        "//dpf:status_macros",
        "//dpf/internal:array_conversions",
        "//dpf/internal:pseudorandom_generator",
        "@boringssl//:crypto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "distributed_point_function_test",
    size = "small",
    srcs = ["distributed_point_function_test.cc"],
    deps = [
        ":distributed_point_function",
        "//dpf/internal:status_matchers",
        "@com_github_google_googletest//:gtest_main",
    ],
)

proto_library(
    name = "distributed_point_function_proto",
    srcs = ["distributed_point_function.proto"],
)

cc_proto_library(
    name = "distributed_point_function_cc_proto",
    deps = [":distributed_point_function_proto"],
)

go_proto_library(
    name = "distributed_point_function_go_proto",
    deps = [":distributed_point_function_proto"],
)

cc_test(
    name = "distributed_point_function_benchmark",
    srcs = [
        "distributed_point_function_benchmark.cc",
    ],
    deps = [
        ":distributed_point_function",
        "@com_github_google_benchmark//:benchmark_main",
    ],
)

cc_library(
    name = "status_macros",
    hdrs = ["status_macros.h"],
)