# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "key_generation_protocol",
    srcs = ["key_generation_protocol.cc"],
    hdrs = ["key_generation_protocol.h"],
    deps = [
        ":key_generation_protocol_cc_proto",
        "//dcf/fss_gates/prng:basic_rng",
        "//dpf:distributed_point_function",
        "//dpf:distributed_point_function_cc_proto",
        "//dpf:status_macros",
        "//dpf/internal:evaluate_prg_hwy",
        "//dpf/internal:get_hwy_mode",
        "//dpf/internal:proto_validator",
        "//dpf/internal:value_type_helpers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "key_generation_protocol_test",
    srcs = ["key_generation_protocol_test.cc"],
    deps = [
        ":key_generation_protocol",
        "//dcf/fss_gates/prng:basic_rng",
        "//dpf:distributed_point_function_cc_proto",
        "//dpf/internal:evaluate_prg_hwy",
        "//dpf/internal:get_hwy_mode",
        "//dpf/internal:proto_validator",
        "//dpf/internal:status_matchers",
        "//dpf/internal:value_type_helpers",
        "@com_github_google_googletest//:gtest_main",
    ],
)

cc_proto_library(
    name = "key_generation_protocol_cc_proto",
    deps = [":key_generation_protocol_proto"],
)

proto_library(
    name = "key_generation_protocol_proto",
    srcs = ["key_generation_protocol.proto"],
    deps = ["//dpf:distributed_point_function_proto"],
)

proto_library(
    name = "key_generation_protocol_rpc_proto",
    srcs = [":key_generation_protocol_rpc.proto"],
    deps = [
        ":key_generation_protocol_proto",
    ],
)
cc_proto_library(
    name = "key_generation_protocol_rpc_cc_proto",
    deps = [
        ":key_generation_protocol_rpc_proto",
		]
)
cc_grpc_library(
    name = "key_generation_protocol_rpc_grpc_proto",
    srcs = [":key_generation_protocol_rpc_proto"],
    grpc_only = True,
    deps = [":key_generation_protocol_rpc_cc_proto"],
)


cc_library(
    name = "key_generation_protocol_rpc_impl",
    srcs = ["key_generation_protocol_rpc_impl.cc"],
    hdrs = [
        "key_generation_protocol_rpc_impl.h",
    ],
    deps = [
        ":key_generation_protocol",
        ":key_generation_protocol_cc_proto",
        ":key_generation_protocol_rpc_grpc_proto",
        "@com_github_grpc_grpc//:grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_binary(
    name = "key_generation_protocol_server",
    srcs = ["key_generation_protocol_server.cc"],
    deps = [
        ":key_generation_protocol",
        ":key_generation_protocol_rpc_impl",
        ":key_generation_protocol_rpc_grpc_proto",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_github_grpc_grpc//:grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_binary(
    name = "key_generation_protocol_client",
    srcs = ["key_generation_protocol_client.cc"],
    deps = [
        ":key_generation_protocol",
        ":key_generation_protocol_cc_proto",
        ":key_generation_protocol_rpc_grpc_proto",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_github_grpc_grpc//:grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)


