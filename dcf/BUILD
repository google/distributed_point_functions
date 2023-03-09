# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "distributed_comparison_function",
    srcs = ["distributed_comparison_function.cc"],
    hdrs = ["distributed_comparison_function.h"],
    deps = [
        ":distributed_comparison_function_cc_proto",
        "//dpf:distributed_point_function",
        "//dpf:distributed_point_function_cc_proto",
        "//dpf:status_macros",
        "@com_github_google_highway//:hwy",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status:statusor",
    ],
)

proto_library(
    name = "distributed_comparison_function_proto",
    srcs = ["distributed_comparison_function.proto"],
    deps = [
        "//dpf:distributed_point_function_proto",
    ],
)

cc_proto_library(
    name = "distributed_comparison_function_cc_proto",
    deps = [":distributed_comparison_function_proto"],
)

cc_test(
    name = "distributed_comparison_function_test",
    srcs = ["distributed_comparison_function_test.cc"],
    deps = [
        ":distributed_comparison_function",
        "//dpf/internal:status_matchers",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/utility",
    ],
)

cc_test(
    name = "distributed_comparison_function_benchmark",
    srcs = [
        "distributed_comparison_function_benchmark.cc",
    ],
    tags = ["benchmark"],
    deps = [
        ":distributed_comparison_function",
        ":distributed_comparison_function_cc_proto",
        "//dpf:distributed_point_function",
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/random",
    ],
)
