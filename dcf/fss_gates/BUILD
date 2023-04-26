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

# This package contains implementation of various Function Secret Sharing (FSS)
# gates as specified in https://eprint.iacr.org/2020/1392. The implementation
# uses the Distributed Comparison Function (as implemented in
# distributed_comparison_function.cc) as a central component.

load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Multiple Interval Containment

cc_library(
    name = "multiple_interval_containment",
    srcs = ["multiple_interval_containment.cc"],
    hdrs = ["multiple_interval_containment.h"],
    deps = [
        ":multiple_interval_containment_cc_proto",
        "//dcf:distributed_comparison_function",
        "//dcf/fss_gates/prng:basic_rng",
        "//dpf:distributed_point_function_cc_proto",
        "//dpf:status_macros",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

proto_library(
    name = "multiple_interval_containment_proto",
    srcs = ["multiple_interval_containment.proto"],
    deps = [
        "//dcf:distributed_comparison_function_proto",
        "//dpf:distributed_point_function_proto",
    ],
)

cc_proto_library(
    name = "multiple_interval_containment_cc_proto",
    deps = [":multiple_interval_containment_proto"],
)

cc_test(
    name = "multiple_interval_containment_test",
    srcs = ["multiple_interval_containment_test.cc"],
    deps = [
        ":multiple_interval_containment",
        ":multiple_interval_containment_cc_proto",
        "//dcf/fss_gates/prng:basic_rng",
        "//dpf:distributed_point_function_cc_proto",
        "//dpf:status_macros",
        "//dpf/internal:status_matchers",
        "//dpf/internal:value_type_helpers",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)
