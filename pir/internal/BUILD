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
load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

package(
    default_visibility = ["//:__subpackages__"],
)

licenses(["notice"])

cc_library(
    name = "inner_product_hwy",
    srcs = ["inner_product_hwy.cc"],
    hdrs = ["inner_product_hwy.h"],
    deps = [
        "//dpf:xor_wrapper",
        "//pir:canonical_status_payload_uris",
        "//pir:private_information_retrieval_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "inner_product_hwy_test",
    srcs = ["inner_product_hwy_test.cc"],
    deps = [
        ":inner_product_hwy",
        "//dpf/internal:status_matchers",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@highway//:hwy",
        "@highway//:hwy_test_util",
    ],
)

# Build and test :inner_product_hwy on platforms without vector intrinsics.
cc_library(
    name = "inner_product_hwy_scalar",
    srcs = ["inner_product_hwy.cc"],
    hdrs = ["inner_product_hwy.h"],
    local_defines = ["HWY_COMPILE_ONLY_SCALAR"],
    deps = [
        "//dpf:xor_wrapper",
        "//pir:canonical_status_payload_uris",
        "//pir:private_information_retrieval_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "inner_product_hwy_scalar_test",
    srcs = ["inner_product_hwy_test.cc"],
    local_defines = ["HWY_COMPILE_ONLY_SCALAR"],
    deps = [
        ":inner_product_hwy_scalar",
        "//dpf/internal:status_matchers",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@highway//:hwy",
        "@highway//:hwy_test_util",
    ],
)
