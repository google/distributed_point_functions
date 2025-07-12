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
    default_testonly = True,
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "mock_pir_database",
    srcs = ["mock_pir_database.cc"],
    hdrs = ["mock_pir_database.h"],
    deps = [
        "//pir:pir_database_interface",
        "@boringssl//:crypto",
        "@com_github_google_googletest//:gtest",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "mock_pir_client",
    hdrs = ["mock_pir_client.h"],
    deps = [
        "//pir:pir_client",
        "//pir:private_information_retrieval_cc_proto",
        "@com_github_google_googletest//:gtest",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "mock_pir_database_test",
    srcs = ["mock_pir_database_test.cc"],
    deps = [
        ":mock_pir_database",
        "//dpf/internal:status_matchers",
        "//pir:dense_dpf_pir_database",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "pir_selection_bits",
    srcs = ["pir_selection_bits.cc"],
    hdrs = ["pir_selection_bits.h"],
    deps = [
        "//pir:dense_dpf_pir_database",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "pir_selection_bits_test",
    srcs = ["pir_selection_bits_test.cc"],
    deps = [
        ":pir_selection_bits",
        "//dpf/internal:status_matchers",
        "//pir:dense_dpf_pir_database",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "encrypt_decrypt",
    srcs = ["encrypt_decrypt.cc"],
    hdrs = ["encrypt_decrypt.h"],
    deps = [
        "//dpf:status_macros",
        "//pir/testing/data:embedded_private_key",
        "//pir/testing/data:embedded_public_key",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tink_cc//tink:cleartext_keyset_handle_testonly",
        "@tink_cc//tink:hybrid_decrypt",
        "@tink_cc//tink:hybrid_encrypt",
        "@tink_cc//tink:keyset_handle",
        "@tink_cc//tink:keyset_reader",
        "@tink_cc//tink/hybrid:hybrid_config",
        "@tink_cc//tink/json:json_keyset_reader",
    ],
)

cc_test(
    name = "encrypt_decrypt_test",
    srcs = ["encrypt_decrypt_test.cc"],
    deps = [
        ":encrypt_decrypt",
        "//dpf/internal:status_matchers",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "request_generator",
    srcs = ["request_generator.cc"],
    hdrs = ["request_generator.h"],
    deps = [
        ":encrypt_decrypt",
        "//dpf:distributed_point_function",
        "//dpf:distributed_point_function_cc_proto",
        "//dpf:status_macros",
        "//pir:private_information_retrieval_cc_proto",
        "//pir/prng:aes_128_ctr_seeded_prng",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "mock_pir_server",
    hdrs = ["mock_pir_server.h"],
    deps = [
        "//pir:pir_server",
        "//pir:private_information_retrieval_cc_proto",
        "@com_github_google_googletest//:gtest",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "request_generator_test",
    srcs = ["request_generator_test.cc"],
    deps = [
        ":encrypt_decrypt",
        ":request_generator",
        "//dpf:distributed_point_function",
        "//dpf/internal:status_matchers",
        "//pir:private_information_retrieval_cc_proto",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
