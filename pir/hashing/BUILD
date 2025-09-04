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

load("@com_google_protobuf//bazel:java_lite_proto_library.bzl", "java_lite_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "cuckoo_hash_table",
    srcs = [
        "cuckoo_hash_table.cc",
    ],
    hdrs = [
        "cuckoo_hash_table.h",
    ],
    deps = [
        ":hash_family",
        "//dpf:status_macros",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hash_family",
    srcs = [
        "hash_family.cc",
    ],
    hdrs = [
        "hash_family.h",
    ],
    deps = [
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "multiple_choice_hash_table",
    srcs = [
        "multiple_choice_hash_table.cc",
    ],
    hdrs = [
        "multiple_choice_hash_table.h",
    ],
    deps = [
        ":hash_family",
        "//dpf:status_macros",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "simple_hash_table",
    srcs = [
        "simple_hash_table.cc",
    ],
    hdrs = [
        "simple_hash_table.h",
    ],
    deps = [
        ":hash_family",
        "//dpf:status_macros",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "cuckoo_hash_table_test",
    size = "medium",
    srcs = [
        "cuckoo_hash_table_test.cc",
    ],
    deps = [
        ":cuckoo_hash_table",
        ":farm_hash_family",
        ":sha256_hash_family",
        "//dpf/internal:status_matchers",
        "@com_github_google_benchmark//:benchmark",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "hash_family_test",
    size = "medium",
    srcs = [
        "hash_family_test.cc",
    ],
    deps = [
        ":hash_family",
        "//dpf/internal:status_matchers",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "multiple_choice_hash_table_test",
    size = "medium",
    srcs = [
        "multiple_choice_hash_table_test.cc",
    ],
    deps = [
        ":farm_hash_family",
        ":multiple_choice_hash_table",
        ":sha256_hash_family",
        "//dpf/internal:status_matchers",
        "@com_github_google_benchmark//:benchmark",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "simple_hash_table_test",
    size = "medium",
    srcs = [
        "simple_hash_table_test.cc",
    ],
    deps = [
        ":farm_hash_family",
        ":sha256_hash_family",
        ":simple_hash_table",
        "//dpf/internal:status_matchers",
        "@com_github_google_benchmark//:benchmark",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "sha256_hash_family",
    srcs = ["sha256_hash_family.cc"],
    hdrs = ["sha256_hash_family.h"],
    deps = [
        ":hash_family",
        "@boringssl//:crypto",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "sha256_hash_family_test",
    srcs = ["sha256_hash_family_test.cc"],
    deps = [
        ":sha256_hash_family",
        "@com_github_google_benchmark//:benchmark",
        "@com_github_google_googletest//:gtest_main",
        "@com_github_google_shell-encryption//shell_encryption:int256",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "farm_hash_family",
    srcs = ["farm_hash_family.cc"],
    hdrs = ["farm_hash_family.h"],
    deps = [
        ":hash_family",
        "@com_github_google_farmhash//:farmhash",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "farm_hash_family_test",
    srcs = ["farm_hash_family_test.cc"],
    deps = [
        ":farm_hash_family",
        "@com_github_google_benchmark//:benchmark",
        "@com_github_google_farmhash//:farmhash",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
    ],
)

cc_proto_library(
    name = "hash_family_config_cc_proto",
    deps = [":hash_family_config_proto"],
)

java_lite_proto_library(
    name = "hash_family_config_java_proto_lite",
    deps = [":hash_family_config_proto"],
)

proto_library(
    name = "hash_family_config_proto",
    srcs = ["hash_family_config.proto"],
)

cc_library(
    name = "hash_family_config",
    srcs = ["hash_family_config.cc"],
    hdrs = ["hash_family_config.h"],
    deps = [
        ":hash_family",
        ":hash_family_config_cc_proto",
        ":sha256_hash_family",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "hash_family_config_test",
    srcs = ["hash_family_config_test.cc"],
    deps = [
        ":hash_family",
        ":hash_family_config",
        ":hash_family_config_cc_proto",
        ":sha256_hash_family",
        "//dpf/internal:status_matchers",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)
