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
    name = "pir_client",
    hdrs = ["pir_client.h"],
    deps = [
        ":private_information_retrieval_cc_proto",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "pir_server",
    hdrs = ["pir_server.h"],
    deps = [
        ":private_information_retrieval_cc_proto",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_proto_library(
    name = "private_information_retrieval_cc_proto",
    deps = [":private_information_retrieval_proto"],
)

proto_library(
    name = "private_information_retrieval_proto",
    srcs = ["private_information_retrieval.proto"],
    deps = [
        "//dpf:distributed_point_function_proto",
        "//pir/hashing:hash_family_config_proto",
    ],
)

# Database interface.
cc_library(
    name = "pir_key_value_pair",
    hdrs = ["pir_key_value_pair.h"],
)

cc_library(
    name = "pir_database_interface",
    hdrs = ["pir_database_interface.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

# Dense database to be used with DPF PIR
cc_library(
    name = "dense_dpf_pir_database",
    srcs = ["dense_dpf_pir_database.cc"],
    hdrs = ["dense_dpf_pir_database.h"],
    deps = [
        ":pir_database_interface",
        "//dpf:status_macros",
        "//dpf:xor_wrapper",
        "//pir/internal:inner_product_hwy",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "dense_dpf_pir_database_test",
    srcs = ["dense_dpf_pir_database_test.cc"],
    deps = [
        ":dense_dpf_pir_database",
        "//dpf/internal:status_matchers",
        "//pir/testing:mock_pir_database",
        "//pir/testing:pir_selection_bits",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "dense_dpf_pir_database_benchmark",
    srcs = ["dense_dpf_pir_database_benchmark.cc"],
    tags = ["benchmark"],
    deps = [
        ":dense_dpf_pir_database",
        "//dpf/internal:status_matchers",
        "//pir/testing:mock_pir_database",
        "//pir/testing:pir_selection_bits",
        "@com_github_google_benchmark//:benchmark",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dense_dpf_pir_server",
    srcs = ["dense_dpf_pir_server.cc"],
    hdrs = ["dense_dpf_pir_server.h"],
    deps = [
        ":dpf_pir_server",
        ":pir_database_interface",
        ":private_information_retrieval_cc_proto",
        "//dpf:distributed_point_function",
        "//dpf:status_macros",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "dense_dpf_pir_server_test",
    srcs = ["dense_dpf_pir_server_test.cc"],
    deps = [
        ":dense_dpf_pir_server",
        ":private_information_retrieval_cc_proto",
        "//dpf:distributed_point_function",
        "//dpf:distributed_point_function_cc_proto",
        "//dpf:xor_wrapper",
        "//dpf/internal:status_matchers",
        "//pir/testing:encrypt_decrypt",
        "//pir/testing:mock_pir_database",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "dense_dpf_pir_server_benchmark",
    srcs = ["dense_dpf_pir_server_benchmark.cc"],
    tags = ["benchmark"],
    deps = [
        ":dense_dpf_pir_database",
        ":dense_dpf_pir_server",
        "//dpf/internal:status_matchers",
        "//pir/testing:mock_pir_database",
        "//pir/testing:request_generator",
        "@com_github_google_benchmark//:benchmark",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dense_dpf_pir_client",
    srcs = ["dense_dpf_pir_client.cc"],
    hdrs = ["dense_dpf_pir_client.h"],
    deps = [
        ":dense_dpf_pir_server",
        ":dpf_pir_client",
        ":private_information_retrieval_cc_proto",
        "//dpf:distributed_point_function",
        "//dpf:distributed_point_function_cc_proto",
        "//dpf:status_macros",
        "//pir/prng:aes_128_ctr_seeded_prng",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "dense_dpf_pir_client_test",
    srcs = ["dense_dpf_pir_client_test.cc"],
    deps = [
        ":dense_dpf_pir_client",
        ":dense_dpf_pir_database",
        ":dense_dpf_pir_server",
        ":private_information_retrieval_cc_proto",
        "//dpf/internal:status_matchers",
        "//pir/prng:aes_128_ctr_seeded_prng",
        "//pir/testing:encrypt_decrypt",
        "//pir/testing:mock_pir_database",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dpf_pir_server",
    srcs = ["dpf_pir_server.cc"],
    hdrs = ["dpf_pir_server.h"],
    deps = [
        ":pir_server",
        ":private_information_retrieval_cc_proto",
        "//dpf:status_macros",
        "//pir/prng:aes_128_ctr_seeded_prng",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
        "@tink_cc//:hybrid_decrypt",
    ],
)

cc_test(
    name = "dpf_pir_server_test",
    srcs = ["dpf_pir_server_test.cc"],
    deps = [
        ":dense_dpf_pir_database",
        ":dpf_pir_server",
        ":private_information_retrieval_cc_proto",
        "//dpf:status_macros",
        "//dpf/internal:status_matchers",
        "//pir/prng:aes_128_ctr_seeded_prng",
        "//pir/testing:encrypt_decrypt",
        "//pir/testing:mock_pir_database",
        "//pir/testing:request_generator",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "cuckoo_hashed_dpf_pir_database",
    srcs = ["cuckoo_hashed_dpf_pir_database.cc"],
    hdrs = ["cuckoo_hashed_dpf_pir_database.h"],
    deps = [
        ":dense_dpf_pir_database",
        ":pir_database_interface",
        ":private_information_retrieval_cc_proto",
        "//dpf:status_macros",
        "//dpf:xor_wrapper",
        "//pir/hashing:cuckoo_hash_table",
        "//pir/hashing:hash_family",
        "//pir/hashing:hash_family_config",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "cuckoo_hashed_dpf_pir_database_test",
    srcs = ["cuckoo_hashed_dpf_pir_database_test.cc"],
    deps = [
        ":cuckoo_hashed_dpf_pir_database",
        ":dense_dpf_pir_database",
        ":private_information_retrieval_cc_proto",
        "//dpf:xor_wrapper",
        "//dpf/internal:status_matchers",
        "//pir/hashing:hash_family_config_cc_proto",
        "//pir/testing:mock_pir_database",
        "//pir/testing:pir_selection_bits",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "cuckoo_hashing_sparse_dpf_pir_server",
    srcs = ["cuckoo_hashing_sparse_dpf_pir_server.cc"],
    hdrs = ["cuckoo_hashing_sparse_dpf_pir_server.h"],
    deps = [
        ":dpf_pir_server",
        ":pir_database_interface",
        ":private_information_retrieval_cc_proto",
        "//dpf:distributed_point_function",
        "//dpf:distributed_point_function_cc_proto",
        "//dpf:status_macros",
        "//dpf:xor_wrapper",
        "//pir/hashing:hash_family_config_cc_proto",
        "//pir/hashing:sha256_hash_family",
        "@boringssl//:crypto",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "cuckoo_hashing_sparse_dpf_pir_server_test",
    srcs = ["cuckoo_hashing_sparse_dpf_pir_server_test.cc"],
    deps = [
        ":cuckoo_hashed_dpf_pir_database",
        ":cuckoo_hashing_sparse_dpf_pir_server",
        ":private_information_retrieval_cc_proto",
        "//dpf/internal:status_matchers",
        "//pir/hashing:hash_family",
        "//pir/hashing:hash_family_config",
        "//pir/hashing:hash_family_config_cc_proto",
        "//pir/testing:encrypt_decrypt",
        "//pir/testing:mock_pir_database",
        "//pir/testing:request_generator",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "cuckoo_hashing_sparse_dpf_pir_server_benchmark",
    srcs = ["cuckoo_hashing_sparse_dpf_pir_server_benchmark.cc"],
    tags = ["benchmark"],
    deps = [
        ":cuckoo_hashed_dpf_pir_database",
        ":cuckoo_hashing_sparse_dpf_pir_server",
        "//dpf/internal:status_matchers",
        "//pir/hashing:hash_family",
        "//pir/hashing:hash_family_config",
        "//pir/hashing:hash_family_config_cc_proto",
        "//pir/testing:mock_pir_database",
        "//pir/testing:request_generator",
        "@com_github_google_benchmark//:benchmark",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "cuckoo_hashing_sparse_dpf_pir_client",
    srcs = ["cuckoo_hashing_sparse_dpf_pir_client.cc"],
    hdrs = ["cuckoo_hashing_sparse_dpf_pir_client.h"],
    deps = [
        ":cuckoo_hashing_sparse_dpf_pir_server",
        ":dense_dpf_pir_client",
        ":dpf_pir_client",
        ":private_information_retrieval_cc_proto",
        "//dpf:status_macros",
        "//pir/hashing:hash_family",
        "//pir/hashing:hash_family_config",
        "//pir/hashing:sha256_hash_family",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dpf_pir_client",
    hdrs = ["dpf_pir_client.h"],
    deps = [
        ":pir_client",
        ":private_information_retrieval_cc_proto",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "cuckoo_hashing_sparse_dpf_pir_client_test",
    srcs = ["cuckoo_hashing_sparse_dpf_pir_client_test.cc"],
    deps = [
        ":cuckoo_hashed_dpf_pir_database",
        ":cuckoo_hashing_sparse_dpf_pir_client",
        ":cuckoo_hashing_sparse_dpf_pir_server",
        ":private_information_retrieval_cc_proto",
        "//dpf/internal:status_matchers",
        "//pir/hashing:hash_family_config_cc_proto",
        "//pir/testing:encrypt_decrypt",
        "//pir/testing:mock_pir_database",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@tink_cc//:hybrid_encrypt",
    ],
)

cc_library(
    name = "simple_hashed_dpf_pir_database",
    srcs = ["simple_hashed_dpf_pir_database.cc"],
    hdrs = ["simple_hashed_dpf_pir_database.h"],
    deps = [
        ":dense_dpf_pir_database",
        ":pir_database_interface",
        ":private_information_retrieval_cc_proto",
        "//dpf:status_macros",
        "//dpf:xor_wrapper",
        "//pir/hashing:hash_family",
        "//pir/hashing:hash_family_config",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "simple_hashed_dpf_pir_database_test",
    srcs = ["simple_hashed_dpf_pir_database_test.cc"],
    deps = [
        ":private_information_retrieval_cc_proto",
        ":simple_hashed_dpf_pir_database",
        "//dpf:xor_wrapper",
        "//dpf/internal:status_matchers",
        "//pir/hashing:hash_family",
        "//pir/hashing:hash_family_config",
        "//pir/hashing:hash_family_config_cc_proto",
        "//pir/testing:mock_pir_database",
        "//pir/testing:pir_selection_bits",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_protobuf//:protobuf",
    ],
)
