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

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "aes_128_ctr_seeded_prng",
    srcs = ["aes_128_ctr_seeded_prng.cc"],
    hdrs = ["aes_128_ctr_seeded_prng.h"],
    deps = [
        "@boringssl//:crypto",
        "@com_google_absl//absl/base:config",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "aes_128_ctr_seeded_prng_test",
    srcs = ["aes_128_ctr_seeded_prng_test.cc"],
    deps = [
        ":aes_128_ctr_seeded_prng",
        "//dpf/internal:status_matchers",
        "@boringssl//:crypto",
        "@com_github_google_benchmark//:benchmark",
        "@com_github_google_googletest//:gtest_main",
    ],
)
