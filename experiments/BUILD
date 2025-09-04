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
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

cc_binary(
    name = "synthetic_data_benchmarks",
    srcs = ["synthetic_data_benchmarks.cc"],
    deps = [
        "//dpf:distributed_point_function",
        "//dpf:distributed_point_function_cc_proto",
        "@com_github_google_benchmark//:benchmark",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@cppitertools",
        "@riegeli//riegeli/bytes:fd_reader",
        "@riegeli//riegeli/lines:line_reading",
    ],
)
