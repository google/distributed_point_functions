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
load("@com_github_google_iree//build_tools/embed_data:build_defs.bzl", "cc_embed_data")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//pir/testing:__subpackages__"],
)

licenses(["notice"])

cc_embed_data(
    name = "embedded_private_key",
    srcs = [
        "hybrid_test_private_keyset.json",
    ],
    cc_file_output = "embedded_private_key.cc",
    h_file_output = "embedded_private_key.h",
)

cc_embed_data(
    name = "embedded_public_key",
    srcs = [
        "hybrid_test_public_keyset.json",
    ],
    cc_file_output = "embedded_public_key.cc",
    h_file_output = "embedded_public_key.h",
)
