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
load("@rules_license//rules:license.bzl", "license")
load("//testing/fuzzing/build_defs:guitar.bzl", "fuzztest_ci_workflows")

package(
    default_visibility = [":allowlist"],
)

license(
    name = "license",
    package_name = "distributed_point_functions",
)

licenses(["notice"])

exports_files(["LICENSE"])

# http://go/fuzztest-setting-up-ci
fuzztest_ci_workflows(
    name = "fuzz_testing_workflows",
    # Privacy > PIE > Private Statistics > Distributed Point Functions
    buganizer_component_id = 1044360,
    metadata = ":METADATA",
    notification_email = "fss-dev+fuzztest@google.com",
    project_name = "distributed_point_functions",
)
