load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")

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
        "@com_github_google_glog//:glog",
        "@com_github_google_riegeli//riegeli/bytes:fd_reader",
        "@com_github_google_riegeli//riegeli/lines:line_reading",
        "@com_github_ryanhaining_cppitertools//:cppitertools",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)
