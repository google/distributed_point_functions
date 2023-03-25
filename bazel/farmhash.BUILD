package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "farmhash",
    hdrs = ["farmhash/farmhash.h"],
    srcs = ["farmhash/farmhash.cc"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/numeric:int128",
    ],
)
