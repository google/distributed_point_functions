// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "absl/container/btree_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/random/random.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "benchmark/benchmark.h"  // third_party/benchmark
#include "dpf/distributed_point_function.h"
#include "dpf/distributed_point_function.pb.h"
#include "glog/logging.h"
#include "imap.hpp"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/lines/line_reading.h"

ABSL_FLAG(std::string, input, "",
          "CSV file containing non-zero buckets in the first column.");
ABSL_FLAG(int, log_domain_size, 20,
          "Logarithm of the domain size. All non-zeros in `input` must be in "
          "[0, 2^log_domain_size).");
ABSL_FLAG(int, num_iterations, 20, "Number of iterations to benchmark.");
ABSL_FLAG(int, max_expansion_factor, 2,
          "Limits the maximum number of elements the expansion at any "
          "hierarchy level can have to a multiple of the number of unique "
          "buckets in the input file. Must be at least 2.");
ABSL_FLAG(bool, only_nonzeros, false,
          "Only evaluates at the nonzero indices of the input file passed via "
          "--input, instead of performing hierarchical evaluation. If true, "
          "all flags related to hierarchy levels will be ignored");
ABSL_FLAG(std::vector<std::string>, levels_to_evaluate, {},
          "List of integers specifying the log domain sizes at which to insert "
          "hierarchy levels.");

#ifndef QCHECK
#define QCHECK(x) CHECK(x)
#endif

namespace {

const char* Usage() {
  return "synthetic_data_benchmarks [OPTIONS]\n\n"
         "Runs a single DPF key evaluation on the specified domain. If an "
         "input file is specified with --input, it is read as a CSV file "
         "containing the bucket IDs to expand in the first column. Otherwise, "
         "the full domain will be expanded.";
}

void ValidateFlags() {
  int log_domain_size = absl::GetFlag(FLAGS_log_domain_size);
  QCHECK(log_domain_size >= 0) << "--log_domain_size must be non-negative";
  int num_iterations = absl::GetFlag(FLAGS_num_iterations);
  QCHECK(num_iterations > 0) << "--num_iterations must be positive";
  if (absl::GetFlag(FLAGS_only_nonzeros)) {
    QCHECK(!absl::GetFlag(FLAGS_input).empty())
        << "--input is required when --only_nonzeros is true";
  }
  int max_expansion_factor = absl::GetFlag(FLAGS_max_expansion_factor);
  QCHECK(max_expansion_factor >= 2)
      << "--max_expansion_factor must be at least 2";
  std::vector<std::string> levels_to_evaluate =
      absl::GetFlag(FLAGS_levels_to_evaluate);
  for (absl::string_view level_str : levels_to_evaluate) {
    int level;
    QCHECK(absl::SimpleAtoi(level_str, &level));
    QCHECK(level > 0 && level <= log_domain_size)
        << "--levels_to_evaluate must be in [1, log_domain_size]";
  }
}

// Returns the prefixes of the given `last_level_prefixes` for each bit-length
// in {1, ..., `log_domain_size`}.
std::vector<std::vector<absl::uint128>> ComputePrefixes(
    const absl::btree_set<absl::uint128>& last_level_prefixes,
    int log_domain_size) {
  std::vector<std::vector<absl::uint128>> result(log_domain_size + 1);
  result.back() = std::vector<absl::uint128>(last_level_prefixes.begin(),
                                             last_level_prefixes.end());

  // Iterate backwards through previous levels, computing prefixes by
  // appropriately shifting the ones from higher levels.
  for (int i = static_cast<int>(result.size()) - 1; i > 1; --i) {
    absl::btree_set<absl::uint128> current_level_prefixes;
    for (const auto& x : result[i]) {
      current_level_prefixes.insert(x >> 1);
    }
    result[i - 1] = std::vector<absl::uint128>(current_level_prefixes.begin(),
                                               current_level_prefixes.end());
  }
  return result;
}

// Parses `input_file` as a CSV file and returns the unique integers in the
// first column as a set.
absl::btree_set<absl::uint128> ReadUniqueValuesFromFile(
    absl::string_view input_file) {
  absl::btree_set<absl::uint128> nonzeros;
  LOG(INFO) << "Reading input file...";
  int line_number = 0;
  riegeli::FdReader reader(input_file);
  absl::string_view line;
  while (riegeli::ReadLine(reader, line)) {
    std::vector<absl::string_view> fields =
        absl::StrSplit(line, ',', absl::SkipWhitespace());
    QCHECK(!fields.empty()) << "Line " << line_number << " is empty";
    absl::uint128 nonzero;
    QCHECK(absl::SimpleAtoi(fields[0], &nonzero))
        << "Invalid bucket ID on line " << line_number;
    nonzeros.insert(nonzero);
    ++line_number;
  }
  QCHECK(reader.ok());
  LOG(INFO) << "Read " << nonzeros.size() << " nonzeros from " << line_number
            << " lines";
  return nonzeros;
}

// Selects bit prefix lengths in {1, ..., `log_domain_size`}, such that for the
// given `prefixes` at each bit, the full domain evaluation from one level to
// the next never exceeds the last level size by a factor of more than
// `max_expansion_factor`.
std::vector<int> ComputeLevelsToEvaluate(
    absl::Span<const std::vector<absl::uint128>> prefixes, int log_domain_size,
    int max_expansion_factor) {
  int num_nonzeros = prefixes.back().size();
  CHECK_GT(num_nonzeros, 0);
  std::vector<int> levels_to_evaluate;
  // The first level is chosen such that it has size at most expansion_factor
  // * num_nonzeros.
  int first_level =
      std::min(log_domain_size,
               static_cast<int>(std::log2(num_nonzeros) +
                                std::log2(max_expansion_factor))) -
      1;
  levels_to_evaluate.push_back(first_level);
  while (levels_to_evaluate.back() < log_domain_size) {
    int nonzeros_at_last_level = prefixes[levels_to_evaluate.back() + 1].size();
    // We want to evaluate as many levels as possible so that we get no more
    // than expansion_factor * num_nonzeros. So 2^bits_to_next_level *
    // nonzeros_at_last_level < expansion_factor * num_nonzeros.
    levels_to_evaluate.push_back(std::min(
        log_domain_size,
        static_cast<int>(levels_to_evaluate.back() + std::log2(num_nonzeros) +
                         std::log2(max_expansion_factor) -
                         std::log2(nonzeros_at_last_level))));
  }
  return levels_to_evaluate;
}

// Evaluates the given `key` for `dpf` at each hierarchy level, using the given
// `prefixes` for each level. Repeats `num_iterations` times.
template <typename T>
void RunHierarchicalEvaluation(
    const distributed_point_functions::DistributedPointFunction& dpf,
    const distributed_point_functions::DpfKey& key,
    absl::Span<const std::vector<absl::uint128>> prefixes, int num_iterations) {
  const distributed_point_functions::EvaluationContext ctx =
      dpf.CreateEvaluationContext(key).value();
  CHECK_EQ(prefixes.size(), ctx.parameters_size());
  for (int i = 0; i < num_iterations; ++i) {
    distributed_point_functions::EvaluationContext ctx_copy = ctx;
    for (int level = 0; level < static_cast<int>(prefixes.size()); ++level) {
      std::vector<T> result =
          dpf.EvaluateUntil<T>(level, prefixes[level], ctx_copy).value();
      if (i == 0) {
        LOG(INFO) << "Number of outputs at " << level
                  << "-th level: " << result.size();
        LOG(INFO) << "log_domain_size="
                  << ctx.parameters(level).log_domain_size();
      }
      benchmark::DoNotOptimize(result);
    }
  }
}

// Evaluates the given `key` for `dpf` at the points in `nonzeros`, repeating
// `num_iterations` times.
template <typename T>
void RunBatchedSinglePointEvaluation(
    const distributed_point_functions::DistributedPointFunction& dpf,
    const distributed_point_functions::DpfKey& key,
    absl::Span<const absl::uint128> nonzeros, int num_iterations) {
  // Check that we have a single hierarchy level.
  CHECK_EQ(dpf.parameters().size(), 1);
  for (int i = 0; i < num_iterations; ++i) {
    std::vector<T> result = dpf.EvaluateAt<T>(key, 0, nonzeros).value();
    CHECK_EQ(result.size(), nonzeros.size());
    benchmark::DoNotOptimize(result);
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  absl::SetProgramUsageMessage(Usage());
  absl::ParseCommandLine(argc, argv);
  FLAGS_logtostderr = 1;
  ValidateFlags();

  // Read nonzeros from input file, compute prefixes,
  std::string input_file = absl::GetFlag(FLAGS_input);
  const int log_domain_size = absl::GetFlag(FLAGS_log_domain_size);
  std::vector<std::vector<absl::uint128>> prefixes(1);
  if (!input_file.empty()) {
    absl::btree_set<absl::uint128> nonzeros =
        ReadUniqueValuesFromFile(input_file);
    prefixes = ComputePrefixes(nonzeros, log_domain_size);
  }
  int num_nonzeros = prefixes.back().size();
  LOG(INFO) << "Number of nonzeros: " << num_nonzeros;

  // Compute levels to evaluate and choose the correct prefixes.
  std::vector<std::string> levels_to_evaluate_str =
      absl::GetFlag(FLAGS_levels_to_evaluate);
  std::vector<int> levels_to_evaluate(levels_to_evaluate_str.size());
  bool only_nonzeros = absl::GetFlag(FLAGS_only_nonzeros);
  for (int i = 0; i < static_cast<int>(levels_to_evaluate.size()); ++i) {
    CHECK(absl::SimpleAtoi(levels_to_evaluate_str[i], &levels_to_evaluate[i]));
  }
  if (levels_to_evaluate.empty()) {
    if (!only_nonzeros && !prefixes.back().empty()) {
      levels_to_evaluate = ComputeLevelsToEvaluate(
          prefixes, log_domain_size, absl::GetFlag(FLAGS_max_expansion_factor));
    } else {
      levels_to_evaluate = {log_domain_size};
    }
  }
  LOG(INFO) << "Levels to evaluate: " << absl::StrJoin(levels_to_evaluate, ",");
  std::vector<std::vector<absl::uint128>> prefixes_to_evaluate(1);
  prefixes_to_evaluate.reserve(levels_to_evaluate.size());
  for (int i = 1; i < levels_to_evaluate.size(); ++i) {
    prefixes_to_evaluate.push_back(prefixes[levels_to_evaluate[i - 1]]);
  }
  LOG(INFO) << "Numbers of prefixes per level: "
            << absl::StrJoin(iter::imap([](auto& c) { return c.size(); },
                                        prefixes_to_evaluate),
                             ",");
  LOG(INFO) << "Numbers of prefixes per bit: "
            << absl::StrJoin(
                   iter::imap([](auto& c) { return c.size(); }, prefixes), ",");

  // Set up parameters and create DPF instance.
  std::vector<distributed_point_functions::DpfParameters> parameters(
      levels_to_evaluate.size());
  const int element_bitsize = 32;  // TODO(schoppmann): Make this a flag?
  for (int i = 0; i < static_cast<int>(parameters.size()); ++i) {
    parameters[i].mutable_value_type()->mutable_integer()->set_bitsize(
        element_bitsize);
    parameters[i].set_log_domain_size(levels_to_evaluate[i]);
  }
  std::unique_ptr<distributed_point_functions::DistributedPointFunction> dpf =
      distributed_point_functions::DistributedPointFunction::CreateIncremental(
          parameters)
          .value();

  // Generate DPF key.
  absl::BitGen rng;
  absl::uint128 alpha = absl::MakeUint128(absl::Uniform<uint64_t>(rng),
                                          absl::Uniform<uint64_t>(rng));
  if (log_domain_size < 128) {
    alpha %= absl::uint128{1} << log_domain_size;
  }
  std::vector<absl::uint128> beta(parameters.size(), 1);
  distributed_point_functions::DpfKey key;
  std::tie(key, std::ignore) =
      dpf->GenerateKeysIncremental(alpha, beta).value();
  LOG(INFO) << "Key size: " << key.ByteSizeLong() << " bytes";

  // Run the experiment and measure time.
  int num_iterations = absl::GetFlag(FLAGS_num_iterations);
  using T = uint32_t;
  absl::Time start = absl::Now();
  if (only_nonzeros) {
    RunBatchedSinglePointEvaluation<T>(*dpf, key, prefixes.back(),
                                       num_iterations);
  } else {
    RunHierarchicalEvaluation<T>(*dpf, key, prefixes_to_evaluate,
                                 num_iterations);
  }
  absl::Duration wallclock = absl::Now() - start;
  LOG(INFO) << "Wallclock time per iteration: " << wallclock / num_iterations;
}
