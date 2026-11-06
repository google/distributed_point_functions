# DPF Microbenchmarks

This folder contains a binary for running DPF microbenchmarks for a two-party
sparse histogram aggregation. See below for usage and options. In the following,
we report the results on a fixed set of synthetic input files.

# Parameters

We fix the number of non-zero points in the entire histogram to 2<sup>20</sup>,
and choose three sets of non-zeros using the following three distributions:

1.  Power law with 90% of nonzeros in 10% of the domain
2.  Power law with 90% of nonzeros in 50% of the domain
3.  Uniform

We then evaluate in two evaluation modes:

1.  Hierarchical evaluation. Here, we assume the non-zeros are not known in
    advance, but instead have to be discovered during the evaluation. To
    simulate this using microbenchmarks, we identify a prefix hierarchy of the
    histogram domain, such that the full evaluation of each hierarchy level
    contains no more evaluation points than 4 times the number of non-zeros
    (i.e., 2<sup>22</sup>). We then perform a hierarchical DPF evaluation using
    the resulting hierarchy.
2.  Direct evaluation. Here, we assume that the set of non-zero indices is known
    in advance. Thus, we can do a direct DPF evaluation at the given set of
    non-zeros.

All entries in the tables below record the time needed to expand a single DPF
key at one of the two servers in the given setting. The evaluation is
single-threaded and runs on an Intel(R) Xeon(R) CPU @ 2.30GHz.

The size of the values being aggregated is fixed to 32 bits.

## Domain size: 2<sup>32</sup>

Prefix bit lengths to evaluate for hierarchical evaluation: 21,23,25,27,29,31,32

<table>
  <tr>
   <td>Distribution
   </td>
   <td>Key size
   </td>
   <td>1
   </td>
   <td>2
   </td>
   <td>3
   </td>
  </tr>
  <tr>
   <td>Hierarchical Evaluation
   </td>
   <td>1105 bytes
   </td>
   <td>1.36s
   </td>
   <td>2.22s
   </td>
   <td>3.31s
   </td>
  </tr>
  <tr>
   <td>Direct Evaluation
   </td>
   <td>883 bytes
   </td>
   <td>0.67s
   </td>
   <td>0.68s
   </td>
   <td>0.70s
   </td>
  </tr>
</table>

## Domain size: 2<sup>128</sup>

Levels to evaluate for hierarchical evaluation:
21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127,128

<table>
  <tr>
   <td>Distribution
   </td>
   <td>Key Size
   </td>
   <td>1
   </td>
   <td>2
   </td>
   <td>3
   </td>
  </tr>
  <tr>
   <td>Hierarchical Evaluation
   </td>
   <td>5603 bytes
   </td>
   <td>32.68s
   </td>
   <td>35.07s
   </td>
   <td>35.95s
   </td>
  </tr>
  <tr>
   <td>Direct Evaluation
   </td>
   <td>3504 bytes
   </td>
   <td>3.08s
   </td>
   <td>3.13s
   </td>
   <td>3.13s
   </td>
  </tr>
</table>



## Key Generation

We evaluate the efficiency of the DPF generation and the length of the DPF keys with _value_ bit length 32 bits and several different settings for the _index_ bit length. The keys are generated for non-hierarchical DPFs (i.e., for direct evaluation only).


<table>
  <tr>
   <td>Bit length
   </td>
   <td>20
   </td>
   <td>21
   </td>
   <td>22
   </td>
   <td>23
   </td>
   <td>24
   </td>
   <td>25
   </td>
   <td>26
   </td>
   <td>27
   </td>
   <td>28
   </td>
   <td>29
   </td>
   <td>30
   </td>
   <td>31
   </td>
   <td>32
   </td>
  </tr>
  <tr>
   <td>KeyGen (µs)
   </td>
   <td>6.9
   </td>
   <td>7.2
   </td>
   <td>7.3
   </td>
   <td>7.6
   </td>
   <td>7.8
   </td>
   <td>8.0
   </td>
   <td>8.3
   </td>
   <td>8.4
   </td>
   <td>8.5
   </td>
   <td>8.7
   </td>
   <td>9.0
   </td>
   <td>9.2
   </td>
   <td>9.6
   </td>
  </tr>
  <tr>
   <td>Size (Bytes)
   </td>
   <td>547
   </td>
   <td>575
   </td>
   <td>609
   </td>
   <td>637
   </td>
   <td>651
   </td>
   <td>680
   </td>
   <td>721
   </td>
   <td>737
   </td>
   <td>772
   </td>
   <td>797
   </td>
   <td>828
   </td>
   <td>839
   </td>
   <td>892
   </td>
  </tr>
</table>


## Full Domain Evaluation

<table>
  <tr>
   <td>Domain Size
   </td>
   <td>2<sup>10</sup>
   </td>
   <td>2<sup>15</sup>
   </td>
   <td>2<sup>20</sup>
   </td>
  </tr>
  <tr>
   <td>Time
   </td>
   <td>18.53µs
   </td>
   <td>511.24µs
   </td>
   <td>17.04ms
   </td>
  </tr>
</table>

## Reproducing the benchmarks

Usage:

```bash
bazel run --cxxopt=-std=c++17 -c opt --dynamic_mode=off experiments:synthetic_data_benchmarks -- [options]
```

Options:

```none
--input (CSV file containing non-zero buckets in the first column.);
  default: "";
--levels_to_evaluate (List of integers specifying the log domain sizes at
  which to insert hierarchy levels.); default: ;
--log_domain_size (Logarithm of the domain size. All non-zeros in `input`
  must be in [0, 2^log_domain_size).); default: 20;
--max_expansion_factor (Limits the maximum number of elements the expansion
  at any hierarchy level can have to a multiple of the number of unique
  buckets in the input file. Must be at least 2.); default: 2;
--num_iterations (Number of iterations to benchmark.); default: 20;
--only_nonzeros (Only evaluates at the nonzero indices of the input file
  passed via --input, instead of performing hierarchical evaluation. If
  true, all flags related to hierarchy levels will be ignored);
  default: false;
```
