# DPClass: An Effective but Concise Discriminative Patterns-Based Classification Framework
## Publication

* Jingbo Shang, Wenzhu Tong, Jian Peng, and Jiawei Han, "**[DPClass: An Effective but Concise Discriminative Patterns-Based Classification Framework](http://web.engr.illinois.edu/~shang7/papers/DPClass.pdf)**", In Proceedings of 2016 SIAM International Conference on Data Mining (SDM 2016), 2016

## Requirements

* g++ 4.8
* matlab

The current executables require OpenMP, which does not come by default on OS X. To be able to run it on OS X, follow <a href="http://stackoverflow.com/questions/20321988/error-enabling-openmp-ld-library-not-found-for-lgomp-and-clang-errors">this stackoverflow post</a>.

## Build

DPClass can be easily built by Makefile in the terminal.

```
$ make
```

## Running the code

You could execute the code in the following way:

```
./run_forward.sh dataset_name l2_norm_coefficient
./run_glmnet.sh dataset_name
```

Example:

```
./run_forward.sh adult 0.5
./run_glmnet.sh adult
```

## Parameters
All the parameters are located in run_forward.sh and run_glmnet.sh. Some important parameters are as follows.

```
TOPK=20
```
Top-k discriminative patterns.

```
MAX_DEPTH=6
```
The maximum number of conditions in a single pattern.

```
MIN_SUP=10
```
Minimum support threshold.

####For other parameters regarding each individual module, please check the corresponding cpp files.
