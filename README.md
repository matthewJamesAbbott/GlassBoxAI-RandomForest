# GlassBoxAI-RandomForest

## **Random Forest Suite**

### *GPU-Accelerated Random Forest Implementations with Formal Verification*

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCL](https://img.shields.io/badge/OpenCL-3.0-blue.svg)](https://www.khronos.org/opencl/)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Kani](https://img.shields.io/badge/Kani-Verified-brightgreen.svg)](https://model-checking.github.io/kani/)
[![CISA Compliant](https://img.shields.io/badge/CISA-Secure%20by%20Design-blue.svg)](https://www.cisa.gov/securebydesign)

---

## **Overview**

GlassBoxAI-RandomForest is a comprehensive, production-ready Random Forest implementation suite featuring:

- **Multiple GPU backends**: CUDA and OpenCL acceleration
- **Multiple language implementations**: C++ and Rust
- **Facade pattern architecture**: Clean API separation with deep introspection capabilities
- **Formal verification**: Kani-verified Rust implementation for memory safety guarantees
- **CISA/NSA Secure by Design compliance**: Built following government cybersecurity standards

This project demonstrates enterprise-grade software engineering practices including comprehensive testing, formal verification, cross-platform compatibility, and security-first development.

---

## **Table of Contents**

1. [Features](#features)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Prerequisites](#prerequisites)
5. [Installation & Compilation](#installation--compilation)
6. [CLI Reference](#cli-reference)
   - [Standard Random Forest Commands](#standard-random-forest-commands)
   - [Facade Random Forest Commands](#facade-random-forest-commands)
7. [Testing](#testing)
8. [Formal Verification with Kani](#formal-verification-with-kani)
9. [CISA/NSA Compliance](#cisansa-compliance)
10. [License](#license)
11. [Author](#author)

---

## **Features**

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Classification** | Multi-class classification with majority voting |
| **Regression** | Continuous value prediction with mean aggregation |
| **Split Criteria** | Gini impurity, Entropy, MSE, Variance Reduction |
| **Bootstrap Sampling** | Random sampling with replacement for tree diversity |
| **Feature Subsampling** | Random feature selection at each split |
| **Out-of-Bag Error** | Built-in validation using OOB samples |
| **Feature Importance** | Impurity-based feature ranking |
| **Model Persistence** | Binary serialization for model save/load |

### GPU Acceleration

| Backend | Implementation | Performance |
|---------|---------------|-------------|
| **CUDA** | Native CUDA kernels | Optimal for NVIDIA GPUs |
| **OpenCL** | Cross-platform GPU | AMD, Intel, NVIDIA support |

### Safety & Security

| Feature | Technology |
|---------|------------|
| **Memory Safety** | Rust ownership model |
| **Formal Verification** | Kani proof harnesses |
| **Bounds Checking** | Verified array access |
| **Input Validation** | CLI argument validation |

---

## **Architecture**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      GlassBoxAI-RandomForest                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │   C++ CUDA  │  │ C++ OpenCL  │  │         Rust CUDA           │  │
│  ├─────────────┤  ├─────────────┤  ├─────────────────────────────┤  │
│  │ • random_   │  │ • random_   │  │ • rust_cuda/                │  │
│  │   forest.cu │  │   forest_   │  │ • facaded_rust_cuda/        │  │
│  │ • facaded_  │  │   opencl.cpp│  │   └─ kani/ (Formal          │  │
│  │   random_   │  │ • facaded_  │  │      Verification)          │  │
│  │   forest.cu │  │   random_   │  │                             │  │
│  │             │  │   forest_   │  │                             │  │
│  │             │  │   opencl.cpp│  │                             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                     Shared Features                             ││
│  │  • Consistent CLI interface across all implementations          ││
│  │  • Binary compatible model formats                              ││
│  │  • Comprehensive test suites                                    ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## **File Structure**

```
GlassBoxAI-RandomForest/
│
├── random_forest.cu                  # C++ CUDA Random Forest implementation
├── random_forest_opencl.cpp          # C++ OpenCL Random Forest implementation
├── facaded_random_forest.cu          # C++ CUDA Random Forest with Facade pattern
├── facaded_random_forest_opencl.cpp  # C++ OpenCL Random Forest with Facade pattern
│
├── rust_cuda/                        # Rust CUDA Random Forest implementation
│   ├── Cargo.toml
│   └── src/
│       └── main.rs
│
├── facaded_rust_cuda/                # Rust CUDA Random Forest with Facade pattern
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       └── kani/                     # Kani proof harnesses
│
├── random_forest_cpp_tests.sh        # C++ test suite
├── random_forest_opencl_tests.sh     # OpenCL test suite
│
├── license.md                        # MIT License
└── README.md                         # This file
```

---

## **Prerequisites**

### Required

| Dependency | Version | Purpose |
|------------|---------|---------|
| **GCC/G++** | 11+ | C++ compilation |
| **CUDA Toolkit** | 12.0+ | CUDA compilation |
| **Rust** | 1.75+ | Rust compilation |

### Optional

| Dependency | Version | Purpose |
|------------|---------|---------|
| **OpenCL SDK** | 3.0 | OpenCL compilation |
| **Kani** | 0.67+ | Formal verification |

---

## **Installation & Compilation**

### **C++ CUDA Implementation**

```bash
# Standard Random Forest
nvcc -O2 -o forest_cuda random_forest.cu

# Facade Random Forest
nvcc -O2 -o forest_facade_cuda facaded_random_forest.cu
```

### **C++ OpenCL Implementation**

```bash
# Standard Random Forest
g++ -O2 -std=c++14 -o forest_opencl random_forest_opencl.cpp -lOpenCL

# Facade Random Forest
g++ -O2 -std=c++14 -o forest_facade_opencl facaded_random_forest_opencl.cpp -lOpenCL
```

### **Rust CUDA Implementation**

```bash
# Standard Random Forest
cd rust_cuda
cargo build --release

# Facade Random Forest
cd facaded_rust_cuda
cargo build --release
```

### **Build All**

```bash
# Build everything
nvcc -O2 -o forest_cuda random_forest.cu
nvcc -O2 -o forest_facade_cuda facaded_random_forest.cu
g++ -O2 -std=c++14 -o forest_opencl random_forest_opencl.cpp -lOpenCL
g++ -O2 -std=c++14 -o forest_facade_opencl facaded_random_forest_opencl.cpp -lOpenCL
(cd rust_cuda && cargo build --release)
(cd facaded_rust_cuda && cargo build --release)
```

---

## **CLI Reference**

### **Standard Random Forest Commands**

The standard Random Forest implementations provide core ensemble learning functionality.

#### Usage

```
forest_cuda <command> [options]
forest_opencl <command> [options]
rust_cuda/target/release/forest_cuda <command> [options]
```

#### Commands

| Command | Description |
|---------|-------------|
| `create` | Create a new Random Forest model |
| `train` | Train a Random Forest model |
| `predict` | Make predictions with a trained model |
| `info` | Display model information |
| `help` | Show help message |

#### Create Options

| Option | Description |
|--------|-------------|
| `--trees=N` | Number of trees (default: 100) |
| `--max-depth=N` | Maximum tree depth (default: 10) |
| `--min-leaf=N` | Minimum samples per leaf (default: 1) |
| `--min-split=N` | Minimum samples to split (default: 2) |
| `--max-features=N` | Maximum features per split (default: sqrt(n)) |
| `--criterion=C` | Split criterion: gini, entropy, mse, variancereduction |
| `--task=T` | Task type: classification, regression |
| `--save=FILE` | Save model to file (required) |

#### Train Options

| Option | Description |
|--------|-------------|
| `--model=FILE` | Model file to load (required) |
| `--data=FILE` | Training data CSV file (required) |
| `--save=FILE` | Save trained model to file (required) |

#### Predict Options

| Option | Description |
|--------|-------------|
| `--model=FILE` | Model file to load (required) |
| `--data=FILE` | Data file for predictions (required) |
| `--output=FILE` | Save predictions to file (optional) |

#### Standard Examples

```bash
# Create a new model
forest_cuda create --trees=50 --max-depth=15 --save=model.bin

# Train the model
forest_cuda train --model=model.bin --data=train.csv --save=model_trained.bin

# Make predictions
forest_cuda predict --model=model_trained.bin --data=test.csv --output=predictions.csv

# Get model information
forest_cuda info --model=model_trained.bin
```

---

### **Facade Random Forest Commands**

The facade implementations provide deep introspection and tree manipulation capabilities.

#### Usage

```
forest_facade_cuda <command> [options]
forest_facade_opencl <command> [options]
facaded_rust_cuda/target/release/forest_facade <command> [options]
```

#### Core Commands

| Command | Description |
|---------|-------------|
| `create` | Create a new empty forest model |
| `train` | Train a random forest model |
| `predict` | Make predictions using a trained model |
| `evaluate` | Evaluate model on test data |
| `info` | Show forest hyperparameters |
| `gpu-info` | Show GPU device information |
| `help` | Show help message |

#### Tree Inspection Commands

| Command | Description |
|---------|-------------|
| `inspect-tree` | Inspect tree structure and nodes |
| `tree-depth` | Get depth of a specific tree |
| `tree-nodes` | Get node count of a specific tree |
| `tree-leaves` | Get leaf count of a specific tree |
| `node-details` | Get details of a specific node |

#### Tree Manipulation Commands

| Command | Description |
|---------|-------------|
| `add-tree` | Add a new tree to the forest |
| `remove-tree` | Remove a tree from the forest |
| `replace-tree` | Replace a tree with new bootstrap sample |
| `retrain-tree` | Retrain a specific tree |
| `prune-tree` | Prune subtree at specified node |
| `modify-split` | Modify split threshold at node |
| `modify-leaf` | Modify leaf prediction value |
| `convert-to-leaf` | Convert node to leaf |

#### Feature Control Commands

| Command | Description |
|---------|-------------|
| `enable-feature` | Enable a feature for predictions |
| `disable-feature` | Disable a feature for predictions |
| `reset-features` | Reset all feature filters |
| `feature-usage` | Show feature usage summary |
| `importance` | Show feature importances |

#### Aggregation Commands

| Command | Description |
|---------|-------------|
| `set-aggregation` | Set prediction aggregation method |
| `get-aggregation` | Get current aggregation method |
| `set-weight` | Set weight for specific tree |
| `get-weight` | Get weight of specific tree |
| `reset-weights` | Reset all tree weights to 1.0 |

#### Performance Analysis Commands

| Command | Description |
|---------|-------------|
| `oob-summary` | Show OOB error summary per tree |
| `track-sample` | Track which trees influence a sample |
| `metrics` | Calculate accuracy/MSE/F1 etc. |
| `misclassified` | Highlight misclassified samples |
| `worst-trees` | Find trees with highest error |

#### Facade Options

| Option | Description |
|--------|-------------|
| `--input=<file>` | Training input data (CSV) |
| `--target=<file>` | Training targets (CSV) |
| `--data=<file>` | Test/prediction data (CSV) |
| `--model=<file>` | Model file (default: forest.bin) |
| `--output=<file>` | Output predictions file |
| `--trees=<n>` | Number of trees (default: 100) |
| `--depth=<n>` | Max tree depth (default: 10) |
| `--tree=<id>` | Tree ID for operations |
| `--node=<id>` | Node ID for operations |
| `--feature=<id>` | Feature ID for operations |
| `--weight=<val>` | Tree weight (0.0-1.0) |
| `--aggregation=<method>` | majority, weighted, mean, weighted-mean |

#### Facade Examples

```bash
# Create and train forest
forest_facade_cuda create --trees=100 --depth=10 --model=rf.bin
forest_facade_cuda train --input=data.csv --target=labels.csv --model=rf.bin

# Make predictions and evaluate
forest_facade_cuda predict --data=test.csv --model=rf.bin --output=preds.csv
forest_facade_cuda evaluate --data=test.csv --model=rf.bin

# Tree inspection
forest_facade_cuda inspect-tree --tree=5 --model=rf.bin
forest_facade_cuda tree-depth --tree=5 --model=rf.bin

# Feature analysis
forest_facade_cuda feature-usage --model=rf.bin
forest_facade_cuda importance --model=rf.bin

# Tree manipulation
forest_facade_cuda add-tree --model=rf.bin
forest_facade_cuda remove-tree --tree=5 --model=rf.bin
forest_facade_cuda disable-feature --feature=3 --model=rf.bin

# Aggregation control
forest_facade_cuda set-aggregation --aggregation=weighted-mean --model=rf.bin
forest_facade_cuda set-weight --tree=5 --weight=1.5 --model=rf.bin
```

---

## **Testing**

### Running All Tests

```bash
# Run C++ tests
./random_forest_cpp_tests.sh

# Run OpenCL tests
./random_forest_opencl_tests.sh

# Run Rust tests
cd rust_cuda && cargo test
cd facaded_rust_cuda && cargo test
```

### Test Categories

Each test suite covers:

| Category | Tests |
|----------|-------|
| **Help & Usage** | Command-line interface verification |
| **Model Creation** | Various hyperparameter configurations |
| **Hyperparameters** | Trees, depth, split criteria, task types |
| **Model Info** | Metadata retrieval |
| **Save & Load** | Model persistence |
| **Introspection** | Tree, node, feature inspection |
| **Error Handling** | Invalid input handling |
| **Cross-Implementation** | API compatibility |
| **Train & Predict** | End-to-end workflows |

### Test Output Example

```
=========================================
Random Forest Test Suite
=========================================

Group: Help & Usage
Test 1: Help command... PASS
Test 2: --help flag... PASS
Test 3: -h flag... PASS
...

=========================================
Test Summary
=========================================
Total tests: 50
Passed: 50
Failed: 0

All tests passed!
```

---

## **Formal Verification with Kani**

### Overview

The Rust Facade implementation includes **Kani formal verification proofs** that mathematically prove the absence of certain classes of bugs. This goes beyond traditional testing to provide **mathematical guarantees** about code correctness.

### Verification Categories

| Category | Description |
|----------|-------------|
| **Strict Bound Checks** | Array/collection indexing safety |
| **Pointer Validity** | Slice-to-pointer conversion safety |
| **No-Panic Guarantee** | Enum and command handling safety |
| **Integer Overflow Prevention** | Tree size, dimension calculations |
| **Division-by-Zero Exclusion** | Bootstrap sampling, feature selection |
| **Global State Consistency** | Training mode state tracking |
| **Input Sanitization Bounds** | Loop iteration limits |
| **Memory Leak Prevention** | Vector allocation bounds |
| **Floating-Point Sanity** | NaN/Infinity prevention |

### Key Kani Proofs

#### Bound Checking Proofs
- `verify_tree_indexing` ✓
- `verify_node_indexing` ✓
- `verify_feature_indexing` ✓

#### Overflow Prevention Proofs
- `verify_tree_size_no_overflow` ✓
- `verify_bootstrap_no_overflow` ✓
- `verify_feature_selection_no_overflow` ✓

#### Safety Proofs
- `verify_criterion_type_no_panic` ✓
- `verify_task_type_no_panic` ✓
- `verify_command_parsing_no_panic` ✓
- `verify_gini_no_nan` ✓

### Running Kani Verification

```bash
# CLI Version
cd facaded_rust_cuda
cargo kani

# Run specific proof
cargo kani --harness verify_tree_indexing
```

### Why Formal Verification Matters

Traditional testing can only verify specific test cases. Formal verification with Kani:

- **Exhaustively checks all possible inputs** within defined bounds
- **Mathematically proves** absence of panics, buffer overflows, and undefined behavior
- **Catches edge cases** that random testing might miss
- **Provides cryptographic-level assurance** for safety-critical code

---

## **CISA/NSA Compliance**

### Secure by Design

This project follows **CISA (Cybersecurity and Infrastructure Security Agency)** and **NSA (National Security Agency)** Secure by Design principles:

| Principle | Implementation |
|-----------|---------------|
| **Memory Safety** | Rust ownership model eliminates buffer overflows, use-after-free, and data races |
| **Formal Verification** | Kani proofs mathematically verify absence of critical bugs |
| **Input Validation** | All CLI inputs validated before processing |
| **Defense in Depth** | Multiple layers of safety (language, compiler, runtime checks) |
| **Secure Defaults** | Safe default configurations throughout |
| **Transparency** | Open source with full code visibility |

### Compliance Checklist

- [x] **Memory-safe language** (Rust implementation)
- [x] **Static analysis** (Rust compiler + Clippy)
- [x] **Formal verification** (Kani proof harnesses)
- [x] **Comprehensive testing** (Unit tests + integration tests)
- [x] **Bounds checking** (Verified array access)
- [x] **Input validation** (CLI argument parsing)
- [x] **No unsafe code in critical paths** (Where possible)
- [x] **Documentation** (Inline docs + README)
- [x] **Version control** (Git)
- [x] **License clarity** (MIT License)

### Attestation

This codebase has been developed following secure software development lifecycle (SSDLC) practices and demonstrates:

- **Formal verification proofs passed** (Kani proofs)
- **Zero warnings** compilation across all implementations
- **Consistent API** across all language/backend combinations
- **Production-ready** code quality

---

## **License**

MIT License

Copyright (c) 2025 Matthew Abbott

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## **Author**

**Matthew Abbott**  
Email: mattbachg@gmail.com

---

*Built with precision. Verified with rigor. Secured by design.*
