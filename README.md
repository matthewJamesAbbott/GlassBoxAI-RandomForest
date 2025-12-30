# GlassBoxAI-RandomForest

**Author:** Matthew Abbott (2025)

GlassBoxAI-RandomForest is a transparent, research-grade random forest implementation for both classification and regression, emphasizing full GPU acceleration and maximal algorithmic introspection. Core modules feature CUDA, C++, and OpenCL support, and a powerful facade API for classroom diagnostics, hacking, and exploratory scripting.

---

## Table of Contents

- [Features](#features)
- [Module Overview](#module-overview)
- [Requirements](#requirements)
- [Quickstart: Compiling & Running](#quickstart-compiling--running)
- [CLI Usage and Help](#cli-usage-and-help)
  - [1. CUDA Core Forest (`random_forest.cu`)](#1-cuda-core-forest-random_forestcu)
  - [2. OpenCL Core Forest (`random_forest_opencl.cpp`)](#2-opencl-core-forest-random_forest_openclcpp)
  - [3. Facade CLI (Introspectable, `facaded_random_forest.cu`/`.cpp`)](#3-facade-cli-introspectable-facaded_random_forestcucpp)
    - [Facade CLI Example Commands](#facade-cli-example-commands)
    - [All Advanced CLI/Introspection Options](#all-advanced-cliintrospection-options)
- [Architecture Notes](#architecture-notes)
- [Core Data Structures](#core-data-structures)
- [License](#license)

---

## Features

- **Fully transparent, research- and classroom-grade random forest algorithm**
- CPU/Core, CUDA, and OpenCL backend support
- Supports regression & classification (Gini/Entropy/MSE/Variance Reduction)
- GPU-accelerated batch prediction
- CLI and facade modes for maximal introspection and model hacking
- Out-of-bag error, feature importance, per-tree & per-node statistics
- Utility for CSV I/O, model save/load, metrics (accuracy, precision, recall, F1, MSE, R²)
- Facade API enables: 
  - Tree/leaf editing, pruning, filtering, introspection
  - Feature selection, tracking, per-tree weights, aggregation method switching
  - Sample path tracking, OOB/error analysis, highlighting misclassified/hard samples
- **No external ML framework required—just CUDA, OpenCL, or STL**

---

## Module Overview

| Type      | Core Forest        | Facade/Introspectable   |
|-----------|--------------------|-------------------------|
| CUDA      | `random_forest.cu` | `facaded_random_forest.cu`        |
| OpenCL    | `random_forest_opencl.cpp` | `facaded_random_forest.cpp`        |
| C++ (no GPU) | `random_forest.cpp` | `facaded_random_forest.cpp`      |

**Core** = minimal scripting/production API  
**Facade** = detailed, hackable CLI for research/teaching

---

## Requirements

- CUDA (for `random_forest.cu`, `facaded_random_forest.cu`): NVIDIA GPU, CUDA Toolkit 11+, C++14
- OpenCL (for `random_forest_opencl.cpp`, `facaded_random_forest.cpp`): OpenCL 1.2+ device, C++14
- C++ (for core and facade, no GPU): basic C++14 compiler
- No additional Python or ML libraries required

---

## Quickstart: Compiling & Running

**CUDA:**
```bash
nvcc -O2 -std=c++14 -o rf_cuda random_forest.cu
nvcc -O2 -std=c++17 -o facaded_rf_cuda facaded_random_forest.cu
```

**OpenCL:**
```bash
g++ -O2 -std=c++14 -o rf_opencl random_forest_opencl.cpp -lOpenCL
g++ -O2 -std=c++14 -o facaded_rf_opencl facaded_random_forest.cpp -lOpenCL
```

**C++ (no GPU):**
```bash
g++ -O2 -std=c++14 -o rf_cpp random_forest.cpp
g++ -O2 -std=c++14 -o facaded_rf_cpp facaded_random_forest.cpp
```

---

## CLI Usage and Help

For all modes, running with `help` or no arguments prints full command and option info.

---

### 1. CUDA Core Forest (`random_forest.cu`)

Minimal, scriptable CLI for fast CUDA forests.  
**Show help:**
```bash
./rf_cuda help
```

#### Example Usage

```bash
# Create model
./rf_cuda create --trees=100 --max-depth=10 --save=rf_model.bin

# Train on data
./rf_cuda train --model=rf_model.bin --data=train.csv --save=rf_trained.bin

# Predict
./rf_cuda predict --model=rf_trained.bin --data=test.csv --output=preds.csv

# Show Info
./rf_cuda info --model=rf_trained.bin
```

---

### 2. OpenCL Core Forest (`random_forest_opencl.cpp`)

Identical logic to CUDA core, with OpenCL backend.

```bash
./rf_opencl help
```

---

### 3. Facade CLI (Introspectable, `facaded_random_forest.cu`/`.cpp`)

All minimal commands **plus dozens of scripting/diagnostic tools** for tree manipulation, feature control, and detailed introspection.

**Show help:**
```bash
./facaded_rf_cuda help
# or
./facaded_rf_opencl help
```

#### Example Facade/Introspection Commands

```
[core]
create, train, predict, info, help
[inspection]
inspect-tree, tree-depth, node-details, feature-usage, importance, oob-summary
[manipulation]
add-tree, remove-tree, retrain-tree, prune-tree, modify-split, modify-leaf, convert-to-leaf
[feature]
enable-feature, disable-feature, reset-features
[aggregation]
set-aggregation, set-weight, reset-weights
[analysis]
track-sample, metrics, misclassified, worst-trees
```

#### Facade CLI Example Commands

```bash
./facaded_rf_cuda create --trees 100 --depth 10 --model rf.bin
./facaded_rf_cuda train --input data.csv --target labels.csv --model rf.bin
./facaded_rf_cuda predict --data test.csv --model rf.bin --output preds.csv
./facaded_rf_cuda inspect-tree --tree 5 --model rf.bin
./facaded_rf_cuda prune-tree --tree 5 --node 12 --model rf.bin
./facaded_rf_cuda importance --model rf.bin
./facaded_rf_cuda set-aggregation --aggregation weighted --model rf.bin
./facaded_rf_cuda set-weight --tree 5 --weight 1.5 --model rf.bin
./facaded_rf_cuda track-sample --sample 13 --model rf.bin
./facaded_rf_cuda oob-summary --model rf.bin
```

#### All Advanced CLI/Introspection Options

- Model/Forest: `info`, `save`, `load`, `gpu-info`
- Tree: `inspect-tree`, `tree-depth`, `tree-nodes`, `tree-leaves`, `node-details`
- Tree manipulation: `add-tree`, `remove-tree`, `retrain-tree`, `prune-tree`, `modify-split`, `modify-leaf`, `convert-to-leaf`
- Feature control: `enable-feature`, `disable-feature`, `reset-features`, `feature-usage`, `importance`
- Aggregation: `set-aggregation`, `set-weight`, `get-weight`, `reset-weights`
- Diagnostics: `oob-summary`, `metrics`, `track-sample`, `misclassified`, `worst-trees`

_(See CLI help output for all options & arguments)_

---

## Architecture Notes

- **Modular Source:** Each backend (CUDA, OpenCL, C++) provides a native, high-performance kernel. The facade modules wrap these for full classroom-style model exploration.
- **GPU Acceleration:** Both `random_forest.cu` and `random_forest_opencl.cpp` offer GPU-parallelized prediction; fallback to CPU if GPU is unavailable.
- **CLI Facade:** The facade (`facaded_random_forest.cu`/`.cpp`) makes model structure, feature usage, and every prediction path inspectable and hackable.  
- **Pure C++ Option:** For environments without GPU, `random_forest.cpp`/`facaded_random_forest.cpp` provides 100% C++ support (production-grade logic, minus GPU speed).

---

## Core Data Structures

- **TreeNode/FlatTreeNode:** Core node with split, threshold, impurity, prediction, class, left/right indices
- **TDecisionTree/FlatTree:** Tree root and OOB indicators; Flat array for GPU batch kernels
- **RandomForest(TRandomForest):** Forest of trees, fit/predict logic, metrics, CSV I/O
- **Facade classes:** Add full tree introspection API, feature and sample trackers, per-tree/statistics and manipulation methods

---

## License

MIT License  
© 2025 Matthew Abbott

---
