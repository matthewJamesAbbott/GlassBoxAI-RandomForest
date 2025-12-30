# GlassBoxAI-RandomForest

**Author:** Matthew Abbott (2025)

GlassBoxAI-RandomForest is a transparent, deeply introspectable random forest library for both classification and regression, featuring CUDA, OpenCL, and pure C++ implementations. It’s designed for teaching, hacking, reproducible research, and high-performance GPU prediction. A rich facade CLI brings full feature and tree introspection, interactive model modification, and advanced diagnostics.

---

## Table of Contents

- [Features](#features)
- [Module Overview](#module-overview)
- [Requirements](#requirements)
- [Quickstart: Compiling & Running](#quickstart-compiling--running)
- [CLI Usage and Help](#cli-usage-and-help)
  - [Core CLI Model (`random_forest.cu`, `random_forest.cpp`, `random_forest_opencl.cpp`)](#core-cli-model-random_forestcu-random_forestcpp-random_forest_openclcpp)
    - [Sample Help Output (Core)](#sample-help-output-core)
    - [Core CLI Command/Options Explained](#core-cli-commandoptions-explained)
  - [Facade CLI (`facaded_random_forest.cu`, `facaded_random_forest.cpp`)](#facade-cli-facaded_random_forestcu-facaded_random_forestcpp)
    - [Sample Help Output (Facade)](#sample-help-output-facade)
    - [Facade CLI Command/Options Explained](#facade-cli-commandoptions-explained)
- [Architecture Notes](#architecture-notes)
- [Core Data Structures](#core-data-structures)
- [License](#license)

---

## Features

- **Pure, dependency-free CUDA, OpenCL and C++ random forest implementations**
- Supports **classification & regression** (Gini/Entropy/MSE/Variance Reduction criteria)
- GPU-accelerated **batch prediction**; batch and interactive use
- CLI and facade modes: 
  - Core scriptable API (fast training/prediction/metrics)
  - **Facade API for deep inspection, manipulation, and troubleshooting**
- Out-of-bag error, feature importances, per-tree/node statistics
- Save/load models, CSV utilities, hyperparameter control
- Advanced feature selection, aggregation modes, per-tree weighting/filtering, sample-path tracking (Facade)
- **Full model “glass box”: All trees, features, and splits visible & modifiable**

---

## Module Overview

| Type      | Core Forest         | Facade/Introspectable   |
|-----------|---------------------|-------------------------|
| CUDA      | `random_forest.cu`  | `facaded_random_forest.cu`        |
| OpenCL    | `random_forest_opencl.cpp` | `facaded_random_forest.cpp`        |
| C++ (no GPU) | `random_forest.cpp` | `facaded_random_forest.cpp`      |

---

## Requirements

- CUDA tools 11+ and NVIDIA GPU for CUDA (`random_forest.cu`, `facaded_random_forest.cu`)
- OpenCL 1.2+ and C++14 compiler for OpenCL (`random_forest_opencl.cpp`, `facaded_random_forest.cpp`)
- C++14 for CPU-only build
- *No Python or external ML packages required*

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

**CPU/C++ only:**  
```bash
g++ -O2 -std=c++14 -o rf_cpp random_forest.cpp
g++ -O2 -std=c++14 -o facaded_rf_cpp facaded_random_forest.cpp
```

---

## CLI Usage and Help

### Core CLI Model (`random_forest.cu`, `random_forest.cpp`, `random_forest_opencl.cpp`)

#### Sample Help Output (Core)
```
Random Forest CLI Tool
======================

Usage: forest <command> [options]

Commands:

  create   Create a new Random Forest model
  train    Train a Random Forest model
  predict  Make predictions with a trained model
  info     Display information about a model
  help     Show this help message

CREATE Options:
  --trees=N              Number of trees (default: 100)
  --max-depth=N          Maximum tree depth (default: 10)
  --min-leaf=N           Minimum samples per leaf (default: 1)
  --min-split=N          Minimum samples to split (default: 2)
  --max-features=N       Maximum features to consider
  --criterion=CRITERION  Split criterion: gini, entropy, mse, variancereduction
  --task=TASK            Task type: classification, regression
  --save=FILE            Save model to file (required)

TRAIN Options:
  --model=FILE           Model file to train (required)
  --data=FILE            Data file for training (required)
  --save=FILE            Save trained model to file (required)

PREDICT Options:
  --model=FILE           Model file to use (required)
  --data=FILE            Data file for prediction (required)
  --output=FILE          Save predictions to file (optional)

INFO Options:
  --model=FILE           Model file to inspect (required)

Examples:
  forest create --trees=50 --max-depth=15 --save=model.bin
  forest train --model=model.bin --data=train.csv --save=model_trained.bin
  forest predict --model=model_trained.bin --data=test.csv --output=predictions.csv
  forest info --model=model_trained.bin
```

#### Core CLI Command/Options Explained

- `create`: Build and configure a new random forest model, specifying trees/depth/criterion/task, and saving to disk.
- `train`: Load an existing model file, fit on training data, save trained model.
- `predict`: Apply a trained model to new data (`test.csv`) and optionally save predictions as CSV.
- `info`: Print a summary of the forest configuration, such as hyperparameters, tree count, and more.
- `help`: Show command/option listing.

**Common options:**
- `--trees=N`: Number of trees in the forest
- `--max-depth=N`: Maximum tree depth
- `--min-leaf=N`: Minimum samples per leaf node
- `--max-features=N`: Maximum features to consider per split
- `--criterion=CRITERION`: `gini`, `entropy`, `mse`, or `variancereduction`
- `--task=TASK`: `classification` or `regression`
- `--save=FILE`: Path to save the model
- `--model=FILE`: Path of an existing model file

---

### Facade CLI (`facaded_random_forest.cu`, `facaded_random_forest.cpp`)

#### Sample Help Output (Facade)
```
Random Forest Facade CLI (CUDA GPU) - Matthew Abbott 2025
Advanced Random Forest with Introspection, Tree Manipulation, and Feature Control

Usage: forest_facade <command> [options]

=== Core Commands ===
  create              Create a new empty forest model
  train               Train a random forest model
  predict             Make predictions using a trained model
  evaluate            Evaluate model on test data
  save                Save model to file
  load                Load model from file
  info                Show forest hyperparameters
  gpu-info            Show GPU device information
  help                Show this help message

=== Tree Inspection & Manipulation ===
  inspect-tree        Inspect tree structure and nodes
  tree-depth          Get depth of a specific tree
  tree-nodes          Get node count of a specific tree
  tree-leaves         Get leaf count of a specific tree
  node-details        Get details of a specific node
  prune-tree          Prune subtree at specified node
  modify-split        Modify split threshold at node
  modify-leaf         Modify leaf prediction value
  convert-to-leaf     Convert node to leaf

=== Tree Management ===
  add-tree            Add a new tree to the forest
  remove-tree         Remove a tree from the forest
  replace-tree        Replace a tree with new bootstrap sample
  retrain-tree        Retrain a specific tree

=== Feature Control ===
  enable-feature      Enable a feature for predictions
  disable-feature     Disable a feature for predictions
  reset-features      Reset all feature filters
  feature-usage       Show feature usage summary
  importance          Show feature importances

=== Aggregation Control ===
  set-aggregation     Set prediction aggregation method
  get-aggregation     Get current aggregation method
  set-weight          Set weight for specific tree
  get-weight          Get weight of specific tree
  reset-weights       Reset all tree weights to 1.0

=== Performance Analysis ===
  oob-summary         Show OOB error summary per tree
  track-sample        Track which trees influence a sample
  metrics             Calculate accuracy/MSE/F1 etc.
  misclassified       Highlight misclassified samples
  worst-trees         Find trees with highest error

=== Options ===

Data & Model:
  --input <file>          Training input data (CSV)
  --target <file>         Training targets (CSV)
  --data <file>           Test/prediction data (CSV)
  --model <file>          Model file (default: forest.bin)
  --output <file>         Output predictions file

Hyperparameters:
  --trees <n>             Number of trees (default: 100)
  --depth <n>             Max tree depth (default: 10)
  --min-leaf <n>          Min samples per leaf (default: 1)
  --min-split <n>         Min samples to split node (default: 2)
  --max-features <n>      Max features per split (0=auto)
  --task <class|reg>      Task type (default: class)
  --criterion <c>         Split criterion: gini/entropy/mse/var

Tree Manipulation:
  --tree <id>             Tree ID for operations
  --node <id>             Node ID for operations
  --threshold <val>       New split threshold
  --value <val>           New leaf value

Feature/Weight Control:
  --feature <id>          Feature ID for operations
  --weight <val>          Tree weight (0.0-1.0)
  --aggregation <method>  majority|weighted|mean|weighted-mean
  --sample <id>           Sample ID for tracking

=== Examples ===
  # Create and train forest
  forest_facade create --trees 100 --depth 10 --model rf.bin
  forest_facade train --input data.csv --target labels.csv --model rf.bin
  # Make predictions and evaluate
  forest_facade predict --data test.csv --model rf.bin --output preds.csv
  forest_facade evaluate --data test.csv --model rf.bin
  # Tree inspection
  forest_facade inspect-tree --tree 5 --model rf.bin
  forest_facade tree-depth --tree 5 --model rf.bin
  # Feature analysis
  forest_facade feature-usage --model rf.bin
  forest_facade importance --model rf.bin
  # Tree manipulation
  forest_facade add-tree --model rf.bin
  forest_facade remove-tree --tree 5 --model rf.bin
  forest_facade disable-feature --feature 3 --model rf.bin
  # Aggregation control
  forest_facade set-aggregation --aggregation weighted-mean --model rf.bin
  forest_facade set-weight --tree 5 --weight 1.5 --model rf.bin
```

#### Facade CLI Command/Options Explained

**Core (Scriptable) Commands:**
- `create`, `train`, `predict`, `evaluate`, `save`, `load`, `info`, `help`: Same as core CLI, but available through facade for scripting and batch jobs.
- `gpu-info`: Print detected GPU details and config.

**Tree Inspection & Manipulation:**
- `inspect-tree`: Show split structure, depths, split values, and leaf predictions for any tree.
- `tree-depth`, `tree-nodes`, `tree-leaves`: Report depth, node, or leaf count for any tree.
- `node-details`: Inspect split/leaf info for a specific node in a tree.
- `prune-tree`: Remove all descendants of a node, converting to a leaf.
- `modify-split`, `modify-leaf`, `convert-to-leaf`: Alter split parameters or forcibly set leaf values.

**Tree Management:**  
- `add-tree`, `remove-tree`, `replace-tree`, `retrain-tree`: Dynamically adjust forest structure without full retrain.

**Feature Control:**  
- `enable-feature`, `disable-feature`, `reset-features`: Filter which input features are used at prediction time.
- `feature-usage`, `importance`: Print detailed feature split statistics and normalized importances.

**Aggregation Control:**  
- `set-aggregation`, `get-aggregation`: Switch between majority/weighted voting and mean/weighted-mean for regression.
- `set-weight`, `get-weight`, `reset-weights`: Adjust weights/contributions per tree when aggregating predictions.

**Performance & Diagnostics:**  
- `oob-summary`: Detailed out-of-bag error per tree.
- `track-sample`: Print all trees that used or mispredicted a given sample (for debugging hard samples).
- `metrics`: Print summary metrics for classification or regression (`accuracy`, `precision`, `recall`, `F1`, `MSE`, `R²`).
- `misclassified`: Show misclassified samples.
- `worst-trees`: List trees most contributing to forest errors.

---

## License

MIT License  
© 2025 Matthew Abbott

---
