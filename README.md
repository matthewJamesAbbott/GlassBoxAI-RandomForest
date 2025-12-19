# GlassBoxAI-RandomForest

**Author:** Matthew Abbott (2025)

A fully-transparent, educationally-minded CUDA implementation of Random Forests for both classification and regression. This repository:
- Provides a direct, research-grade random forest (random_forest.cu) with full feature importance, forest introspection, and GPU-prediction kernels.
- Includes a "facade" API (facaded_random_forest.cu) for classroom/diagnostic exploration and scripting, giving maximal access to every tree, split, feature, and sample—including advanced tracking, modification, and analytics.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [random_forest.cu](#random_forestcu)
  - [Design](#design)
  - [Usage](#usage)
  - [Arguments](#arguments)
  - [Public Methods](#public-methods)
- [facaded_random_forest.cu (Facade)](#facaded_random_forestcu-facade)
  - [Design](#design-1)
  - [Usage](#usage-1)
  - [Arguments](#arguments-1)
  - [Public Methods](#public-methods-1)
- [Data Structures](#data-structures)
- [Overview & Notes](#overview--notes)

---

## Features

- **Random Forests for Regression & Classification** (Gini, Entropy, MSE, Variance Reduction criteria).
- Efficient CUDA-batch prediction for large datasets.
- Bootstrapping, out-of-bag error, feature importance, split diagnostics.
- Full tree introspection: every split, threshold, impurity, etc.
- Utility for CSV I/O, model save/load, metrics (Accuracy, Precision, Recall, F1, MSE, R²).
- Facade exposes progressive pruning, tree/leaf modification, feature selection and filters, OOB/per-tree diagnostics, advanced sample tracking.
- **No external ML framework required—just CUDA and STL.**

---

## Requirements

- NVIDIA GPU with CUDA (compute 5.0+ recommended)
- CUDA toolkit (tested: CUDA 11-12)
- C++14 (main) or C++17 (facade)
- No extra dependencies

---

## random_forest.cu

### Design

Implements a direct Random Forest from algorithm basics:
- Tree-building via bootstrapped samples, with full control of stopping criteria
- Shrink-to-fit data arrays, per-tree impurity and split gain computation
- Classification and regression all supported
- Out-of-bag prediction for unbiased error measurement
- Prediction kernels fully CUDA-parallelized for batch prediction (both tasks)
- Feature importance gathered and printed after training

### Usage

```bash
nvcc -O3 -std=c++14 random_forest.cu -o random_forest_cuda
```

**Typical workflow (in your own CLI/driver):**
- Create `TRandomForest` object
- Load CSV or arrays
- Call `fit()`
- Use `predict(sample)` or `predictBatch(samples, n, predictions)` for CPU
- Use `predictBatchGPU(samples, n, predictions)` for CUDA batch

### Arguments

#### Hyperparameters (setters)

- `setNumTrees(n)`: integer
- `setMaxDepth(d)`: integer
- `setMinSamplesLeaf(m)`: integer
- `setMinSamplesSplit(m)`: integer
- `setMaxFeatures(m)`: integer
- `setTaskType(TaskType::Classification/Regression)`
- `setCriterion(SplitCriterion::Gini/Entropy/MSE/VarianceReduction)`
- `setRandomSeed(seed)`

#### Data Loading

- `loadCSV(filename, targetColumn, hasHeader)`
- `loadData(double* data, double* targets, nSamples, nFeatures)`

#### Model Persistence

- `saveModel(filename)`
- `loadModel(filename)`

### Public Methods

#### Main Class: `TRandomForest`

- `fit()` — Builds all trees.
- `predict(sample)` — Single prediction (CPU).
- `predictBatch(samples, n, predictions)` — Batched prediction (CPU).
- `predictBatchGPU(samples, n, predictions)` — Batched prediction (CUDA).
- `calculateOOBError()` — Out-of-bag error estimate.
- `printForestInfo()` — Displays full hyperparameters and forest config.
- `accuracy`, `precision`, `recall`, `f1Score`, `meanSquaredError`, `rSquared` — Full test metrics.
- `printFeatureImportances()` — Prints feature usage summary.
- (Advanced) Tree flattening, loading from/saving to disk, direct manipulation/API.

#### Utility/Introspection

- All trees, split statistics, leaf predictions, and impurity are accessible; see class methods and structs in code for DIY diagnostics.

---

## facaded_random_forest.cu (Facade)

### Design

A much more interactive C++/CUDA interface to the random forest model:
- Ready for classroom, research, and exploratory scripting.
- Lets you inspect, modify, prune, and track any tree, split, or feature.
- Progressive/feature filtering, aggregation methods, per-tree weighting, and tracking of every sample and OOB result.

### Usage

```bash
nvcc -O3 -std=c++17 facaded_random_forest.cu -o facaded_random_forest_cuda
```
Use in your C++ project, scripting context, or standalone as a model exploration shell.

### Arguments

#### Facade API Hyperparameters

- `setHyperparameter(param, value)`
- `setTaskType(TaskType)`
- `setCriterion(SplitCriterion)`
- `enableFeature(idx)` / `disableFeature(idx)` / `resetFeatureFilters()`
- `addTree()`, `removeTree(id)`, `pruneTree(treeId, nodeId)`
- `setAggregationMethod(method)` — MajorityVote, WeightedVote, Mean, WeightedMean
- `setTreeWeight(id, weight)` / `getTreeWeight(id)`
- Model save/load: `saveModel(filename)` / `loadModel(filename)`

#### Data Loading & Training

- `loadData(vector<vector<double>>, vector<double>)`
- `trainForest()`

#### Introspection

- `printForestOverview()`
- `printTreeStructure(id)` / `printNodeDetails(treeId, nodeId)`
- `inspectTree(id): TreeInfo`
- `featureUsageSummary()` / `printFeatureUsageSummary()` / `printFeatureHeatmap()`
- OOB summary: `printOOBSummary()`, `getGlobalOOBError()`
- `trackSample(sampleIdx)` — See all trees affecting sample
- Metrics for accuracy, MSE, precision, recall, F1, etc.

#### Visualization/Diagnostics

- `visualizeTree(id)`, `visualizeSplitDistribution(treeId, nodeId)`
- `highlightMisclassified(preds, actual)` / `highlightHighResidual(preds, actual, threshold)`
- Find worst trees, problematic splits, overfitting, etc.

### Public Methods

Class: `RandomForestFacade`

- Full forest access: `getForest()`
- Tree add/remove/prune/replace/retrain
- Node modification: `modifySplit`, `modifyLeafValue`, `convertToLeaf`
- Track and visualize every sample's path through forest
- Aggregated prediction: weighted, majority, mean, per-tree
- Global and per-tree OOB error queries, feature and node stats

---

## Data Structures

- **TreeNode / DecisionTree / FlatTreeNode:** Low-level tree, split, impurity, sample stats.
- **TreeInfo / NodeInfo / FeatureStats:** Full recursive tree reports, including feature use.
- **SampleTrackInfo:** Tracks sample's passage and influence across all trees.
- **OOBTreeInfo:** Per-tree out-of-bag error details.
- **RandomForestFacade:** Wraps all above for extensible, scriptable forest introspection.

---

## Overview & Notes

- Maximal transparency: every tree, split, prediction, feature, and metric available via the API—no algorithm is "hidden" from you.
- Designed to support code and research curiosity—change anything, print anything, compare and measure.
- Facade is ideal for classroom demos, research diagnostics, auto-pruning, and criterion experimentation.
- For CSV/data formatting and detailed results, see code comments and structs.

---

## License

MIT License, Copyright © 2025 Matthew Abbott
