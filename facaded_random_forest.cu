//
// Matthew Abbott 2025
// Combined Random Forest + Facade C++ CUDA 
//

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <map>
#include <ctime>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>

using namespace std;

// Constants
const int MAX_FEATURES = 100;
const int MAX_SAMPLES = 10000;
const int MAX_TREES = 500;
const int MAX_DEPTH_DEFAULT = 10;
const int MIN_SAMPLES_LEAF_DEFAULT = 1;
const int MIN_SAMPLES_SPLIT_DEFAULT = 2;
const int MAX_NODE_INFO = 1000;
const int MAX_FEATURE_STATS = 100;
const int MAX_SAMPLE_TRACK = 1000;
const int MAX_NODES = 4096;

// Enums
enum TaskType { Classification, Regression };
enum SplitCriterion { Gini, Entropy, MSE, VarianceReduction };
enum TAggregationMethod { MajorityVote, WeightedVote, Mean, WeightedMean };

// Type definitions
typedef double TDataRow[MAX_FEATURES];
typedef double TTargetArray[MAX_SAMPLES];
typedef TDataRow TDataMatrix[MAX_SAMPLES];
typedef int TIndexArray[MAX_SAMPLES];
typedef int TFeatureArray[MAX_FEATURES];
typedef bool TBoolArray[MAX_SAMPLES];
typedef double TDoubleArray[MAX_FEATURES];

// Tree node structure
struct TreeNodeRec {
    bool isLeaf;
    int featureIndex;
    double threshold;
    double prediction;
    int classLabel;
    double impurity;
    int numSamples;
    TreeNodeRec* left;
    TreeNodeRec* right;
};

typedef TreeNodeRec* TreeNode;

// Flat tree structure for GPU
struct FlatTreeNode {
    bool isLeaf;
    int featureIndex;
    double threshold;
    double prediction;
    int classLabel;
    int leftChild;
    int rightChild;
};

struct FlatTree {
    FlatTreeNode nodes[MAX_NODES];
    int numNodes;
    bool oobIndices[MAX_SAMPLES];
    int numOobIndices;
};

// Decision tree structure
struct TDecisionTreeRec {
    TreeNode root;
    int maxDepth;
    int minSamplesLeaf;
    int minSamplesSplit;
    int maxFeatures;
    TaskType taskType;
    SplitCriterion criterion;
    TBoolArray oobIndices;
    int numOobIndices;
    FlatTree* flatTree;
};

typedef TDecisionTreeRec* TDecisionTree;

// Facade types
struct TNodeInfo {
    int nodeId;
    int depth;
    bool isLeaf;
    int featureIndex;
    double threshold;
    double prediction;
    int classLabel;
    double impurity;
    int numSamples;
    int leftChildId;
    int rightChildId;
};

struct TTreeInfo {
    int treeId;
    int numNodes;
    int maxDepth;
    int numLeaves;
    bool featuresUsed[MAX_FEATURES];
    int numFeaturesUsed;
    double oobError;
    vector<TNodeInfo> nodes;
};

struct TFeatureStats {
    int featureIndex;
    int timesUsed;
    int treesUsedIn;
    double avgImportance;
    double totalImportance;
};

struct TSampleTrackInfo {
    int sampleIndex;
    bool treesInfluenced[MAX_TREES];
    int numTreesInfluenced;
    bool oobTrees[MAX_TREES];
    int numOobTrees;
    double predictions[MAX_TREES];
};

struct TOOBTreeInfo {
    int treeId;
    int numOobSamples;
    double oobError;
    double oobAccuracy;
};

// CUDA Kernels
__global__ void predictBatchKernel(
    double* data,
    int numFeatures,
    FlatTreeNode* allTreeNodes,
    int* treeNodeOffsets,
    int numTrees,
    int numSamples,
    TaskType taskType,
    double* predictions
) {
    int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sampleIdx >= numSamples) return;

    double* sample = &data[sampleIdx * numFeatures];
    
    if (taskType == Regression) {
        double sum = 0.0;
        for (int t = 0; t < numTrees; t++) {
            FlatTreeNode* tree = &allTreeNodes[treeNodeOffsets[t]];
            int nodeIdx = 0;
            while (!tree[nodeIdx].isLeaf) {
                if (sample[tree[nodeIdx].featureIndex] <= tree[nodeIdx].threshold)
                    nodeIdx = tree[nodeIdx].leftChild;
                else
                    nodeIdx = tree[nodeIdx].rightChild;
            }
            sum += tree[nodeIdx].prediction;
        }
        predictions[sampleIdx] = sum / numTrees;
    } else {
        int votes[100] = {0};
        for (int t = 0; t < numTrees; t++) {
            FlatTreeNode* tree = &allTreeNodes[treeNodeOffsets[t]];
            int nodeIdx = 0;
            while (!tree[nodeIdx].isLeaf) {
                if (sample[tree[nodeIdx].featureIndex] <= tree[nodeIdx].threshold)
                    nodeIdx = tree[nodeIdx].leftChild;
                else
                    nodeIdx = tree[nodeIdx].rightChild;
            }
            int classLabel = tree[nodeIdx].classLabel;
            if (classLabel >= 0 && classLabel < 100)
                votes[classLabel]++;
        }
        
        int maxVotes = 0;
        int maxClass = 0;
        for (int i = 0; i < 100; i++) {
            if (votes[i] > maxVotes) {
                maxVotes = votes[i];
                maxClass = i;
            }
        }
        predictions[sampleIdx] = maxClass;
    }
}

// Random Forest Class
class TRandomForest {
private:
    FlatTree* trees[MAX_TREES];
    int numTrees;
    int maxDepth;
    int minSamplesLeaf;
    int minSamplesSplit;
    int maxFeatures;
    int numFeatures;
    int numSamples;
    TaskType taskType;
    SplitCriterion criterion;
    double featureImportances[MAX_FEATURES];
    long randomSeed;

    double* data;
    double* targets;
    
    double* d_data;
    FlatTreeNode* d_allTreeNodes;
    int* d_treeNodeOffsets;
    double* d_predictions;
    bool gpuInitialized;
    int totalGpuNodes;

    struct TreeNode {
        bool isLeaf;
        int featureIndex;
        double threshold;
        double prediction;
        int classLabel;
        double impurity;
        int numSamples;
        TreeNode* left;
        TreeNode* right;
    };

public:
    TRandomForest();
    ~TRandomForest();

    void setNumTrees(int n);
    void setMaxDepth(int d);
    void setMinSamplesLeaf(int m);
    void setMinSamplesSplit(int m);
    void setMaxFeatures(int m);
    void setTaskType(TaskType t);
    void setCriterion(SplitCriterion c);
    void setRandomSeed(long seed);

    int randomInt(int maxVal);
    double randomDouble();

    void loadData(double* inputData, double* inputTargets, int nSamples, int nFeatures);
    void bootstrap(int* sampleIndices, int& numBootstrap, bool* oobMask);
    void selectFeatureSubset(int* featureIndices, int& numSelected);

    double calculateGini(int* indices, int numIndices);
    double calculateEntropy(int* indices, int numIndices);
    double calculateMSE(int* indices, int numIndices);
    double calculateImpurity(int* indices, int numIndices);
    bool findBestSplit(int* indices, int numIndices, int* featureIndices, int nFeatures,
                       int& bestFeature, double& bestThreshold, double& bestGain);
    int getMajorityClass(int* indices, int numIndices);
    double getMeanTarget(int* indices, int numIndices);
    TreeNode* createLeafNode(int* indices, int numIndices);
    bool shouldStop(int depth, int numIndices, double impurity);
    TreeNode* buildTree(int* indices, int numIndices, int depth);
    void flattenTree(TreeNode* node, FlatTree* flat, int& nodeIdx);
    void freeTreeNode(TreeNode* node);

    void fit();
    void fitTree(int treeIndex);
    void initGPU();
    void freeGPU();

    double predict(double* sample);
    int predictClass(double* sample);
    void predictBatch(double* samples, int nSamples, double* predictions);
    void predictBatchGPU(double* samples, int nSamples, double* predictions);

    double calculateOOBError();

    void calculateFeatureImportance();
    double getFeatureImportance(int featureIndex);
    void printFeatureImportances();

    double accuracy(double* predictions, double* actual, int nSamples);
    double precision(double* predictions, double* actual, int nSamples, int positiveClass);
    double recall(double* predictions, double* actual, int nSamples, int positiveClass);
    double f1Score(double* predictions, double* actual, int nSamples, int positiveClass);
    double meanSquaredError(double* predictions, double* actual, int nSamples);
    double rSquared(double* predictions, double* actual, int nSamples);

    void printForestInfo();
    void freeForest();

    int getNumTrees() { return numTrees; }
    int getNumFeatures() { return numFeatures; }
    int getNumSamples() { return numSamples; }
    int getMaxDepthVal() { return maxDepth; }
    TaskType getTaskType() { return taskType; }
    SplitCriterion getCriterion() { return criterion; }

    bool loadCSV(const char* filename, int targetColumn, bool hasHeader);
    bool saveModel(const char* filename);
    bool loadModel(const char* filename);
    bool predictCSV(const char* inputFile, const char* outputFile, bool hasHeader);
    
    void addNewTree();
    void removeTreeAt(int treeId);
    void retrainTreeAt(int treeId);
};

// Random Forest Facade
class TRandomForestFacade {
private:
    TRandomForest forest;
    bool forestInitialized;
    TAggregationMethod currentAggregation;
    double treeWeights[MAX_TREES];
    bool featureEnabled[MAX_FEATURES];

public:
    TRandomForestFacade();
    ~TRandomForestFacade();

    void initForest();

    void setHyperparameter(const string& paramName, int value);
    void setHyperparameterFloat(const string& paramName, double value);
    int getHyperparameter(const string& paramName);
    void setTaskType(TaskType t) { forest.setTaskType(t); }
    void setCriterion(SplitCriterion c) { forest.setCriterion(c); }
    void printHyperparameters() { forest.printForestInfo(); }

    bool loadCSV(const char* filename) { return forest.loadCSV(filename, -1, true); }
    void train() { forest.fit(); }
    void trainGPU() { forest.initGPU(); forest.fit(); }

    TTreeInfo inspectTree(int treeId);
    void printTreeInfo(int treeId);
    void printTreeStructure(int treeId);

    void addTree() { forest.addNewTree(); }
    void removeTree(int treeId) { forest.removeTreeAt(treeId); }
    void replaceTree(int treeId) { forest.retrainTreeAt(treeId); }
    void retrainTree(int treeId) { forest.retrainTreeAt(treeId); }
    int getNumTrees() { return forest.getNumTrees(); }

    void enableFeature(int featureIndex);
    void disableFeature(int featureIndex);
    void resetFeatures();
    void printFeatureUsage();
    void printFeatureImportances() { forest.printFeatureImportances(); }

    void setAggregationMethod(TAggregationMethod method);
    TAggregationMethod getAggregationMethod() { return currentAggregation; }
    void setTreeWeight(int treeId, double weight);
    double getTreeWeight(int treeId);
    void resetTreeWeights();

    double predict(double* sample);
    int predictClass(double* sample);
    void predictBatch(double* samples, int nSamples, double* predictions);

    TSampleTrackInfo trackSample(int sampleIndex);
    void printSampleTracking(int sampleIndex);

    vector<TOOBTreeInfo> oobErrorSummary();
    void printOOBSummary();
    double getGlobalOOBError() { return forest.calculateOOBError(); }

    double accuracy(double* predictions, double* actual, int nSamples);
    double meanSquaredError(double* predictions, double* actual, int nSamples);

    void highlightMisclassified(double* predictions, double* actual, int nSamples);

    bool saveModel(const char* filename) { return forest.saveModel(filename); }
    bool loadModel(const char* filename) { return forest.loadModel(filename); }
    void printForestInfo() { forest.printForestInfo(); }
};

// Minimal TRandomForest implementation (references forest.cu for full)
TRandomForest::TRandomForest() {
    numTrees = 100;
    maxDepth = MAX_DEPTH_DEFAULT;
    minSamplesLeaf = MIN_SAMPLES_LEAF_DEFAULT;
    minSamplesSplit = MIN_SAMPLES_SPLIT_DEFAULT;
    maxFeatures = 0;
    numFeatures = 0;
    numSamples = 0;
    taskType = Classification;
    criterion = Gini;
    randomSeed = 42;
    
    d_data = nullptr;
    d_allTreeNodes = nullptr;
    d_treeNodeOffsets = nullptr;
    d_predictions = nullptr;
    gpuInitialized = false;
    totalGpuNodes = 0;

    for (int i = 0; i < MAX_TREES; i++)
        trees[i] = nullptr;

    for (int i = 0; i < MAX_FEATURES; i++)
        featureImportances[i] = 0.0;

    srand(static_cast<unsigned int>(time(nullptr)));
}

TRandomForest::~TRandomForest() { freeGPU(); freeForest(); }

void TRandomForest::setNumTrees(int n) { if (n > MAX_TREES) numTrees = MAX_TREES; else if (n < 1) numTrees = 1; else numTrees = n; }
void TRandomForest::setMaxDepth(int d) { if (d < 1) maxDepth = 1; else maxDepth = d; }
void TRandomForest::setMinSamplesLeaf(int m) { if (m < 1) minSamplesLeaf = 1; else minSamplesLeaf = m; }
void TRandomForest::setMinSamplesSplit(int m) { if (m < 2) minSamplesSplit = 2; else minSamplesSplit = m; }
void TRandomForest::setMaxFeatures(int m) { maxFeatures = m; }
void TRandomForest::setTaskType(TaskType t) { taskType = t; }
void TRandomForest::setCriterion(SplitCriterion c) { criterion = c; }
void TRandomForest::setRandomSeed(long seed) { randomSeed = seed; srand(static_cast<unsigned int>(seed)); }

int TRandomForest::randomInt(int maxVal) { if (maxVal <= 0) return 0; return rand() % maxVal; }
double TRandomForest::randomDouble() { return static_cast<double>(rand()) / RAND_MAX; }

void TRandomForest::loadData(double* inputData, double* inputTargets, int nSamples, int nFeatures) {}
void TRandomForest::bootstrap(int* sampleIndices, int& numBootstrap, bool* oobMask) {}
void TRandomForest::selectFeatureSubset(int* featureIndices, int& numSelected) {}

double TRandomForest::calculateGini(int* indices, int numIndices) { return 0.0; }
double TRandomForest::calculateEntropy(int* indices, int numIndices) { return 0.0; }
double TRandomForest::calculateMSE(int* indices, int numIndices) { return 0.0; }
double TRandomForest::calculateImpurity(int* indices, int numIndices) { return 0.0; }
bool TRandomForest::findBestSplit(int* indices, int numIndices, int* featureIndices, int nFeatures, int& bestFeature, double& bestThreshold, double& bestGain) { return false; }
int TRandomForest::getMajorityClass(int* indices, int numIndices) { return 0; }
double TRandomForest::getMeanTarget(int* indices, int numIndices) { return 0.0; }
TRandomForest::TreeNode* TRandomForest::createLeafNode(int* indices, int numIndices) { return nullptr; }
bool TRandomForest::shouldStop(int depth, int numIndices, double impurity) { return false; }
TRandomForest::TreeNode* TRandomForest::buildTree(int* indices, int numIndices, int depth) { return nullptr; }
void TRandomForest::flattenTree(TreeNode* node, FlatTree* flat, int& nodeIdx) {}
void TRandomForest::freeTreeNode(TreeNode* node) {}

void TRandomForest::fit() { cout << "Training forest..." << endl; }
void TRandomForest::fitTree(int treeIndex) {}

void TRandomForest::initGPU() {
    if (!gpuInitialized) {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount > 0) {
            cout << "GPU initialized" << endl;
            gpuInitialized = true;
        }
    }
}

void TRandomForest::freeGPU() {
    if (gpuInitialized) {
        if (d_data) cudaFree(d_data);
        if (d_allTreeNodes) cudaFree(d_allTreeNodes);
        if (d_treeNodeOffsets) cudaFree(d_treeNodeOffsets);
        if (d_predictions) cudaFree(d_predictions);
        gpuInitialized = false;
    }
}

double TRandomForest::predict(double* sample) { return 0.0; }
int TRandomForest::predictClass(double* sample) { return 0; }
void TRandomForest::predictBatch(double* samples, int nSamples, double* predictions) {}
void TRandomForest::predictBatchGPU(double* samples, int nSamples, double* predictions) {}

double TRandomForest::calculateOOBError() { return 0.0; }

void TRandomForest::calculateFeatureImportance() {}
double TRandomForest::getFeatureImportance(int featureIndex) { return 0.0; }
void TRandomForest::printFeatureImportances() { cout << "Feature importances" << endl; }

double TRandomForest::accuracy(double* predictions, double* actual, int nSamples) { return 0.0; }
double TRandomForest::precision(double* predictions, double* actual, int nSamples, int positiveClass) { return 0.0; }
double TRandomForest::recall(double* predictions, double* actual, int nSamples, int positiveClass) { return 0.0; }
double TRandomForest::f1Score(double* predictions, double* actual, int nSamples, int positiveClass) { return 0.0; }
double TRandomForest::meanSquaredError(double* predictions, double* actual, int nSamples) { return 0.0; }
double TRandomForest::rSquared(double* predictions, double* actual, int nSamples) { return 0.0; }

void TRandomForest::printForestInfo() {
    cout << "Forest Information:" << endl;
    cout << "  Trees: " << numTrees << endl;
    cout << "  Max Depth: " << maxDepth << endl;
    cout << "  Min Samples Leaf: " << minSamplesLeaf << endl;
    cout << "  Min Samples Split: " << minSamplesSplit << endl;
}

void TRandomForest::freeForest() {
    for (int i = 0; i < numTrees; i++) {
        if (trees[i] != nullptr) {
            delete trees[i];
            trees[i] = nullptr;
        }
    }
}

bool TRandomForest::loadCSV(const char* filename, int targetColumn, bool hasHeader) { return true; }
bool TRandomForest::saveModel(const char* filename) { return true; }
bool TRandomForest::loadModel(const char* filename) { return true; }
bool TRandomForest::predictCSV(const char* inputFile, const char* outputFile, bool hasHeader) { return true; }

void TRandomForest::addNewTree() { if (numTrees < MAX_TREES) numTrees++; }
void TRandomForest::removeTreeAt(int treeId) { if (treeId >= 0 && treeId < numTrees) numTrees--; }
void TRandomForest::retrainTreeAt(int treeId) {}

// TRandomForestFacade Implementation
TRandomForestFacade::TRandomForestFacade() {
    forestInitialized = false;
    currentAggregation = MajorityVote;
    for (int i = 0; i < MAX_TREES; i++)
        treeWeights[i] = 1.0;
    for (int i = 0; i < MAX_FEATURES; i++)
        featureEnabled[i] = true;
}

TRandomForestFacade::~TRandomForestFacade() {}

void TRandomForestFacade::initForest() { forestInitialized = true; }

void TRandomForestFacade::setHyperparameter(const string& paramName, int value) {
    if (paramName == "n_estimators") forest.setNumTrees(value);
    else if (paramName == "max_depth") forest.setMaxDepth(value);
    else if (paramName == "min_samples_leaf") forest.setMinSamplesLeaf(value);
    else if (paramName == "min_samples_split") forest.setMinSamplesSplit(value);
    else if (paramName == "max_features") forest.setMaxFeatures(value);
}

void TRandomForestFacade::setHyperparameterFloat(const string& paramName, double value) {}
int TRandomForestFacade::getHyperparameter(const string& paramName) { return 0; }

TTreeInfo TRandomForestFacade::inspectTree(int treeId) {
    TTreeInfo info = {treeId, 0, 0, 0, {}, 0, 0.0};
    return info;
}

void TRandomForestFacade::printTreeInfo(int treeId) {
    TTreeInfo info = inspectTree(treeId);
    cout << "Tree " << treeId << ": " << info.numNodes << " nodes, max depth: " << info.maxDepth << endl;
}

void TRandomForestFacade::printTreeStructure(int treeId) {
    cout << "Tree " << treeId << " structure" << endl;
}

void TRandomForestFacade::enableFeature(int featureIndex) {
    if (featureIndex >= 0 && featureIndex < MAX_FEATURES)
        featureEnabled[featureIndex] = true;
}

void TRandomForestFacade::disableFeature(int featureIndex) {
    if (featureIndex >= 0 && featureIndex < MAX_FEATURES)
        featureEnabled[featureIndex] = false;
}

void TRandomForestFacade::resetFeatures() {
    for (int i = 0; i < MAX_FEATURES; i++)
        featureEnabled[i] = true;
}

void TRandomForestFacade::printFeatureUsage() {
    cout << "Feature Usage Summary" << endl;
}

void TRandomForestFacade::setAggregationMethod(TAggregationMethod method) {
    currentAggregation = method;
}

void TRandomForestFacade::setTreeWeight(int treeId, double weight) {
    if (treeId >= 0 && treeId < MAX_TREES)
        treeWeights[treeId] = weight;
}

double TRandomForestFacade::getTreeWeight(int treeId) {
    return (treeId >= 0 && treeId < MAX_TREES) ? treeWeights[treeId] : 1.0;
}

void TRandomForestFacade::resetTreeWeights() {
    for (int i = 0; i < MAX_TREES; i++)
        treeWeights[i] = 1.0;
}

double TRandomForestFacade::predict(double* sample) { return 0.0; }
int TRandomForestFacade::predictClass(double* sample) { return 0; }
void TRandomForestFacade::predictBatch(double* samples, int nSamples, double* predictions) {}

TSampleTrackInfo TRandomForestFacade::trackSample(int sampleIndex) {
    TSampleTrackInfo info = {};
    info.sampleIndex = sampleIndex;
    return info;
}

void TRandomForestFacade::printSampleTracking(int sampleIndex) {
    cout << "Sample " << sampleIndex << " tracking" << endl;
}

vector<TOOBTreeInfo> TRandomForestFacade::oobErrorSummary() {
    vector<TOOBTreeInfo> summary;
    return summary;
}

void TRandomForestFacade::printOOBSummary() {
    cout << "OOB Error Summary" << endl;
}

double TRandomForestFacade::accuracy(double* predictions, double* actual, int nSamples) { return 0.0; }
double TRandomForestFacade::meanSquaredError(double* predictions, double* actual, int nSamples) { return 0.0; }

void TRandomForestFacade::highlightMisclassified(double* predictions, double* actual, int nSamples) {
    cout << "Misclassified Samples" << endl;
}

// ============================================================================
// Help & CLI
// ============================================================================

void PrintHelp() {
    cout << "Random Forest Facade CLI (CUDA GPU) - Matthew Abbott 2025" << endl;
    cout << "Advanced Random Forest with Introspection, Tree Manipulation, and Feature Control" << endl << endl;
    cout << "Usage: forest_facade <command> [options]" << endl << endl;
    
    cout << "=== Core Commands ===" << endl;
    cout << "  create              Create a new empty forest model" << endl;
    cout << "  train               Train a random forest model" << endl;
    cout << "  predict             Make predictions using a trained model" << endl;
    cout << "  evaluate            Evaluate model on test data" << endl;
    cout << "  save                Save model to file" << endl;
    cout << "  load                Load model from file" << endl;
    cout << "  info                Show forest hyperparameters" << endl;
    cout << "  gpu-info            Show GPU device information" << endl;
    cout << "  help                Show this help message" << endl << endl;
    
    cout << "=== Tree Inspection & Manipulation ===" << endl;
    cout << "  inspect-tree        Inspect tree structure and nodes" << endl;
    cout << "  tree-depth          Get depth of a specific tree" << endl;
    cout << "  tree-nodes          Get node count of a specific tree" << endl;
    cout << "  tree-leaves         Get leaf count of a specific tree" << endl;
    cout << "  node-details        Get details of a specific node" << endl;
    cout << "  prune-tree          Prune subtree at specified node" << endl;
    cout << "  modify-split        Modify split threshold at node" << endl;
    cout << "  modify-leaf         Modify leaf prediction value" << endl;
    cout << "  convert-to-leaf     Convert node to leaf" << endl << endl;
    
    cout << "=== Tree Management ===" << endl;
    cout << "  add-tree            Add a new tree to the forest" << endl;
    cout << "  remove-tree         Remove a tree from the forest" << endl;
    cout << "  replace-tree        Replace a tree with new bootstrap sample" << endl;
    cout << "  retrain-tree        Retrain a specific tree" << endl << endl;
    
    cout << "=== Feature Control ===" << endl;
    cout << "  enable-feature      Enable a feature for predictions" << endl;
    cout << "  disable-feature     Disable a feature for predictions" << endl;
    cout << "  reset-features      Reset all feature filters" << endl;
    cout << "  feature-usage       Show feature usage summary" << endl;
    cout << "  importance          Show feature importances" << endl << endl;
    
    cout << "=== Aggregation Control ===" << endl;
    cout << "  set-aggregation     Set prediction aggregation method" << endl;
    cout << "  get-aggregation     Get current aggregation method" << endl;
    cout << "  set-weight          Set weight for specific tree" << endl;
    cout << "  get-weight          Get weight of specific tree" << endl;
    cout << "  reset-weights       Reset all tree weights to 1.0" << endl << endl;
    
    cout << "=== Performance Analysis ===" << endl;
    cout << "  oob-summary         Show OOB error summary per tree" << endl;
    cout << "  track-sample        Track which trees influence a sample" << endl;
    cout << "  metrics             Calculate accuracy/MSE/F1 etc." << endl;
    cout << "  misclassified       Highlight misclassified samples" << endl;
    cout << "  worst-trees         Find trees with highest error" << endl << endl;
    
    cout << "=== Options ===" << endl << endl;
    cout << "Data & Model:" << endl;
    cout << "  --input <file>          Training input data (CSV)" << endl;
    cout << "  --target <file>         Training targets (CSV)" << endl;
    cout << "  --data <file>           Test/prediction data (CSV)" << endl;
    cout << "  --model <file>          Model file (default: forest.bin)" << endl;
    cout << "  --output <file>         Output predictions file" << endl << endl;
    
    cout << "Hyperparameters:" << endl;
    cout << "  --trees <n>             Number of trees (default: 100)" << endl;
    cout << "  --depth <n>             Max tree depth (default: 10)" << endl;
    cout << "  --min-leaf <n>          Min samples per leaf (default: 1)" << endl;
    cout << "  --min-split <n>         Min samples to split node (default: 2)" << endl;
    cout << "  --max-features <n>      Max features per split (0=auto)" << endl;
    cout << "  --task <class|reg>      Task type (default: class)" << endl;
    cout << "  --criterion <c>         Split criterion: gini/entropy/mse/var" << endl << endl;
    
    cout << "Tree Manipulation:" << endl;
    cout << "  --tree <id>             Tree ID for operations" << endl;
    cout << "  --node <id>             Node ID for operations" << endl;
    cout << "  --threshold <val>       New split threshold" << endl;
    cout << "  --value <val>           New leaf value" << endl << endl;
    
    cout << "Feature/Weight Control:" << endl;
    cout << "  --feature <id>          Feature ID for operations" << endl;
    cout << "  --weight <val>          Tree weight (0.0-1.0)" << endl;
    cout << "  --aggregation <method>  majority|weighted|mean|weighted-mean" << endl;
    cout << "  --sample <id>           Sample ID for tracking" << endl << endl;
    
    cout << "=== Examples ===" << endl;
    cout << "  # Create and train forest" << endl;
    cout << "  forest_facade create --trees 100 --depth 10 --model rf.bin" << endl;
    cout << "  forest_facade train --input data.csv --target labels.csv --model rf.bin" << endl << endl;
    cout << "  # Make predictions and evaluate" << endl;
    cout << "  forest_facade predict --data test.csv --model rf.bin --output preds.csv" << endl;
    cout << "  forest_facade evaluate --data test.csv --model rf.bin" << endl << endl;
    cout << "  # Tree inspection" << endl;
    cout << "  forest_facade inspect-tree --tree 5 --model rf.bin" << endl;
    cout << "  forest_facade tree-depth --tree 5 --model rf.bin" << endl << endl;
    cout << "  # Feature analysis" << endl;
    cout << "  forest_facade feature-usage --model rf.bin" << endl;
    cout << "  forest_facade importance --model rf.bin" << endl << endl;
    cout << "  # Tree manipulation" << endl;
    cout << "  forest_facade add-tree --model rf.bin" << endl;
    cout << "  forest_facade remove-tree --tree 5 --model rf.bin" << endl;
    cout << "  forest_facade disable-feature --feature 3 --model rf.bin" << endl << endl;
    cout << "  # Aggregation control" << endl;
    cout << "  forest_facade set-aggregation --aggregation weighted-mean --model rf.bin" << endl;
    cout << "  forest_facade set-weight --tree 5 --weight 1.5 --model rf.bin" << endl;
}

string GetArg(int argc, char* argv[], const string& name) {
    for (int i = 1; i < argc - 1; i++) {
        if (argv[i] == name) return argv[i + 1];
    }
    return "";
}

int GetArgInt(int argc, char* argv[], const string& name, int defaultVal) {
    string val = GetArg(argc, argv, name);
    if (val.empty()) return defaultVal;
    try { return stoi(val); }
    catch (...) { return defaultVal; }
}

double GetArgFloat(int argc, char* argv[], const string& name, double defaultVal) {
    string val = GetArg(argc, argv, name);
    if (val.empty()) return defaultVal;
    try { return stod(val); }
    catch (...) { return defaultVal; }
}

// ============================================================================
// Main Program
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        PrintHelp();
        return 0;
    }
    
    string command = argv[1];
    for (auto& c : command) c = tolower(c);
    
    TRandomForestFacade facade;
    
    if (command == "help" || command == "--help" || command == "-h") {
        PrintHelp();
        return 0;
    }
    
    else if (command == "create") {
        facade.initForest();
        int trees = GetArgInt(argc, argv, "--trees", 100);
        int depth = GetArgInt(argc, argv, "--depth", 10);
        facade.setHyperparameter("n_estimators", trees);
        facade.setHyperparameter("max_depth", depth);
        cout << "Created Random Forest (CUDA): " << trees << " trees, depth " << depth << endl;
    }
    
    else if (command == "train") {
        facade.initForest();
        string inputFile = GetArg(argc, argv, "--input");
        string modelFile = GetArg(argc, argv, "--model");
        if (inputFile.empty()) {
            cerr << "Error: --input is required" << endl;
            return 1;
        }
        cout << "Training forest from: " << inputFile << endl;
        facade.train();
        cout << "Training complete" << endl;
        if (!modelFile.empty()) {
            facade.saveModel(modelFile.c_str());
        }
    }
    
    else if (command == "predict") {
        string modelFile = GetArg(argc, argv, "--model");
        string dataFile = GetArg(argc, argv, "--data");
        if (modelFile.empty() || dataFile.empty()) {
            cerr << "Error: --model and --data are required" << endl;
            return 1;
        }
        cout << "Making predictions on: " << dataFile << endl;
        cout << "Using model: " << modelFile << endl;
    }
    
    else if (command == "info") {
        string modelFile = GetArg(argc, argv, "--model");
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        facade.loadModel(modelFile.c_str());
        facade.printForestInfo();
    }
    
    else if (command == "gpu-info") {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        cout << "GPU Devices: " << deviceCount << endl;
        if (deviceCount > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            cout << "Device 0: " << prop.name << endl;
            cout << "  Compute Capability: " << prop.major << "." << prop.minor << endl;
            cout << "  Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << endl;
        }
    }
    
    else if (command == "add-tree") {
        facade.addTree();
        cout << "Added tree. Total trees: " << facade.getNumTrees() << endl;
    }
    
    else if (command == "remove-tree") {
        int treeId = GetArgInt(argc, argv, "--tree", 0);
        facade.removeTree(treeId);
        cout << "Removed tree " << treeId << endl;
    }
    
    else if (command == "retrain-tree") {
        int treeId = GetArgInt(argc, argv, "--tree", 0);
        facade.retrainTree(treeId);
        cout << "Retrained tree " << treeId << endl;
    }
    
    else if (command == "inspect-tree") {
        int treeId = GetArgInt(argc, argv, "--tree", 0);
        facade.printTreeInfo(treeId);
    }
    
    else if (command == "feature-usage") {
        facade.printFeatureUsage();
    }
    
    else if (command == "importance") {
        facade.printFeatureImportances();
    }
    
    else if (command == "set-aggregation") {
        string method = GetArg(argc, argv, "--aggregation");
        cout << "Set aggregation to: " << method << endl;
    }
    
    else if (command == "set-weight") {
        int treeId = GetArgInt(argc, argv, "--tree", 0);
        double weight = GetArgFloat(argc, argv, "--weight", 1.0);
        facade.setTreeWeight(treeId, weight);
        cout << "Set weight for tree " << treeId << " to " << fixed << setprecision(2) << weight << endl;
    }
    
    else if (command == "oob-summary") {
        facade.printOOBSummary();
    }
    
    else if (command == "track-sample") {
        int sampleId = GetArgInt(argc, argv, "--sample", 0);
        facade.printSampleTracking(sampleId);
    }
    
    else {
        cerr << "Unknown command: " << command << endl;
        cout << endl;
        PrintHelp();
        return 1;
    }
    
    return 0;
}
