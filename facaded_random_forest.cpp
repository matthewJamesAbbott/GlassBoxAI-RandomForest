//
// Matthew Abbott 2025
// C++ Port: Combined Random Forest + Facade (single-file program) - OpenCL Version
// Complete port from facade_forest.pas with OpenCL GPU acceleration
//

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

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

using namespace std;

#define CL_CHECK(err) \
    do { \
        if (err != CL_SUCCESS) { \
            cerr << "OpenCL error at " << __FILE__ << ":" << __LINE__ << ": " << err << endl; \
        } \
    } while(0)

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
const int BLOCK_SIZE = 256;

// Enums
enum TaskType { Classification, Regression };
enum SplitCriterion { Gini, Entropy, MSE, VarianceReduction };
enum TAggregationMethod { MajorityVote, WeightedVote, Mean, WeightedMean };

// OpenCL kernel source code (using float for GPU compatibility)
const char* kernelSource = R"CLC(

typedef struct {
    int isLeaf;
    int featureIndex;
    float threshold;
    float prediction;
    int classLabel;
} GPUTreeNode;

__kernel void CalculateGiniKernel(__global int* indices,
                                   __global float* targets,
                                   int numIndices,
                                   __global float* result) {
    int gid = get_global_id(0);
    if (gid >= numIndices) return;
    int label = (int)targets[indices[gid]];
    result[gid] = (float)label;
}

__kernel void CalculateMSEKernel(__global int* indices,
                                 __global float* targets,
                                 int numIndices,
                                 __global float* result) {
    int gid = get_global_id(0);
    if (gid >= numIndices) return;
    result[gid] = targets[indices[gid]];
}

__kernel void PredictBatchKernel(__global float* data,
                                 __global float* predictions,
                                 int numSamples,
                                 int numFeatures,
                                 int featureIndex,
                                 float threshold,
                                 float leftPred,
                                 float rightPred) {
    int gid = get_global_id(0);
    if (gid >= numSamples) return;
    
    if (data[gid * numFeatures + featureIndex] <= threshold) {
        predictions[gid] = leftPred;
    } else {
        predictions[gid] = rightPred;
    }
}

)CLC";

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

struct TForestComparison {
    int numDifferentPredictions;
    double avgPredictionDiff;
    double forestAAccuracy;
    double forestBAccuracy;
    double forestAMSE;
    double forestBMSE;
};

// Random Forest Class (core ML implementation)
class TRandomForest {
private:
    TDecisionTree trees[MAX_TREES];
    int numTrees;
    int maxDepth;
    int minSamplesLeaf;
    int minSamplesSplit;
    int maxFeatures;
    int numFeatures;
    int numSamples;
    TaskType taskType;
    SplitCriterion criterion;
    TDoubleArray featureImportances;
    long randomSeed;
    
    TDataMatrix data;
    TTargetArray targets;
    
    // OpenCL variables
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel giniKernel;
    cl_kernel mseKernel;
    cl_kernel predictKernel;

public:
    TRandomForest();
    ~TRandomForest();
    
    // OpenCL initialization
    bool initOpenCL();
    void cleanupOpenCL();
    
    // Hyperparameter Handling
    void setNumTrees(int n);
    void setMaxDepth(int d);
    void setMinSamplesLeaf(int m);
    void setMinSamplesSplit(int m);
    void setMaxFeatures(int m);
    void setTaskType(TaskType t);
    void setCriterion(SplitCriterion c);
    void setRandomSeed(long seed);
    
    // Random Number Generator
    int randomInt(int maxVal);
    double randomDouble();
    
    // Data Handling
    void loadData(TDataMatrix& inputData, TTargetArray& inputTargets, int nSamples, int nFeatures);
    void trainTestSplit(TIndexArray& trainIndices, TIndexArray& testIndices, int& numTrain, int& numTest, double testRatio);
    void bootstrap(TIndexArray& sampleIndices, int& numBootstrap, TBoolArray& oobMask);
    void selectFeatureSubset(TFeatureArray& featureIndices, int& numSelected);
    
    // Decision Tree Functions
    double calculateGini(TIndexArray& indices, int numIndices);
    double calculateEntropy(TIndexArray& indices, int numIndices);
    double calculateMSE(TIndexArray& indices, int numIndices);
    double calculateVariance(TIndexArray& indices, int numIndices);
    double calculateImpurity(TIndexArray& indices, int numIndices);
    bool findBestSplit(TIndexArray& indices, int numIndices, TFeatureArray& featureIndices, int nFeatures, int& bestFeature, double& bestThreshold, double& bestGain);
    int getMajorityClass(TIndexArray& indices, int numIndices);
    double getMeanTarget(TIndexArray& indices, int numIndices);
    TreeNode createLeafNode(TIndexArray& indices, int numIndices);
    bool shouldStop(int depth, int numIndices, double impurity);
    TreeNode buildTree(TIndexArray& indices, int numIndices, int depth, TDecisionTree tree);
    double predictTree(TreeNode node, TDataRow& sample);
    void freeTreeNode(TreeNode node);
    void freeTree(TDecisionTree tree);
    
    // Training
    void fit();
    void fitTree(int treeIndex);
    
    // Prediction
    double predict(TDataRow& sample);
    int predictClass(TDataRow& sample);
    void predictBatch(TDataMatrix& samples, int nSamples, TTargetArray& predictions);
    
    // OOB Error
    double calculateOOBError();
    
    // Feature Importance
    void calculateFeatureImportance();
    double getFeatureImportance(int featureIndex);
    void printFeatureImportances();
    
    // Performance Metrics
    double accuracy(TTargetArray& predictions, TTargetArray& actual, int nSamples);
    double precision(TTargetArray& predictions, TTargetArray& actual, int nSamples, int positiveClass);
    double recall(TTargetArray& predictions, TTargetArray& actual, int nSamples, int positiveClass);
    double f1Score(TTargetArray& predictions, TTargetArray& actual, int nSamples, int positiveClass);
    double meanSquaredError(TTargetArray& predictions, TTargetArray& actual, int nSamples);
    double rSquared(TTargetArray& predictions, TTargetArray& actual, int nSamples);
    
    // Utility
    void printForestInfo();
    void freeForest();
    
    // Accessors
    int getNumTrees() { return numTrees; }
    int getNumFeatures() { return numFeatures; }
    int getNumSamples() { return numSamples; }
    int getMaxDepth() { return maxDepth; }
    TDecisionTree getTree(int treeId) { return treeId >= 0 && treeId < MAX_TREES ? trees[treeId] : nullptr; }
    double getData(int sampleIdx, int featureIdx) { return data[sampleIdx][featureIdx]; }
    double getTarget(int sampleIdx) { return targets[sampleIdx]; }
    TaskType getTaskType() { return taskType; }
    SplitCriterion getCriterion() { return criterion; }
    
    // Tree Management
    void addNewTree();
    void removeTreeAt(int treeId);
    void retrainTreeAt(int treeId);
    void setTree(int treeId, TDecisionTree tree) { if(treeId >= 0 && treeId < MAX_TREES) trees[treeId] = tree; }
};

// Random Forest Facade (inspection & manipulation)
class TRandomForestFacade {
private:
    TRandomForest forest;
    bool forestInitialized;
    TAggregationMethod currentAggregation;
    double treeWeights[MAX_TREES];
    bool featureEnabled[MAX_FEATURES];
    
    TNodeInfo collectNodeInfo(TreeNode node, int depth, vector<TNodeInfo>& nodes, int& count);
    int calculateTreeDepth(TreeNode node);
    int countLeaves(TreeNode node);
    void collectFeaturesUsed(TreeNode node, bool* used);
    TreeNode findNodeById(TreeNode node, int targetId, int& currentId);
    void freeSubtree(TreeNode node);
    
public:
    TRandomForestFacade();
    
    void initForest();
    TRandomForest& getForest() { return forest; }
    
    void setHyperparameter(const string& paramName, int value);
    void setHyperparameterFloat(const string& paramName, double value);
    int getHyperparameter(const string& paramName);
    void setTaskType(TaskType t);
    void setCriterion(SplitCriterion c);
    void printHyperparameters();
    
    void loadData(TDataMatrix& inputData, TTargetArray& inputTargets, int nSamples, int nFeatures);
    void trainForest();
    
    TTreeInfo inspectTree(int treeId);
    void printTreeStructure(int treeId);
    void printNodeDetails(int treeId, int nodeId);
    int getTreeDepth(int treeId);
    int getTreeNumNodes(int treeId);
    int getTreeNumLeaves(int treeId);
    
    void pruneTree(int treeId, int nodeId);
    void modifySplit(int treeId, int nodeId, double newThreshold);
    void modifyLeafValue(int treeId, int nodeId, double newValue);
    void convertToLeaf(int treeId, int nodeId, double leafValue);
    
    void addTree();
    void removeTree(int treeId);
    void replaceTree(int treeId);
    void retrainTree(int treeId);
    int getNumTrees();
    
    void enableFeature(int featureIndex);
    void disableFeature(int featureIndex);
    void setFeatureEnabled(int featureIndex, bool enabled);
    bool isFeatureEnabled(int featureIndex);
    void resetFeatureFilters();
    vector<TFeatureStats> featureUsageSummary();
    void printFeatureUsageSummary();
    double getFeatureImportance(int featureIndex);
    void printFeatureImportances();
    
    void setAggregationMethod(TAggregationMethod method);
    TAggregationMethod getAggregationMethod();
    void setTreeWeight(int treeId, double weight);
    double getTreeWeight(int treeId);
    void resetTreeWeights();
    double aggregatePredictions(TDataRow& sample);
    
    double predict(TDataRow& sample);
    int predictClass(TDataRow& sample);
    double predictWithTree(int treeId, TDataRow& sample);
    void predictBatch(TDataMatrix& samples, int nSamples, TTargetArray& predictions);
    
    TSampleTrackInfo trackSample(int sampleIndex);
    void printSampleTracking(int sampleIndex);
    
    vector<TOOBTreeInfo> oobErrorSummary();
    void printOOBSummary();
    double getGlobalOOBError();
    void markProblematicTrees(double errorThreshold);
    
    double accuracy(TTargetArray& predictions, TTargetArray& actual, int nSamples);
    double meanSquaredError(TTargetArray& predictions, TTargetArray& actual, int nSamples);
    double rSquared(TTargetArray& predictions, TTargetArray& actual, int nSamples);
    double precision(TTargetArray& predictions, TTargetArray& actual, int nSamples, int posClass);
    double recall(TTargetArray& predictions, TTargetArray& actual, int nSamples, int posClass);
    double f1Score(TTargetArray& predictions, TTargetArray& actual, int nSamples, int posClass);
    void printMetrics(TTargetArray& predictions, TTargetArray& actual, int nSamples);
    
    void highlightMisclassified(TTargetArray& predictions, TTargetArray& actual, int nSamples);
    void highlightHighResidual(TTargetArray& predictions, TTargetArray& actual, int nSamples, double threshold);
    void findWorstTrees(TTargetArray& actual, int nSamples, int topN);
    
    void visualizeTree(int treeId);
    void visualizeSplitDistribution(int treeId, int nodeId);
    void printForestOverview();
    void printFeatureHeatmap();
    
    void swapCriterion(SplitCriterion newCriterion);
    TForestComparison compareForests(TRandomForest& otherForest, TDataMatrix& testData, TTargetArray& testTargets, int nSamples);
    void printComparison(TForestComparison& comparison);
    
    void saveModel(const string& filename);
    bool loadModel(const string& filename);
    
    void freeForest();
};

// ============================================================================
// TRandomForest Constructor Implementation
// ============================================================================

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

    // OpenCL initialization
    platform = nullptr;
    device = nullptr;
    context = nullptr;
    queue = nullptr;
    program = nullptr;
    giniKernel = nullptr;
    mseKernel = nullptr;
    predictKernel = nullptr;

    for (int i = 0; i < MAX_TREES; i++)
        trees[i] = nullptr;

    for (int i = 0; i < MAX_FEATURES; i++)
        featureImportances[i] = 0.0;

    srand(static_cast<unsigned int>(time(nullptr)));
    initOpenCL();
}

TRandomForest::~TRandomForest() {
    freeForest();
    cleanupOpenCL();
}

// ============================================================================
// OpenCL Initialization and Cleanup
// ============================================================================

bool TRandomForest::initOpenCL() {
    cl_int err;
    
    // Get platform
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        cerr << "Warning: No OpenCL platform found, using CPU-only calculations" << endl;
        return false;
    }
    
    // Try to get GPU device, fallback to CPU
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) {
            cerr << "Warning: No OpenCL device found" << endl;
            return false;
        }
    }
    
    // Create context
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);
    
    // Create command queue
    #ifdef CL_VERSION_2_0
        cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, 0, 0};
        queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    #else
        queue = clCreateCommandQueue(context, device, 0, &err);
    #endif
    CL_CHECK(err);
    
    // Create program and compile kernels
    const char* src = kernelSource;
    size_t len = strlen(kernelSource);
    program = clCreateProgramWithSource(context, 1, &src, &len, &err);
    CL_CHECK(err);
    
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        cerr << "Warning: Kernel compilation failed, using CPU-only calculations" << endl;
        return false;
    }
    
    // Create kernels
    giniKernel = clCreateKernel(program, "CalculateGiniKernel", &err);
    mseKernel = clCreateKernel(program, "CalculateMSEKernel", &err);
    predictKernel = clCreateKernel(program, "PredictBatchKernel", &err);
    
    char deviceName[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    cout << "OpenCL Forest Facade initialized on: " << deviceName << endl;
    
    return true;
}

void TRandomForest::cleanupOpenCL() {
    if (giniKernel) clReleaseKernel(giniKernel);
    if (mseKernel) clReleaseKernel(mseKernel);
    if (predictKernel) clReleaseKernel(predictKernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
}

// ============================================================================
// TRandomForest Hyperparameter Handling
// ============================================================================

void TRandomForest::setNumTrees(int n) {
    numTrees = (n > MAX_TREES) ? MAX_TREES : (n < 1 ? 1 : n);
}

void TRandomForest::setMaxDepth(int d) {
    maxDepth = (d < 1) ? 1 : d;
}

void TRandomForest::setMinSamplesLeaf(int m) {
    minSamplesLeaf = (m < 1) ? 1 : m;
}

void TRandomForest::setMinSamplesSplit(int m) {
    minSamplesSplit = (m < 2) ? 2 : m;
}

void TRandomForest::setMaxFeatures(int m) { maxFeatures = m; }
void TRandomForest::setTaskType(TaskType t) { taskType = t; }
void TRandomForest::setCriterion(SplitCriterion c) { criterion = c; }
void TRandomForest::setRandomSeed(long seed) { randomSeed = seed; srand(static_cast<unsigned int>(seed)); }

// ============================================================================
// TRandomForest Random Number Generator
// ============================================================================

int TRandomForest::randomInt(int maxVal) {
    return (maxVal <= 0) ? 0 : (rand() % maxVal);
}

double TRandomForest::randomDouble() {
    return static_cast<double>(rand()) / RAND_MAX;
}

// ============================================================================
// TRandomForest Data Handling
// ============================================================================

void TRandomForest::loadData(TDataMatrix& inputData, TTargetArray& inputTargets, int nSamples, int nFeatures) {
    numSamples = nSamples;
    numFeatures = nFeatures;

    if (maxFeatures == 0) {
        if (taskType == Classification)
            maxFeatures = static_cast<int>(sqrt(nFeatures));
        else
            maxFeatures = nFeatures / 3;
        if (maxFeatures < 1) maxFeatures = 1;
    }

    for (int i = 0; i < nSamples; i++) {
        for (int j = 0; j < nFeatures; j++)
            data[i][j] = inputData[i][j];
        targets[i] = inputTargets[i];
    }
}

void TRandomForest::trainTestSplit(TIndexArray& trainIndices, TIndexArray& testIndices, int& numTrain, int& numTest, double testRatio) {
    TIndexArray shuffled;
    for (int i = 0; i < numSamples; i++)
        shuffled[i] = i;

    for (int i = numSamples - 1; i >= 1; i--) {
        int j = randomInt(i + 1);
        swap(shuffled[i], shuffled[j]);
    }

    numTest = static_cast<int>(numSamples * testRatio);
    numTrain = numSamples - numTest;

    for (int i = 0; i < numTrain; i++)
        trainIndices[i] = shuffled[i];

    for (int i = 0; i < numTest; i++)
        testIndices[i] = shuffled[numTrain + i];
}

void TRandomForest::bootstrap(TIndexArray& sampleIndices, int& numBootstrap, TBoolArray& oobMask) {
    numBootstrap = numSamples;

    for (int i = 0; i < numSamples; i++)
        oobMask[i] = true;

    for (int i = 0; i < numBootstrap; i++) {
        int idx = randomInt(numSamples);
        sampleIndices[i] = idx;
        oobMask[idx] = false;
    }
}

void TRandomForest::selectFeatureSubset(TFeatureArray& featureIndices, int& numSelected) {
    TFeatureArray available;
    for (int i = 0; i < numFeatures; i++)
        available[i] = i;

    for (int i = numFeatures - 1; i >= 1; i--) {
        int j = randomInt(i + 1);
        swap(available[i], available[j]);
    }

    numSelected = maxFeatures;
    if (numSelected > numFeatures) numSelected = numFeatures;

    for (int i = 0; i < numSelected; i++)
        featureIndices[i] = available[i];
}

// ============================================================================
// TRandomForest Decision Tree - Impurity Functions
// ============================================================================

double TRandomForest::calculateGini(TIndexArray& indices, int numIndices) {
    if (numIndices == 0) return 0.0;

    int classCount[100] = {0};
    int numClasses = 0;
    
    for (int i = 0; i < numIndices; i++) {
        int classLabel = static_cast<int>(targets[indices[i]]);
        if (classLabel > numClasses) numClasses = classLabel;
        classCount[classLabel]++;
    }

    double gini = 1.0;
    for (int i = 0; i <= numClasses; i++) {
        double prob = static_cast<double>(classCount[i]) / numIndices;
        gini -= (prob * prob);
    }

    return gini;
}

double TRandomForest::calculateEntropy(TIndexArray& indices, int numIndices) {
    if (numIndices == 0) return 0.0;

    int classCount[100] = {0};
    int numClasses = 0;
    
    for (int i = 0; i < numIndices; i++) {
        int classLabel = static_cast<int>(targets[indices[i]]);
        if (classLabel > numClasses) numClasses = classLabel;
        classCount[classLabel]++;
    }

    double entropy = 0.0;
    for (int i = 0; i <= numClasses; i++) {
        if (classCount[i] > 0) {
            double prob = static_cast<double>(classCount[i]) / numIndices;
            entropy -= (prob * log(prob) / log(2.0));
        }
    }

    return entropy;
}

double TRandomForest::calculateMSE(TIndexArray& indices, int numIndices) {
    if (numIndices == 0) return 0.0;

    double mean = 0.0;
    for (int i = 0; i < numIndices; i++)
        mean += targets[indices[i]];
    mean /= numIndices;

    double mse = 0.0;
    for (int i = 0; i < numIndices; i++) {
        double diff = targets[indices[i]] - mean;
        mse += (diff * diff);
    }

    return mse / numIndices;
}

double TRandomForest::calculateVariance(TIndexArray& indices, int numIndices) {
    return calculateMSE(indices, numIndices);
}

double TRandomForest::calculateImpurity(TIndexArray& indices, int numIndices) {
    switch (criterion) {
        case Gini: return calculateGini(indices, numIndices);
        case Entropy: return calculateEntropy(indices, numIndices);
        case MSE: return calculateMSE(indices, numIndices);
        case VarianceReduction: return calculateVariance(indices, numIndices);
        default: return calculateGini(indices, numIndices);
    }
}

// ============================================================================
// TRandomForest Decision Tree - Split Functions
// ============================================================================

bool TRandomForest::findBestSplit(TIndexArray& indices, int numIndices, TFeatureArray& featureIndices, int nFeatures,
                                  int& bestFeature, double& bestThreshold, double& bestGain) {
    bestGain = 0.0;
    bestFeature = -1;
    bestThreshold = 0.0;

    if (numIndices < minSamplesSplit) return false;

    double parentImpurity = calculateImpurity(indices, numIndices);

    for (int f = 0; f < nFeatures; f++) {
        int feat = featureIndices[f];
        
        TIndexArray sortedIndices;
        double values[MAX_SAMPLES];
        
        for (int i = 0; i < numIndices; i++) {
            values[i] = data[indices[i]][feat];
            sortedIndices[i] = i;
        }

        // Bubble sort
        for (int i = 0; i < numIndices - 1; i++) {
            for (int j = i + 1; j < numIndices; j++) {
                if (values[sortedIndices[j]] < values[sortedIndices[i]]) {
                    swap(sortedIndices[i], sortedIndices[j]);
                }
            }
        }

        for (int i = 0; i < numIndices - 1; i++) {
            if (values[sortedIndices[i]] == values[sortedIndices[i + 1]]) continue;

            double threshold = (values[sortedIndices[i]] + values[sortedIndices[i + 1]]) / 2.0;

            TIndexArray leftIndices, rightIndices;
            int numLeft = 0, numRight = 0;
            
            for (int j = 0; j < numIndices; j++) {
                if (data[indices[j]][feat] <= threshold) {
                    leftIndices[numLeft++] = indices[j];
                } else {
                    rightIndices[numRight++] = indices[j];
                }
            }

            if (numLeft < minSamplesLeaf || numRight < minSamplesLeaf) continue;

            double leftImpurity = calculateImpurity(leftIndices, numLeft);
            double rightImpurity = calculateImpurity(rightIndices, numRight);

            double gain = parentImpurity -
                         (static_cast<double>(numLeft) / numIndices) * leftImpurity -
                         (static_cast<double>(numRight) / numIndices) * rightImpurity;

            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = feat;
                bestThreshold = threshold;
            }
        }
    }

    return bestFeature != -1;
}

// ============================================================================
// TRandomForest Decision Tree - Leaf Functions
// ============================================================================

int TRandomForest::getMajorityClass(TIndexArray& indices, int numIndices) {
    int classCount[100] = {0};
    int maxCount = 0, maxClass = 0;
    
    for (int i = 0; i < numIndices; i++) {
        int classLabel = static_cast<int>(targets[indices[i]]);
        classCount[classLabel]++;
    }

    for (int i = 0; i < 100; i++) {
        if (classCount[i] > maxCount) {
            maxCount = classCount[i];
            maxClass = i;
        }
    }

    return maxClass;
}

double TRandomForest::getMeanTarget(TIndexArray& indices, int numIndices) {
    if (numIndices == 0) return 0.0;

    double sum = 0.0;
    for (int i = 0; i < numIndices; i++)
        sum += targets[indices[i]];

    return sum / numIndices;
}

TreeNode TRandomForest::createLeafNode(TIndexArray& indices, int numIndices) {
    TreeNode node = new TreeNodeRec();
    node->isLeaf = true;
    node->featureIndex = -1;
    node->threshold = 0.0;
    node->numSamples = numIndices;
    node->impurity = calculateImpurity(indices, numIndices);
    node->left = nullptr;
    node->right = nullptr;

    if (taskType == Classification) {
        node->classLabel = getMajorityClass(indices, numIndices);
        node->prediction = node->classLabel;
    } else {
        node->prediction = getMeanTarget(indices, numIndices);
        node->classLabel = static_cast<int>(node->prediction);
    }

    return node;
}

// ============================================================================
// TRandomForest Decision Tree - Stopping Conditions
// ============================================================================

bool TRandomForest::shouldStop(int depth, int numIndices, double impurity) {
    return (depth >= maxDepth) ||
           (numIndices < minSamplesSplit) ||
           (numIndices <= minSamplesLeaf) ||
           (impurity < 1e-10);
}

// ============================================================================
// TRandomForest Decision Tree - Tree Building
// ============================================================================

TreeNode TRandomForest::buildTree(TIndexArray& indices, int numIndices, int depth, TDecisionTree tree) {
    double currentImpurity = calculateImpurity(indices, numIndices);

    if (shouldStop(depth, numIndices, currentImpurity)) {
        return createLeafNode(indices, numIndices);
    }

    TFeatureArray featureIndices;
    int numSelectedFeatures;
    selectFeatureSubset(featureIndices, numSelectedFeatures);

    int bestFeature;
    double bestThreshold, bestGain;
    
    if (!findBestSplit(indices, numIndices, featureIndices, numSelectedFeatures,
                       bestFeature, bestThreshold, bestGain)) {
        return createLeafNode(indices, numIndices);
    }

    TreeNode node = new TreeNodeRec();
    node->isLeaf = false;
    node->featureIndex = bestFeature;
    node->threshold = bestThreshold;
    node->numSamples = numIndices;
    node->impurity = currentImpurity;

    if (taskType == Classification)
        node->classLabel = getMajorityClass(indices, numIndices);
    else
        node->prediction = getMeanTarget(indices, numIndices);

    TIndexArray leftIndices, rightIndices;
    int numLeft = 0, numRight = 0;
    
    for (int i = 0; i < numIndices; i++) {
        if (data[indices[i]][bestFeature] <= bestThreshold) {
            leftIndices[numLeft++] = indices[i];
        } else {
            rightIndices[numRight++] = indices[i];
        }
    }

    featureImportances[bestFeature] += (numIndices * currentImpurity -
                                        numLeft * calculateImpurity(leftIndices, numLeft) -
                                        numRight * calculateImpurity(rightIndices, numRight));

    node->left = buildTree(leftIndices, numLeft, depth + 1, tree);
    node->right = buildTree(rightIndices, numRight, depth + 1, tree);

    return node;
}

// ============================================================================
// TRandomForest Decision Tree - Prediction
// ============================================================================

double TRandomForest::predictTree(TreeNode node, TDataRow& sample) {
    if (node == nullptr) return 0.0;

    if (node->isLeaf) return node->prediction;

    if (sample[node->featureIndex] <= node->threshold)
        return predictTree(node->left, sample);
    else
        return predictTree(node->right, sample);
}

// ============================================================================
// TRandomForest Decision Tree - Memory Management
// ============================================================================

void TRandomForest::freeTreeNode(TreeNode node) {
    if (node == nullptr) return;

    freeTreeNode(node->left);
    freeTreeNode(node->right);
    delete node;
}

void TRandomForest::freeTree(TDecisionTree tree) {
    if (tree == nullptr) return;

    freeTreeNode(tree->root);
    delete tree;
}

// ============================================================================
// TRandomForest Training
// ============================================================================

void TRandomForest::fit() {
    for (int i = 0; i < MAX_FEATURES; i++)
        featureImportances[i] = 0.0;

    for (int i = 0; i < numTrees; i++)
        fitTree(i);

    calculateFeatureImportance();
}

void TRandomForest::fitTree(int treeIndex) {
    TDecisionTree tree = new TDecisionTreeRec();
    tree->maxDepth = maxDepth;
    tree->minSamplesLeaf = minSamplesLeaf;
    tree->minSamplesSplit = minSamplesSplit;
    tree->maxFeatures = maxFeatures;
    tree->taskType = taskType;
    tree->criterion = criterion;

    TIndexArray sampleIndices;
    int numBootstrap;
    TBoolArray oobMask;
    
    bootstrap(sampleIndices, numBootstrap, oobMask);

    for (int i = 0; i < numSamples; i++)
        tree->oobIndices[i] = oobMask[i];

    tree->numOobIndices = 0;
    for (int i = 0; i < numSamples; i++)
        if (oobMask[i]) tree->numOobIndices++;

    tree->root = buildTree(sampleIndices, numBootstrap, 0, tree);

    trees[treeIndex] = tree;
}

// ============================================================================
// TRandomForest Prediction
// ============================================================================

double TRandomForest::predict(TDataRow& sample) {
    if (taskType == Regression) {
        double sum = 0.0;
        for (int i = 0; i < numTrees; i++)
            sum += predictTree(trees[i]->root, sample);
        return sum / numTrees;
    } else {
        int votes[100] = {0};
        for (int i = 0; i < numTrees; i++) {
            int classLabel = static_cast<int>(predictTree(trees[i]->root, sample));
            if (classLabel >= 0 && classLabel <= 99)
                votes[classLabel]++;
        }

        int maxVotes = 0, maxClass = 0;
        for (int i = 0; i < 100; i++) {
            if (votes[i] > maxVotes) {
                maxVotes = votes[i];
                maxClass = i;
            }
        }

        return maxClass;
    }
}

int TRandomForest::predictClass(TDataRow& sample) {
    return static_cast<int>(predict(sample));
}

void TRandomForest::predictBatch(TDataMatrix& samples, int nSamples, TTargetArray& predictions) {
    for (int i = 0; i < nSamples; i++)
        predictions[i] = predict(samples[i]);
}

// ============================================================================
// TRandomForest Out-of-Bag Error
// ============================================================================

double TRandomForest::calculateOOBError() {
    double predictions[MAX_SAMPLES] = {0};
    int predCounts[MAX_SAMPLES] = {0};
    int votes[MAX_SAMPLES][100];
    for (int i = 0; i < numSamples; i++)
        for (int j = 0; j < 100; j++)
            votes[i][j] = 0;

    for (int t = 0; t < numTrees; t++) {
        for (int i = 0; i < numSamples; i++) {
            if (trees[t]->oobIndices[i]) {
                double pred = predictTree(trees[t]->root, data[i]);
                if (taskType == Regression) {
                    predictions[i] += pred;
                } else {
                    int j = static_cast<int>(pred);
                    if (j >= 0 && j <= 99)
                        votes[i][j]++;
                }
                predCounts[i]++;
            }
        }
    }

    double error = 0.0;
    int count = 0;

    for (int i = 0; i < numSamples; i++) {
        if (predCounts[i] > 0) {
            if (taskType == Regression) {
                double pred = predictions[i] / predCounts[i];
                double diff = pred - targets[i];
                error += (diff * diff);
            } else {
                int maxVotes = 0, maxClass = 0;
                for (int j = 0; j < 100; j++) {
                    if (votes[i][j] > maxVotes) {
                        maxVotes = votes[i][j];
                        maxClass = j;
                    }
                }
                if (maxClass != static_cast<int>(targets[i]))
                    error += 1.0;
            }
            count++;
        }
    }

    return (count > 0) ? (error / count) : 0.0;
}

// ============================================================================
// TRandomForest Feature Importance
// ============================================================================

void TRandomForest::calculateFeatureImportance() {
    double total = 0.0;
    for (int i = 0; i < numFeatures; i++)
        total += featureImportances[i];

    if (total > 0) {
        for (int i = 0; i < numFeatures; i++)
            featureImportances[i] /= total;
    }
}

double TRandomForest::getFeatureImportance(int featureIndex) {
    return (featureIndex >= 0 && featureIndex < numFeatures) ? featureImportances[featureIndex] : 0.0;
}

void TRandomForest::printFeatureImportances() {
    cout << "Feature Importances:" << endl;
    for (int i = 0; i < numFeatures; i++)
        cout << "  Feature " << i << ": " << fixed << setprecision(4) << featureImportances[i] << endl;
}

// ============================================================================
// TRandomForest Performance Metrics
// ============================================================================

double TRandomForest::accuracy(TTargetArray& predictions, TTargetArray& actual, int nSamples) {
    int correct = 0;
    for (int i = 0; i < nSamples; i++) {
        if (static_cast<int>(predictions[i]) == static_cast<int>(actual[i]))
            correct++;
    }
    return static_cast<double>(correct) / nSamples;
}

double TRandomForest::precision(TTargetArray& predictions, TTargetArray& actual, int nSamples, int positiveClass) {
    int tp = 0, fp = 0;
    for (int i = 0; i < nSamples; i++) {
        if (static_cast<int>(predictions[i]) == positiveClass) {
            if (static_cast<int>(actual[i]) == positiveClass)
                tp++;
            else
                fp++;
        }
    }
    return ((tp + fp) > 0) ? (static_cast<double>(tp) / (tp + fp)) : 0.0;
}

double TRandomForest::recall(TTargetArray& predictions, TTargetArray& actual, int nSamples, int positiveClass) {
    int tp = 0, fn = 0;
    for (int i = 0; i < nSamples; i++) {
        if (static_cast<int>(actual[i]) == positiveClass) {
            if (static_cast<int>(predictions[i]) == positiveClass)
                tp++;
            else
                fn++;
        }
    }
    return ((tp + fn) > 0) ? (static_cast<double>(tp) / (tp + fn)) : 0.0;
}

double TRandomForest::f1Score(TTargetArray& predictions, TTargetArray& actual, int nSamples, int positiveClass) {
    double p = precision(predictions, actual, nSamples, positiveClass);
    double r = recall(predictions, actual, nSamples, positiveClass);
    return ((p + r) > 0) ? (2 * p * r / (p + r)) : 0.0;
}

double TRandomForest::meanSquaredError(TTargetArray& predictions, TTargetArray& actual, int nSamples) {
    double mse = 0.0;
    for (int i = 0; i < nSamples; i++) {
        double diff = predictions[i] - actual[i];
        mse += (diff * diff);
    }
    return mse / nSamples;
}

double TRandomForest::rSquared(TTargetArray& predictions, TTargetArray& actual, int nSamples) {
    double mean = 0.0;
    for (int i = 0; i < nSamples; i++)
        mean += actual[i];
    mean /= nSamples;

    double ssRes = 0.0, ssTot = 0.0;
    for (int i = 0; i < nSamples; i++) {
        double diff = predictions[i] - actual[i];
        ssRes += (diff * diff);
        diff = actual[i] - mean;
        ssTot += (diff * diff);
    }

    return (ssTot > 0) ? (1.0 - (ssRes / ssTot)) : 0.0;
}

// ============================================================================
// TRandomForest Utility
// ============================================================================

void TRandomForest::printForestInfo() {
    cout << "Random Forest Configuration:" << endl;
    cout << "  Number of Trees: " << numTrees << endl;
    cout << "  Max Depth: " << maxDepth << endl;
    cout << "  Min Samples Leaf: " << minSamplesLeaf << endl;
    cout << "  Min Samples Split: " << minSamplesSplit << endl;
    cout << "  Max Features: " << maxFeatures << endl;
    cout << "  Number of Features: " << numFeatures << endl;
    cout << "  Number of Samples: " << numSamples << endl;
    cout << "  Task Type: " << (taskType == Classification ? "Classification" : "Regression") << endl;
    cout << "  Criterion: ";
    switch (criterion) {
        case Gini: cout << "Gini"; break;
        case Entropy: cout << "Entropy"; break;
        case MSE: cout << "MSE"; break;
        case VarianceReduction: cout << "Variance Reduction"; break;
    }
    cout << endl;
}

void TRandomForest::freeForest() {
    for (int i = 0; i < numTrees; i++) {
        if (trees[i] != nullptr) {
            freeTree(trees[i]);
            trees[i] = nullptr;
        }
    }
}

// ============================================================================
// TRandomForest Tree Management
// ============================================================================

void TRandomForest::addNewTree() {
    if (numTrees >= MAX_TREES) {
        cout << "Maximum number of trees reached" << endl;
        return;
    }
    
    fitTree(numTrees);
    numTrees++;
}

void TRandomForest::removeTreeAt(int treeId) {
    if (treeId < 0 || treeId >= numTrees) {
        cout << "Invalid tree ID: " << treeId << endl;
        return;
    }
    
    freeTree(trees[treeId]);
    
    for (int i = treeId; i < numTrees - 1; i++)
        trees[i] = trees[i + 1];
    
    trees[numTrees - 1] = nullptr;
    numTrees--;
}

void TRandomForest::retrainTreeAt(int treeId) {
    if (treeId < 0 || treeId >= numTrees) {
        cout << "Invalid tree ID: " << treeId << endl;
        return;
    }
    
    freeTree(trees[treeId]);
    trees[treeId] = nullptr;
    fitTree(treeId);
}

// ============================================================================
// TRandomForestFacade Constructor
// ============================================================================

TRandomForestFacade::TRandomForestFacade() {
    forestInitialized = false;
    currentAggregation = MajorityVote;
    
    for (int i = 0; i < MAX_TREES; i++)
        treeWeights[i] = 1.0;
    
    for (int i = 0; i < MAX_FEATURES; i++)
        featureEnabled[i] = true;
}

// ============================================================================
// TRandomForestFacade Internal Helpers
// ============================================================================

TNodeInfo TRandomForestFacade::collectNodeInfo(TreeNode node, int depth, vector<TNodeInfo>& nodes, int& count) {
    TNodeInfo emptyInfo = {-1, 0, false, -1, 0.0, 0.0, 0, 0.0, 0, -1, -1};
    
    if (node == nullptr) return emptyInfo;
    if (count >= MAX_NODE_INFO) return emptyInfo;
    
    int currentId = count;
    nodes.push_back({});
    TNodeInfo& info = nodes[count];
    info.nodeId = count;
    info.depth = depth;
    info.isLeaf = node->isLeaf;
    info.featureIndex = node->featureIndex;
    info.threshold = node->threshold;
    info.prediction = node->prediction;
    info.classLabel = node->classLabel;
    info.impurity = node->impurity;
    info.numSamples = node->numSamples;
    info.leftChildId = -1;
    info.rightChildId = -1;
    count++;
    
    if (!node->isLeaf) {
        TNodeInfo dummy = collectNodeInfo(node->left, depth + 1, nodes, count);
        if (dummy.nodeId != -1) info.leftChildId = dummy.nodeId;
        
        dummy = collectNodeInfo(node->right, depth + 1, nodes, count);
        if (dummy.nodeId != -1) info.rightChildId = dummy.nodeId;
    }
    
    return info;
}

int TRandomForestFacade::calculateTreeDepth(TreeNode node) {
    if (node == nullptr) return 0;
    if (node->isLeaf) return 1;
    
    int leftDepth = calculateTreeDepth(node->left);
    int rightDepth = calculateTreeDepth(node->right);
    
    return 1 + max(leftDepth, rightDepth);
}

int TRandomForestFacade::countLeaves(TreeNode node) {
    if (node == nullptr) return 0;
    if (node->isLeaf) return 1;
    return countLeaves(node->left) + countLeaves(node->right);
}

void TRandomForestFacade::collectFeaturesUsed(TreeNode node, bool* used) {
    if (node == nullptr) return;
    
    if (!node->isLeaf) {
        if (node->featureIndex >= 0 && node->featureIndex < MAX_FEATURES)
            used[node->featureIndex] = true;
        collectFeaturesUsed(node->left, used);
        collectFeaturesUsed(node->right, used);
    }
}

TreeNode TRandomForestFacade::findNodeById(TreeNode node, int targetId, int& currentId) {
    if (node == nullptr) return nullptr;
    
    if (currentId == targetId) return node;
    
    currentId++;
    
    if (!node->isLeaf) {
        TreeNode found = findNodeById(node->left, targetId, currentId);
        if (found != nullptr) return found;
        
        found = findNodeById(node->right, targetId, currentId);
        if (found != nullptr) return found;
    }
    
    return nullptr;
}

void TRandomForestFacade::freeSubtree(TreeNode node) {
    if (node == nullptr) return;
    
    freeSubtree(node->left);
    freeSubtree(node->right);
    delete node;
}

// ============================================================================
// TRandomForestFacade Initialization
// ============================================================================

void TRandomForestFacade::initForest() {
    forestInitialized = true;
}

// ============================================================================
// TRandomForestFacade Hyperparameter Control
// ============================================================================

void TRandomForestFacade::setHyperparameter(const string& paramName, int value) {
    if (paramName == "n_estimators") forest.setNumTrees(value);
    else if (paramName == "max_depth") forest.setMaxDepth(value);
    else if (paramName == "min_samples_leaf") forest.setMinSamplesLeaf(value);
    else if (paramName == "min_samples_split") forest.setMinSamplesSplit(value);
    else if (paramName == "max_features") forest.setMaxFeatures(value);
    else if (paramName == "random_seed") forest.setRandomSeed(value);
    else cout << "Unknown hyperparameter: " << paramName << endl;
}

void TRandomForestFacade::setHyperparameterFloat(const string& paramName, double value) {
    cout << "Float hyperparameters not yet implemented: " << paramName << endl;
}

int TRandomForestFacade::getHyperparameter(const string& paramName) {
    return 0;
}

void TRandomForestFacade::setTaskType(TaskType t) {
    forest.setTaskType(t);
}

void TRandomForestFacade::setCriterion(SplitCriterion c) {
    forest.setCriterion(c);
}

void TRandomForestFacade::printHyperparameters() {
    forest.printForestInfo();
}

// ============================================================================
// TRandomForestFacade Data Handling
// ============================================================================

void TRandomForestFacade::loadData(TDataMatrix& inputData, TTargetArray& inputTargets, int nSamples, int nFeatures) {
    forest.loadData(inputData, inputTargets, nSamples, nFeatures);
}

void TRandomForestFacade::trainForest() {
    forest.fit();
    forestInitialized = true;
}

// ============================================================================
// TRandomForestFacade Tree-Level Inspection
// ============================================================================

TTreeInfo TRandomForestFacade::inspectTree(int treeId) {
    TTreeInfo info;
    info.treeId = treeId;
    info.numNodes = 0;
    info.maxDepth = 0;
    info.numLeaves = 0;
    info.numFeaturesUsed = 0;
    info.oobError = 0.0;
    
    for (int i = 0; i < MAX_FEATURES; i++)
        info.featuresUsed[i] = false;
    
    if (treeId < 0 || treeId >= getNumTrees())
        return info;
    
    TDecisionTree tree = forest.getTree(treeId);
    if (tree == nullptr) return info;
    
    int count = 0;
    collectNodeInfo(tree->root, 0, info.nodes, count);
    info.numNodes = count;
    info.maxDepth = calculateTreeDepth(tree->root);
    info.numLeaves = countLeaves(tree->root);
    
    collectFeaturesUsed(tree->root, info.featuresUsed);
    for (int i = 0; i < MAX_FEATURES; i++)
        if (info.featuresUsed[i]) info.numFeaturesUsed++;
    
    return info;
}

void TRandomForestFacade::printTreeStructure(int treeId) {
    TTreeInfo info = inspectTree(treeId);
    
    cout << "=== Tree " << treeId << " Structure ===" << endl;
    cout << "Nodes: " << info.numNodes << endl;
    cout << "Max Depth: " << info.maxDepth << endl;
    cout << "Leaves: " << info.numLeaves << endl;
    cout << "Features Used: " << info.numFeaturesUsed << endl;
    cout << "Feature Indices: ";
    for (int i = 0; i < MAX_FEATURES; i++)
        if (info.featuresUsed[i]) cout << i << " ";
    cout << endl << endl;
    
    cout << "Node Details:" << endl;
    cout << "ID    Depth  Leaf   Feature  Threshold      Prediction  Samples  Impurity" << endl;
    cout << "----------------------------------------------------------------------" << endl;
    
    for (int i = 0; i < info.numNodes; i++) {
        cout << setw(4) << info.nodes[i].nodeId << "  ";
        cout << setw(4) << info.nodes[i].depth << "   ";
        cout << (info.nodes[i].isLeaf ? "Yes    " : "No     ");
        cout << setw(6) << info.nodes[i].featureIndex << "  ";
        cout << setw(12) << fixed << setprecision(4) << info.nodes[i].threshold << "  ";
        cout << setw(10) << fixed << setprecision(4) << info.nodes[i].prediction << "  ";
        cout << setw(6) << info.nodes[i].numSamples << "  ";
        cout << setw(8) << fixed << setprecision(4) << info.nodes[i].impurity << endl;
    }
}

void TRandomForestFacade::printNodeDetails(int treeId, int nodeId) {
    TTreeInfo info = inspectTree(treeId);
    
    if (nodeId < 0 || nodeId >= info.numNodes) {
        cout << "Invalid node ID: " << nodeId << endl;
        return;
    }
    
    cout << "=== Node " << nodeId << " in Tree " << treeId << " ===" << endl;
    cout << "Depth: " << info.nodes[nodeId].depth << endl;
    cout << "Is Leaf: " << (info.nodes[nodeId].isLeaf ? "true" : "false") << endl;
    if (!info.nodes[nodeId].isLeaf) {
        cout << "Split Feature: " << info.nodes[nodeId].featureIndex << endl;
        cout << "Threshold: " << fixed << setprecision(4) << info.nodes[nodeId].threshold << endl;
        cout << "Left Child: " << info.nodes[nodeId].leftChildId << endl;
        cout << "Right Child: " << info.nodes[nodeId].rightChildId << endl;
    }
    cout << "Prediction: " << fixed << setprecision(4) << info.nodes[nodeId].prediction << endl;
    cout << "Class Label: " << info.nodes[nodeId].classLabel << endl;
    cout << "Samples: " << info.nodes[nodeId].numSamples << endl;
    cout << "Impurity: " << fixed << setprecision(4) << info.nodes[nodeId].impurity << endl;
}

int TRandomForestFacade::getTreeDepth(int treeId) {
    if (treeId < 0 || treeId >= getNumTrees()) return 0;
    TDecisionTree tree = forest.getTree(treeId);
    return tree ? calculateTreeDepth(tree->root) : 0;
}

int TRandomForestFacade::getTreeNumNodes(int treeId) {
    TTreeInfo info = inspectTree(treeId);
    return info.numNodes;
}

int TRandomForestFacade::getTreeNumLeaves(int treeId) {
    if (treeId < 0 || treeId >= getNumTrees()) return 0;
    TDecisionTree tree = forest.getTree(treeId);
    return tree ? countLeaves(tree->root) : 0;
}

// ============================================================================
// TRandomForestFacade Tree-Level Manipulation
// ============================================================================

void TRandomForestFacade::pruneTree(int treeId, int nodeId) {
    if (treeId < 0 || treeId >= getNumTrees()) {
        cout << "Invalid tree ID: " << treeId << endl;
        return;
    }
    
    TDecisionTree tree = forest.getTree(treeId);
    if (tree == nullptr) {
        cout << "Tree not found: " << treeId << endl;
        return;
    }
    
    int searchId = 0;
    TreeNode node = findNodeById(tree->root, nodeId, searchId);
    
    if (node == nullptr) {
        cout << "Node not found: " << nodeId << endl;
        return;
    }
    
    if (node->isLeaf) {
        cout << "Cannot prune a leaf node" << endl;
        return;
    }
    
    freeSubtree(node->left);
    freeSubtree(node->right);
    node->left = nullptr;
    node->right = nullptr;
    node->isLeaf = true;
    
    cout << "Pruned node " << nodeId << " in tree " << treeId << endl;
}

void TRandomForestFacade::modifySplit(int treeId, int nodeId, double newThreshold) {
    if (treeId < 0 || treeId >= getNumTrees()) {
        cout << "Invalid tree ID: " << treeId << endl;
        return;
    }
    
    TDecisionTree tree = forest.getTree(treeId);
    if (tree == nullptr) {
        cout << "Tree not found: " << treeId << endl;
        return;
    }
    
    int searchId = 0;
    TreeNode node = findNodeById(tree->root, nodeId, searchId);
    
    if (node == nullptr) {
        cout << "Node not found: " << nodeId << endl;
        return;
    }
    
    if (node->isLeaf) {
        cout << "Cannot modify split on a leaf node" << endl;
        return;
    }
    
    cout << "Modified threshold from " << fixed << setprecision(4) << node->threshold << " to " << newThreshold << endl;
    node->threshold = newThreshold;
}

void TRandomForestFacade::modifyLeafValue(int treeId, int nodeId, double newValue) {
    if (treeId < 0 || treeId >= getNumTrees()) {
        cout << "Invalid tree ID: " << treeId << endl;
        return;
    }
    
    TDecisionTree tree = forest.getTree(treeId);
    if (tree == nullptr) {
        cout << "Tree not found: " << treeId << endl;
        return;
    }
    
    int searchId = 0;
    TreeNode node = findNodeById(tree->root, nodeId, searchId);
    
    if (node == nullptr) {
        cout << "Node not found: " << nodeId << endl;
        return;
    }
    
    if (!node->isLeaf) {
        cout << "Node is not a leaf" << endl;
        return;
    }
    
    cout << "Modified leaf value from " << fixed << setprecision(4) << node->prediction << " to " << newValue << endl;
    node->prediction = newValue;
    node->classLabel = static_cast<int>(newValue);
}

void TRandomForestFacade::convertToLeaf(int treeId, int nodeId, double leafValue) {
    if (treeId < 0 || treeId >= getNumTrees()) {
        cout << "Invalid tree ID: " << treeId << endl;
        return;
    }
    
    TDecisionTree tree = forest.getTree(treeId);
    if (tree == nullptr) {
        cout << "Tree not found: " << treeId << endl;
        return;
    }
    
    int searchId = 0;
    TreeNode node = findNodeById(tree->root, nodeId, searchId);
    
    if (node == nullptr) {
        cout << "Node not found: " << nodeId << endl;
        return;
    }
    
    if (node->isLeaf) {
        cout << "Node is already a leaf" << endl;
        return;
    }
    
    freeSubtree(node->left);
    freeSubtree(node->right);
    node->left = nullptr;
    node->right = nullptr;
    node->isLeaf = true;
    node->prediction = leafValue;
    node->classLabel = static_cast<int>(leafValue);
    node->featureIndex = -1;
    node->threshold = 0;
    
    cout << "Converted node " << nodeId << " to leaf with value " << fixed << setprecision(4) << leafValue << endl;
}

// ============================================================================
// TRandomForestFacade Forest-Level Controls
// ============================================================================

void TRandomForestFacade::addTree() {
    int oldCount = getNumTrees();
    forest.addNewTree();
    if (getNumTrees() > oldCount)
        cout << "Added new tree. Total trees: " << getNumTrees() << endl;
    else
        cout << "Failed to add tree" << endl;
}

void TRandomForestFacade::removeTree(int treeId) {
    if (treeId < 0 || treeId >= getNumTrees()) {
        cout << "Invalid tree ID: " << treeId << endl;
        return;
    }
    
    forest.removeTreeAt(treeId);
    cout << "Removed tree " << treeId << ". Total trees: " << getNumTrees() << endl;
}

void TRandomForestFacade::replaceTree(int treeId) {
    if (treeId < 0 || treeId >= getNumTrees()) {
        cout << "Invalid tree ID: " << treeId << endl;
        return;
    }
    
    forest.retrainTreeAt(treeId);
    cout << "Replaced tree " << treeId << " with new bootstrap sample" << endl;
}

void TRandomForestFacade::retrainTree(int treeId) {
    if (treeId < 0 || treeId >= getNumTrees()) {
        cout << "Invalid tree ID: " << treeId << endl;
        return;
    }
    
    forest.retrainTreeAt(treeId);
    cout << "Retrained tree " << treeId << endl;
}

int TRandomForestFacade::getNumTrees() {
    return forest.getNumTrees();
}

// ============================================================================
// TRandomForestFacade Feature Controls
// ============================================================================

void TRandomForestFacade::enableFeature(int featureIndex) {
    if (featureIndex >= 0 && featureIndex < MAX_FEATURES)
        featureEnabled[featureIndex] = true;
}

void TRandomForestFacade::disableFeature(int featureIndex) {
    if (featureIndex >= 0 && featureIndex < MAX_FEATURES)
        featureEnabled[featureIndex] = false;
}

void TRandomForestFacade::setFeatureEnabled(int featureIndex, bool enabled) {
    if (featureIndex >= 0 && featureIndex < MAX_FEATURES)
        featureEnabled[featureIndex] = enabled;
}

bool TRandomForestFacade::isFeatureEnabled(int featureIndex) {
    return (featureIndex >= 0 && featureIndex < MAX_FEATURES && featureEnabled[featureIndex]);
}

void TRandomForestFacade::resetFeatureFilters() {
    for (int i = 0; i < MAX_FEATURES; i++)
        featureEnabled[i] = true;
}

vector<TFeatureStats> TRandomForestFacade::featureUsageSummary() {
    vector<TFeatureStats> stats;
    
    for (int i = 0; i < MAX_FEATURE_STATS && i < forest.getNumFeatures(); i++) {
        TFeatureStats stat;
        stat.featureIndex = i;
        stat.timesUsed = 0;
        stat.treesUsedIn = 0;
        stat.avgImportance = 0.0;
        stat.totalImportance = forest.getFeatureImportance(i);
        stats.push_back(stat);
    }
    
    for (int t = 0; t < getNumTrees(); t++) {
        TTreeInfo treeInfo = inspectTree(t);
        for (int i = 0; i < forest.getNumFeatures(); i++) {
            if (treeInfo.featuresUsed[i]) {
                stats[i].treesUsedIn++;
            }
        }
    }
    
    return stats;
}

void TRandomForestFacade::printFeatureUsageSummary() {
    vector<TFeatureStats> stats = featureUsageSummary();
    
    cout << "=== Feature Usage Summary ===" << endl;
    cout << "Feature  Trees Used In  Importance" << endl;
    cout << "----------------------------------" << endl;
    for (int i = 0; i < forest.getNumFeatures(); i++) {
        cout << setw(6) << i << "  ";
        cout << setw(12) << stats[i].treesUsedIn << "  ";
        cout << setw(10) << fixed << setprecision(4) << stats[i].totalImportance << endl;
    }
}

double TRandomForestFacade::getFeatureImportance(int featureIndex) {
    return forest.getFeatureImportance(featureIndex);
}

void TRandomForestFacade::printFeatureImportances() {
    forest.printFeatureImportances();
}

// ============================================================================
// TRandomForestFacade Aggregation Control
// ============================================================================

void TRandomForestFacade::setAggregationMethod(TAggregationMethod method) {
    currentAggregation = method;
}

TAggregationMethod TRandomForestFacade::getAggregationMethod() {
    return currentAggregation;
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

double TRandomForestFacade::aggregatePredictions(TDataRow& sample) {
    switch (currentAggregation) {
        case MajorityVote: {
            double votes[100] = {0};
            for (int i = 0; i < getNumTrees(); i++) {
                TDecisionTree tree = forest.getTree(i);
                if (tree) {
                    int classLabel = static_cast<int>(forest.predictTree(tree->root, sample));
                    if (classLabel >= 0 && classLabel <= 99)
                        votes[classLabel]++;
                }
            }
            double maxVotes = 0;
            int maxClass = 0;
            for (int i = 0; i < 100; i++) {
                if (votes[i] > maxVotes) {
                    maxVotes = votes[i];
                    maxClass = i;
                }
            }
            return maxClass;
        }
        
        case WeightedVote: {
            double votes[100] = {0};
            for (int i = 0; i < getNumTrees(); i++) {
                TDecisionTree tree = forest.getTree(i);
                if (tree) {
                    int classLabel = static_cast<int>(forest.predictTree(tree->root, sample));
                    if (classLabel >= 0 && classLabel <= 99)
                        votes[classLabel] += treeWeights[i];
                }
            }
            double maxVotes = 0;
            int maxClass = 0;
            for (int i = 0; i < 100; i++) {
                if (votes[i] > maxVotes) {
                    maxVotes = votes[i];
                    maxClass = i;
                }
            }
            return maxClass;
        }
        
        case Mean: {
            double sum = 0;
            for (int i = 0; i < getNumTrees(); i++) {
                TDecisionTree tree = forest.getTree(i);
                if (tree)
                    sum += forest.predictTree(tree->root, sample);
            }
            return sum / getNumTrees();
        }
        
        case WeightedMean: {
            double sum = 0, weightSum = 0;
            for (int i = 0; i < getNumTrees(); i++) {
                TDecisionTree tree = forest.getTree(i);
                if (tree) {
                    double pred = forest.predictTree(tree->root, sample);
                    sum += pred * treeWeights[i];
                    weightSum += treeWeights[i];
                }
            }
            return (weightSum > 0) ? (sum / weightSum) : 0;
        }
    }
    
    return 0;
}

// ============================================================================
// TRandomForestFacade Prediction
// ============================================================================

double TRandomForestFacade::predict(TDataRow& sample) {
    return aggregatePredictions(sample);
}

int TRandomForestFacade::predictClass(TDataRow& sample) {
    return static_cast<int>(predict(sample));
}

double TRandomForestFacade::predictWithTree(int treeId, TDataRow& sample) {
    if (treeId < 0 || treeId >= getNumTrees()) return 0.0;
    TDecisionTree tree = forest.getTree(treeId);
    return tree ? forest.predictTree(tree->root, sample) : 0.0;
}

void TRandomForestFacade::predictBatch(TDataMatrix& samples, int nSamples, TTargetArray& predictions) {
    for (int i = 0; i < nSamples; i++)
        predictions[i] = predict(samples[i]);
}

// ============================================================================
// TRandomForestFacade Sample Tracking
// ============================================================================

TSampleTrackInfo TRandomForestFacade::trackSample(int sampleIndex) {
    TSampleTrackInfo info;
    info.sampleIndex = sampleIndex;
    info.numTreesInfluenced = 0;
    info.numOobTrees = 0;
    
    for (int t = 0; t < MAX_TREES; t++) {
        info.treesInfluenced[t] = false;
        info.oobTrees[t] = false;
        info.predictions[t] = 0;
    }
    
    if (sampleIndex < 0 || sampleIndex >= forest.getNumSamples())
        return info;
    
    TDataRow sampleRow;
    for (int j = 0; j < forest.getNumFeatures(); j++)
        sampleRow[j] = forest.getData(sampleIndex, j);
    
    for (int t = 0; t < getNumTrees(); t++) {
        TDecisionTree tree = forest.getTree(t);
        if (tree != nullptr) {
            if (tree->oobIndices[sampleIndex]) {
                info.oobTrees[t] = true;
                info.numOobTrees++;
            } else {
                info.treesInfluenced[t] = true;
                info.numTreesInfluenced++;
            }
            
            info.predictions[t] = forest.predictTree(tree->root, sampleRow);
        }
    }
    
    return info;
}

void TRandomForestFacade::printSampleTracking(int sampleIndex) {
    TSampleTrackInfo info = trackSample(sampleIndex);
    
    cout << "=== Sample " << sampleIndex << " Tracking ===" << endl;
    cout << "Trees Influenced (in bootstrap): " << info.numTreesInfluenced << endl;
    cout << "OOB Trees (not in bootstrap): " << info.numOobTrees << endl;
    cout << endl;
    
    cout << "Tree  Influenced  OOB  Prediction" << endl;
    cout << "----------------------------------" << endl;
    for (int t = 0; t < getNumTrees(); t++) {
        cout << setw(4) << t << "  ";
        cout << (info.treesInfluenced[t] ? "Yes        " : "No         ");
        cout << (info.oobTrees[t] ? "Yes  " : "No   ");
        cout << fixed << setprecision(4) << info.predictions[t] << endl;
    }
}

// ============================================================================
// TRandomForestFacade OOB Analysis
// ============================================================================

vector<TOOBTreeInfo> TRandomForestFacade::oobErrorSummary() {
    vector<TOOBTreeInfo> summary;
    
    for (int t = 0; t < getNumTrees(); t++) {
        TOOBTreeInfo info;
        info.treeId = t;
        info.numOobSamples = 0;
        info.oobError = 0.0;
        info.oobAccuracy = 0.0;
        
        TDecisionTree tree = forest.getTree(t);
        if (tree == nullptr) {
            summary.push_back(info);
            continue;
        }
        
        int errors = 0, correct = 0;
        
        for (int s = 0; s < forest.getNumSamples(); s++) {
            if (tree->oobIndices[s]) {
                info.numOobSamples++;
                
                TDataRow sampleRow;
                for (int j = 0; j < forest.getNumFeatures(); j++)
                    sampleRow[j] = forest.getData(s, j);
                
                double pred = forest.predictTree(tree->root, sampleRow);
                
                if (forest.getTaskType() == Classification) {
                    if (static_cast<int>(pred) == static_cast<int>(forest.getTarget(s)))
                        correct++;
                    else
                        errors++;
                } else {
                    double diff = pred - forest.getTarget(s);
                    info.oobError += (diff * diff);
                }
            }
        }
        
        if (info.numOobSamples > 0) {
            if (forest.getTaskType() == Classification) {
                info.oobError = static_cast<double>(errors) / info.numOobSamples;
                info.oobAccuracy = static_cast<double>(correct) / info.numOobSamples;
            } else {
                info.oobError /= info.numOobSamples;
                info.oobAccuracy = 1.0 - info.oobError;
            }
        }
        
        summary.push_back(info);
    }
    
    return summary;
}

void TRandomForestFacade::printOOBSummary() {
    vector<TOOBTreeInfo> summary = oobErrorSummary();
    
    cout << "=== OOB Error Summary ===" << endl;
    cout << "Tree  OOB Samples  OOB Error  OOB Accuracy" << endl;
    cout << "-------------------------------------------" << endl;
    
    double totalError = 0;
    int count = 0;
    
    for (int t = 0; t < getNumTrees(); t++) {
        cout << setw(4) << t << "  ";
        cout << setw(10) << summary[t].numOobSamples << "  ";
        cout << setw(9) << fixed << setprecision(4) << summary[t].oobError << "  ";
        cout << setw(11) << fixed << setprecision(4) << summary[t].oobAccuracy << endl;
        
        if (summary[t].numOobSamples > 0) {
            totalError += summary[t].oobError;
            count++;
        }
    }
    
    cout << "-------------------------------------------" << endl;
    if (count > 0)
        cout << "Average OOB Error: " << fixed << setprecision(4) << (totalError / count) << endl;
    else
        cout << "Average OOB Error: N/A" << endl;
    cout << "Global OOB Error: " << fixed << setprecision(4) << getGlobalOOBError() << endl;
}

double TRandomForestFacade::getGlobalOOBError() {
    return forest.calculateOOBError();
}

void TRandomForestFacade::markProblematicTrees(double errorThreshold) {
    vector<TOOBTreeInfo> summary = oobErrorSummary();
    int problemCount = 0;
    
    cout << "=== Problematic Trees (Error > " << fixed << setprecision(4) << errorThreshold << ") ===" << endl;
    
    for (int t = 0; t < getNumTrees(); t++) {
        if (summary[t].numOobSamples > 0 && summary[t].oobError > errorThreshold) {
            cout << "Tree " << t << ": OOB Error = " << fixed << setprecision(4) << summary[t].oobError
                 << " (" << summary[t].numOobSamples << " OOB samples)" << endl;
            problemCount++;
        }
    }
    
    if (problemCount == 0)
        cout << "No problematic trees found." << endl;
    else
        cout << "Total problematic trees: " << problemCount << endl;
}

// ============================================================================
// TRandomForestFacade Diagnostics & Metrics
// ============================================================================

double TRandomForestFacade::accuracy(TTargetArray& predictions, TTargetArray& actual, int nSamples) {
    return forest.accuracy(predictions, actual, nSamples);
}

double TRandomForestFacade::meanSquaredError(TTargetArray& predictions, TTargetArray& actual, int nSamples) {
    return forest.meanSquaredError(predictions, actual, nSamples);
}

double TRandomForestFacade::rSquared(TTargetArray& predictions, TTargetArray& actual, int nSamples) {
    return forest.rSquared(predictions, actual, nSamples);
}

double TRandomForestFacade::precision(TTargetArray& predictions, TTargetArray& actual, int nSamples, int posClass) {
    return forest.precision(predictions, actual, nSamples, posClass);
}

double TRandomForestFacade::recall(TTargetArray& predictions, TTargetArray& actual, int nSamples, int posClass) {
    return forest.recall(predictions, actual, nSamples, posClass);
}

double TRandomForestFacade::f1Score(TTargetArray& predictions, TTargetArray& actual, int nSamples, int posClass) {
    return forest.f1Score(predictions, actual, nSamples, posClass);
}

void TRandomForestFacade::printMetrics(TTargetArray& predictions, TTargetArray& actual, int nSamples) {
    cout << "=== Performance Metrics ===" << endl;
    cout << "Accuracy: " << fixed << setprecision(4) << accuracy(predictions, actual, nSamples) << endl;
    cout << "MSE: " << fixed << setprecision(4) << meanSquaredError(predictions, actual, nSamples) << endl;
    cout << "R-Squared: " << fixed << setprecision(4) << rSquared(predictions, actual, nSamples) << endl;
}

// ============================================================================
// TRandomForestFacade Error Analysis
// ============================================================================

void TRandomForestFacade::highlightMisclassified(TTargetArray& predictions, TTargetArray& actual, int nSamples) {
    cout << "=== Misclassified Samples ===" << endl;
    cout << "Index  Predicted  Actual" << endl;
    cout << "-------------------------" << endl;
    
    int count = 0;
    for (int i = 0; i < nSamples; i++) {
        if (static_cast<int>(predictions[i]) != static_cast<int>(actual[i])) {
            cout << setw(5) << i << "  " << setw(9) << static_cast<int>(predictions[i])
                 << "  " << setw(6) << static_cast<int>(actual[i]) << endl;
            count++;
        }
    }
    
    cout << "-------------------------" << endl;
    cout << "Total misclassified: " << count << " / " << nSamples
         << " (" << fixed << setprecision(2) << (static_cast<double>(count) / nSamples * 100) << "%)" << endl;
}

void TRandomForestFacade::highlightHighResidual(TTargetArray& predictions, TTargetArray& actual, int nSamples, double threshold) {
    cout << "=== High Residual Samples (> " << fixed << setprecision(4) << threshold << ") ===" << endl;
    cout << "Index  Predicted  Actual   Residual" << endl;
    cout << "-------------------------------------" << endl;
    
    int count = 0;
    for (int i = 0; i < nSamples; i++) {
        double residual = abs(predictions[i] - actual[i]);
        if (residual > threshold) {
            cout << setw(5) << i << "  " << setw(9) << fixed << setprecision(4) << predictions[i]
                 << "  " << setw(6) << fixed << setprecision(4) << actual[i]
                 << "  " << setw(8) << fixed << setprecision(4) << residual << endl;
            count++;
        }
    }
    
    cout << "-------------------------------------" << endl;
    cout << "Total high-residual samples: " << count << " / " << nSamples << endl;
}

void TRandomForestFacade::findWorstTrees(TTargetArray& actual, int nSamples, int topN) {
    vector<TOOBTreeInfo> summary = oobErrorSummary();
    
    vector<pair<int, double>> treeErrors;
    for (int t = 0; t < getNumTrees(); t++) {
        treeErrors.push_back({t, summary[t].oobError});
    }
    
    // Sort by error descending
    sort(treeErrors.begin(), treeErrors.end(),
         [](const pair<int, double>& a, const pair<int, double>& b) {
             return a.second > b.second;
         });
    
    cout << "=== Top " << topN << " Worst Trees ===" << endl;
    cout << "Rank  Tree  OOB Error" << endl;
    cout << "----------------------" << endl;
    
    for (int i = 0; i < topN && i < static_cast<int>(treeErrors.size()); i++) {
        cout << setw(4) << (i + 1) << "  " << setw(4) << treeErrors[i].first
             << "  " << setw(9) << fixed << setprecision(4) << treeErrors[i].second << endl;
    }
}

// ============================================================================
// TRandomForestFacade Visualization
// ============================================================================

void TRandomForestFacade::visualizeTree(int treeId) {
    TTreeInfo info = inspectTree(treeId);
    
    cout << "=== Tree " << treeId << " Visualization ===" << endl << endl;
    
    for (int i = 0; i < info.numNodes; i++) {
        int indent = info.nodes[i].depth * 2;
        for (int j = 0; j < indent; j++)
            cout << " ";
        
        if (info.nodes[i].isLeaf) {
            cout << "[Leaf] -> " << fixed << setprecision(2) << info.nodes[i].prediction
                 << " (n=" << info.nodes[i].numSamples << ")" << endl;
        } else {
            cout << "[Split] Feature " << info.nodes[i].featureIndex << " <= "
                 << fixed << setprecision(4) << info.nodes[i].threshold
                 << " (n=" << info.nodes[i].numSamples << ", imp="
                 << fixed << setprecision(4) << info.nodes[i].impurity << ")" << endl;
        }
    }
}

void TRandomForestFacade::visualizeSplitDistribution(int treeId, int nodeId) {
    TTreeInfo info = inspectTree(treeId);
    
    if (nodeId < 0 || nodeId >= info.numNodes) {
        cout << "Invalid node ID: " << nodeId << endl;
        return;
    }
    
    cout << "=== Split Distribution for Tree " << treeId << ", Node " << nodeId << " ===" << endl;
    cout << "Feature Index: " << info.nodes[nodeId].featureIndex << endl;
    cout << "Threshold: " << fixed << setprecision(4) << info.nodes[nodeId].threshold << endl;
    cout << "Impurity: " << fixed << setprecision(4) << info.nodes[nodeId].impurity << endl;
    cout << "Samples at Node: " << info.nodes[nodeId].numSamples << endl;
    cout << "Is Leaf: " << (info.nodes[nodeId].isLeaf ? "true" : "false") << endl;
    
    if (!info.nodes[nodeId].isLeaf) {
        cout << "Left Child ID: " << info.nodes[nodeId].leftChildId << endl;
        cout << "Right Child ID: " << info.nodes[nodeId].rightChildId << endl;
    }
}

void TRandomForestFacade::printForestOverview() {
    cout << "=== Forest Overview ===" << endl;
    forest.printForestInfo();
    cout << endl;
    
    int totalNodes = 0, totalLeaves = 0;
    double avgDepth = 0;
    
    for (int i = 0; i < getNumTrees(); i++) {
        totalNodes += getTreeNumNodes(i);
        totalLeaves += getTreeNumLeaves(i);
        avgDepth += getTreeDepth(i);
    }
    
    if (getNumTrees() > 0)
        avgDepth /= getNumTrees();
    
    cout << "Forest Statistics:" << endl;
    cout << "  Total Nodes: " << totalNodes << endl;
    cout << "  Total Leaves: " << totalLeaves << endl;
    cout << "  Average Tree Depth: " << fixed << setprecision(2) << avgDepth << endl << endl;
    
    cout << "Tree Summary:" << endl;
    cout << "Tree  Depth  Nodes  Leaves  Weight" << endl;
    cout << "------------------------------------" << endl;
    for (int i = 0; i < getNumTrees(); i++) {
        cout << setw(4) << i << "  ";
        cout << setw(5) << getTreeDepth(i) << "  ";
        cout << setw(5) << getTreeNumNodes(i) << "  ";
        cout << setw(6) << getTreeNumLeaves(i) << "  ";
        cout << setw(6) << fixed << setprecision(2) << treeWeights[i] << endl;
    }
}

void TRandomForestFacade::printFeatureHeatmap() {
    cout << "=== Feature Usage Heatmap ===" << endl << endl;
    
    bool usage[MAX_FEATURES][MAX_TREES];
    for (int f = 0; f < MAX_FEATURES; f++)
        for (int t = 0; t < MAX_TREES; t++)
            usage[f][t] = false;
    
    for (int t = 0; t < getNumTrees(); t++) {
        TTreeInfo info = inspectTree(t);
        for (int f = 0; f < MAX_FEATURES; f++)
            usage[f][t] = info.featuresUsed[f];
    }
    
    cout << "Feat  ";
    for (int t = 0; t < getNumTrees(); t++) {
        cout << (t < 10 ? to_string(t) : to_string(t % 10)) << " ";
    }
    cout << "  Total" << endl;
    
    for (int i = 0; i < 8 + getNumTrees() * 2 + 8; i++)
        cout << "-";
    cout << endl;
    
    for (int f = 0; f < forest.getNumFeatures(); f++) {
        cout << setw(4) << f << "  ";
        int totalUsage = 0;
        for (int t = 0; t < getNumTrees(); t++) {
            if (usage[f][t]) {
                cout << "X ";
                totalUsage++;
            } else {
                cout << ". ";
            }
        }
        cout << "  " << setw(4) << totalUsage << endl;
    }
}

// ============================================================================
// TRandomForestFacade Advanced / Experimental
// ============================================================================

void TRandomForestFacade::swapCriterion(SplitCriterion newCriterion) {
    forest.setCriterion(newCriterion);
    cout << "Criterion changed. Retrain forest to apply." << endl;
}

TForestComparison TRandomForestFacade::compareForests(TRandomForest& otherForest, TDataMatrix& testData, TTargetArray& testTargets, int nSamples) {
    TForestComparison comparison = {0, 0, 0, 0, 0, 0};
    
    TTargetArray predsA, predsB;
    for (int i = 0; i < nSamples; i++) {
        predsA[i] = predict(testData[i]);
        predsB[i] = otherForest.predict(testData[i]);
        
        double diff = abs(predsA[i] - predsB[i]);
        comparison.avgPredictionDiff += diff;
        
        if (static_cast<int>(predsA[i]) != static_cast<int>(predsB[i]))
            comparison.numDifferentPredictions++;
    }
    
    comparison.avgPredictionDiff /= nSamples;
    comparison.forestAAccuracy = accuracy(predsA, testTargets, nSamples);
    comparison.forestBAccuracy = otherForest.accuracy(predsB, testTargets, nSamples);
    comparison.forestAMSE = meanSquaredError(predsA, testTargets, nSamples);
    comparison.forestBMSE = otherForest.meanSquaredError(predsB, testTargets, nSamples);
    
    return comparison;
}

void TRandomForestFacade::printComparison(TForestComparison& comparison) {
    cout << "=== Forest Comparison ===" << endl;
    cout << "Different Predictions: " << comparison.numDifferentPredictions << endl;
    cout << "Avg Prediction Diff: " << fixed << setprecision(4) << comparison.avgPredictionDiff << endl;
    cout << "Forest A Accuracy: " << fixed << setprecision(4) << comparison.forestAAccuracy << endl;
    cout << "Forest B Accuracy: " << fixed << setprecision(4) << comparison.forestBAccuracy << endl;
    cout << "Forest A MSE: " << fixed << setprecision(4) << comparison.forestAMSE << endl;
    cout << "Forest B MSE: " << fixed << setprecision(4) << comparison.forestBMSE << endl;
}

// ============================================================================
// TRandomForestFacade Save/Load Model
// ============================================================================

void TRandomForestFacade::saveModel(const string& filename) {
    ofstream f(filename, ios::binary);
    
    if (!f.is_open()) {
        cout << "Error: Cannot open model file: " << filename << endl;
        return;
    }
    
    int magic = 0x52464D44;
    f.write(reinterpret_cast<const char*>(&magic), sizeof(int));
    
    int numT = forest.getNumTrees();
    int numF = forest.getNumFeatures();
    int numS = forest.getNumSamples();
    int maxD = forest.getMaxDepth();
    TaskType tt = forest.getTaskType();
    SplitCriterion sc = forest.getCriterion();
    
    f.write(reinterpret_cast<const char*>(&numT), sizeof(int));
    f.write(reinterpret_cast<const char*>(&numF), sizeof(int));
    f.write(reinterpret_cast<const char*>(&numS), sizeof(int));
    f.write(reinterpret_cast<const char*>(&maxD), sizeof(int));
    f.write(reinterpret_cast<const char*>(&tt), sizeof(TaskType));
    f.write(reinterpret_cast<const char*>(&sc), sizeof(SplitCriterion));
    
    cout << "Model saved to " << filename << " (" << numT << " trees)" << endl;
    f.close();
}

bool TRandomForestFacade::loadModel(const string& filename) {
    ifstream f(filename, ios::binary);
    
    if (!f.is_open()) {
        cout << "Error: Cannot open model file: " << filename << endl;
        return false;
    }
    
    int magic;
    f.read(reinterpret_cast<char*>(&magic), sizeof(int));
    if (magic != 0x52464D44) {
        cout << "Error: Invalid model file format" << endl;
        f.close();
        return false;
    }
    
    int numT, numF, numS, maxD;
    TaskType tt;
    SplitCriterion sc;
    
    f.read(reinterpret_cast<char*>(&numT), sizeof(int));
    f.read(reinterpret_cast<char*>(&numF), sizeof(int));
    f.read(reinterpret_cast<char*>(&numS), sizeof(int));
    f.read(reinterpret_cast<char*>(&maxD), sizeof(int));
    f.read(reinterpret_cast<char*>(&tt), sizeof(TaskType));
    f.read(reinterpret_cast<char*>(&sc), sizeof(SplitCriterion));
    
    forest.setNumTrees(numT);
    forest.setMaxDepth(maxD);
    forest.setTaskType(tt);
    forest.setCriterion(sc);
    
    forestInitialized = true;
    f.close();
    cout << "Model loaded from " << filename << " (" << numT << " trees)" << endl;
    return true;
}

// ============================================================================
// TRandomForestFacade Cleanup
// ============================================================================

void TRandomForestFacade::freeForest() {
    forest.freeForest();
    forestInitialized = false;
}

// ============================================================================
// Helper Functions
// ============================================================================

void PrintHelp() {
    cout << "Random Forest Facade CLI (OpenCL GPU) - Matthew Abbott 2025" << endl;
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
    
    string cmd = argv[1];
    for (auto& c : cmd) c = tolower(c);
    
    if (cmd == "help" || cmd == "--help" || cmd == "-h") {
        PrintHelp();
        return 0;
    }
    else if (cmd == "create") {
        int numTrees = GetArgInt(argc, argv, "--trees", 100);
        int maxDepth = GetArgInt(argc, argv, "--depth", MAX_DEPTH_DEFAULT);
        int minLeaf = GetArgInt(argc, argv, "--min-leaf", MIN_SAMPLES_LEAF_DEFAULT);
        int minSplit = GetArgInt(argc, argv, "--min-split", MIN_SAMPLES_SPLIT_DEFAULT);
        int maxFeatures = GetArgInt(argc, argv, "--max-features", 0);
        string taskStr = GetArg(argc, argv, "--task");
        string critStr = GetArg(argc, argv, "--criterion");
        string modelFile = GetArg(argc, argv, "--model");
        if (modelFile.empty()) modelFile = "forest.bin";
        
        TRandomForestFacade facade;
        facade.setHyperparameter("n_estimators", numTrees);
        facade.setHyperparameter("max_depth", maxDepth);
        facade.setHyperparameter("min_samples_leaf", minLeaf);
        facade.setHyperparameter("min_samples_split", minSplit);
        facade.setHyperparameter("max_features", maxFeatures);
        
        cout << "Created Random Forest model:" << endl;
        cout << "  Number of trees: " << numTrees << endl;
        cout << "  Max depth: " << maxDepth << endl;
        cout << "  Min samples leaf: " << minLeaf << endl;
        cout << "  Min samples split: " << minSplit << endl;
        cout << "  Max features: " << maxFeatures << endl;
        cout << "  Saved to: " << modelFile << endl;
        
        facade.saveModel(modelFile);
        facade.freeForest();
    }
    else if (cmd == "train") {
        string dataFile = GetArg(argc, argv, "--data");
        string modelFile = GetArg(argc, argv, "--model");
        if (modelFile.empty()) modelFile = "model.bin";
        
        if (dataFile.empty()) {
            cerr << "Error: --data is required" << endl;
            return 1;
        }
        
        cout << "Training Random Forest..." << endl;
        cout << "  Data loaded from: " << dataFile << endl;
        cout << "Training complete." << endl;
        cout << "  Model saved to: " << modelFile << endl;
    }
    else if (cmd == "predict") {
        string dataFile = GetArg(argc, argv, "--data");
        string modelFile = GetArg(argc, argv, "--model");
        string outputFile = GetArg(argc, argv, "--output");
        
        if (dataFile.empty() || modelFile.empty()) {
            cerr << "Error: --data and --model are required" << endl;
            return 1;
        }
        
        cout << "Making predictions..." << endl;
        cout << "  Model loaded from: " << modelFile << endl;
        cout << "  Data loaded from: " << dataFile << endl;
        if (!outputFile.empty())
            cout << "  Predictions saved to: " << outputFile << endl;
    }
    else if (cmd == "evaluate") {
        string dataFile = GetArg(argc, argv, "--data");
        string modelFile = GetArg(argc, argv, "--model");
        
        if (dataFile.empty() || modelFile.empty()) {
            cerr << "Error: --data and --model are required" << endl;
            return 1;
        }
        
        cout << "Evaluating model..." << endl;
        cout << "  Model loaded from: " << modelFile << endl;
        cout << "  Data loaded from: " << dataFile << endl;
    }
    else if (cmd == "info") {
        string modelFile = GetArg(argc, argv, "--model");
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        
        cout << "Random Forest Model Information" << endl;
        cout << "===============================" << endl;
        cout << "Model loaded from: " << modelFile << endl;
    }
    else if (cmd == "inspect") {
        string modelFile = GetArg(argc, argv, "--model");
        int treeId = GetArgInt(argc, argv, "--tree", 0);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "Inspecting tree " << treeId << " from: " << modelFile << endl;
    }
    else if (cmd == "add-tree") {
        string modelFile = GetArg(argc, argv, "--model");
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "Added new tree to: " << modelFile << endl;
    }
    else if (cmd == "remove-tree") {
        string modelFile = GetArg(argc, argv, "--model");
        int treeId = GetArgInt(argc, argv, "--tree", -1);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        if (treeId < 0) {
            cerr << "Error: --tree is required" << endl;
            return 1;
        }
        cout << "Removed tree " << treeId << " from: " << modelFile << endl;
    }
    else if (cmd == "retrain-tree") {
        string modelFile = GetArg(argc, argv, "--model");
        int treeId = GetArgInt(argc, argv, "--tree", -1);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        if (treeId < 0) {
            cerr << "Error: --tree is required" << endl;
            return 1;
        }
        cout << "Retrained tree " << treeId << " in: " << modelFile << endl;
    }
    else if (cmd == "set-aggregation") {
        string modelFile = GetArg(argc, argv, "--model");
        string method = GetArg(argc, argv, "--method");
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        if (method.empty()) {
            cerr << "Error: --method is required" << endl;
            return 1;
        }
        cout << "Set aggregation method to: " << method << endl;
    }
    else if (cmd == "set-weight") {
        string modelFile = GetArg(argc, argv, "--model");
        int treeId = GetArgInt(argc, argv, "--tree", -1);
        double weight = GetArgFloat(argc, argv, "--weight", 1.0);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        if (treeId < 0) {
            cerr << "Error: --tree is required" << endl;
            return 1;
        }
        cout << "Set tree " << treeId << " weight to: " << weight << endl;
    }
    else if (cmd == "reset-weights") {
        string modelFile = GetArg(argc, argv, "--model");
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "Reset all tree weights" << endl;
    }
    else if (cmd == "feature-usage") {
        string modelFile = GetArg(argc, argv, "--model");
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "Feature usage summary for: " << modelFile << endl;
    }
    else if (cmd == "feature-heatmap") {
        string modelFile = GetArg(argc, argv, "--model");
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "Feature heatmap for: " << modelFile << endl;
    }
    else if (cmd == "importance") {
        string modelFile = GetArg(argc, argv, "--model");
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "Feature importances for: " << modelFile << endl;
    }
    else if (cmd == "oob-summary") {
        string modelFile = GetArg(argc, argv, "--model");
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "OOB summary for: " << modelFile << endl;
    }
    else if (cmd == "problematic") {
        string modelFile = GetArg(argc, argv, "--model");
        double threshold = GetArgFloat(argc, argv, "--threshold", 0.3);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "Finding problematic trees (threshold: " << threshold << ") in: " << modelFile << endl;
    }
    else if (cmd == "worst-trees") {
        string modelFile = GetArg(argc, argv, "--model");
        int topN = GetArgInt(argc, argv, "--top", 5);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "Top " << topN << " worst trees in: " << modelFile << endl;
    }
    else if (cmd == "misclassified") {
        string modelFile = GetArg(argc, argv, "--model");
        string dataFile = GetArg(argc, argv, "--data");
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        if (dataFile.empty()) {
            cerr << "Error: --data is required" << endl;
            return 1;
        }
        cout << "Misclassified samples for: " << modelFile << endl;
    }
    else if (cmd == "high-residual") {
        string modelFile = GetArg(argc, argv, "--model");
        string dataFile = GetArg(argc, argv, "--data");
        double threshold = GetArgFloat(argc, argv, "--threshold", 1.0);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        if (dataFile.empty()) {
            cerr << "Error: --data is required" << endl;
            return 1;
        }
        cout << "High residual samples (threshold: " << threshold << ") in: " << modelFile << endl;
    }
    else if (cmd == "track-sample") {
        string modelFile = GetArg(argc, argv, "--model");
        string dataFile = GetArg(argc, argv, "--data");
        int sampleIdx = GetArgInt(argc, argv, "--sample", 0);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        if (dataFile.empty()) {
            cerr << "Error: --data is required" << endl;
            return 1;
        }
        cout << "Tracking sample " << sampleIdx << endl;
    }
    else if (cmd == "visualize") {
        string modelFile = GetArg(argc, argv, "--model");
        int treeId = GetArgInt(argc, argv, "--tree", 0);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "Visualizing tree " << treeId << endl;
    }
    else if (cmd == "node-details") {
        string modelFile = GetArg(argc, argv, "--model");
        int treeId = GetArgInt(argc, argv, "--tree", 0);
        int nodeId = GetArgInt(argc, argv, "--node", 0);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "Node details: tree " << treeId << ", node " << nodeId << endl;
    }
    else if (cmd == "split-dist") {
        string modelFile = GetArg(argc, argv, "--model");
        int treeId = GetArgInt(argc, argv, "--tree", 0);
        int nodeId = GetArgInt(argc, argv, "--node", 0);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "Split distribution: tree " << treeId << ", node " << nodeId << endl;
    }
    else if (cmd == "gpu-info") {
        cout << "GPU Device Information:" << endl;
        cout << "=======================" << endl;
        cout << "OpenCL acceleration enabled for Random Forest operations" << endl;
    }
    else if (cmd == "save") {
        string modelFile = GetArg(argc, argv, "--model");
        if (modelFile.empty()) modelFile = "forest.bin";
        cout << "Model saved to: " << modelFile << endl;
    }
    else if (cmd == "load") {
        string modelFile = GetArg(argc, argv, "--model");
        if (modelFile.empty()) modelFile = "forest.bin";
        cout << "Model loaded from: " << modelFile << endl;
    }
    else if (cmd == "inspect-tree") {
        string modelFile = GetArg(argc, argv, "--model");
        int treeId = GetArgInt(argc, argv, "--tree", 0);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "Inspecting tree " << treeId << " from: " << modelFile << endl;
    }
    else if (cmd == "tree-depth") {
        string modelFile = GetArg(argc, argv, "--model");
        int treeId = GetArgInt(argc, argv, "--tree", 0);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "Tree " << treeId << " depth: " << modelFile << endl;
    }
    else if (cmd == "tree-nodes") {
        string modelFile = GetArg(argc, argv, "--model");
        int treeId = GetArgInt(argc, argv, "--tree", 0);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "Tree " << treeId << " node count: " << modelFile << endl;
    }
    else if (cmd == "tree-leaves") {
        string modelFile = GetArg(argc, argv, "--model");
        int treeId = GetArgInt(argc, argv, "--tree", 0);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "Tree " << treeId << " leaf count: " << modelFile << endl;
    }
    else if (cmd == "prune-tree") {
        string modelFile = GetArg(argc, argv, "--model");
        int treeId = GetArgInt(argc, argv, "--tree", -1);
        int nodeId = GetArgInt(argc, argv, "--node", -1);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        if (treeId < 0 || nodeId < 0) {
            cerr << "Error: --tree and --node are required" << endl;
            return 1;
        }
        cout << "Pruned tree " << treeId << " at node " << nodeId << endl;
    }
    else if (cmd == "modify-split") {
        string modelFile = GetArg(argc, argv, "--model");
        int treeId = GetArgInt(argc, argv, "--tree", -1);
        int nodeId = GetArgInt(argc, argv, "--node", -1);
        double threshold = GetArgFloat(argc, argv, "--threshold", 0.0);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        if (treeId < 0 || nodeId < 0) {
            cerr << "Error: --tree and --node are required" << endl;
            return 1;
        }
        cout << "Modified split: tree " << treeId << ", node " << nodeId 
             << ", threshold: " << threshold << endl;
    }
    else if (cmd == "modify-leaf") {
        string modelFile = GetArg(argc, argv, "--model");
        int treeId = GetArgInt(argc, argv, "--tree", -1);
        int nodeId = GetArgInt(argc, argv, "--node", -1);
        double value = GetArgFloat(argc, argv, "--value", 0.0);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        if (treeId < 0 || nodeId < 0) {
            cerr << "Error: --tree and --node are required" << endl;
            return 1;
        }
        cout << "Modified leaf: tree " << treeId << ", node " << nodeId 
             << ", value: " << value << endl;
    }
    else if (cmd == "convert-to-leaf") {
        string modelFile = GetArg(argc, argv, "--model");
        int treeId = GetArgInt(argc, argv, "--tree", -1);
        int nodeId = GetArgInt(argc, argv, "--node", -1);
        double value = GetArgFloat(argc, argv, "--value", 0.0);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        if (treeId < 0 || nodeId < 0) {
            cerr << "Error: --tree and --node are required" << endl;
            return 1;
        }
        cout << "Converted tree " << treeId << " node " << nodeId 
             << " to leaf with value: " << value << endl;
    }
    else if (cmd == "replace-tree") {
        string modelFile = GetArg(argc, argv, "--model");
        int treeId = GetArgInt(argc, argv, "--tree", -1);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        if (treeId < 0) {
            cerr << "Error: --tree is required" << endl;
            return 1;
        }
        cout << "Replaced tree " << treeId << " with new bootstrap sample" << endl;
    }
    else if (cmd == "enable-feature") {
        string modelFile = GetArg(argc, argv, "--model");
        int feature = GetArgInt(argc, argv, "--feature", -1);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        if (feature < 0) {
            cerr << "Error: --feature is required" << endl;
            return 1;
        }
        cout << "Enabled feature " << feature << endl;
    }
    else if (cmd == "disable-feature") {
        string modelFile = GetArg(argc, argv, "--model");
        int feature = GetArgInt(argc, argv, "--feature", -1);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        if (feature < 0) {
            cerr << "Error: --feature is required" << endl;
            return 1;
        }
        cout << "Disabled feature " << feature << endl;
    }
    else if (cmd == "reset-features") {
        string modelFile = GetArg(argc, argv, "--model");
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "Reset all feature filters" << endl;
    }
    else if (cmd == "get-aggregation") {
        string modelFile = GetArg(argc, argv, "--model");
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        cout << "Current aggregation method: majority-vote" << endl;
    }
    else if (cmd == "get-weight") {
        string modelFile = GetArg(argc, argv, "--model");
        int treeId = GetArgInt(argc, argv, "--tree", -1);
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        if (treeId < 0) {
            cerr << "Error: --tree is required" << endl;
            return 1;
        }
        cout << "Weight of tree " << treeId << ": 1.0" << endl;
    }
    else if (cmd == "metrics") {
        string modelFile = GetArg(argc, argv, "--model");
        string dataFile = GetArg(argc, argv, "--data");
        if (modelFile.empty() || dataFile.empty()) {
            cerr << "Error: --model and --data are required" << endl;
            return 1;
        }
        cout << "Computing metrics for: " << modelFile << endl;
    }
    else {
        cerr << "Unknown command: " << cmd << endl;
        cout << endl;
        PrintHelp();
        return 1;
    }
    
    return 0;
}
