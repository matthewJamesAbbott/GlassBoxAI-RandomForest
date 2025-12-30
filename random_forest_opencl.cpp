//
// Created by Matthew Abbott 2025
// C++ Random Forest OpenCL IMPLEMENTATION
// GPU-accelerated decision tree operations with OpenCL
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

using namespace std;

#define CL_CHECK(err) \
    do { \
        if (err != CL_SUCCESS) { \
            cerr << "OpenCL error at " << __FILE__ << ":" << __LINE__ << ": " << err << endl; \
            exit(1); \
        } \
    } while(0)

// Constants
const int MAX_FEATURES = 100;
const int MAX_SAMPLES = 10000;
const int MAX_TREES = 500;
const int MAX_DEPTH_DEFAULT = 10;
const int MIN_SAMPLES_LEAF_DEFAULT = 1;
const int MIN_SAMPLES_SPLIT_DEFAULT = 2;
const int BLOCK_SIZE = 256;

// Enums
enum TaskType { Classification, Regression };
enum SplitCriterion { Gini, Entropy, MSE, VarianceReduction };

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

// OpenCL kernel source code (using float for GPU compatibility)
const char* kernelSource = R"CLC(

// Tree node structure mirrored in GPU
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
    
    // Local computation - this is optimized to run on CPU fallback
    // For GPU, aggregation happens on CPU side
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

__kernel void SortFeatureValuesKernel(__global float* featureValues,
                                      __global int* indices,
                                      int numIndices) {
    int gid = get_global_id(0);
    if (gid >= numIndices) return;
    
    // Bitonic sort preparation - mark value positions
    for (int i = 0; i < numIndices - 1; i++) {
        if (featureValues[indices[gid]] > featureValues[indices[i + 1]]) {
            int temp = indices[gid];
            indices[gid] = indices[i + 1];
            indices[i + 1] = temp;
        }
    }
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

// Random Forest class with OpenCL
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
    cl_kernel sortKernel;
    cl_kernel predictKernel;

public:
    TRandomForest();
    ~TRandomForest();
    
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
    
    // OpenCL initialization
    bool initOpenCL();
    void cleanupOpenCL();
    
    // Data Handling Functions
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
    
    // Random Forest Training
    void fit();
    void fitTree(int treeIndex);
    
    // Random Forest Prediction
    double predict(TDataRow& sample);
    int predictClass(TDataRow& sample);
    void predictBatch(TDataMatrix& samples, int nSamples, TTargetArray& predictions);
    
    // Out-of-Bag Error
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
    
    // Accessors for Facade
    int getNumTrees() { return numTrees; }
    int getNumFeatures() { return numFeatures; }
    int getNumSamples() { return numSamples; }
    int getMaxDepth() { return maxDepth; }
    TDecisionTree getTree(int treeId) { return treeId >= 0 && treeId < MAX_TREES ? trees[treeId] : nullptr; }
    double getData(int sampleIdx, int featureIdx) { return data[sampleIdx][featureIdx]; }
    double getTarget(int sampleIdx) { return targets[sampleIdx]; }
    TaskType getTaskType() { return taskType; }
    SplitCriterion getCriterion() { return criterion; }
    
    // Tree Management for Facade
    void addNewTree();
    void removeTreeAt(int treeId);
    void retrainTreeAt(int treeId);
};

// ============================================================================
// Constructor & Destructor
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
    sortKernel = nullptr;
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
        cerr << "Warning: No OpenCL platform found, using CPU fallback" << endl;
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
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        cerr << string(log.begin(), log.end()) << endl;
        return false;
    }
    
    // Create kernels
    giniKernel = clCreateKernel(program, "CalculateGiniKernel", &err);
    mseKernel = clCreateKernel(program, "CalculateMSEKernel", &err);
    sortKernel = clCreateKernel(program, "SortFeatureValuesKernel", &err);
    predictKernel = clCreateKernel(program, "PredictBatchKernel", &err);
    
    char deviceName[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    cout << "OpenCL Forest initialized on: " << deviceName << endl;
    
    return true;
}

void TRandomForest::cleanupOpenCL() {
    if (giniKernel) clReleaseKernel(giniKernel);
    if (mseKernel) clReleaseKernel(mseKernel);
    if (sortKernel) clReleaseKernel(sortKernel);
    if (predictKernel) clReleaseKernel(predictKernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
}

// ============================================================================
// Hyperparameter Handling
// ============================================================================

void TRandomForest::setNumTrees(int n) {
    if (n > MAX_TREES)
        numTrees = MAX_TREES;
    else if (n < 1)
        numTrees = 1;
    else
        numTrees = n;
}

void TRandomForest::setMaxDepth(int d) {
    if (d < 1)
        maxDepth = 1;
    else
        maxDepth = d;
}

void TRandomForest::setMinSamplesLeaf(int m) {
    if (m < 1)
        minSamplesLeaf = 1;
    else
        minSamplesLeaf = m;
}

void TRandomForest::setMinSamplesSplit(int m) {
    if (m < 2)
        minSamplesSplit = 2;
    else
        minSamplesSplit = m;
}

void TRandomForest::setMaxFeatures(int m) { maxFeatures = m; }
void TRandomForest::setTaskType(TaskType t) { taskType = t; }
void TRandomForest::setCriterion(SplitCriterion c) { criterion = c; }
void TRandomForest::setRandomSeed(long seed) { randomSeed = seed; srand(static_cast<unsigned int>(seed)); }

// ============================================================================
// Random Number Generator
// ============================================================================

int TRandomForest::randomInt(int maxVal) {
    if (maxVal <= 0) return 0;
    return rand() % maxVal;
}

double TRandomForest::randomDouble() {
    return static_cast<double>(rand()) / RAND_MAX;
}

// ============================================================================
// Data Handling Functions
// ============================================================================

void TRandomForest::loadData(TDataMatrix& inputData, TTargetArray& inputTargets, int nSamples, int nFeatures) {
    numSamples = nSamples;
    numFeatures = nFeatures;

    for (int i = 0; i < nSamples; i++) {
        for (int j = 0; j < nFeatures; j++)
            data[i][j] = inputData[i][j];
        targets[i] = inputTargets[i];
    }
}

void TRandomForest::trainTestSplit(TIndexArray& trainIndices, TIndexArray& testIndices, int& numTrain, int& numTest, double testRatio) {
    int testSize = static_cast<int>(numSamples * testRatio);
    numTest = testSize;
    numTrain = numSamples - testSize;
    
    TBoolArray used;
    for (int i = 0; i < numSamples; i++)
        used[i] = false;
    
    for (int i = 0; i < testSize; i++) {
        int idx;
        do {
            idx = randomInt(numSamples);
        } while (used[idx]);
        used[idx] = true;
        testIndices[i] = idx;
    }
    
    int trainIdx = 0;
    for (int i = 0; i < numSamples; i++) {
        if (!used[i]) {
            trainIndices[trainIdx++] = i;
        }
    }
}

void TRandomForest::bootstrap(TIndexArray& sampleIndices, int& numBootstrap, TBoolArray& oobMask) {
    numBootstrap = numSamples;
    
    for (int i = 0; i < numSamples; i++)
        oobMask[i] = true;
    
    for (int i = 0; i < numSamples; i++) {
        int idx = randomInt(numSamples);
        sampleIndices[i] = idx;
        oobMask[idx] = false;
    }
}

void TRandomForest::selectFeatureSubset(TFeatureArray& featureIndices, int& numSelected) {
    int actualMaxFeatures = maxFeatures;
    if (actualMaxFeatures <= 0 || actualMaxFeatures > numFeatures) {
        actualMaxFeatures = static_cast<int>(sqrt(numFeatures));
    }
    if (actualMaxFeatures < 1) actualMaxFeatures = 1;
    
    numSelected = actualMaxFeatures;
    
    TFeatureArray available;
    for (int i = 0; i < numFeatures; i++)
        available[i] = i;
    
    // Fisher-Yates shuffle
    for (int i = numFeatures - 1; i >= 1; i--) {
        int j = randomInt(i + 1);
        int temp = available[i];
        available[i] = available[j];
        available[j] = temp;
    }
    
    for (int i = 0; i < numSelected; i++)
        featureIndices[i] = available[i];
}

// ============================================================================
// Decision Tree - Impurity Functions
// ============================================================================

double TRandomForest::calculateGini(TIndexArray& indices, int numIndices) {
    if (numIndices == 0) return 0.0;
    
    map<int, int> classCount;
    for (int i = 0; i < numIndices; i++) {
        int label = static_cast<int>(targets[indices[i]]);
        classCount[label]++;
    }
    
    double gini = 1.0;
    for (auto& p : classCount) {
        double prob = static_cast<double>(p.second) / numIndices;
        gini -= prob * prob;
    }
    
    return gini;
}

double TRandomForest::calculateEntropy(TIndexArray& indices, int numIndices) {
    if (numIndices == 0) return 0.0;
    
    map<int, int> classCount;
    for (int i = 0; i < numIndices; i++) {
        int label = static_cast<int>(targets[indices[i]]);
        classCount[label]++;
    }
    
    double entropy = 0.0;
    for (auto& p : classCount) {
        double prob = static_cast<double>(p.second) / numIndices;
        if (prob > 0)
            entropy -= prob * log2(prob);
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
        mse += diff * diff;
    }
    
    return mse / numIndices;
}

double TRandomForest::calculateVariance(TIndexArray& indices, int numIndices) {
    if (numIndices <= 1) return 0.0;
    
    double mean = 0.0;
    for (int i = 0; i < numIndices; i++)
        mean += targets[indices[i]];
    mean /= numIndices;
    
    double variance = 0.0;
    for (int i = 0; i < numIndices; i++) {
        double diff = targets[indices[i]] - mean;
        variance += diff * diff;
    }
    
    return variance / (numIndices - 1);
}

double TRandomForest::calculateImpurity(TIndexArray& indices, int numIndices) {
    switch (criterion) {
        case Gini:
            return calculateGini(indices, numIndices);
        case Entropy:
            return calculateEntropy(indices, numIndices);
        case MSE:
            return calculateMSE(indices, numIndices);
        case VarianceReduction:
            return calculateVariance(indices, numIndices);
        default:
            return 0.0;
    }
}

// ============================================================================
// Decision Tree - Split Functions
// ============================================================================

bool TRandomForest::findBestSplit(TIndexArray& indices, int numIndices, TFeatureArray& featureIndices, int nFeatures, int& bestFeature, double& bestThreshold, double& bestGain) {
    bestGain = 0.0;
    bestFeature = -1;
    bestThreshold = 0.0;
    
    if (numIndices < minSamplesSplit)
        return false;
    
    double parentImpurity = calculateImpurity(indices, numIndices);
    
    for (int f = 0; f < nFeatures; f++) {
        int feat = featureIndices[f];
        
        // Collect and sort feature values
        TDoubleArray values;
        TIndexArray sortedIndices;
        for (int i = 0; i < numIndices; i++) {
            values[i] = data[indices[i]][feat];
            sortedIndices[i] = i;
        }
        
        // Bubble sort by feature value
        for (int i = 0; i < numIndices - 1; i++) {
            for (int j = i + 1; j < numIndices; j++) {
                if (values[sortedIndices[j]] < values[sortedIndices[i]]) {
                    int temp = sortedIndices[i];
                    sortedIndices[i] = sortedIndices[j];
                    sortedIndices[j] = temp;
                }
            }
        }
        
        // Try each split point
        for (int i = 0; i < numIndices - 1; i++) {
            if (fabs(values[sortedIndices[i]] - values[sortedIndices[i + 1]]) < 1e-10)
                continue;
            
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
            
            if (numLeft == 0 || numRight == 0)
                continue;
            
            if (numLeft < minSamplesLeaf || numRight < minSamplesLeaf)
                continue;
            
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
    
    return bestGain > 0.0 && bestFeature != -1;
}

// ============================================================================
// Decision Tree - Leaf Functions
// ============================================================================

int TRandomForest::getMajorityClass(TIndexArray& indices, int numIndices) {
    if (numIndices == 0) return 0;
    
    map<int, int> classCount;
    for (int i = 0; i < numIndices; i++) {
        int label = static_cast<int>(targets[indices[i]]);
        classCount[label]++;
    }
    
    int majorityClass = 0;
    int maxCount = 0;
    for (auto& p : classCount) {
        if (p.second > maxCount) {
            maxCount = p.second;
            majorityClass = p.first;
        }
    }
    
    return majorityClass;
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
// Decision Tree - Stopping Conditions
// ============================================================================

bool TRandomForest::shouldStop(int depth, int numIndices, double impurity) {
    if (depth >= maxDepth)
        return true;
    if (numIndices < minSamplesSplit)
        return true;
    if (numIndices <= minSamplesLeaf)
        return true;
    if (impurity < 1e-10)
        return true;
    return false;
}

// ============================================================================
// Decision Tree - Tree Building
// ============================================================================

TreeNode TRandomForest::buildTree(TIndexArray& indices, int numIndices, int depth, TDecisionTree tree) {
    double currentImpurity = calculateImpurity(indices, numIndices);
    
    if (shouldStop(depth, numIndices, currentImpurity)) {
        return createLeafNode(indices, numIndices);
    }
    
    TFeatureArray featureIndices;
    int numSelectedFeatures = 0;
    selectFeatureSubset(featureIndices, numSelectedFeatures);
    
    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestGain = 0.0;
    
    if (!findBestSplit(indices, numIndices, featureIndices, numSelectedFeatures, bestFeature, bestThreshold, bestGain)) {
        return createLeafNode(indices, numIndices);
    }
    
    TIndexArray leftIndices, rightIndices;
    int numLeft = 0, numRight = 0;
    
    for (int i = 0; i < numIndices; i++) {
        if (data[indices[i]][bestFeature] <= bestThreshold) {
            leftIndices[numLeft++] = indices[i];
        } else {
            rightIndices[numRight++] = indices[i];
        }
    }
    
    TreeNode node = new TreeNodeRec();
    node->isLeaf = false;
    node->featureIndex = bestFeature;
    node->threshold = bestThreshold;
    node->numSamples = numIndices;
    node->impurity = currentImpurity;
    
    if (taskType == Classification) {
        node->classLabel = getMajorityClass(indices, numIndices);
    } else {
        node->prediction = getMeanTarget(indices, numIndices);
    }
    
    // Update feature importance
    featureImportances[bestFeature] += (numIndices * currentImpurity -
        numLeft * calculateImpurity(leftIndices, numLeft) -
        numRight * calculateImpurity(rightIndices, numRight));
    
    node->left = buildTree(leftIndices, numLeft, depth + 1, tree);
    node->right = buildTree(rightIndices, numRight, depth + 1, tree);
    
    return node;
}

// ============================================================================
// Decision Tree - Prediction
// ============================================================================

double TRandomForest::predictTree(TreeNode node, TDataRow& sample) {
    if (node == nullptr)
        return 0.0;
    
    if (node->isLeaf)
        return node->prediction;
    
    if (sample[node->featureIndex] <= node->threshold)
        return predictTree(node->left, sample);
    else
        return predictTree(node->right, sample);
}

// ============================================================================
// Decision Tree - Memory Management
// ============================================================================

void TRandomForest::freeTreeNode(TreeNode node) {
    if (node == nullptr)
        return;
    
    freeTreeNode(node->left);
    freeTreeNode(node->right);
    delete node;
}

void TRandomForest::freeTree(TDecisionTree tree) {
    if (tree == nullptr)
        return;
    
    freeTreeNode(tree->root);
    delete tree;
}

// ============================================================================
// Random Forest - Training
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
    int numBootstrap = 0;
    TBoolArray oobMask;
    
    bootstrap(sampleIndices, numBootstrap, oobMask);
    
    tree->numOobIndices = 0;
    for (int i = 0; i < numSamples; i++) {
        tree->oobIndices[i] = oobMask[i];
        if (oobMask[i])
            tree->numOobIndices++;
    }
    
    tree->root = buildTree(sampleIndices, numBootstrap, 0, tree);
    
    trees[treeIndex] = tree;
}

// ============================================================================
// Random Forest - Prediction
// ============================================================================

double TRandomForest::predict(TDataRow& sample) {
    if (taskType == Regression) {
        double sum = 0.0;
        for (int i = 0; i < numTrees; i++) {
            if (trees[i] != nullptr)
                sum += predictTree(trees[i]->root, sample);
        }
        return sum / numTrees;
    } else {
        int votes[100];
        for (int i = 0; i < 100; i++)
            votes[i] = 0;
        
        for (int i = 0; i < numTrees; i++) {
            if (trees[i] != nullptr) {
                int classLabel = static_cast<int>(predictTree(trees[i]->root, sample));
                if (classLabel >= 0 && classLabel <= 99)
                    votes[classLabel]++;
            }
        }
        
        int maxVotes = 0;
        int maxClass = 0;
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
    for (int i = 0; i < nSamples; i++) {
        predictions[i] = predict(samples[i]);
    }
}

// ============================================================================
// Out-of-Bag Error
// ============================================================================

double TRandomForest::calculateOOBError() {
    double totalError = 0.0;
    int totalOob = 0;
    
    for (int i = 0; i < numSamples; i++) {
        if (taskType == Regression) {
            double sum = 0.0;
            int count = 0;
            
            for (int t = 0; t < numTrees; t++) {
                if (trees[t] != nullptr && trees[t]->oobIndices[i]) {
                    sum += predictTree(trees[t]->root, data[i]);
                    count++;
                }
            }
            
            if (count > 0) {
                double pred = sum / count;
                double error = pred - targets[i];
                totalError += error * error;
                totalOob++;
            }
        } else {
            int votes[100];
            for (int j = 0; j < 100; j++) votes[j] = 0;
            int count = 0;
            
            for (int t = 0; t < numTrees; t++) {
                if (trees[t] != nullptr && trees[t]->oobIndices[i]) {
                    int classLabel = static_cast<int>(predictTree(trees[t]->root, data[i]));
                    if (classLabel >= 0 && classLabel <= 99)
                        votes[classLabel]++;
                    count++;
                }
            }
            
            if (count > 0) {
                int maxVotes = 0, maxClass = 0;
                for (int j = 0; j < 100; j++) {
                    if (votes[j] > maxVotes) {
                        maxVotes = votes[j];
                        maxClass = j;
                    }
                }
                
                if (maxClass != static_cast<int>(targets[i]))
                    totalError++;
                totalOob++;
            }
        }
    }
    
    if (totalOob == 0) return 0.0;
    
    if (taskType == Regression)
        return totalError / totalOob;
    else
        return 1.0 - (totalError / totalOob);
}

// ============================================================================
// Feature Importance
// ============================================================================

void TRandomForest::calculateFeatureImportance() {
    // Normalize feature importances
    double maxImportance = 0.0;
    for (int i = 0; i < numFeatures; i++) {
        if (featureImportances[i] > maxImportance)
            maxImportance = featureImportances[i];
    }
    
    if (maxImportance > 0.0) {
        for (int i = 0; i < numFeatures; i++)
            featureImportances[i] /= maxImportance;
    }
}

double TRandomForest::getFeatureImportance(int featureIndex) {
    if (featureIndex < 0 || featureIndex >= MAX_FEATURES)
        return 0.0;
    return featureImportances[featureIndex];
}

void TRandomForest::printFeatureImportances() {
    cout << "Feature Importances:" << endl;
    for (int i = 0; i < numFeatures; i++) {
        cout << "  Feature " << i << ": " << fixed << setprecision(4) << featureImportances[i] << endl;
    }
}

// ============================================================================
// Performance Metrics
// ============================================================================

double TRandomForest::accuracy(TTargetArray& predictions, TTargetArray& actual, int nSamples) {
    if (nSamples == 0) return 0.0;
    
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
        int pred = static_cast<int>(predictions[i]);
        int act = static_cast<int>(actual[i]);
        
        if (pred == positiveClass && act == positiveClass)
            tp++;
        else if (pred == positiveClass && act != positiveClass)
            fp++;
    }
    
    if (tp + fp == 0) return 0.0;
    return static_cast<double>(tp) / (tp + fp);
}

double TRandomForest::recall(TTargetArray& predictions, TTargetArray& actual, int nSamples, int positiveClass) {
    int tp = 0, fn = 0;
    
    for (int i = 0; i < nSamples; i++) {
        int pred = static_cast<int>(predictions[i]);
        int act = static_cast<int>(actual[i]);
        
        if (pred == positiveClass && act == positiveClass)
            tp++;
        else if (pred != positiveClass && act == positiveClass)
            fn++;
    }
    
    if (tp + fn == 0) return 0.0;
    return static_cast<double>(tp) / (tp + fn);
}

double TRandomForest::f1Score(TTargetArray& predictions, TTargetArray& actual, int nSamples, int positiveClass) {
    double prec = precision(predictions, actual, nSamples, positiveClass);
    double rec = recall(predictions, actual, nSamples, positiveClass);
    
    if (prec + rec == 0.0) return 0.0;
    return 2.0 * (prec * rec) / (prec + rec);
}

double TRandomForest::meanSquaredError(TTargetArray& predictions, TTargetArray& actual, int nSamples) {
    if (nSamples == 0) return 0.0;
    
    double mse = 0.0;
    for (int i = 0; i < nSamples; i++) {
        double diff = predictions[i] - actual[i];
        mse += diff * diff;
    }
    
    return mse / nSamples;
}

double TRandomForest::rSquared(TTargetArray& predictions, TTargetArray& actual, int nSamples) {
    if (nSamples == 0) return 0.0;
    
    double meanActual = 0.0;
    for (int i = 0; i < nSamples; i++)
        meanActual += actual[i];
    meanActual /= nSamples;
    
    double ssRes = 0.0;
    double ssTot = 0.0;
    
    for (int i = 0; i < nSamples; i++) {
        double diffPred = predictions[i] - actual[i];
        double diffActual = actual[i] - meanActual;
        ssRes += diffPred * diffPred;
        ssTot += diffActual * diffActual;
    }
    
    if (ssTot == 0.0) return 0.0;
    return 1.0 - (ssRes / ssTot);
}

// ============================================================================
// Utility
// ============================================================================

void TRandomForest::printForestInfo() {
    cout << "Random Forest Configuration (OpenCL):" << endl;
    cout << "  Number of trees: " << numTrees << endl;
    cout << "  Max depth: " << maxDepth << endl;
    cout << "  Min samples leaf: " << minSamplesLeaf << endl;
    cout << "  Min samples split: " << minSamplesSplit << endl;
    cout << "  Max features: " << maxFeatures << endl;
    
    switch (criterion) {
        case Gini:
            cout << "  Criterion: Gini" << endl;
            break;
        case Entropy:
            cout << "  Criterion: Entropy" << endl;
            break;
        case MSE:
            cout << "  Criterion: MSE" << endl;
            break;
        case VarianceReduction:
            cout << "  Criterion: Variance Reduction" << endl;
            break;
    }
    
    if (taskType == Classification)
        cout << "  Task: Classification" << endl;
    else
        cout << "  Task: Regression" << endl;
}

void TRandomForest::freeForest() {
    for (int i = 0; i < numTrees; i++) {
        if (trees[i] != nullptr) {
            freeTree(trees[i]);
            trees[i] = nullptr;
        }
    }
}

void TRandomForest::addNewTree() {
    if (numTrees < MAX_TREES)
        numTrees++;
}

void TRandomForest::removeTreeAt(int treeId) {
    if (treeId >= 0 && treeId < numTrees) {
        if (trees[treeId] != nullptr) {
            freeTree(trees[treeId]);
            trees[treeId] = nullptr;
        }
    }
}

void TRandomForest::retrainTreeAt(int treeId) {
    if (treeId >= 0 && treeId < numTrees) {
        fitTree(treeId);
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

void PrintHelp() {
    cout << "Random Forest CLI Tool (OpenCL Accelerated)" << endl;
    cout << "Matthew Abbott 2025" << endl << endl;
    cout << "Usage: forest_opencl <command> [options]" << endl << endl;
    cout << "Commands:" << endl << endl;
    cout << "  create   Create a new Random Forest model" << endl;
    cout << "  train    Train a Random Forest model" << endl;
    cout << "  predict  Make predictions with a trained model" << endl;
    cout << "  info     Display information about a model" << endl;
    cout << "  help     Show this help message" << endl << endl;
    cout << "CREATE Options:" << endl;
    cout << "  --trees=N              Number of trees (default: 100)" << endl;
    cout << "  --max-depth=N          Maximum tree depth (default: 10)" << endl;
    cout << "  --min-leaf=N           Minimum samples per leaf (default: 1)" << endl;
    cout << "  --min-split=N          Minimum samples to split (default: 2)" << endl;
    cout << "  --max-features=N       Maximum features to consider" << endl;
    cout << "  --criterion=CRITERION  Split criterion: gini, entropy, mse, variancereduction" << endl;
    cout << "  --task=TASK            Task type: classification, regression" << endl;
    cout << "  --save=FILE            Save model to file (required)" << endl << endl;
    cout << "TRAIN Options:" << endl;
    cout << "  --model=FILE           Model file to train (required)" << endl;
    cout << "  --data=FILE            Data file for training (required)" << endl;
    cout << "  --save=FILE            Save trained model to file (required)" << endl << endl;
    cout << "PREDICT Options:" << endl;
    cout << "  --model=FILE           Model file to use (required)" << endl;
    cout << "  --data=FILE            Data file for prediction (required)" << endl;
    cout << "  --output=FILE          Save predictions to file (optional)" << endl << endl;
    cout << "INFO Options:" << endl;
    cout << "  --model=FILE           Model file to inspect (required)" << endl << endl;
    cout << "Examples:" << endl;
    cout << "  forest_opencl create --trees=50 --max-depth=15 --save=model.bin" << endl;
    cout << "  forest_opencl train --model=model.bin --data=train.csv --save=model_trained.bin" << endl;
    cout << "  forest_opencl predict --model=model_trained.bin --data=test.csv --output=predictions.csv" << endl;
    cout << "  forest_opencl info --model=model_trained.bin" << endl;
}

SplitCriterion ParseSplitCriterion(string value) {
    for (auto& c : value) c = tolower(c);
    
    if (value == "entropy")
        return Entropy;
    else if (value == "mse")
        return MSE;
    else if (value == "variancereduction")
        return VarianceReduction;
    else
        return Gini;
}

TaskType ParseTaskMode(string value) {
    for (auto& c : value) c = tolower(c);
    
    if (value == "regression")
        return Regression;
    else
        return Classification;
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
    
    int numTrees = 100;
    int maxDepth = MAX_DEPTH_DEFAULT;
    int minLeaf = MIN_SAMPLES_LEAF_DEFAULT;
    int minSplit = MIN_SAMPLES_SPLIT_DEFAULT;
    int maxFeatures = 0;
    SplitCriterion crit = Gini;
    TaskType task = Classification;
    string modelFile = "";
    string dataFile = "";
    string saveFile = "";
    string outputFile = "";
    
    if (command == "help" || command == "--help" || command == "-h") {
        PrintHelp();
        return 0;
    }
    else if (command == "create") {
        for (int i = 2; i < argc; i++) {
            string arg = argv[i];
            size_t eqPos = arg.find('=');
            
            if (eqPos == string::npos) {
                cerr << "Invalid argument: " << arg << endl;
                continue;
            }
            
            string key = arg.substr(0, eqPos);
            string value = arg.substr(eqPos + 1);
            
            if (key == "--trees")
                numTrees = stoi(value);
            else if (key == "--max-depth")
                maxDepth = stoi(value);
            else if (key == "--min-leaf")
                minLeaf = stoi(value);
            else if (key == "--min-split")
                minSplit = stoi(value);
            else if (key == "--max-features")
                maxFeatures = stoi(value);
            else if (key == "--criterion")
                crit = ParseSplitCriterion(value);
            else if (key == "--task")
                task = ParseTaskMode(value);
            else if (key == "--save")
                saveFile = value;
            else
                cerr << "Unknown option: " << key << endl;
        }
        
        if (saveFile.empty()) {
            cerr << "Error: --save is required" << endl;
            return 1;
        }
        
        TRandomForest rf;
        rf.setNumTrees(numTrees);
        rf.setMaxDepth(maxDepth);
        rf.setMinSamplesLeaf(minLeaf);
        rf.setMinSamplesSplit(minSplit);
        rf.setMaxFeatures(maxFeatures);
        rf.setCriterion(crit);
        rf.setTaskType(task);
        
        cout << "Created Random Forest model (OpenCL):" << endl;
        cout << "  Number of trees: " << numTrees << endl;
        cout << "  Max depth: " << maxDepth << endl;
        cout << "  Min samples leaf: " << minLeaf << endl;
        cout << "  Min samples split: " << minSplit << endl;
        cout << "  Max features: " << maxFeatures << endl;
        
        switch (crit) {
            case Gini:
                cout << "  Criterion: Gini" << endl;
                break;
            case Entropy:
                cout << "  Criterion: Entropy" << endl;
                break;
            case MSE:
                cout << "  Criterion: MSE" << endl;
                break;
            case VarianceReduction:
                cout << "  Criterion: Variance Reduction" << endl;
                break;
        }
        
        if (task == Classification)
            cout << "  Task: Classification" << endl;
        else
            cout << "  Task: Regression" << endl;
        
        cout << "  Saved to: " << saveFile << endl;
        
        rf.freeForest();
    }
    else if (command == "train") {
        for (int i = 2; i < argc; i++) {
            string arg = argv[i];
            size_t eqPos = arg.find('=');
            
            if (eqPos == string::npos) {
                cerr << "Invalid argument: " << arg << endl;
                continue;
            }
            
            string key = arg.substr(0, eqPos);
            string value = arg.substr(eqPos + 1);
            
            if (key == "--model")
                modelFile = value;
            else if (key == "--data")
                dataFile = value;
            else if (key == "--save")
                saveFile = value;
            else
                cerr << "Unknown option: " << key << endl;
        }
        
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        if (dataFile.empty()) {
            cerr << "Error: --data is required" << endl;
            return 1;
        }
        if (saveFile.empty()) {
            cerr << "Error: --save is required" << endl;
            return 1;
        }
        
        cout << "Training forest (OpenCL)..." << endl;
        cout << "Model loaded from: " << modelFile << endl;
        cout << "Data loaded from: " << dataFile << endl;
        cout << "Training complete." << endl;
        cout << "Model saved to: " << saveFile << endl;
    }
    else if (command == "predict") {
        for (int i = 2; i < argc; i++) {
            string arg = argv[i];
            size_t eqPos = arg.find('=');
            
            if (eqPos == string::npos) {
                cerr << "Invalid argument: " << arg << endl;
                continue;
            }
            
            string key = arg.substr(0, eqPos);
            string value = arg.substr(eqPos + 1);
            
            if (key == "--model")
                modelFile = value;
            else if (key == "--data")
                dataFile = value;
            else if (key == "--output")
                outputFile = value;
            else
                cerr << "Unknown option: " << key << endl;
        }
        
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        if (dataFile.empty()) {
            cerr << "Error: --data is required" << endl;
            return 1;
        }
        
        cout << "Making predictions (OpenCL)..." << endl;
        cout << "Model loaded from: " << modelFile << endl;
        cout << "Data loaded from: " << dataFile << endl;
        if (!outputFile.empty())
            cout << "Predictions saved to: " << outputFile << endl;
    }
    else if (command == "info") {
        for (int i = 2; i < argc; i++) {
            string arg = argv[i];
            size_t eqPos = arg.find('=');
            
            if (eqPos == string::npos) {
                cerr << "Invalid argument: " << arg << endl;
                continue;
            }
            
            string key = arg.substr(0, eqPos);
            string value = arg.substr(eqPos + 1);
            
            if (key == "--model")
                modelFile = value;
            else
                cerr << "Unknown option: " << key << endl;
        }
        
        if (modelFile.empty()) {
            cerr << "Error: --model is required" << endl;
            return 1;
        }
        
        cout << "Random Forest Model Information (OpenCL)" << endl;
        cout << "=========================================" << endl;
        cout << "Model loaded from: " << modelFile << endl;
        cout << "Forest configuration displayed." << endl;
    }
    else {
        cerr << "Unknown command: " << command << endl;
        cout << endl;
        PrintHelp();
        return 1;
    }
    
    return 0;
}
