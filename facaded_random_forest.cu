//
// Random Forest + Facade - CUDA Implementation
// Ported from C++ by Matthew Abbott 2025
//

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>
#include <iomanip>
#include <map>

// Constants
const int MAX_FEATURES = 100;
const int MAX_SAMPLES = 10000;
const int MAX_TREES = 500;
const int MAX_DEPTH_DEFAULT = 10;
const int MIN_SAMPLES_LEAF_DEFAULT = 1;
const int MIN_SAMPLES_SPLIT_DEFAULT = 2;
const int MAX_NODE_INFO = 1000;
const int MAX_FEATURE_STATS = 100;
const int CUDA_BLOCK_SIZE = 256;

// Enums
enum class TaskType { Classification, Regression };
enum class SplitCriterion { Gini, Entropy, MSE, VarianceReduction };
enum class AggregationMethod { MajorityVote, WeightedVote, Mean, WeightedMean };

// Custom atomicAdd for double (needed for GPUs with compute capability < 6.0)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__device__ double atomicAddDouble(double* address, double val) {
    return atomicAdd(address, val);
}
#endif

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
        } \
    } while(0)

// Forward declarations
struct TreeNode;
struct DecisionTree;
class RandomForest;
class RandomForestFacade;

// ============================================================================
// Data Structures
// ============================================================================

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
    
    TreeNode() : isLeaf(true), featureIndex(-1), threshold(0.0), 
                 prediction(0.0), classLabel(0), impurity(0.0), 
                 numSamples(0), left(nullptr), right(nullptr) {}
};

struct DecisionTree {
    TreeNode* root;
    int maxDepth;
    int minSamplesLeaf;
    int minSamplesSplit;
    int maxFeatures;
    TaskType taskType;
    SplitCriterion criterion;
    std::vector<bool> oobIndices;
    int numOobIndices;
    
    DecisionTree() : root(nullptr), maxDepth(MAX_DEPTH_DEFAULT),
                     minSamplesLeaf(MIN_SAMPLES_LEAF_DEFAULT),
                     minSamplesSplit(MIN_SAMPLES_SPLIT_DEFAULT),
                     maxFeatures(0), taskType(TaskType::Classification),
                     criterion(SplitCriterion::Gini), numOobIndices(0) {}
};

struct NodeInfo {
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

struct TreeInfo {
    int treeId;
    int numNodes;
    int maxDepth;
    int numLeaves;
    std::vector<bool> featuresUsed;
    int numFeaturesUsed;
    double oobError;
    std::vector<NodeInfo> nodes;
    
    TreeInfo() : treeId(0), numNodes(0), maxDepth(0), numLeaves(0),
                 featuresUsed(MAX_FEATURES, false), numFeaturesUsed(0), oobError(0.0) {}
};

struct FeatureStats {
    int featureIndex;
    int timesUsed;
    int treesUsedIn;
    double avgImportance;
    double totalImportance;
};

struct SampleTrackInfo {
    int sampleIndex;
    std::vector<bool> treesInfluenced;
    int numTreesInfluenced;
    std::vector<bool> oobTrees;
    int numOobTrees;
    std::vector<double> predictions;
};

struct OOBTreeInfo {
    int treeId;
    int numOobSamples;
    double oobError;
    double oobAccuracy;
};

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void kernelInitRNG(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void kernelBootstrap(curandState* states, int* indices, bool* oobMask, 
                                 int numSamples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) {
        int sampledIdx = curand(&states[idx]) % numSamples;
        indices[idx] = sampledIdx;
        atomicExch((int*)&oobMask[sampledIdx], 0);
    }
}

__global__ void kernelCalculateGini(double* d_targets, int* d_indices, int numIndices,
                                     int* d_classCounts, int numClasses, double* d_result) {
    __shared__ int sharedCounts[32];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (tid < numClasses) sharedCounts[tid] = 0;
    __syncthreads();
    
    if (idx < numIndices) {
        int label = (int)(d_targets[d_indices[idx]] + 0.5);
        if (label >= 0 && label < numClasses) {
            atomicAdd(&sharedCounts[label], 1);
        }
    }
    __syncthreads();
    
    if (tid == 0) {
        double gini = 1.0;
        for (int c = 0; c < numClasses; c++) {
            double prob = (double)sharedCounts[c] / numIndices;
            gini -= prob * prob;
        }
        *d_result = gini;
    }
}

__global__ void kernelCalculateMSE(double* d_targets, int* d_indices, int numIndices,
                                    double* d_mean, double* d_result) {
    __shared__ double sharedSum[CUDA_BLOCK_SIZE];
    __shared__ double sharedSqSum[CUDA_BLOCK_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    sharedSum[tid] = 0.0;
    sharedSqSum[tid] = 0.0;
    
    if (idx < numIndices) {
        double val = d_targets[d_indices[idx]];
        sharedSum[tid] = val;
        sharedSqSum[tid] = val * val;
    }
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedSum[tid] += sharedSum[tid + s];
            sharedSqSum[tid] += sharedSqSum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        double mean = sharedSum[0] / numIndices;
        *d_mean = mean;
        *d_result = (sharedSqSum[0] / numIndices) - (mean * mean);
    }
}

__global__ void kernelSplitData(double* d_data, int* d_indices, int numIndices,
                                 int featureIndex, double threshold, int numFeatures,
                                 int* d_leftIndices, int* d_rightIndices,
                                 int* d_leftCount, int* d_rightCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numIndices) {
        int sampleIdx = d_indices[idx];
        double featureVal = d_data[sampleIdx * numFeatures + featureIndex];
        
        if (featureVal <= threshold) {
            int pos = atomicAdd(d_leftCount, 1);
            d_leftIndices[pos] = sampleIdx;
        } else {
            int pos = atomicAdd(d_rightCount, 1);
            d_rightIndices[pos] = sampleIdx;
        }
    }
}

__global__ void kernelPredictBatch(double* d_data, int numSamples, int numFeatures,
                                    int* d_nodeFeatures, double* d_nodeThresholds,
                                    double* d_nodePredictions, bool* d_nodeIsLeaf,
                                    int* d_nodeLeft, int* d_nodeRight, int numNodes,
                                    double* d_predictions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) {
        int nodeIdx = 0;
        while (!d_nodeIsLeaf[nodeIdx] && nodeIdx < numNodes) {
            int feat = d_nodeFeatures[nodeIdx];
            double thresh = d_nodeThresholds[nodeIdx];
            double val = d_data[idx * numFeatures + feat];
            
            if (val <= thresh)
                nodeIdx = d_nodeLeft[nodeIdx];
            else
                nodeIdx = d_nodeRight[nodeIdx];
        }
        d_predictions[idx] = d_nodePredictions[nodeIdx];
    }
}

__global__ void kernelAccuracy(double* d_predictions, double* d_targets, 
                                int numSamples, int* d_correct) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) {
        int pred = (int)(d_predictions[idx] + 0.5);
        int actual = (int)(d_targets[idx] + 0.5);
        if (pred == actual) {
            atomicAdd(d_correct, 1);
        }
    }
}

__global__ void kernelMSE(double* d_predictions, double* d_targets,
                           int numSamples, double* d_mse) {
    __shared__ double sharedMSE[CUDA_BLOCK_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    sharedMSE[tid] = 0.0;
    if (idx < numSamples) {
        double diff = d_predictions[idx] - d_targets[idx];
        sharedMSE[tid] = diff * diff;
    }
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sharedMSE[tid] += sharedMSE[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) atomicAddDouble(d_mse, sharedMSE[0]);
}

// ============================================================================
// RandomForest Class Declaration
// ============================================================================

class RandomForest {
private:
    std::vector<DecisionTree*> trees;
    int numTrees;
    int maxDepth;
    int minSamplesLeaf;
    int minSamplesSplit;
    int maxFeatures;
    int numFeatures;
    int numSamples;
    TaskType taskType;
    SplitCriterion criterion;
    std::vector<double> featureImportances;
    std::mt19937 rng;
    
    std::vector<std::vector<double>> data;
    std::vector<double> targets;
    
    // CUDA device pointers
    double* d_data;
    double* d_targets;
    curandState* d_rngStates;
    bool cudaInitialized;

public:
    RandomForest();
    ~RandomForest();
    
    void initCUDA();
    void freeCUDA();
    void copyDataToDevice();
    
    void setNumTrees(int n);
    void setMaxDepth(int d);
    void setMinSamplesLeaf(int m);
    void setMinSamplesSplit(int m);
    void setMaxFeatures(int m);
    void setTaskType(TaskType t);
    void setCriterion(SplitCriterion c);
    void setRandomSeed(int seed);
    
    void loadData(const std::vector<std::vector<double>>& inputData,
                  const std::vector<double>& inputTargets);
    void bootstrap(std::vector<int>& sampleIndices, std::vector<bool>& oobMask);
    void selectFeatureSubset(std::vector<int>& featureIndices);
    
    double calculateGini(const std::vector<int>& indices);
    double calculateEntropy(const std::vector<int>& indices);
    double calculateMSE(const std::vector<int>& indices);
    double calculateImpurity(const std::vector<int>& indices);
    
    bool findBestSplit(const std::vector<int>& indices, const std::vector<int>& featureIndices,
                       int& bestFeature, double& bestThreshold, double& bestGain);
    int getMajorityClass(const std::vector<int>& indices);
    double getMeanTarget(const std::vector<int>& indices);
    TreeNode* createLeafNode(const std::vector<int>& indices);
    bool shouldStop(int depth, int numIndices, double impurity);
    TreeNode* buildTree(const std::vector<int>& indices, int depth);
    
    double predictTree(TreeNode* node, const std::vector<double>& sample);
    double predict(const std::vector<double>& sample);
    int predictClass(const std::vector<double>& sample);
    void predictBatch(const std::vector<std::vector<double>>& samples,
                      std::vector<double>& predictions);
    void predictBatchGPU(const std::vector<std::vector<double>>& samples,
                         std::vector<double>& predictions);
    
    void fit();
    void fitTree(int treeIndex);
    
    double calculateOOBError();
    void calculateFeatureImportance();
    double getFeatureImportance(int featureIndex);
    void printFeatureImportances();
    
    double accuracy(const std::vector<double>& predictions, const std::vector<double>& actual);
    double precision(const std::vector<double>& predictions, const std::vector<double>& actual, int positiveClass);
    double recall(const std::vector<double>& predictions, const std::vector<double>& actual, int positiveClass);
    double f1Score(const std::vector<double>& predictions, const std::vector<double>& actual, int positiveClass);
    double meanSquaredError(const std::vector<double>& predictions, const std::vector<double>& actual);
    double rSquared(const std::vector<double>& predictions, const std::vector<double>& actual);
    
    void printForestInfo();
    void freeTreeNode(TreeNode* node);
    void freeTree(DecisionTree* tree);
    void freeForest();
    
    int getNumTrees() const { return numTrees; }
    int getNumFeatures() const { return numFeatures; }
    int getNumSamples() const { return numSamples; }
    int getMaxDepth() const { return maxDepth; }
    DecisionTree* getTree(int treeId);
    double getData(int sampleIdx, int featureIdx);
    double getTarget(int sampleIdx);
    TaskType getTaskType() const { return taskType; }
    SplitCriterion getCriterion() const { return criterion; }
    
    void addNewTree();
    void removeTreeAt(int treeId);
    void retrainTreeAt(int treeId);
    void setTree(int treeId, DecisionTree* tree);
};

// ============================================================================
// RandomForestFacade Class Declaration
// ============================================================================

class RandomForestFacade {
private:
    RandomForest forest;
    bool forestInitialized;
    AggregationMethod currentAggregation;
    std::vector<double> treeWeights;
    std::vector<bool> featureEnabled;
    
    int collectNodeInfo(TreeNode* node, int depth, std::vector<NodeInfo>& nodes);
    int calculateTreeDepth(TreeNode* node);
    int countLeaves(TreeNode* node);
    void collectFeaturesUsed(TreeNode* node, std::vector<bool>& used);
    TreeNode* findNodeById(TreeNode* node, int targetId, int& currentId);
    void freeSubtree(TreeNode* node);

public:
    RandomForestFacade();
    
    void initForest();
    RandomForest& getForest() { return forest; }
    
    void setHyperparameter(const std::string& paramName, int value);
    void setTaskType(TaskType t);
    void setCriterion(SplitCriterion c);
    void printHyperparameters();
    
    void loadData(const std::vector<std::vector<double>>& inputData,
                  const std::vector<double>& inputTargets);
    void trainForest();
    
    TreeInfo inspectTree(int treeId);
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
    void resetFeatureFilters();
    std::vector<FeatureStats> featureUsageSummary();
    void printFeatureUsageSummary();
    double getFeatureImportance(int featureIndex);
    void printFeatureImportances();
    
    void setAggregationMethod(AggregationMethod method);
    AggregationMethod getAggregationMethod();
    void setTreeWeight(int treeId, double weight);
    double getTreeWeight(int treeId);
    void resetTreeWeights();
    double aggregatePredictions(const std::vector<double>& sample);
    
    double predict(const std::vector<double>& sample);
    int predictClass(const std::vector<double>& sample);
    double predictWithTree(int treeId, const std::vector<double>& sample);
    void predictBatch(const std::vector<std::vector<double>>& samples,
                      std::vector<double>& predictions);
    
    SampleTrackInfo trackSample(int sampleIndex);
    void printSampleTracking(int sampleIndex);
    
    std::vector<OOBTreeInfo> oobErrorSummary();
    void printOOBSummary();
    double getGlobalOOBError();
    void markProblematicTrees(double errorThreshold);
    
    double accuracy(const std::vector<double>& predictions, const std::vector<double>& actual);
    double meanSquaredError(const std::vector<double>& predictions, const std::vector<double>& actual);
    double rSquared(const std::vector<double>& predictions, const std::vector<double>& actual);
    double precision(const std::vector<double>& predictions, const std::vector<double>& actual, int posClass);
    double recall(const std::vector<double>& predictions, const std::vector<double>& actual, int posClass);
    double f1Score(const std::vector<double>& predictions, const std::vector<double>& actual, int posClass);
    
    void highlightMisclassified(const std::vector<double>& predictions, const std::vector<double>& actual);
    void highlightHighResidual(const std::vector<double>& predictions, const std::vector<double>& actual, double threshold);
    void findWorstTrees(const std::vector<double>& actual, int topN);
    
    void visualizeTree(int treeId);
    void visualizeSplitDistribution(int treeId, int nodeId);
    void printForestOverview();
    void printFeatureHeatmap();
    
    void saveModel(const std::string& filename);
    bool loadModel(const std::string& filename);
    
    void freeForest();
};

// ============================================================================
// RandomForest Implementation
// ============================================================================

RandomForest::RandomForest() : numTrees(100), maxDepth(MAX_DEPTH_DEFAULT),
    minSamplesLeaf(MIN_SAMPLES_LEAF_DEFAULT), minSamplesSplit(MIN_SAMPLES_SPLIT_DEFAULT),
    maxFeatures(0), numFeatures(0), numSamples(0), 
    taskType(TaskType::Classification), criterion(SplitCriterion::Gini),
    d_data(nullptr), d_targets(nullptr), d_rngStates(nullptr), cudaInitialized(false) {
    trees.resize(MAX_TREES, nullptr);
    featureImportances.resize(MAX_FEATURES, 0.0);
    rng.seed(42);
}

RandomForest::~RandomForest() {
    freeForest();
    freeCUDA();
}

void RandomForest::initCUDA() {
    if (cudaInitialized) return;
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaSetDevice(0);
        cudaInitialized = true;
        std::cout << "CUDA initialized with " << deviceCount << " device(s)\n";
    }
}

void RandomForest::freeCUDA() {
    if (d_data) { cudaFree(d_data); d_data = nullptr; }
    if (d_targets) { cudaFree(d_targets); d_targets = nullptr; }
    if (d_rngStates) { cudaFree(d_rngStates); d_rngStates = nullptr; }
}

void RandomForest::copyDataToDevice() {
    if (!cudaInitialized) initCUDA();
    if (numSamples == 0 || numFeatures == 0) return;
    
    freeCUDA();
    
    size_t dataSize = numSamples * numFeatures * sizeof(double);
    size_t targetSize = numSamples * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_data, dataSize));
    CUDA_CHECK(cudaMalloc(&d_targets, targetSize));
    CUDA_CHECK(cudaMalloc(&d_rngStates, numSamples * sizeof(curandState)));
    
    std::vector<double> flatData(numSamples * numFeatures);
    for (int i = 0; i < numSamples; i++)
        for (int j = 0; j < numFeatures; j++)
            flatData[i * numFeatures + j] = data[i][j];
    
    CUDA_CHECK(cudaMemcpy(d_data, flatData.data(), dataSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, targets.data(), targetSize, cudaMemcpyHostToDevice));
    
    int blocks = (numSamples + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernelInitRNG<<<blocks, CUDA_BLOCK_SIZE>>>(d_rngStates, 42, numSamples);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void RandomForest::setNumTrees(int n) {
    if (n > MAX_TREES) numTrees = MAX_TREES;
    else if (n < 1) numTrees = 1;
    else numTrees = n;
}

void RandomForest::setMaxDepth(int d) { maxDepth = (d < 1) ? 1 : d; }
void RandomForest::setMinSamplesLeaf(int m) { minSamplesLeaf = (m < 1) ? 1 : m; }
void RandomForest::setMinSamplesSplit(int m) { minSamplesSplit = (m < 2) ? 2 : m; }
void RandomForest::setMaxFeatures(int m) { maxFeatures = m; }

void RandomForest::setTaskType(TaskType t) {
    taskType = t;
    criterion = (t == TaskType::Classification) ? SplitCriterion::Gini : SplitCriterion::MSE;
}

void RandomForest::setCriterion(SplitCriterion c) { criterion = c; }
void RandomForest::setRandomSeed(int seed) { rng.seed(seed); }

void RandomForest::loadData(const std::vector<std::vector<double>>& inputData,
                            const std::vector<double>& inputTargets) {
    numSamples = inputData.size();
    numFeatures = numSamples > 0 ? inputData[0].size() : 0;
    
    data = inputData;
    targets = inputTargets;
    
    if (maxFeatures == 0) {
        if (taskType == TaskType::Classification)
            maxFeatures = std::max(1, (int)std::sqrt(numFeatures));
        else
            maxFeatures = std::max(1, numFeatures / 3);
    }
    
    copyDataToDevice();
}

void RandomForest::bootstrap(std::vector<int>& sampleIndices, std::vector<bool>& oobMask) {
    sampleIndices.resize(numSamples);
    oobMask.assign(numSamples, true);
    
    std::uniform_int_distribution<int> dist(0, numSamples - 1);
    for (int i = 0; i < numSamples; i++) {
        int idx = dist(rng);
        sampleIndices[i] = idx;
        oobMask[idx] = false;
    }
}

void RandomForest::selectFeatureSubset(std::vector<int>& featureIndices) {
    std::vector<int> available(numFeatures);
    for (int i = 0; i < numFeatures; i++) available[i] = i;
    std::shuffle(available.begin(), available.end(), rng);
    int nSelect = std::min(maxFeatures, numFeatures);
    featureIndices.assign(available.begin(), available.begin() + nSelect);
}

double RandomForest::calculateGini(const std::vector<int>& indices) {
    if (indices.empty()) return 0.0;
    
    std::map<int, int> classCount;
    for (int idx : indices) {
        int label = (int)std::round(targets[idx]);
        classCount[label]++;
    }
    
    double gini = 1.0;
    int n = indices.size();
    for (auto& p : classCount) {
        double prob = (double)p.second / n;
        gini -= prob * prob;
    }
    return gini;
}

double RandomForest::calculateEntropy(const std::vector<int>& indices) {
    if (indices.empty()) return 0.0;
    
    std::map<int, int> classCount;
    for (int idx : indices) {
        int label = (int)std::round(targets[idx]);
        classCount[label]++;
    }
    
    double entropy = 0.0;
    int n = indices.size();
    for (auto& p : classCount) {
        if (p.second > 0) {
            double prob = (double)p.second / n;
            entropy -= prob * std::log2(prob);
        }
    }
    return entropy;
}

double RandomForest::calculateMSE(const std::vector<int>& indices) {
    if (indices.empty()) return 0.0;
    
    double mean = 0.0;
    for (int idx : indices) mean += targets[idx];
    mean /= indices.size();
    
    double mse = 0.0;
    for (int idx : indices) {
        double diff = targets[idx] - mean;
        mse += diff * diff;
    }
    return mse / indices.size();
}

double RandomForest::calculateImpurity(const std::vector<int>& indices) {
    switch (criterion) {
        case SplitCriterion::Gini: return calculateGini(indices);
        case SplitCriterion::Entropy: return calculateEntropy(indices);
        case SplitCriterion::MSE:
        case SplitCriterion::VarianceReduction: return calculateMSE(indices);
        default: return calculateGini(indices);
    }
}

bool RandomForest::findBestSplit(const std::vector<int>& indices, 
                                  const std::vector<int>& featureIndices,
                                  int& bestFeature, double& bestThreshold, double& bestGain) {
    bestGain = 0.0;
    bestFeature = -1;
    bestThreshold = 0.0;
    
    if ((int)indices.size() < minSamplesSplit) return false;
    
    double parentImpurity = calculateImpurity(indices);
    
    for (int feat : featureIndices) {
        std::vector<std::pair<double, int>> valIdx;
        for (int idx : indices) {
            valIdx.push_back({data[idx][feat], idx});
        }
        std::sort(valIdx.begin(), valIdx.end());
        
        for (size_t i = 0; i < valIdx.size() - 1; i++) {
            if (valIdx[i].first == valIdx[i+1].first) continue;
            
            double threshold = (valIdx[i].first + valIdx[i+1].first) / 2.0;
            
            std::vector<int> leftIdx, rightIdx;
            for (int idx : indices) {
                if (data[idx][feat] <= threshold)
                    leftIdx.push_back(idx);
                else
                    rightIdx.push_back(idx);
            }
            
            if ((int)leftIdx.size() < minSamplesLeaf || (int)rightIdx.size() < minSamplesLeaf)
                continue;
            
            double leftImp = calculateImpurity(leftIdx);
            double rightImp = calculateImpurity(rightIdx);
            
            double gain = parentImpurity 
                - ((double)leftIdx.size() / indices.size()) * leftImp
                - ((double)rightIdx.size() / indices.size()) * rightImp;
            
            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = feat;
                bestThreshold = threshold;
            }
        }
    }
    return bestFeature >= 0;
}

int RandomForest::getMajorityClass(const std::vector<int>& indices) {
    std::map<int, int> classCount;
    for (int idx : indices) {
        int label = (int)std::round(targets[idx]);
        classCount[label]++;
    }
    
    int maxCount = 0, maxClass = 0;
    for (auto& p : classCount) {
        if (p.second > maxCount) {
            maxCount = p.second;
            maxClass = p.first;
        }
    }
    return maxClass;
}

double RandomForest::getMeanTarget(const std::vector<int>& indices) {
    if (indices.empty()) return 0.0;
    double sum = 0.0;
    for (int idx : indices) sum += targets[idx];
    return sum / indices.size();
}

TreeNode* RandomForest::createLeafNode(const std::vector<int>& indices) {
    TreeNode* node = new TreeNode();
    node->isLeaf = true;
    node->numSamples = indices.size();
    node->impurity = calculateImpurity(indices);
    
    if (taskType == TaskType::Classification) {
        node->classLabel = getMajorityClass(indices);
        node->prediction = node->classLabel;
    } else {
        node->prediction = getMeanTarget(indices);
        node->classLabel = (int)std::round(node->prediction);
    }
    return node;
}

bool RandomForest::shouldStop(int depth, int numIndices, double impurity) {
    return depth >= maxDepth || numIndices < minSamplesSplit 
           || numIndices <= minSamplesLeaf || impurity < 1e-10;
}

TreeNode* RandomForest::buildTree(const std::vector<int>& indices, int depth) {
    double currentImpurity = calculateImpurity(indices);
    
    if (shouldStop(depth, indices.size(), currentImpurity)) {
        return createLeafNode(indices);
    }
    
    std::vector<int> featureIndices;
    selectFeatureSubset(featureIndices);
    
    int bestFeature;
    double bestThreshold, bestGain;
    if (!findBestSplit(indices, featureIndices, bestFeature, bestThreshold, bestGain)) {
        return createLeafNode(indices);
    }
    
    TreeNode* node = new TreeNode();
    node->isLeaf = false;
    node->featureIndex = bestFeature;
    node->threshold = bestThreshold;
    node->numSamples = indices.size();
    node->impurity = currentImpurity;
    
    if (taskType == TaskType::Classification)
        node->classLabel = getMajorityClass(indices);
    else
        node->prediction = getMeanTarget(indices);
    
    std::vector<int> leftIdx, rightIdx;
    for (int idx : indices) {
        if (data[idx][bestFeature] <= bestThreshold)
            leftIdx.push_back(idx);
        else
            rightIdx.push_back(idx);
    }
    
    double leftImp = calculateImpurity(leftIdx);
    double rightImp = calculateImpurity(rightIdx);
    featureImportances[bestFeature] += indices.size() * currentImpurity 
        - leftIdx.size() * leftImp - rightIdx.size() * rightImp;
    
    node->left = buildTree(leftIdx, depth + 1);
    node->right = buildTree(rightIdx, depth + 1);
    
    return node;
}

double RandomForest::predictTree(TreeNode* node, const std::vector<double>& sample) {
    if (!node) return 0.0;
    if (node->isLeaf) return node->prediction;
    
    if (sample[node->featureIndex] <= node->threshold)
        return predictTree(node->left, sample);
    else
        return predictTree(node->right, sample);
}

double RandomForest::predict(const std::vector<double>& sample) {
    if (taskType == TaskType::Regression) {
        double sum = 0.0;
        for (int i = 0; i < numTrees; i++) {
            if (trees[i]) sum += predictTree(trees[i]->root, sample);
        }
        return sum / numTrees;
    } else {
        std::map<int, int> votes;
        for (int i = 0; i < numTrees; i++) {
            if (trees[i]) {
                int label = (int)std::round(predictTree(trees[i]->root, sample));
                votes[label]++;
            }
        }
        int maxVotes = 0, maxClass = 0;
        for (auto& p : votes) {
            if (p.second > maxVotes) {
                maxVotes = p.second;
                maxClass = p.first;
            }
        }
        return maxClass;
    }
}

int RandomForest::predictClass(const std::vector<double>& sample) {
    return (int)std::round(predict(sample));
}

void RandomForest::predictBatch(const std::vector<std::vector<double>>& samples,
                                 std::vector<double>& predictions) {
    predictions.resize(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        predictions[i] = predict(samples[i]);
    }
}

void RandomForest::fit() {
    std::fill(featureImportances.begin(), featureImportances.end(), 0.0);
    
    for (int i = 0; i < numTrees; i++) {
        fitTree(i);
    }
    calculateFeatureImportance();
}

void RandomForest::fitTree(int treeIndex) {
    DecisionTree* tree = new DecisionTree();
    tree->maxDepth = maxDepth;
    tree->minSamplesLeaf = minSamplesLeaf;
    tree->minSamplesSplit = minSamplesSplit;
    tree->maxFeatures = maxFeatures;
    tree->taskType = taskType;
    tree->criterion = criterion;
    
    std::vector<int> sampleIndices;
    std::vector<bool> oobMask;
    bootstrap(sampleIndices, oobMask);
    
    tree->oobIndices = oobMask;
    tree->numOobIndices = std::count(oobMask.begin(), oobMask.end(), true);
    
    tree->root = buildTree(sampleIndices, 0);
    
    if (trees[treeIndex]) freeTree(trees[treeIndex]);
    trees[treeIndex] = tree;
}

double RandomForest::calculateOOBError() {
    std::vector<double> predictions(numSamples, 0.0);
    std::vector<int> predCounts(numSamples, 0);
    std::vector<std::map<int, int>> votes(numSamples);
    
    for (int t = 0; t < numTrees; t++) {
        if (!trees[t]) continue;
        for (int i = 0; i < numSamples; i++) {
            if (trees[t]->oobIndices[i]) {
                double pred = predictTree(trees[t]->root, data[i]);
                if (taskType == TaskType::Regression) {
                    predictions[i] += pred;
                } else {
                    votes[i][(int)std::round(pred)]++;
                }
                predCounts[i]++;
            }
        }
    }
    
    double error = 0.0;
    int count = 0;
    
    for (int i = 0; i < numSamples; i++) {
        if (predCounts[i] > 0) {
            if (taskType == TaskType::Regression) {
                double pred = predictions[i] / predCounts[i];
                double diff = pred - targets[i];
                error += diff * diff;
            } else {
                int maxVotes = 0, maxClass = 0;
                for (auto& p : votes[i]) {
                    if (p.second > maxVotes) {
                        maxVotes = p.second;
                        maxClass = p.first;
                    }
                }
                if (maxClass != (int)std::round(targets[i])) error += 1.0;
            }
            count++;
        }
    }
    
    return count > 0 ? error / count : 0.0;
}

void RandomForest::calculateFeatureImportance() {
    double total = 0.0;
    for (int i = 0; i < numFeatures; i++) total += featureImportances[i];
    
    if (total > 0) {
        for (int i = 0; i < numFeatures; i++)
            featureImportances[i] /= total;
    }
}

double RandomForest::getFeatureImportance(int featureIndex) {
    if (featureIndex >= 0 && featureIndex < numFeatures)
        return featureImportances[featureIndex];
    return 0.0;
}

void RandomForest::printFeatureImportances() {
    std::cout << "Feature Importances:\n";
    for (int i = 0; i < numFeatures; i++) {
        std::cout << "  Feature " << i << ": " << std::fixed 
                  << std::setprecision(4) << featureImportances[i] << "\n";
    }
}

double RandomForest::accuracy(const std::vector<double>& predictions, 
                               const std::vector<double>& actual) {
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        if ((int)std::round(predictions[i]) == (int)std::round(actual[i]))
            correct++;
    }
    return (double)correct / predictions.size();
}

double RandomForest::precision(const std::vector<double>& predictions,
                                const std::vector<double>& actual, int positiveClass) {
    int tp = 0, fp = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        if ((int)std::round(predictions[i]) == positiveClass) {
            if ((int)std::round(actual[i]) == positiveClass) tp++;
            else fp++;
        }
    }
    return (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
}

double RandomForest::recall(const std::vector<double>& predictions,
                             const std::vector<double>& actual, int positiveClass) {
    int tp = 0, fn = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        if ((int)std::round(actual[i]) == positiveClass) {
            if ((int)std::round(predictions[i]) == positiveClass) tp++;
            else fn++;
        }
    }
    return (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
}

double RandomForest::f1Score(const std::vector<double>& predictions,
                              const std::vector<double>& actual, int positiveClass) {
    double p = precision(predictions, actual, positiveClass);
    double r = recall(predictions, actual, positiveClass);
    return (p + r > 0) ? 2 * p * r / (p + r) : 0.0;
}

double RandomForest::meanSquaredError(const std::vector<double>& predictions,
                                       const std::vector<double>& actual) {
    double mse = 0.0;
    for (size_t i = 0; i < predictions.size(); i++) {
        double diff = predictions[i] - actual[i];
        mse += diff * diff;
    }
    return mse / predictions.size();
}

double RandomForest::rSquared(const std::vector<double>& predictions,
                               const std::vector<double>& actual) {
    double mean = 0.0;
    for (double v : actual) mean += v;
    mean /= actual.size();
    
    double ssRes = 0.0, ssTot = 0.0;
    for (size_t i = 0; i < actual.size(); i++) {
        double diff = predictions[i] - actual[i];
        ssRes += diff * diff;
        diff = actual[i] - mean;
        ssTot += diff * diff;
    }
    return ssTot > 0 ? 1.0 - ssRes / ssTot : 0.0;
}

void RandomForest::printForestInfo() {
    std::cout << "Random Forest Configuration:\n";
    std::cout << "  Number of Trees: " << numTrees << "\n";
    std::cout << "  Max Depth: " << maxDepth << "\n";
    std::cout << "  Min Samples Leaf: " << minSamplesLeaf << "\n";
    std::cout << "  Min Samples Split: " << minSamplesSplit << "\n";
    std::cout << "  Max Features: " << maxFeatures << "\n";
    std::cout << "  Number of Features: " << numFeatures << "\n";
    std::cout << "  Number of Samples: " << numSamples << "\n";
    std::cout << "  Task Type: " << (taskType == TaskType::Classification ? "Classification" : "Regression") << "\n";
    std::cout << "  CUDA Enabled: " << (cudaInitialized ? "Yes" : "No") << "\n";
    std::cout << "  Criterion: ";
    switch (criterion) {
        case SplitCriterion::Gini: std::cout << "Gini"; break;
        case SplitCriterion::Entropy: std::cout << "Entropy"; break;
        case SplitCriterion::MSE: std::cout << "MSE"; break;
        case SplitCriterion::VarianceReduction: std::cout << "Variance Reduction"; break;
    }
    std::cout << "\n";
}

void RandomForest::freeTreeNode(TreeNode* node) {
    if (!node) return;
    freeTreeNode(node->left);
    freeTreeNode(node->right);
    delete node;
}

void RandomForest::freeTree(DecisionTree* tree) {
    if (!tree) return;
    freeTreeNode(tree->root);
    delete tree;
}

void RandomForest::freeForest() {
    for (int i = 0; i < MAX_TREES; i++) {
        if (trees[i]) {
            freeTree(trees[i]);
            trees[i] = nullptr;
        }
    }
}

DecisionTree* RandomForest::getTree(int treeId) {
    if (treeId >= 0 && treeId < numTrees) return trees[treeId];
    return nullptr;
}

double RandomForest::getData(int sampleIdx, int featureIdx) {
    if (sampleIdx >= 0 && sampleIdx < numSamples && 
        featureIdx >= 0 && featureIdx < numFeatures)
        return data[sampleIdx][featureIdx];
    return 0.0;
}

double RandomForest::getTarget(int sampleIdx) {
    if (sampleIdx >= 0 && sampleIdx < numSamples)
        return targets[sampleIdx];
    return 0.0;
}

void RandomForest::addNewTree() {
    if (numTrees >= MAX_TREES) {
        std::cout << "Maximum number of trees reached\n";
        return;
    }
    fitTree(numTrees);
    numTrees++;
}

void RandomForest::removeTreeAt(int treeId) {
    if (treeId < 0 || treeId >= numTrees) {
        std::cout << "Invalid tree ID: " << treeId << "\n";
        return;
    }
    freeTree(trees[treeId]);
    for (int i = treeId; i < numTrees - 1; i++)
        trees[i] = trees[i + 1];
    trees[numTrees - 1] = nullptr;
    numTrees--;
}

void RandomForest::retrainTreeAt(int treeId) {
    if (treeId < 0 || treeId >= numTrees) {
        std::cout << "Invalid tree ID: " << treeId << "\n";
        return;
    }
    freeTree(trees[treeId]);
    trees[treeId] = nullptr;
    fitTree(treeId);
}

void RandomForest::setTree(int treeId, DecisionTree* tree) {
    if (treeId >= 0 && treeId < MAX_TREES)
        trees[treeId] = tree;
}

// ============================================================================
// RandomForestFacade Implementation
// ============================================================================

RandomForestFacade::RandomForestFacade() : forestInitialized(false),
    currentAggregation(AggregationMethod::MajorityVote) {
    treeWeights.resize(MAX_TREES, 1.0);
    featureEnabled.resize(MAX_FEATURES, true);
}

void RandomForestFacade::initForest() { forestInitialized = true; }

void RandomForestFacade::setHyperparameter(const std::string& paramName, int value) {
    if (paramName == "n_estimators") forest.setNumTrees(value);
    else if (paramName == "max_depth") forest.setMaxDepth(value);
    else if (paramName == "min_samples_leaf") forest.setMinSamplesLeaf(value);
    else if (paramName == "min_samples_split") forest.setMinSamplesSplit(value);
    else if (paramName == "max_features") forest.setMaxFeatures(value);
    else if (paramName == "random_seed") forest.setRandomSeed(value);
    else std::cout << "Unknown hyperparameter: " << paramName << "\n";
}

void RandomForestFacade::setTaskType(TaskType t) { forest.setTaskType(t); }
void RandomForestFacade::setCriterion(SplitCriterion c) { forest.setCriterion(c); }
void RandomForestFacade::printHyperparameters() { forest.printForestInfo(); }

void RandomForestFacade::loadData(const std::vector<std::vector<double>>& inputData,
                                   const std::vector<double>& inputTargets) {
    forest.loadData(inputData, inputTargets);
}

void RandomForestFacade::trainForest() {
    forest.fit();
    forestInitialized = true;
}

int RandomForestFacade::collectNodeInfo(TreeNode* node, int depth, std::vector<NodeInfo>& nodes) {
    if (!node || nodes.size() >= MAX_NODE_INFO) return -1;
    
    int currentId = nodes.size();
    NodeInfo info;
    info.nodeId = currentId;
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
    nodes.push_back(info);
    
    if (!node->isLeaf) {
        nodes[currentId].leftChildId = collectNodeInfo(node->left, depth + 1, nodes);
        nodes[currentId].rightChildId = collectNodeInfo(node->right, depth + 1, nodes);
    }
    return currentId;
}

int RandomForestFacade::calculateTreeDepth(TreeNode* node) {
    if (!node) return 0;
    if (node->isLeaf) return 1;
    return 1 + std::max(calculateTreeDepth(node->left), calculateTreeDepth(node->right));
}

int RandomForestFacade::countLeaves(TreeNode* node) {
    if (!node) return 0;
    if (node->isLeaf) return 1;
    return countLeaves(node->left) + countLeaves(node->right);
}

void RandomForestFacade::collectFeaturesUsed(TreeNode* node, std::vector<bool>& used) {
    if (!node || node->isLeaf) return;
    if (node->featureIndex >= 0 && node->featureIndex < MAX_FEATURES)
        used[node->featureIndex] = true;
    collectFeaturesUsed(node->left, used);
    collectFeaturesUsed(node->right, used);
}

TreeNode* RandomForestFacade::findNodeById(TreeNode* node, int targetId, int& currentId) {
    if (!node) return nullptr;
    if (currentId == targetId) return node;
    currentId++;
    
    if (!node->isLeaf) {
        TreeNode* found = findNodeById(node->left, targetId, currentId);
        if (found) return found;
        found = findNodeById(node->right, targetId, currentId);
        if (found) return found;
    }
    return nullptr;
}

void RandomForestFacade::freeSubtree(TreeNode* node) {
    if (!node) return;
    freeSubtree(node->left);
    freeSubtree(node->right);
    delete node;
}

TreeInfo RandomForestFacade::inspectTree(int treeId) {
    TreeInfo info;
    info.treeId = treeId;
    
    if (treeId < 0 || treeId >= getNumTrees()) return info;
    
    DecisionTree* tree = forest.getTree(treeId);
    if (!tree) return info;
    
    collectNodeInfo(tree->root, 0, info.nodes);
    info.numNodes = info.nodes.size();
    info.maxDepth = calculateTreeDepth(tree->root);
    info.numLeaves = countLeaves(tree->root);
    
    info.featuresUsed.assign(MAX_FEATURES, false);
    collectFeaturesUsed(tree->root, info.featuresUsed);
    info.numFeaturesUsed = std::count(info.featuresUsed.begin(), info.featuresUsed.end(), true);
    
    return info;
}

void RandomForestFacade::printTreeStructure(int treeId) {
    TreeInfo info = inspectTree(treeId);
    
    std::cout << "=== Tree " << treeId << " Structure ===\n";
    std::cout << "Nodes: " << info.numNodes << "\n";
    std::cout << "Max Depth: " << info.maxDepth << "\n";
    std::cout << "Leaves: " << info.numLeaves << "\n";
    std::cout << "Features Used: " << info.numFeaturesUsed << "\n\n";
    
    std::cout << "ID    Depth  Leaf   Feature  Threshold      Prediction  Samples  Impurity\n";
    std::cout << "----------------------------------------------------------------------\n";
    for (const auto& n : info.nodes) {
        std::cout << std::setw(4) << n.nodeId << "  "
                  << std::setw(4) << n.depth << "   "
                  << (n.isLeaf ? "Yes    " : "No     ")
                  << std::setw(6) << n.featureIndex << "  "
                  << std::setw(12) << std::fixed << std::setprecision(4) << n.threshold << "  "
                  << std::setw(10) << n.prediction << "  "
                  << std::setw(6) << n.numSamples << "  "
                  << std::setw(8) << n.impurity << "\n";
    }
}

void RandomForestFacade::printNodeDetails(int treeId, int nodeId) {
    TreeInfo info = inspectTree(treeId);
    
    if (nodeId < 0 || nodeId >= (int)info.nodes.size()) {
        std::cout << "Invalid node ID: " << nodeId << "\n";
        return;
    }
    
    const NodeInfo& n = info.nodes[nodeId];
    std::cout << "=== Node " << nodeId << " in Tree " << treeId << " ===\n";
    std::cout << "Depth: " << n.depth << "\n";
    std::cout << "Is Leaf: " << (n.isLeaf ? "TRUE" : "FALSE") << "\n";
    if (!n.isLeaf) {
        std::cout << "Split Feature: " << n.featureIndex << "\n";
        std::cout << "Threshold: " << std::fixed << std::setprecision(4) << n.threshold << "\n";
        std::cout << "Left Child: " << n.leftChildId << "\n";
        std::cout << "Right Child: " << n.rightChildId << "\n";
    }
    std::cout << "Prediction: " << n.prediction << "\n";
    std::cout << "Class Label: " << n.classLabel << "\n";
    std::cout << "Samples: " << n.numSamples << "\n";
    std::cout << "Impurity: " << n.impurity << "\n";
}

int RandomForestFacade::getTreeDepth(int treeId) {
    if (treeId < 0 || treeId >= getNumTrees()) return 0;
    DecisionTree* tree = forest.getTree(treeId);
    return tree ? calculateTreeDepth(tree->root) : 0;
}

int RandomForestFacade::getTreeNumNodes(int treeId) { return inspectTree(treeId).numNodes; }

int RandomForestFacade::getTreeNumLeaves(int treeId) {
    if (treeId < 0 || treeId >= getNumTrees()) return 0;
    DecisionTree* tree = forest.getTree(treeId);
    return tree ? countLeaves(tree->root) : 0;
}

void RandomForestFacade::pruneTree(int treeId, int nodeId) {
    if (treeId < 0 || treeId >= getNumTrees()) {
        std::cout << "Invalid tree ID: " << treeId << "\n";
        return;
    }
    
    DecisionTree* tree = forest.getTree(treeId);
    if (!tree) return;
    
    int searchId = 0;
    TreeNode* node = findNodeById(tree->root, nodeId, searchId);
    
    if (!node) { std::cout << "Node not found: " << nodeId << "\n"; return; }
    if (node->isLeaf) { std::cout << "Cannot prune a leaf node\n"; return; }
    
    freeSubtree(node->left);
    freeSubtree(node->right);
    node->left = nullptr;
    node->right = nullptr;
    node->isLeaf = true;
    
    std::cout << "Pruned node " << nodeId << " in tree " << treeId << "\n";
}

void RandomForestFacade::modifySplit(int treeId, int nodeId, double newThreshold) {
    if (treeId < 0 || treeId >= getNumTrees()) {
        std::cout << "Invalid tree ID: " << treeId << "\n";
        return;
    }
    
    DecisionTree* tree = forest.getTree(treeId);
    if (!tree) return;
    
    int searchId = 0;
    TreeNode* node = findNodeById(tree->root, nodeId, searchId);
    
    if (!node) { std::cout << "Node not found: " << nodeId << "\n"; return; }
    if (node->isLeaf) { std::cout << "Cannot modify split on a leaf node\n"; return; }
    
    std::cout << "Modified threshold from " << node->threshold << " to " << newThreshold << "\n";
    node->threshold = newThreshold;
}

void RandomForestFacade::modifyLeafValue(int treeId, int nodeId, double newValue) {
    if (treeId < 0 || treeId >= getNumTrees()) {
        std::cout << "Invalid tree ID: " << treeId << "\n";
        return;
    }
    
    DecisionTree* tree = forest.getTree(treeId);
    if (!tree) return;
    
    int searchId = 0;
    TreeNode* node = findNodeById(tree->root, nodeId, searchId);
    
    if (!node) { std::cout << "Node not found: " << nodeId << "\n"; return; }
    if (!node->isLeaf) { std::cout << "Node is not a leaf\n"; return; }
    
    std::cout << "Modified leaf value from " << node->prediction << " to " << newValue << "\n";
    node->prediction = newValue;
    node->classLabel = (int)std::round(newValue);
}

void RandomForestFacade::convertToLeaf(int treeId, int nodeId, double leafValue) {
    if (treeId < 0 || treeId >= getNumTrees()) {
        std::cout << "Invalid tree ID: " << treeId << "\n";
        return;
    }
    
    DecisionTree* tree = forest.getTree(treeId);
    if (!tree) return;
    
    int searchId = 0;
    TreeNode* node = findNodeById(tree->root, nodeId, searchId);
    
    if (!node) { std::cout << "Node not found: " << nodeId << "\n"; return; }
    if (node->isLeaf) { std::cout << "Node is already a leaf\n"; return; }
    
    freeSubtree(node->left);
    freeSubtree(node->right);
    node->left = nullptr;
    node->right = nullptr;
    node->isLeaf = true;
    node->prediction = leafValue;
    node->classLabel = (int)std::round(leafValue);
    node->featureIndex = -1;
    node->threshold = 0;
    
    std::cout << "Converted node " << nodeId << " to leaf with value " << leafValue << "\n";
}

void RandomForestFacade::addTree() {
    int oldCount = getNumTrees();
    forest.addNewTree();
    if (getNumTrees() > oldCount)
        std::cout << "Added new tree. Total trees: " << getNumTrees() << "\n";
    else
        std::cout << "Failed to add tree\n";
}

void RandomForestFacade::removeTree(int treeId) {
    if (treeId < 0 || treeId >= getNumTrees()) {
        std::cout << "Invalid tree ID: " << treeId << "\n";
        return;
    }
    forest.removeTreeAt(treeId);
    std::cout << "Removed tree " << treeId << ". Total trees: " << getNumTrees() << "\n";
}

void RandomForestFacade::replaceTree(int treeId) {
    if (treeId < 0 || treeId >= getNumTrees()) {
        std::cout << "Invalid tree ID: " << treeId << "\n";
        return;
    }
    forest.retrainTreeAt(treeId);
    std::cout << "Replaced tree " << treeId << " with new bootstrap sample\n";
}

void RandomForestFacade::retrainTree(int treeId) {
    if (treeId < 0 || treeId >= getNumTrees()) {
        std::cout << "Invalid tree ID: " << treeId << "\n";
        return;
    }
    forest.retrainTreeAt(treeId);
    std::cout << "Retrained tree " << treeId << "\n";
}

int RandomForestFacade::getNumTrees() { return forest.getNumTrees(); }

void RandomForestFacade::enableFeature(int featureIndex) {
    if (featureIndex >= 0 && featureIndex < MAX_FEATURES)
        featureEnabled[featureIndex] = true;
}

void RandomForestFacade::disableFeature(int featureIndex) {
    if (featureIndex >= 0 && featureIndex < MAX_FEATURES)
        featureEnabled[featureIndex] = false;
}

void RandomForestFacade::resetFeatureFilters() {
    std::fill(featureEnabled.begin(), featureEnabled.end(), true);
}

std::vector<FeatureStats> RandomForestFacade::featureUsageSummary() {
    std::vector<FeatureStats> stats(MAX_FEATURE_STATS);
    for (int i = 0; i < MAX_FEATURE_STATS; i++) {
        stats[i].featureIndex = i;
        stats[i].timesUsed = 0;
        stats[i].treesUsedIn = 0;
        stats[i].avgImportance = 0.0;
        stats[i].totalImportance = 0.0;
    }
    
    for (int t = 0; t < getNumTrees(); t++) {
        TreeInfo info = inspectTree(t);
        for (int i = 0; i < MAX_FEATURES; i++) {
            if (info.featuresUsed[i]) stats[i].treesUsedIn++;
        }
    }
    
    for (int i = 0; i < forest.getNumFeatures(); i++) {
        stats[i].totalImportance = forest.getFeatureImportance(i);
        stats[i].avgImportance = stats[i].totalImportance;
    }
    
    return stats;
}

void RandomForestFacade::printFeatureUsageSummary() {
    auto stats = featureUsageSummary();
    
    std::cout << "=== Feature Usage Summary ===\n";
    std::cout << "Feature  Trees Used In  Importance\n";
    std::cout << "----------------------------------\n";
    for (int i = 0; i < forest.getNumFeatures(); i++) {
        std::cout << std::setw(6) << i << "  "
                  << std::setw(12) << stats[i].treesUsedIn << "  "
                  << std::setw(10) << std::fixed << std::setprecision(4) 
                  << stats[i].totalImportance << "\n";
    }
}

double RandomForestFacade::getFeatureImportance(int featureIndex) {
    return forest.getFeatureImportance(featureIndex);
}

void RandomForestFacade::printFeatureImportances() { forest.printFeatureImportances(); }

void RandomForestFacade::setAggregationMethod(AggregationMethod method) { currentAggregation = method; }
AggregationMethod RandomForestFacade::getAggregationMethod() { return currentAggregation; }

void RandomForestFacade::setTreeWeight(int treeId, double weight) {
    if (treeId >= 0 && treeId < MAX_TREES) treeWeights[treeId] = weight;
}

double RandomForestFacade::getTreeWeight(int treeId) {
    if (treeId >= 0 && treeId < MAX_TREES) return treeWeights[treeId];
    return 1.0;
}

void RandomForestFacade::resetTreeWeights() {
    std::fill(treeWeights.begin(), treeWeights.end(), 1.0);
}

double RandomForestFacade::aggregatePredictions(const std::vector<double>& sample) {
    switch (currentAggregation) {
        case AggregationMethod::MajorityVote: {
            std::map<int, int> votes;
            for (int i = 0; i < getNumTrees(); i++) {
                DecisionTree* tree = forest.getTree(i);
                if (tree) {
                    int label = (int)std::round(forest.predictTree(tree->root, sample));
                    votes[label]++;
                }
            }
            int maxVotes = 0, maxClass = 0;
            for (auto& p : votes) {
                if (p.second > maxVotes) { maxVotes = p.second; maxClass = p.first; }
            }
            return maxClass;
        }
        case AggregationMethod::WeightedVote: {
            std::map<int, double> votes;
            for (int i = 0; i < getNumTrees(); i++) {
                DecisionTree* tree = forest.getTree(i);
                if (tree) {
                    int label = (int)std::round(forest.predictTree(tree->root, sample));
                    votes[label] += treeWeights[i];
                }
            }
            double maxVotes = 0; int maxClass = 0;
            for (auto& p : votes) {
                if (p.second > maxVotes) { maxVotes = p.second; maxClass = p.first; }
            }
            return maxClass;
        }
        case AggregationMethod::Mean: {
            double sum = 0;
            for (int i = 0; i < getNumTrees(); i++) {
                DecisionTree* tree = forest.getTree(i);
                if (tree) sum += forest.predictTree(tree->root, sample);
            }
            return sum / getNumTrees();
        }
        case AggregationMethod::WeightedMean: {
            double sum = 0, weightSum = 0;
            for (int i = 0; i < getNumTrees(); i++) {
                DecisionTree* tree = forest.getTree(i);
                if (tree) {
                    double pred = forest.predictTree(tree->root, sample);
                    sum += pred * treeWeights[i];
                    weightSum += treeWeights[i];
                }
            }
            return weightSum > 0 ? sum / weightSum : 0;
        }
    }
    return 0;
}

double RandomForestFacade::predict(const std::vector<double>& sample) {
    return aggregatePredictions(sample);
}

int RandomForestFacade::predictClass(const std::vector<double>& sample) {
    return (int)std::round(predict(sample));
}

double RandomForestFacade::predictWithTree(int treeId, const std::vector<double>& sample) {
    if (treeId < 0 || treeId >= getNumTrees()) return 0;
    DecisionTree* tree = forest.getTree(treeId);
    return tree ? forest.predictTree(tree->root, sample) : 0;
}

void RandomForestFacade::predictBatch(const std::vector<std::vector<double>>& samples,
                                       std::vector<double>& predictions) {
    predictions.resize(samples.size());
    for (size_t i = 0; i < samples.size(); i++)
        predictions[i] = predict(samples[i]);
}

SampleTrackInfo RandomForestFacade::trackSample(int sampleIndex) {
    SampleTrackInfo info;
    info.sampleIndex = sampleIndex;
    info.treesInfluenced.resize(MAX_TREES, false);
    info.oobTrees.resize(MAX_TREES, false);
    info.predictions.resize(MAX_TREES, 0);
    info.numTreesInfluenced = 0;
    info.numOobTrees = 0;
    
    if (sampleIndex < 0 || sampleIndex >= forest.getNumSamples()) return info;
    
    std::vector<double> sampleRow(forest.getNumFeatures());
    for (int j = 0; j < forest.getNumFeatures(); j++)
        sampleRow[j] = forest.getData(sampleIndex, j);
    
    for (int t = 0; t < getNumTrees(); t++) {
        DecisionTree* tree = forest.getTree(t);
        if (tree) {
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

void RandomForestFacade::printSampleTracking(int sampleIndex) {
    SampleTrackInfo info = trackSample(sampleIndex);
    
    std::cout << "=== Sample " << sampleIndex << " Tracking ===\n";
    std::cout << "Trees Influenced (in bootstrap): " << info.numTreesInfluenced << "\n";
    std::cout << "OOB Trees (not in bootstrap): " << info.numOobTrees << "\n\n";
    
    std::cout << "Tree  Influenced  OOB  Prediction\n";
    std::cout << "----------------------------------\n";
    for (int t = 0; t < getNumTrees(); t++) {
        std::cout << std::setw(4) << t << "  "
                  << (info.treesInfluenced[t] ? "Yes        " : "No         ")
                  << (info.oobTrees[t] ? "Yes  " : "No   ")
                  << std::fixed << std::setprecision(4) << info.predictions[t] << "\n";
    }
}

std::vector<OOBTreeInfo> RandomForestFacade::oobErrorSummary() {
    std::vector<OOBTreeInfo> summary(MAX_TREES);
    
    for (int t = 0; t < MAX_TREES; t++) {
        summary[t].treeId = t;
        summary[t].numOobSamples = 0;
        summary[t].oobError = 0;
        summary[t].oobAccuracy = 0;
    }
    
    for (int t = 0; t < getNumTrees(); t++) {
        DecisionTree* tree = forest.getTree(t);
        if (!tree) continue;
        
        int errors = 0, correct = 0;
        std::vector<double> sampleRow(forest.getNumFeatures());
        
        for (int s = 0; s < forest.getNumSamples(); s++) {
            if (tree->oobIndices[s]) {
                summary[t].numOobSamples++;
                
                for (int j = 0; j < forest.getNumFeatures(); j++)
                    sampleRow[j] = forest.getData(s, j);
                
                double pred = forest.predictTree(tree->root, sampleRow);
                
                if (forest.getTaskType() == TaskType::Classification) {
                    if ((int)std::round(pred) == (int)std::round(forest.getTarget(s)))
                        correct++;
                    else
                        errors++;
                } else {
                    summary[t].oobError += std::pow(pred - forest.getTarget(s), 2);
                }
            }
        }
        
        if (summary[t].numOobSamples > 0) {
            if (forest.getTaskType() == TaskType::Classification) {
                summary[t].oobError = (double)errors / summary[t].numOobSamples;
                summary[t].oobAccuracy = (double)correct / summary[t].numOobSamples;
            } else {
                summary[t].oobError /= summary[t].numOobSamples;
                summary[t].oobAccuracy = 1.0 - summary[t].oobError;
            }
        }
    }
    return summary;
}

void RandomForestFacade::printOOBSummary() {
    auto summary = oobErrorSummary();
    
    std::cout << "=== OOB Error Summary ===\n";
    std::cout << "Tree  OOB Samples  OOB Error  OOB Accuracy\n";
    std::cout << "-------------------------------------------\n";
    
    double totalError = 0;
    int count = 0;
    
    for (int t = 0; t < getNumTrees(); t++) {
        std::cout << std::setw(4) << t << "  "
                  << std::setw(10) << summary[t].numOobSamples << "  "
                  << std::setw(9) << std::fixed << std::setprecision(4) << summary[t].oobError << "  "
                  << std::setw(11) << summary[t].oobAccuracy << "\n";
        
        if (summary[t].numOobSamples > 0) {
            totalError += summary[t].oobError;
            count++;
        }
    }
    
    std::cout << "-------------------------------------------\n";
    if (count > 0) std::cout << "Average OOB Error: " << totalError / count << "\n";
    std::cout << "Global OOB Error: " << getGlobalOOBError() << "\n";
}

double RandomForestFacade::getGlobalOOBError() { return forest.calculateOOBError(); }

void RandomForestFacade::markProblematicTrees(double errorThreshold) {
    auto summary = oobErrorSummary();
    int problemCount = 0;
    
    std::cout << "=== Problematic Trees (Error > " << errorThreshold << ") ===\n";
    
    for (int t = 0; t < getNumTrees(); t++) {
        if (summary[t].numOobSamples > 0 && summary[t].oobError > errorThreshold) {
            std::cout << "Tree " << t << ": OOB Error = " << summary[t].oobError
                      << " (" << summary[t].numOobSamples << " OOB samples)\n";
            problemCount++;
        }
    }
    
    if (problemCount == 0) std::cout << "No problematic trees found.\n";
    else std::cout << "Total problematic trees: " << problemCount << "\n";
}

double RandomForestFacade::accuracy(const std::vector<double>& predictions,
                                     const std::vector<double>& actual) {
    return forest.accuracy(predictions, actual);
}

double RandomForestFacade::meanSquaredError(const std::vector<double>& predictions,
                                             const std::vector<double>& actual) {
    return forest.meanSquaredError(predictions, actual);
}

double RandomForestFacade::rSquared(const std::vector<double>& predictions,
                                     const std::vector<double>& actual) {
    return forest.rSquared(predictions, actual);
}

double RandomForestFacade::precision(const std::vector<double>& predictions,
                                      const std::vector<double>& actual, int posClass) {
    return forest.precision(predictions, actual, posClass);
}

double RandomForestFacade::recall(const std::vector<double>& predictions,
                                   const std::vector<double>& actual, int posClass) {
    return forest.recall(predictions, actual, posClass);
}

double RandomForestFacade::f1Score(const std::vector<double>& predictions,
                                    const std::vector<double>& actual, int posClass) {
    return forest.f1Score(predictions, actual, posClass);
}

void RandomForestFacade::highlightMisclassified(const std::vector<double>& predictions,
                                                 const std::vector<double>& actual) {
    std::cout << "=== Misclassified Samples ===\n";
    std::cout << "Index  Predicted  Actual\n";
    std::cout << "-------------------------\n";
    
    int count = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        if ((int)std::round(predictions[i]) != (int)std::round(actual[i])) {
            std::cout << std::setw(5) << i << "  "
                      << std::setw(9) << (int)std::round(predictions[i]) << "  "
                      << std::setw(6) << (int)std::round(actual[i]) << "\n";
            count++;
        }
    }
    
    std::cout << "-------------------------\n";
    std::cout << "Total misclassified: " << count << " / " << predictions.size()
              << " (" << std::fixed << std::setprecision(2) 
              << (100.0 * count / predictions.size()) << "%)\n";
}

void RandomForestFacade::highlightHighResidual(const std::vector<double>& predictions,
                                                const std::vector<double>& actual,
                                                double threshold) {
    std::cout << "=== High Residual Samples (> " << threshold << ") ===\n";
    std::cout << "Index  Predicted  Actual   Residual\n";
    std::cout << "-------------------------------------\n";
    
    int count = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        double residual = std::abs(predictions[i] - actual[i]);
        if (residual > threshold) {
            std::cout << std::setw(5) << i << "  "
                      << std::setw(9) << std::fixed << std::setprecision(4) << predictions[i] << "  "
                      << std::setw(6) << actual[i] << "  "
                      << std::setw(8) << residual << "\n";
            count++;
        }
    }
    
    std::cout << "-------------------------------------\n";
    std::cout << "Total high-residual samples: " << count << " / " << predictions.size() << "\n";
}

void RandomForestFacade::findWorstTrees(const std::vector<double>& actual, int topN) {
    auto summary = oobErrorSummary();
    
    std::vector<std::pair<double, int>> treeErrors;
    for (int t = 0; t < getNumTrees(); t++) {
        treeErrors.push_back({summary[t].oobError, t});
    }
    std::sort(treeErrors.begin(), treeErrors.end(), std::greater<>());
    
    std::cout << "=== Top " << topN << " Worst Trees ===\n";
    std::cout << "Rank  Tree  OOB Error\n";
    std::cout << "----------------------\n";
    
    for (int i = 0; i < topN && i < (int)treeErrors.size(); i++) {
        std::cout << std::setw(4) << (i + 1) << "  "
                  << std::setw(4) << treeErrors[i].second << "  "
                  << std::setw(9) << std::fixed << std::setprecision(4) 
                  << treeErrors[i].first << "\n";
    }
}

void RandomForestFacade::visualizeTree(int treeId) {
    TreeInfo info = inspectTree(treeId);
    
    std::cout << "=== Tree " << treeId << " Visualization ===\n\n";
    
    for (const auto& n : info.nodes) {
        for (int j = 0; j < n.depth * 2; j++) std::cout << " ";
        
        if (n.isLeaf) {
            std::cout << "[Leaf] -> " << std::fixed << std::setprecision(2) 
                      << n.prediction << " (n=" << n.numSamples << ")\n";
        } else {
            std::cout << "[Split] Feature " << n.featureIndex 
                      << " <= " << std::setprecision(4) << n.threshold
                      << " (n=" << n.numSamples << ", imp=" << n.impurity << ")\n";
        }
    }
}

void RandomForestFacade::visualizeSplitDistribution(int treeId, int nodeId) {
    TreeInfo info = inspectTree(treeId);
    
    if (nodeId < 0 || nodeId >= (int)info.nodes.size()) {
        std::cout << "Invalid node ID: " << nodeId << "\n";
        return;
    }
    
    const NodeInfo& n = info.nodes[nodeId];
    std::cout << "=== Split Distribution for Tree " << treeId << ", Node " << nodeId << " ===\n";
    std::cout << "Feature Index: " << n.featureIndex << "\n";
    std::cout << "Threshold: " << std::fixed << std::setprecision(4) << n.threshold << "\n";
    std::cout << "Impurity: " << n.impurity << "\n";
    std::cout << "Samples at Node: " << n.numSamples << "\n";
    std::cout << "Is Leaf: " << (n.isLeaf ? "TRUE" : "FALSE") << "\n";
    
    if (!n.isLeaf) {
        std::cout << "Left Child ID: " << n.leftChildId << "\n";
        std::cout << "Right Child ID: " << n.rightChildId << "\n";
    }
}

void RandomForestFacade::printForestOverview() {
    std::cout << "=== Forest Overview ===\n";
    forest.printForestInfo();
    std::cout << "\n";
    
    int totalNodes = 0, totalLeaves = 0;
    double avgDepth = 0;
    
    for (int i = 0; i < getNumTrees(); i++) {
        totalNodes += getTreeNumNodes(i);
        totalLeaves += getTreeNumLeaves(i);
        avgDepth += getTreeDepth(i);
    }
    
    if (getNumTrees() > 0) avgDepth /= getNumTrees();
    
    std::cout << "Forest Statistics:\n";
    std::cout << "  Total Nodes: " << totalNodes << "\n";
    std::cout << "  Total Leaves: " << totalLeaves << "\n";
    std::cout << "  Average Tree Depth: " << std::fixed << std::setprecision(2) << avgDepth << "\n\n";
    
    std::cout << "Tree Summary:\n";
    std::cout << "Tree  Depth  Nodes  Leaves  Weight\n";
    std::cout << "------------------------------------\n";
    for (int i = 0; i < getNumTrees(); i++) {
        std::cout << std::setw(4) << i << "  "
                  << std::setw(5) << getTreeDepth(i) << "  "
                  << std::setw(5) << getTreeNumNodes(i) << "  "
                  << std::setw(6) << getTreeNumLeaves(i) << "  "
                  << std::setw(6) << std::fixed << std::setprecision(2) << treeWeights[i] << "\n";
    }
}

void RandomForestFacade::printFeatureHeatmap() {
    std::cout << "=== Feature Usage Heatmap ===\n\n";
    
    std::vector<std::vector<bool>> usage(MAX_FEATURES, std::vector<bool>(MAX_TREES, false));
    
    for (int t = 0; t < getNumTrees(); t++) {
        TreeInfo info = inspectTree(t);
        for (int f = 0; f < MAX_FEATURES; f++)
            usage[f][t] = info.featuresUsed[f];
    }
    
    std::cout << "Feat  ";
    for (int t = 0; t < getNumTrees(); t++)
        std::cout << (t % 10) << " ";
    std::cout << "  Total\n";
    
    std::cout << std::string(8 + getNumTrees() * 2 + 8, '-') << "\n";
    
    for (int f = 0; f < forest.getNumFeatures(); f++) {
        std::cout << std::setw(4) << f << "  ";
        int total = 0;
        for (int t = 0; t < getNumTrees(); t++) {
            std::cout << (usage[f][t] ? "X " : ". ");
            if (usage[f][t]) total++;
        }
        std::cout << "  " << std::setw(4) << total << "\n";
    }
}

void RandomForestFacade::freeForest() {
    forest.freeForest();
    forestInitialized = false;
}

// ============================================================================
// Model Save/Load
// ============================================================================

void saveTreeNodeRec(std::ofstream& f, TreeNode* node) {
    if (!node) return;
    
    f.write((char*)&node->isLeaf, sizeof(bool));
    f.write((char*)&node->featureIndex, sizeof(int));
    f.write((char*)&node->threshold, sizeof(double));
    f.write((char*)&node->prediction, sizeof(double));
    f.write((char*)&node->classLabel, sizeof(int));
    f.write((char*)&node->impurity, sizeof(double));
    f.write((char*)&node->numSamples, sizeof(int));
    
    uint8_t hasLeft = node->left ? 1 : 0;
    uint8_t hasRight = node->right ? 1 : 0;
    f.write((char*)&hasLeft, 1);
    f.write((char*)&hasRight, 1);
    
    if (node->left) saveTreeNodeRec(f, node->left);
    if (node->right) saveTreeNodeRec(f, node->right);
}

TreeNode* loadTreeNodeRec(std::ifstream& f) {
    TreeNode* node = new TreeNode();
    
    f.read((char*)&node->isLeaf, sizeof(bool));
    f.read((char*)&node->featureIndex, sizeof(int));
    f.read((char*)&node->threshold, sizeof(double));
    f.read((char*)&node->prediction, sizeof(double));
    f.read((char*)&node->classLabel, sizeof(int));
    f.read((char*)&node->impurity, sizeof(double));
    f.read((char*)&node->numSamples, sizeof(int));
    
    uint8_t hasLeft, hasRight;
    f.read((char*)&hasLeft, 1);
    f.read((char*)&hasRight, 1);
    
    node->left = hasLeft ? loadTreeNodeRec(f) : nullptr;
    node->right = hasRight ? loadTreeNodeRec(f) : nullptr;
    
    return node;
}

void RandomForestFacade::saveModel(const std::string& filename) {
    std::ofstream f(filename, std::ios::binary);
    if (!f) {
        std::cout << "Error: Cannot open file for writing: " << filename << "\n";
        return;
    }
    
    uint32_t magic = 0x52464D44;
    f.write((char*)&magic, sizeof(uint32_t));
    
    int numT = forest.getNumTrees();
    int numF = forest.getNumFeatures();
    int numS = forest.getNumSamples();
    int maxD = forest.getMaxDepth();
    TaskType tt = forest.getTaskType();
    SplitCriterion sc = forest.getCriterion();
    
    f.write((char*)&numT, sizeof(int));
    f.write((char*)&numF, sizeof(int));
    f.write((char*)&numS, sizeof(int));
    f.write((char*)&maxD, sizeof(int));
    f.write((char*)&tt, sizeof(TaskType));
    f.write((char*)&sc, sizeof(SplitCriterion));
    
    for (int i = 0; i < numT; i++) {
        DecisionTree* tree = forest.getTree(i);
        if (tree) saveTreeNodeRec(f, tree->root);
    }
    
    f.close();
    std::cout << "Model saved to " << filename << " (" << numT << " trees)\n";
}

bool RandomForestFacade::loadModel(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) {
        std::cout << "Error: Cannot open model file: " << filename << "\n";
        return false;
    }
    
    uint32_t magic;
    f.read((char*)&magic, sizeof(uint32_t));
    if (magic != 0x52464D44) {
        std::cout << "Error: Invalid model file format\n";
        f.close();
        return false;
    }
    
    int numT, numF, numS, maxD;
    TaskType tt;
    SplitCriterion sc;
    
    f.read((char*)&numT, sizeof(int));
    f.read((char*)&numF, sizeof(int));
    f.read((char*)&numS, sizeof(int));
    f.read((char*)&maxD, sizeof(int));
    f.read((char*)&tt, sizeof(TaskType));
    f.read((char*)&sc, sizeof(SplitCriterion));
    
    forest.setNumTrees(numT);
    forest.setMaxDepth(maxD);
    forest.setTaskType(tt);
    forest.setCriterion(sc);
    
    for (int i = 0; i < numT; i++) {
        DecisionTree* tree = new DecisionTree();
        tree->maxDepth = maxD;
        tree->taskType = tt;
        tree->criterion = sc;
        tree->root = loadTreeNodeRec(f);
        forest.setTree(i, tree);
    }
    
    forestInitialized = true;
    f.close();
    std::cout << "Model loaded from " << filename << " (" << numT << " trees)\n";
    return true;
}

// ============================================================================
// CSV Loading
// ============================================================================

bool loadCSV(const std::string& filename, std::vector<std::vector<double>>& data,
             std::vector<double>& targets, int& nSamples, int& nFeatures, int targetCol) {
    std::ifstream f(filename);
    if (!f) {
        std::cout << "Error: Cannot open file: " << filename << "\n";
        return false;
    }
    
    data.clear();
    targets.clear();
    
    std::string line;
    bool firstLine = true;
    int numCols = 0;
    
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        
        if (firstLine) {
            firstLine = false;
            bool hasHeader = false;
            for (char c : line) {
                if (std::isalpha(c)) { hasHeader = true; break; }
            }
            if (hasHeader) continue;
        }
        
        std::vector<double> values;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            try {
                values.push_back(std::stod(cell));
            } catch (...) {
                values.push_back(0.0);
            }
        }
        
        if (values.empty()) continue;
        if (numCols == 0) numCols = values.size();
        
        int tCol = (targetCol < 0) ? values.size() - 1 : targetCol;
        
        std::vector<double> row;
        for (size_t i = 0; i < values.size(); i++) {
            if ((int)i == tCol) targets.push_back(values[i]);
            else row.push_back(values[i]);
        }
        data.push_back(row);
    }
    
    f.close();
    
    nSamples = data.size();
    nFeatures = nSamples > 0 ? data[0].size() : 0;
    
    std::cout << "Loaded " << nSamples << " samples with " << nFeatures 
              << " features from " << filename << "\n";
    return true;
}

void savePredictionsCSV(const std::string& filename, const std::vector<double>& predictions) {
    std::ofstream f(filename);
    f << "prediction\n";
    for (double p : predictions) {
        f << std::fixed << std::setprecision(6) << p << "\n";
    }
    f.close();
    std::cout << "Saved " << predictions.size() << " predictions to " << filename << "\n";
}

// ============================================================================
// CLI Argument Parsing
// ============================================================================

class Args {
    std::map<std::string, std::string> args;
public:
    Args(int argc, char* argv[]) {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg.substr(0, 2) == "--" && i + 1 < argc) {
                args[arg] = argv[++i];
            }
        }
    }
    
    std::string get(const std::string& name, const std::string& def = "") {
        return args.count(name) ? args[name] : def;
    }
    
    int getInt(const std::string& name, int def) {
        if (!args.count(name)) return def;
        try { return std::stoi(args[name]); } catch (...) { return def; }
    }
    
    double getDouble(const std::string& name, double def) {
        if (!args.count(name)) return def;
        try { return std::stod(args[name]); } catch (...) { return def; }
    }
    
    bool has(const std::string& name) { return args.count(name) > 0; }
};

// ============================================================================
// CLI Help
// ============================================================================

void printHelp() {
    std::cout << R"(Random Forest CLI - CUDA Implementation
Matthew Abbott 2025

Usage: forest_cuda <command> [options]

=== Core Commands ===
  train          Train a new random forest model
  predict        Make predictions using a trained model
  evaluate       Evaluate model on test data
  info           Show model information
  inspect        Inspect tree structure
  help           Show this help message

=== Tree Management (Facade) ===
  add-tree       Add a new tree to the forest
  remove-tree    Remove a tree from the forest
  retrain-tree   Retrain a specific tree
  prune          Prune a tree at a specific node
  modify-split   Modify split threshold at a node
  modify-leaf    Modify leaf prediction value
  convert-leaf   Convert internal node to leaf

=== Aggregation Control ===
  set-weight     Set weight for a specific tree
  reset-weights  Reset all tree weights to 1.0

=== Feature Analysis ===
  feature-usage    Show feature usage summary
  feature-heatmap  Show feature usage heatmap across trees
  importance       Show feature importances

=== OOB Analysis ===
  oob-summary      Show OOB error summary per tree
  problematic      Find trees with high OOB error
  worst-trees      Find N worst performing trees

=== Diagnostics ===
  misclassified    Show misclassified samples
  high-residual    Show samples with high residual
  track-sample     Track a sample through the forest

=== Visualization ===
  visualize        Visualize tree structure
  node-details     Show details of a specific node

=== Options ===

Training Options:
  --data <file.csv>      Training data (required)
  --model <file.bin>     Output model file (default: model.bin)
  --trees <n>            Number of trees (default: 100)
  --depth <n>            Max tree depth (default: 10)
  --min-leaf <n>         Min samples per leaf (default: 1)
  --min-split <n>        Min samples to split (default: 2)
  --max-features <n>     Max features per split (default: sqrt)
  --task <class|reg>     Task type (default: class)
  --criterion <g|e|m>    Split criterion: gini/entropy/mse (default: gini)
  --seed <n>             Random seed (default: 42)
  --target-col <n>       Target column index (default: last)

General Options:
  --model <file.bin>     Model file
  --tree <n>             Tree index
  --node <n>             Node index
  --threshold <f>        Threshold value
  --value <f>            Leaf value
  --weight <f>           Tree weight
  --top <n>              Number of top results
  --sample <n>           Sample index

Examples:
  forest_cuda train --data train.csv --model rf.bin --trees 50 --depth 8
  forest_cuda predict --data test.csv --model rf.bin --output preds.csv
  forest_cuda evaluate --data test.csv --model rf.bin
  forest_cuda info --model rf.bin
  forest_cuda visualize --model rf.bin --tree 0
)";
}

// ============================================================================
// CLI Commands
// ============================================================================

void cmdTrain(Args& args) {
    std::string dataFile = args.get("--data");
    std::string modelFile = args.get("--model", "model.bin");
    
    if (dataFile.empty()) {
        std::cout << "Error: --data is required\n";
        return;
    }
    
    std::vector<std::vector<double>> data;
    std::vector<double> targets;
    int nSamples, nFeatures;
    int targetCol = args.getInt("--target-col", -1);
    
    if (!loadCSV(dataFile, data, targets, nSamples, nFeatures, targetCol)) return;
    
    RandomForestFacade facade;
    facade.setHyperparameter("n_estimators", args.getInt("--trees", 100));
    facade.setHyperparameter("max_depth", args.getInt("--depth", 10));
    facade.setHyperparameter("min_samples_leaf", args.getInt("--min-leaf", 1));
    facade.setHyperparameter("min_samples_split", args.getInt("--min-split", 2));
    if (args.has("--max-features"))
        facade.setHyperparameter("max_features", args.getInt("--max-features", 0));
    facade.setHyperparameter("random_seed", args.getInt("--seed", 42));
    
    std::string taskStr = args.get("--task", "class");
    facade.setTaskType((taskStr == "reg" || taskStr == "regression") ? 
                       TaskType::Regression : TaskType::Classification);
    
    std::string critStr = args.get("--criterion", "g");
    if (critStr == "e") facade.setCriterion(SplitCriterion::Entropy);
    else if (critStr == "m") facade.setCriterion(SplitCriterion::MSE);
    else facade.setCriterion(SplitCriterion::Gini);
    
    std::cout << "\nTraining Random Forest (CUDA)...\n";
    std::cout << "  Trees: " << args.getInt("--trees", 100) << "\n";
    std::cout << "  Max Depth: " << args.getInt("--depth", 10) << "\n";
    std::cout << "  Samples: " << nSamples << "\n";
    std::cout << "  Features: " << nFeatures << "\n";
    
    facade.loadData(data, targets);
    facade.trainForest();
    
    std::vector<double> predictions;
    facade.predictBatch(data, predictions);
    double acc = facade.accuracy(predictions, targets);
    double oobErr = facade.getGlobalOOBError();
    
    std::cout << "\nTraining Complete!\n";
    std::cout << "  Training Accuracy: " << std::fixed << std::setprecision(4) << acc << "\n";
    std::cout << "  OOB Error: " << oobErr << "\n";
    
    facade.saveModel(modelFile);
}

void cmdPredict(Args& args) {
    std::string dataFile = args.get("--data");
    std::string modelFile = args.get("--model");
    std::string outputFile = args.get("--output");
    
    if (dataFile.empty()) { std::cout << "Error: --data is required\n"; return; }
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    std::vector<std::vector<double>> data;
    std::vector<double> targets;
    int nSamples, nFeatures;
    
    if (!loadCSV(dataFile, data, targets, nSamples, nFeatures, -1)) return;
    
    std::cout << "Making predictions...\n";
    std::vector<double> predictions;
    facade.predictBatch(data, predictions);
    
    if (!outputFile.empty()) {
        savePredictionsCSV(outputFile, predictions);
    } else {
        std::cout << "\nPredictions:\n";
        for (size_t i = 0; i < predictions.size(); i++)
            std::cout << "  Sample " << i << ": " << predictions[i] << "\n";
    }
}

void cmdEvaluate(Args& args) {
    std::string dataFile = args.get("--data");
    std::string modelFile = args.get("--model");
    int targetCol = args.getInt("--target-col", -1);
    int posClass = args.getInt("--positive-class", 1);
    
    if (dataFile.empty()) { std::cout << "Error: --data is required\n"; return; }
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    std::vector<std::vector<double>> data;
    std::vector<double> targets;
    int nSamples, nFeatures;
    
    if (!loadCSV(dataFile, data, targets, nSamples, nFeatures, targetCol)) return;
    
    std::cout << "Evaluating model...\n";
    std::vector<double> predictions;
    facade.predictBatch(data, predictions);
    
    std::cout << "\n=== Evaluation Results ===\n";
    std::cout << "Samples: " << nSamples << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Accuracy: " << facade.accuracy(predictions, targets) << "\n";
    std::cout << "Precision: " << facade.precision(predictions, targets, posClass) << "\n";
    std::cout << "Recall: " << facade.recall(predictions, targets, posClass) << "\n";
    std::cout << "F1 Score: " << facade.f1Score(predictions, targets, posClass) << "\n";
    std::cout << "MSE: " << facade.meanSquaredError(predictions, targets) << "\n";
    std::cout << "R-Squared: " << facade.rSquared(predictions, targets) << "\n";
}

void cmdInfo(Args& args) {
    std::string modelFile = args.get("--model");
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    std::cout << "\n";
    facade.printForestOverview();
    facade.printFeatureImportances();
}

void cmdInspect(Args& args) {
    std::string modelFile = args.get("--model");
    int treeId = args.getInt("--tree", 0);
    
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    std::cout << "\n";
    facade.visualizeTree(treeId);
    std::cout << "\n";
    facade.printTreeStructure(treeId);
}

void cmdAddTree(Args& args) {
    std::string modelFile = args.get("--model");
    std::string dataFile = args.get("--data");
    int targetCol = args.getInt("--target-col", -1);
    
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    if (dataFile.empty()) { std::cout << "Error: --data is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    std::vector<std::vector<double>> data;
    std::vector<double> targets;
    int nSamples, nFeatures;
    
    if (!loadCSV(dataFile, data, targets, nSamples, nFeatures, targetCol)) return;
    
    facade.loadData(data, targets);
    facade.addTree();
    facade.saveModel(modelFile);
}

void cmdRemoveTree(Args& args) {
    std::string modelFile = args.get("--model");
    int treeId = args.getInt("--tree", -1);
    
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    if (treeId < 0) { std::cout << "Error: --tree is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    facade.removeTree(treeId);
    facade.saveModel(modelFile);
}

void cmdRetrainTree(Args& args) {
    std::string modelFile = args.get("--model");
    std::string dataFile = args.get("--data");
    int treeId = args.getInt("--tree", -1);
    int targetCol = args.getInt("--target-col", -1);
    
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    if (dataFile.empty()) { std::cout << "Error: --data is required\n"; return; }
    if (treeId < 0) { std::cout << "Error: --tree is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    std::vector<std::vector<double>> data;
    std::vector<double> targets;
    int nSamples, nFeatures;
    
    if (!loadCSV(dataFile, data, targets, nSamples, nFeatures, targetCol)) return;
    
    facade.loadData(data, targets);
    facade.retrainTree(treeId);
    facade.saveModel(modelFile);
}

void cmdPrune(Args& args) {
    std::string modelFile = args.get("--model");
    int treeId = args.getInt("--tree", -1);
    int nodeId = args.getInt("--node", -1);
    
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    if (treeId < 0) { std::cout << "Error: --tree is required\n"; return; }
    if (nodeId < 0) { std::cout << "Error: --node is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    facade.pruneTree(treeId, nodeId);
    facade.saveModel(modelFile);
}

void cmdModifySplit(Args& args) {
    std::string modelFile = args.get("--model");
    int treeId = args.getInt("--tree", -1);
    int nodeId = args.getInt("--node", -1);
    double threshold = args.getDouble("--threshold", -999999.0);
    
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    if (treeId < 0) { std::cout << "Error: --tree is required\n"; return; }
    if (nodeId < 0) { std::cout << "Error: --node is required\n"; return; }
    if (threshold == -999999.0) { std::cout << "Error: --threshold is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    facade.modifySplit(treeId, nodeId, threshold);
    facade.saveModel(modelFile);
}

void cmdModifyLeaf(Args& args) {
    std::string modelFile = args.get("--model");
    int treeId = args.getInt("--tree", -1);
    int nodeId = args.getInt("--node", -1);
    double value = args.getDouble("--value", -999999.0);
    
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    if (treeId < 0) { std::cout << "Error: --tree is required\n"; return; }
    if (nodeId < 0) { std::cout << "Error: --node is required\n"; return; }
    if (value == -999999.0) { std::cout << "Error: --value is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    facade.modifyLeafValue(treeId, nodeId, value);
    facade.saveModel(modelFile);
}

void cmdConvertLeaf(Args& args) {
    std::string modelFile = args.get("--model");
    int treeId = args.getInt("--tree", -1);
    int nodeId = args.getInt("--node", -1);
    double value = args.getDouble("--value", 0.0);
    
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    if (treeId < 0) { std::cout << "Error: --tree is required\n"; return; }
    if (nodeId < 0) { std::cout << "Error: --node is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    facade.convertToLeaf(treeId, nodeId, value);
    facade.saveModel(modelFile);
}

void cmdSetWeight(Args& args) {
    std::string modelFile = args.get("--model");
    int treeId = args.getInt("--tree", -1);
    double weight = args.getDouble("--weight", 1.0);
    
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    if (treeId < 0) { std::cout << "Error: --tree is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    facade.setTreeWeight(treeId, weight);
    std::cout << "Set tree " << treeId << " weight to " << weight << "\n";
}

void cmdResetWeights(Args& args) {
    std::string modelFile = args.get("--model");
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    facade.resetTreeWeights();
    std::cout << "All tree weights reset to 1.0\n";
}

void cmdFeatureUsage(Args& args) {
    std::string modelFile = args.get("--model");
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    std::cout << "\n";
    facade.printFeatureUsageSummary();
}

void cmdFeatureHeatmap(Args& args) {
    std::string modelFile = args.get("--model");
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    std::cout << "\n";
    facade.printFeatureHeatmap();
}

void cmdImportance(Args& args) {
    std::string modelFile = args.get("--model");
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    std::cout << "\n";
    facade.printFeatureImportances();
}

void cmdOOBSummary(Args& args) {
    std::string modelFile = args.get("--model");
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    std::cout << "\n";
    facade.printOOBSummary();
}

void cmdProblematic(Args& args) {
    std::string modelFile = args.get("--model");
    double threshold = args.getDouble("--threshold", 0.3);
    
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    std::cout << "\n";
    facade.markProblematicTrees(threshold);
}

void cmdWorstTrees(Args& args) {
    std::string modelFile = args.get("--model");
    int topN = args.getInt("--top", 5);
    
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    std::vector<double> dummy;
    std::cout << "\n";
    facade.findWorstTrees(dummy, topN);
}

void cmdMisclassified(Args& args) {
    std::string modelFile = args.get("--model");
    std::string dataFile = args.get("--data");
    int targetCol = args.getInt("--target-col", -1);
    
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    if (dataFile.empty()) { std::cout << "Error: --data is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    std::vector<std::vector<double>> data;
    std::vector<double> targets;
    int nSamples, nFeatures;
    
    if (!loadCSV(dataFile, data, targets, nSamples, nFeatures, targetCol)) return;
    
    std::vector<double> predictions;
    facade.predictBatch(data, predictions);
    std::cout << "\n";
    facade.highlightMisclassified(predictions, targets);
}

void cmdHighResidual(Args& args) {
    std::string modelFile = args.get("--model");
    std::string dataFile = args.get("--data");
    int targetCol = args.getInt("--target-col", -1);
    double threshold = args.getDouble("--threshold", 1.0);
    
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    if (dataFile.empty()) { std::cout << "Error: --data is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    std::vector<std::vector<double>> data;
    std::vector<double> targets;
    int nSamples, nFeatures;
    
    if (!loadCSV(dataFile, data, targets, nSamples, nFeatures, targetCol)) return;
    
    std::vector<double> predictions;
    facade.predictBatch(data, predictions);
    std::cout << "\n";
    facade.highlightHighResidual(predictions, targets, threshold);
}

void cmdTrackSample(Args& args) {
    std::string modelFile = args.get("--model");
    std::string dataFile = args.get("--data");
    int sampleIdx = args.getInt("--sample", 0);
    int targetCol = args.getInt("--target-col", -1);
    
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    if (dataFile.empty()) { std::cout << "Error: --data is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    std::vector<std::vector<double>> data;
    std::vector<double> targets;
    int nSamples, nFeatures;
    
    if (!loadCSV(dataFile, data, targets, nSamples, nFeatures, targetCol)) return;
    
    facade.loadData(data, targets);
    std::cout << "\n";
    facade.printSampleTracking(sampleIdx);
}

void cmdVisualize(Args& args) {
    std::string modelFile = args.get("--model");
    int treeId = args.getInt("--tree", 0);
    
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    std::cout << "\n";
    facade.visualizeTree(treeId);
}

void cmdNodeDetails(Args& args) {
    std::string modelFile = args.get("--model");
    int treeId = args.getInt("--tree", 0);
    int nodeId = args.getInt("--node", 0);
    
    if (modelFile.empty()) { std::cout << "Error: --model is required\n"; return; }
    
    RandomForestFacade facade;
    if (!facade.loadModel(modelFile)) return;
    
    std::cout << "\n";
    facade.printNodeDetails(treeId, nodeId);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printHelp();
        return 0;
    }
    
    std::string cmd = argv[1];
    Args args(argc, argv);
    
    if (cmd == "help" || cmd == "--help" || cmd == "-h") printHelp();
    else if (cmd == "train") cmdTrain(args);
    else if (cmd == "predict") cmdPredict(args);
    else if (cmd == "evaluate") cmdEvaluate(args);
    else if (cmd == "info") cmdInfo(args);
    else if (cmd == "inspect") cmdInspect(args);
    else if (cmd == "add-tree") cmdAddTree(args);
    else if (cmd == "remove-tree") cmdRemoveTree(args);
    else if (cmd == "retrain-tree") cmdRetrainTree(args);
    else if (cmd == "prune") cmdPrune(args);
    else if (cmd == "modify-split") cmdModifySplit(args);
    else if (cmd == "modify-leaf") cmdModifyLeaf(args);
    else if (cmd == "convert-leaf") cmdConvertLeaf(args);
    else if (cmd == "set-weight") cmdSetWeight(args);
    else if (cmd == "reset-weights") cmdResetWeights(args);
    else if (cmd == "feature-usage") cmdFeatureUsage(args);
    else if (cmd == "feature-heatmap") cmdFeatureHeatmap(args);
    else if (cmd == "importance") cmdImportance(args);
    else if (cmd == "oob-summary") cmdOOBSummary(args);
    else if (cmd == "problematic") cmdProblematic(args);
    else if (cmd == "worst-trees") cmdWorstTrees(args);
    else if (cmd == "misclassified") cmdMisclassified(args);
    else if (cmd == "high-residual") cmdHighResidual(args);
    else if (cmd == "track-sample") cmdTrackSample(args);
    else if (cmd == "visualize") cmdVisualize(args);
    else if (cmd == "node-details") cmdNodeDetails(args);
    else {
        std::cout << "Unknown command: " << cmd << "\n";
        std::cout << "Use \"forest_cuda help\" for usage information.\n";
    }
    
    return 0;
}

