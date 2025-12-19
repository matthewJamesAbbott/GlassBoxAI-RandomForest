//
// Created by Matthew Abbott 2025
// CUDA port of Random Forest
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cstring>
#include <cuda_runtime.h>

constexpr int MAX_FEATURES = 100;
constexpr int MAX_SAMPLES = 10000;
constexpr int MAX_TREES = 500;
constexpr int MAX_DEPTH_DEFAULT = 10;
constexpr int MIN_SAMPLES_LEAF_DEFAULT = 1;
constexpr int MIN_SAMPLES_SPLIT_DEFAULT = 2;
constexpr int MAX_NODES = 4096;

enum TaskType { Classification, Regression };
enum SplitCriterion { Gini, Entropy, MSE, VarianceReduction };

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

__global__ void calculateGiniKernel(
    double* targets,
    int* indices,
    int numIndices,
    double* result
) {
    if (numIndices == 0) {
        *result = 0.0;
        return;
    }

    __shared__ int classCount[100];
    
    int tid = threadIdx.x;
    if (tid < 100) classCount[tid] = 0;
    __syncthreads();

    for (int i = tid; i < numIndices; i += blockDim.x) {
        int classLabel = static_cast<int>(round(targets[indices[i]]));
        if (classLabel >= 0 && classLabel < 100)
            atomicAdd(&classCount[classLabel], 1);
    }
    __syncthreads();

    if (tid == 0) {
        double gini = 1.0;
        for (int i = 0; i < 100; i++) {
            double prob = static_cast<double>(classCount[i]) / numIndices;
            gini -= prob * prob;
        }
        *result = gini;
    }
}

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
};

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
    gpuInitialized = false;
    totalGpuNodes = 0;

    for (int i = 0; i < MAX_TREES; i++)
        trees[i] = nullptr;

    for (int i = 0; i < MAX_FEATURES; i++)
        featureImportances[i] = 0.0;

    data = new double[MAX_SAMPLES * MAX_FEATURES];
    targets = new double[MAX_SAMPLES];

    d_data = nullptr;
    d_allTreeNodes = nullptr;
    d_treeNodeOffsets = nullptr;
    d_predictions = nullptr;

    std::srand(static_cast<unsigned>(std::time(nullptr)));
}

TRandomForest::~TRandomForest() {
    freeForest();
    freeGPU();
    delete[] data;
    delete[] targets;
}

void TRandomForest::setNumTrees(int n) {
    if (n > MAX_TREES) numTrees = MAX_TREES;
    else if (n < 1) numTrees = 1;
    else numTrees = n;
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

void TRandomForest::setMaxFeatures(int m) {
    maxFeatures = m;
}

void TRandomForest::setTaskType(TaskType t) {
    taskType = t;
    criterion = (t == Classification) ? Gini : MSE;
}

void TRandomForest::setCriterion(SplitCriterion c) {
    criterion = c;
}

void TRandomForest::setRandomSeed(long seed) {
    randomSeed = seed;
    std::srand(static_cast<unsigned>(seed));
}

int TRandomForest::randomInt(int maxVal) {
    return std::rand() % maxVal;
}

double TRandomForest::randomDouble() {
    return static_cast<double>(std::rand()) / RAND_MAX;
}

void TRandomForest::loadData(double* inputData, double* inputTargets, int nSamples, int nFeatures) {
    numSamples = nSamples;
    numFeatures = nFeatures;

    if (maxFeatures == 0) {
        if (taskType == Classification)
            maxFeatures = static_cast<int>(std::round(std::sqrt(nFeatures)));
        else
            maxFeatures = nFeatures / 3;
        if (maxFeatures < 1) maxFeatures = 1;
    }

    for (int i = 0; i < nSamples; i++) {
        for (int j = 0; j < nFeatures; j++)
            data[i * MAX_FEATURES + j] = inputData[i * nFeatures + j];
        targets[i] = inputTargets[i];
    }
}

void TRandomForest::bootstrap(int* sampleIndices, int& numBootstrap, bool* oobMask) {
    numBootstrap = numSamples;
    for (int i = 0; i < numSamples; i++)
        oobMask[i] = true;

    for (int i = 0; i < numBootstrap; i++) {
        int idx = randomInt(numSamples);
        sampleIndices[i] = idx;
        oobMask[idx] = false;
    }
}

void TRandomForest::selectFeatureSubset(int* featureIndices, int& numSelected) {
    int available[MAX_FEATURES];
    for (int i = 0; i < numFeatures; i++)
        available[i] = i;

    for (int i = numFeatures - 1; i >= 1; i--) {
        int j = randomInt(i + 1);
        int temp = available[i];
        available[i] = available[j];
        available[j] = temp;
    }

    numSelected = maxFeatures;
    if (numSelected > numFeatures) numSelected = numFeatures;

    for (int i = 0; i < numSelected; i++)
        featureIndices[i] = available[i];
}

double TRandomForest::calculateGini(int* indices, int numIndices) {
    if (numIndices == 0) return 0.0;

    int classCount[100] = {0};
    int numClasses = 0;

    for (int i = 0; i < numIndices; i++) {
        int classLabel = static_cast<int>(std::round(targets[indices[i]]));
        if (classLabel > numClasses) numClasses = classLabel;
        classCount[classLabel]++;
    }

    double gini = 1.0;
    for (int i = 0; i <= numClasses; i++) {
        double prob = static_cast<double>(classCount[i]) / numIndices;
        gini -= prob * prob;
    }
    return gini;
}

double TRandomForest::calculateEntropy(int* indices, int numIndices) {
    if (numIndices == 0) return 0.0;

    int classCount[100] = {0};
    int numClasses = 0;

    for (int i = 0; i < numIndices; i++) {
        int classLabel = static_cast<int>(std::round(targets[indices[i]]));
        if (classLabel > numClasses) numClasses = classLabel;
        classCount[classLabel]++;
    }

    double entropy = 0.0;
    for (int i = 0; i <= numClasses; i++) {
        if (classCount[i] > 0) {
            double prob = static_cast<double>(classCount[i]) / numIndices;
            entropy -= prob * std::log(prob) / std::log(2.0);
        }
    }
    return entropy;
}

double TRandomForest::calculateMSE(int* indices, int numIndices) {
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

double TRandomForest::calculateImpurity(int* indices, int numIndices) {
    switch (criterion) {
        case Gini: return calculateGini(indices, numIndices);
        case Entropy: return calculateEntropy(indices, numIndices);
        case MSE: return calculateMSE(indices, numIndices);
        case VarianceReduction: return calculateMSE(indices, numIndices);
        default: return calculateGini(indices, numIndices);
    }
}

bool TRandomForest::findBestSplit(int* indices, int numIndices, int* featureIndices, int nFeatures,
                                   int& bestFeature, double& bestThreshold, double& bestGain) {
    bestGain = 0.0;
    bestFeature = -1;
    bestThreshold = 0.0;

    if (numIndices < minSamplesSplit) return false;

    double parentImpurity = calculateImpurity(indices, numIndices);

    int* leftIndices = new int[MAX_SAMPLES];
    int* rightIndices = new int[MAX_SAMPLES];
    double* values = new double[MAX_SAMPLES];
    int* sortedIndices = new int[MAX_SAMPLES];

    for (int f = 0; f < nFeatures; f++) {
        int feat = featureIndices[f];

        for (int i = 0; i < numIndices; i++) {
            values[i] = data[indices[i] * MAX_FEATURES + feat];
            sortedIndices[i] = i;
        }

        for (int i = 0; i < numIndices - 1; i++) {
            for (int j = i + 1; j < numIndices; j++) {
                if (values[sortedIndices[j]] < values[sortedIndices[i]]) {
                    int temp = sortedIndices[i];
                    sortedIndices[i] = sortedIndices[j];
                    sortedIndices[j] = temp;
                }
            }
        }

        for (int i = 0; i < numIndices - 1; i++) {
            if (values[sortedIndices[i]] == values[sortedIndices[i + 1]]) continue;

            double threshold = (values[sortedIndices[i]] + values[sortedIndices[i + 1]]) / 2.0;

            int numLeft = 0, numRight = 0;
            for (int j = 0; j < numIndices; j++) {
                if (data[indices[j] * MAX_FEATURES + feat] <= threshold)
                    leftIndices[numLeft++] = indices[j];
                else
                    rightIndices[numRight++] = indices[j];
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

    delete[] leftIndices;
    delete[] rightIndices;
    delete[] values;
    delete[] sortedIndices;

    return bestFeature != -1;
}

int TRandomForest::getMajorityClass(int* indices, int numIndices) {
    int classCount[100] = {0};

    for (int i = 0; i < numIndices; i++) {
        int classLabel = static_cast<int>(std::round(targets[indices[i]]));
        classCount[classLabel]++;
    }

    int maxCount = 0, maxClass = 0;
    for (int i = 0; i < 100; i++) {
        if (classCount[i] > maxCount) {
            maxCount = classCount[i];
            maxClass = i;
        }
    }
    return maxClass;
}

double TRandomForest::getMeanTarget(int* indices, int numIndices) {
    if (numIndices == 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < numIndices; i++)
        sum += targets[indices[i]];
    return sum / numIndices;
}

TRandomForest::TreeNode* TRandomForest::createLeafNode(int* indices, int numIndices) {
    TreeNode* node = new TreeNode();
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
        node->classLabel = static_cast<int>(std::round(node->prediction));
    }
    return node;
}

bool TRandomForest::shouldStop(int depth, int numIndices, double impurity) {
    return depth >= maxDepth || numIndices < minSamplesSplit || 
           numIndices <= minSamplesLeaf || impurity < 1e-10;
}

TRandomForest::TreeNode* TRandomForest::buildTree(int* indices, int numIndices, int depth) {
    double currentImpurity = calculateImpurity(indices, numIndices);

    if (shouldStop(depth, numIndices, currentImpurity))
        return createLeafNode(indices, numIndices);

    int featureIndices[MAX_FEATURES];
    int numSelectedFeatures;
    selectFeatureSubset(featureIndices, numSelectedFeatures);

    int bestFeature;
    double bestThreshold, bestGain;
    if (!findBestSplit(indices, numIndices, featureIndices, numSelectedFeatures,
                       bestFeature, bestThreshold, bestGain))
        return createLeafNode(indices, numIndices);

    TreeNode* node = new TreeNode();
    node->isLeaf = false;
    node->featureIndex = bestFeature;
    node->threshold = bestThreshold;
    node->numSamples = numIndices;
    node->impurity = currentImpurity;

    if (taskType == Classification)
        node->classLabel = getMajorityClass(indices, numIndices);
    else
        node->prediction = getMeanTarget(indices, numIndices);

    int* leftIndices = new int[MAX_SAMPLES];
    int* rightIndices = new int[MAX_SAMPLES];
    int numLeft = 0, numRight = 0;

    for (int i = 0; i < numIndices; i++) {
        if (data[indices[i] * MAX_FEATURES + bestFeature] <= bestThreshold)
            leftIndices[numLeft++] = indices[i];
        else
            rightIndices[numRight++] = indices[i];
    }

    featureImportances[bestFeature] += numIndices * currentImpurity -
        numLeft * calculateImpurity(leftIndices, numLeft) -
        numRight * calculateImpurity(rightIndices, numRight);

    node->left = buildTree(leftIndices, numLeft, depth + 1);
    node->right = buildTree(rightIndices, numRight, depth + 1);

    delete[] leftIndices;
    delete[] rightIndices;

    return node;
}

void TRandomForest::flattenTree(TreeNode* node, FlatTree* flat, int& nodeIdx) {
    if (node == nullptr || nodeIdx >= MAX_NODES) return;

    int currentIdx = nodeIdx++;
    flat->nodes[currentIdx].isLeaf = node->isLeaf;
    flat->nodes[currentIdx].featureIndex = node->featureIndex;
    flat->nodes[currentIdx].threshold = node->threshold;
    flat->nodes[currentIdx].prediction = node->prediction;
    flat->nodes[currentIdx].classLabel = node->classLabel;

    if (node->isLeaf) {
        flat->nodes[currentIdx].leftChild = -1;
        flat->nodes[currentIdx].rightChild = -1;
    } else {
        flat->nodes[currentIdx].leftChild = nodeIdx;
        flattenTree(node->left, flat, nodeIdx);
        flat->nodes[currentIdx].rightChild = nodeIdx;
        flattenTree(node->right, flat, nodeIdx);
    }
}

void TRandomForest::freeTreeNode(TreeNode* node) {
    if (node == nullptr) return;
    freeTreeNode(node->left);
    freeTreeNode(node->right);
    delete node;
}

void TRandomForest::fit() {
    for (int i = 0; i < MAX_FEATURES; i++)
        featureImportances[i] = 0.0;

    for (int i = 0; i < numTrees; i++)
        fitTree(i);

    calculateFeatureImportance();
    initGPU();
}

void TRandomForest::fitTree(int treeIndex) {
    FlatTree* flat = new FlatTree();
    flat->numNodes = 0;

    int* sampleIndices = new int[MAX_SAMPLES];
    bool* oobMask = new bool[MAX_SAMPLES];
    int numBootstrap;
    bootstrap(sampleIndices, numBootstrap, oobMask);

    for (int i = 0; i < numSamples; i++)
        flat->oobIndices[i] = oobMask[i];

    flat->numOobIndices = 0;
    for (int i = 0; i < numSamples; i++)
        if (oobMask[i]) flat->numOobIndices++;

    TreeNode* root = buildTree(sampleIndices, numBootstrap, 0);
    
    int nodeIdx = 0;
    flattenTree(root, flat, nodeIdx);
    flat->numNodes = nodeIdx;

    freeTreeNode(root);
    delete[] sampleIndices;
    delete[] oobMask;

    trees[treeIndex] = flat;
}

void TRandomForest::initGPU() {
    if (gpuInitialized) freeGPU();

    totalGpuNodes = 0;
    for (int t = 0; t < numTrees; t++)
        totalGpuNodes += trees[t]->numNodes;

    FlatTreeNode* h_allNodes = new FlatTreeNode[totalGpuNodes];
    int* h_offsets = new int[numTrees];

    int offset = 0;
    for (int t = 0; t < numTrees; t++) {
        h_offsets[t] = offset;
        for (int n = 0; n < trees[t]->numNodes; n++)
            h_allNodes[offset + n] = trees[t]->nodes[n];
        offset += trees[t]->numNodes;
    }

    cudaMalloc(&d_data, numSamples * numFeatures * sizeof(double));
    cudaMalloc(&d_allTreeNodes, totalGpuNodes * sizeof(FlatTreeNode));
    cudaMalloc(&d_treeNodeOffsets, numTrees * sizeof(int));
    cudaMalloc(&d_predictions, numSamples * sizeof(double));

    double* h_dataFlat = new double[numSamples * numFeatures];
    for (int i = 0; i < numSamples; i++)
        for (int j = 0; j < numFeatures; j++)
            h_dataFlat[i * numFeatures + j] = data[i * MAX_FEATURES + j];

    cudaMemcpy(d_data, h_dataFlat, numSamples * numFeatures * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_allTreeNodes, h_allNodes, totalGpuNodes * sizeof(FlatTreeNode), cudaMemcpyHostToDevice);
    cudaMemcpy(d_treeNodeOffsets, h_offsets, numTrees * sizeof(int), cudaMemcpyHostToDevice);

    delete[] h_allNodes;
    delete[] h_offsets;
    delete[] h_dataFlat;

    gpuInitialized = true;
}

void TRandomForest::freeGPU() {
    if (!gpuInitialized) return;

    if (d_data) cudaFree(d_data);
    if (d_allTreeNodes) cudaFree(d_allTreeNodes);
    if (d_treeNodeOffsets) cudaFree(d_treeNodeOffsets);
    if (d_predictions) cudaFree(d_predictions);

    d_data = nullptr;
    d_allTreeNodes = nullptr;
    d_treeNodeOffsets = nullptr;
    d_predictions = nullptr;
    gpuInitialized = false;
}

double TRandomForest::predict(double* sample) {
    if (taskType == Regression) {
        double sum = 0.0;
        for (int t = 0; t < numTrees; t++) {
            int nodeIdx = 0;
            while (!trees[t]->nodes[nodeIdx].isLeaf) {
                if (sample[trees[t]->nodes[nodeIdx].featureIndex] <= trees[t]->nodes[nodeIdx].threshold)
                    nodeIdx = trees[t]->nodes[nodeIdx].leftChild;
                else
                    nodeIdx = trees[t]->nodes[nodeIdx].rightChild;
            }
            sum += trees[t]->nodes[nodeIdx].prediction;
        }
        return sum / numTrees;
    } else {
        int votes[100] = {0};
        for (int t = 0; t < numTrees; t++) {
            int nodeIdx = 0;
            while (!trees[t]->nodes[nodeIdx].isLeaf) {
                if (sample[trees[t]->nodes[nodeIdx].featureIndex] <= trees[t]->nodes[nodeIdx].threshold)
                    nodeIdx = trees[t]->nodes[nodeIdx].leftChild;
                else
                    nodeIdx = trees[t]->nodes[nodeIdx].rightChild;
            }
            int classLabel = trees[t]->nodes[nodeIdx].classLabel;
            if (classLabel >= 0 && classLabel < 100) votes[classLabel]++;
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

int TRandomForest::predictClass(double* sample) {
    return static_cast<int>(std::round(predict(sample)));
}

void TRandomForest::predictBatch(double* samples, int nSamples, double* predictions) {
    for (int i = 0; i < nSamples; i++)
        predictions[i] = predict(&samples[i * numFeatures]);
}

void TRandomForest::predictBatchGPU(double* samples, int nSamples, double* predictions) {
    if (!gpuInitialized) {
        predictBatch(samples, nSamples, predictions);
        return;
    }

    double* d_samples;
    double* d_preds;
    cudaMalloc(&d_samples, nSamples * numFeatures * sizeof(double));
    cudaMalloc(&d_preds, nSamples * sizeof(double));

    cudaMemcpy(d_samples, samples, nSamples * numFeatures * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (nSamples + blockSize - 1) / blockSize;

    predictBatchKernel<<<numBlocks, blockSize>>>(
        d_samples, numFeatures, d_allTreeNodes, d_treeNodeOffsets,
        numTrees, nSamples, taskType, d_preds
    );

    cudaMemcpy(predictions, d_preds, nSamples * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_samples);
    cudaFree(d_preds);
}

double TRandomForest::calculateOOBError() {
    double* predictions = new double[MAX_SAMPLES]();
    int* predCounts = new int[MAX_SAMPLES]();
    int (*votes)[100] = new int[MAX_SAMPLES][100]();

    for (int t = 0; t < numTrees; t++) {
        for (int i = 0; i < numSamples; i++) {
            if (trees[t]->oobIndices[i]) {
                double sample[MAX_FEATURES];
                for (int j = 0; j < numFeatures; j++)
                    sample[j] = data[i * MAX_FEATURES + j];

                double pred = predict(sample);
                if (taskType == Regression) {
                    predictions[i] += pred;
                } else {
                    int j = static_cast<int>(std::round(pred));
                    if (j >= 0 && j < 100) votes[i][j]++;
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
                error += diff * diff;
            } else {
                int maxVotes = 0, maxClass = 0;
                for (int j = 0; j < 100; j++) {
                    if (votes[i][j] > maxVotes) {
                        maxVotes = votes[i][j];
                        maxClass = j;
                    }
                }
                if (maxClass != static_cast<int>(std::round(targets[i])))
                    error += 1.0;
            }
            count++;
        }
    }

    delete[] predictions;
    delete[] predCounts;
    delete[] votes;

    return (count > 0) ? error / count : 0.0;
}

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
    if (featureIndex >= 0 && featureIndex < numFeatures)
        return featureImportances[featureIndex];
    return 0.0;
}

void TRandomForest::printFeatureImportances() {
    std::cout << "Feature Importances:" << std::endl;
    for (int i = 0; i < numFeatures; i++)
        std::cout << "  Feature " << i << ": " << std::fixed << std::setprecision(4) << featureImportances[i] << std::endl;
}

double TRandomForest::accuracy(double* predictions, double* actual, int nSamples) {
    int correct = 0;
    for (int i = 0; i < nSamples; i++) {
        if (static_cast<int>(std::round(predictions[i])) == static_cast<int>(std::round(actual[i])))
            correct++;
    }
    return static_cast<double>(correct) / nSamples;
}

double TRandomForest::precision(double* predictions, double* actual, int nSamples, int positiveClass) {
    int tp = 0, fp = 0;
    for (int i = 0; i < nSamples; i++) {
        if (static_cast<int>(std::round(predictions[i])) == positiveClass) {
            if (static_cast<int>(std::round(actual[i])) == positiveClass) tp++;
            else fp++;
        }
    }
    return (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
}

double TRandomForest::recall(double* predictions, double* actual, int nSamples, int positiveClass) {
    int tp = 0, fn = 0;
    for (int i = 0; i < nSamples; i++) {
        if (static_cast<int>(std::round(actual[i])) == positiveClass) {
            if (static_cast<int>(std::round(predictions[i])) == positiveClass) tp++;
            else fn++;
        }
    }
    return (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
}

double TRandomForest::f1Score(double* predictions, double* actual, int nSamples, int positiveClass) {
    double p = precision(predictions, actual, nSamples, positiveClass);
    double r = recall(predictions, actual, nSamples, positiveClass);
    return (p + r > 0) ? 2 * p * r / (p + r) : 0.0;
}

double TRandomForest::meanSquaredError(double* predictions, double* actual, int nSamples) {
    double mse = 0.0;
    for (int i = 0; i < nSamples; i++) {
        double diff = predictions[i] - actual[i];
        mse += diff * diff;
    }
    return mse / nSamples;
}

double TRandomForest::rSquared(double* predictions, double* actual, int nSamples) {
    double mean = 0.0;
    for (int i = 0; i < nSamples; i++)
        mean += actual[i];
    mean /= nSamples;

    double ssRes = 0.0, ssTot = 0.0;
    for (int i = 0; i < nSamples; i++) {
        double diff = predictions[i] - actual[i];
        ssRes += diff * diff;
        diff = actual[i] - mean;
        ssTot += diff * diff;
    }
    return (ssTot > 0) ? 1.0 - (ssRes / ssTot) : 0.0;
}

void TRandomForest::printForestInfo() {
    std::cout << "Random Forest Configuration (CUDA):" << std::endl;
    std::cout << "  Number of Trees: " << numTrees << std::endl;
    std::cout << "  Max Depth: " << maxDepth << std::endl;
    std::cout << "  Min Samples Leaf: " << minSamplesLeaf << std::endl;
    std::cout << "  Min Samples Split: " << minSamplesSplit << std::endl;
    std::cout << "  Max Features: " << maxFeatures << std::endl;
    std::cout << "  Number of Features: " << numFeatures << std::endl;
    std::cout << "  Number of Samples: " << numSamples << std::endl;
    std::cout << "  Task Type: " << (taskType == Classification ? "Classification" : "Regression") << std::endl;
    switch (criterion) {
        case Gini: std::cout << "  Criterion: Gini" << std::endl; break;
        case Entropy: std::cout << "  Criterion: Entropy" << std::endl; break;
        case MSE: std::cout << "  Criterion: MSE" << std::endl; break;
        case VarianceReduction: std::cout << "  Criterion: Variance Reduction" << std::endl; break;
    }
    std::cout << "  GPU Initialized: " << (gpuInitialized ? "Yes" : "No") << std::endl;
    if (gpuInitialized)
        std::cout << "  Total GPU Nodes: " << totalGpuNodes << std::endl;
}

void TRandomForest::freeForest() {
    for (int i = 0; i < numTrees; i++) {
        if (trees[i] != nullptr) {
            delete trees[i];
            trees[i] = nullptr;
        }
    }
}

bool TRandomForest::loadCSV(const char* filename, int targetColumn, bool hasHeader) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    std::vector<std::vector<double>> rows;
    std::string line;
    int lineNum = 0;
    int numCols = 0;

    while (std::getline(file, line)) {
        lineNum++;
        if (hasHeader && lineNum == 1) continue;
        if (line.empty()) continue;

        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stod(cell));
            } catch (...) {
                row.push_back(0.0);
            }
        }

        if (numCols == 0) numCols = row.size();
        if ((int)row.size() == numCols)
            rows.push_back(row);
    }
    file.close();

    if (rows.empty()) {
        std::cerr << "Error: No data loaded from " << filename << std::endl;
        return false;
    }

    int nSamples = std::min((int)rows.size(), MAX_SAMPLES);
    int nFeatures = numCols - 1;
    if (nFeatures > MAX_FEATURES) nFeatures = MAX_FEATURES;

    if (targetColumn < 0) targetColumn = numCols - 1;

    numSamples = nSamples;
    numFeatures = nFeatures;

    if (maxFeatures == 0) {
        if (taskType == Classification)
            maxFeatures = static_cast<int>(std::round(std::sqrt(nFeatures)));
        else
            maxFeatures = nFeatures / 3;
        if (maxFeatures < 1) maxFeatures = 1;
    }

    for (int i = 0; i < nSamples; i++) {
        int featIdx = 0;
        for (int j = 0; j < numCols && featIdx < nFeatures; j++) {
            if (j == targetColumn) {
                targets[i] = rows[i][j];
            } else {
                data[i * MAX_FEATURES + featIdx] = rows[i][j];
                featIdx++;
            }
        }
        if (targetColumn >= numCols) targets[i] = 0;
    }

    std::cout << "Loaded " << nSamples << " samples with " << nFeatures << " features from " << filename << std::endl;
    return true;
}

bool TRandomForest::saveModel(const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return false;
    }

    const char magic[] = "RFCU";
    file.write(magic, 4);

    int version = 1;
    file.write(reinterpret_cast<char*>(&version), sizeof(int));
    file.write(reinterpret_cast<char*>(&numTrees), sizeof(int));
    file.write(reinterpret_cast<char*>(&maxDepth), sizeof(int));
    file.write(reinterpret_cast<char*>(&minSamplesLeaf), sizeof(int));
    file.write(reinterpret_cast<char*>(&minSamplesSplit), sizeof(int));
    file.write(reinterpret_cast<char*>(&maxFeatures), sizeof(int));
    file.write(reinterpret_cast<char*>(&numFeatures), sizeof(int));
    file.write(reinterpret_cast<char*>(&numSamples), sizeof(int));
    file.write(reinterpret_cast<char*>(&taskType), sizeof(TaskType));
    file.write(reinterpret_cast<char*>(&criterion), sizeof(SplitCriterion));

    file.write(reinterpret_cast<char*>(featureImportances), sizeof(double) * MAX_FEATURES);

    for (int t = 0; t < numTrees; t++) {
        file.write(reinterpret_cast<char*>(&trees[t]->numNodes), sizeof(int));
        file.write(reinterpret_cast<char*>(&trees[t]->numOobIndices), sizeof(int));
        file.write(reinterpret_cast<char*>(trees[t]->nodes), sizeof(FlatTreeNode) * trees[t]->numNodes);
        file.write(reinterpret_cast<char*>(trees[t]->oobIndices), sizeof(bool) * MAX_SAMPLES);
    }

    file.close();
    std::cout << "Model saved to " << filename << std::endl;
    return true;
}

bool TRandomForest::loadModel(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for reading" << std::endl;
        return false;
    }

    char magic[4];
    file.read(magic, 4);
    if (std::strncmp(magic, "RFCU", 4) != 0) {
        std::cerr << "Error: Invalid model file format" << std::endl;
        file.close();
        return false;
    }

    freeForest();
    freeGPU();

    int version;
    file.read(reinterpret_cast<char*>(&version), sizeof(int));
    file.read(reinterpret_cast<char*>(&numTrees), sizeof(int));
    file.read(reinterpret_cast<char*>(&maxDepth), sizeof(int));
    file.read(reinterpret_cast<char*>(&minSamplesLeaf), sizeof(int));
    file.read(reinterpret_cast<char*>(&minSamplesSplit), sizeof(int));
    file.read(reinterpret_cast<char*>(&maxFeatures), sizeof(int));
    file.read(reinterpret_cast<char*>(&numFeatures), sizeof(int));
    file.read(reinterpret_cast<char*>(&numSamples), sizeof(int));
    file.read(reinterpret_cast<char*>(&taskType), sizeof(TaskType));
    file.read(reinterpret_cast<char*>(&criterion), sizeof(SplitCriterion));

    file.read(reinterpret_cast<char*>(featureImportances), sizeof(double) * MAX_FEATURES);

    for (int t = 0; t < numTrees; t++) {
        trees[t] = new FlatTree();
        file.read(reinterpret_cast<char*>(&trees[t]->numNodes), sizeof(int));
        file.read(reinterpret_cast<char*>(&trees[t]->numOobIndices), sizeof(int));
        file.read(reinterpret_cast<char*>(trees[t]->nodes), sizeof(FlatTreeNode) * trees[t]->numNodes);
        file.read(reinterpret_cast<char*>(trees[t]->oobIndices), sizeof(bool) * MAX_SAMPLES);
    }

    file.close();
    std::cout << "Model loaded from " << filename << std::endl;

    initGPU();
    return true;
}

bool TRandomForest::predictCSV(const char* inputFile, const char* outputFile, bool hasHeader) {
    std::ifstream inFile(inputFile);
    if (!inFile.is_open()) {
        std::cerr << "Error: Cannot open file " << inputFile << std::endl;
        return false;
    }

    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "Error: Cannot open file " << outputFile << " for writing" << std::endl;
        inFile.close();
        return false;
    }

    std::string line;
    int lineNum = 0;
    std::string headerLine;

    while (std::getline(inFile, line)) {
        lineNum++;
        if (hasHeader && lineNum == 1) {
            headerLine = line;
            outFile << line << ",prediction" << std::endl;
            continue;
        }
        if (line.empty()) continue;

        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stod(cell));
            } catch (...) {
                row.push_back(0.0);
            }
        }

        double sample[MAX_FEATURES];
        for (int j = 0; j < numFeatures && j < (int)row.size(); j++)
            sample[j] = row[j];

        double pred = predict(sample);
        outFile << line << "," << std::fixed << std::setprecision(4) << pred << std::endl;
    }

    inFile.close();
    outFile.close();
    std::cout << "Predictions saved to " << outputFile << std::endl;
    return true;
}

void printUsage(const char* progName) {
    std::cout << "Random Forest CUDA - Usage:" << std::endl;
    std::cout << std::endl;
    std::cout << "Training:" << std::endl;
    std::cout << "  " << progName << " train <data.csv> <model.bin> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Prediction:" << std::endl;
    std::cout << "  " << progName << " predict <model.bin> <input.csv> <output.csv> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --trees N          Number of trees (default: 100)" << std::endl;
    std::cout << "  --depth N          Max depth (default: 10)" << std::endl;
    std::cout << "  --min-leaf N       Min samples per leaf (default: 1)" << std::endl;
    std::cout << "  --min-split N      Min samples to split (default: 2)" << std::endl;
    std::cout << "  --max-features N   Max features per split (default: sqrt(n))" << std::endl;
    std::cout << "  --target N         Target column index (default: last column)" << std::endl;
    std::cout << "  --regression       Use regression instead of classification" << std::endl;
    std::cout << "  --no-header        CSV has no header row" << std::endl;
    std::cout << "  --seed N           Random seed" << std::endl;
    std::cout << "  --gpu              Use GPU for batch prediction" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << progName << " train data.csv model.bin --trees 50 --depth 8" << std::endl;
    std::cout << "  " << progName << " predict model.bin test.csv results.csv --gpu" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    int numTreesArg = 100;
    int maxDepthArg = 10;
    int minLeafArg = 1;
    int minSplitArg = 2;
    int maxFeaturesArg = 0;
    int targetColumn = -1;
    bool regression = false;
    bool hasHeader = true;
    long seed = 42;
    bool useGPU = false;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--trees" && i + 1 < argc) numTreesArg = std::atoi(argv[++i]);
        else if (arg == "--depth" && i + 1 < argc) maxDepthArg = std::atoi(argv[++i]);
        else if (arg == "--min-leaf" && i + 1 < argc) minLeafArg = std::atoi(argv[++i]);
        else if (arg == "--min-split" && i + 1 < argc) minSplitArg = std::atoi(argv[++i]);
        else if (arg == "--max-features" && i + 1 < argc) maxFeaturesArg = std::atoi(argv[++i]);
        else if (arg == "--target" && i + 1 < argc) targetColumn = std::atoi(argv[++i]);
        else if (arg == "--regression") regression = true;
        else if (arg == "--no-header") hasHeader = false;
        else if (arg == "--seed" && i + 1 < argc) seed = std::atol(argv[++i]);
        else if (arg == "--gpu") useGPU = true;
        else if (arg == "--help" || arg == "-h") { printUsage(argv[0]); return 0; }
    }

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "CUDA Device: " << prop.name << std::endl;
    } else {
        std::cout << "No CUDA devices found, using CPU" << std::endl;
        useGPU = false;
    }
    std::cout << std::endl;

    TRandomForest* rf = new TRandomForest();

    if (command == "train") {
        if (argc < 4) {
            std::cerr << "Error: train requires <data.csv> and <model.bin>" << std::endl;
            printUsage(argv[0]);
            delete rf;
            return 1;
        }

        const char* dataFile = argv[2];
        const char* modelFile = argv[3];

        rf->setNumTrees(numTreesArg);
        rf->setMaxDepth(maxDepthArg);
        rf->setMinSamplesLeaf(minLeafArg);
        rf->setMinSamplesSplit(minSplitArg);
        rf->setMaxFeatures(maxFeaturesArg);
        rf->setTaskType(regression ? Regression : Classification);
        rf->setRandomSeed(seed);

        if (!rf->loadCSV(dataFile, targetColumn, hasHeader)) {
            delete rf;
            return 1;
        }

        rf->printForestInfo();
        std::cout << std::endl;

        std::cout << "Training forest..." << std::endl;
        rf->fit();
        std::cout << "Training complete." << std::endl;
        std::cout << std::endl;

        std::cout << "OOB Error: " << std::fixed << std::setprecision(4) << rf->calculateOOBError() << std::endl;
        std::cout << std::endl;

        rf->printFeatureImportances();
        std::cout << std::endl;

        rf->saveModel(modelFile);

    } else if (command == "predict") {
        if (argc < 5) {
            std::cerr << "Error: predict requires <model.bin> <input.csv> <output.csv>" << std::endl;
            printUsage(argv[0]);
            delete rf;
            return 1;
        }

        const char* modelFile = argv[2];
        const char* inputFile = argv[3];
        const char* outputFile = argv[4];

        if (!rf->loadModel(modelFile)) {
            delete rf;
            return 1;
        }

        rf->printForestInfo();
        std::cout << std::endl;

        if (useGPU) {
            std::cout << "Using GPU for prediction..." << std::endl;
        }

        rf->predictCSV(inputFile, outputFile, hasHeader);

    } else if (command == "demo") {
        std::cout << "Running demo..." << std::endl;
        std::cout << std::endl;

        rf->setNumTrees(10);
        rf->setMaxDepth(5);
        rf->setTaskType(Classification);

        const int nSamples = 100;
        const int nFeatures = 4;
        double* testData = new double[nSamples * nFeatures];
        double* testTargets = new double[nSamples];

        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++)
                testData[i * nFeatures + j] = static_cast<double>(std::rand()) / RAND_MAX * 10.0;
            if (testData[i * nFeatures + 0] + testData[i * nFeatures + 1] > 10)
                testTargets[i] = 1;
            else
                testTargets[i] = 0;
        }

        rf->loadData(testData, testTargets, nSamples, nFeatures);
        rf->printForestInfo();
        std::cout << std::endl;

        std::cout << "Training forest..." << std::endl;
        rf->fit();
        std::cout << "Training complete." << std::endl;
        std::cout << std::endl;

        double* predictions = new double[nSamples];

        std::cout << "CPU Prediction:" << std::endl;
        rf->predictBatch(testData, nSamples, predictions);
        double accCPU = rf->accuracy(predictions, testTargets, nSamples);
        std::cout << "  Training accuracy: " << std::fixed << std::setprecision(4) << accCPU << std::endl;

        if (useGPU) {
            std::cout << std::endl;
            std::cout << "GPU Prediction:" << std::endl;
            rf->predictBatchGPU(testData, nSamples, predictions);
            double accGPU = rf->accuracy(predictions, testTargets, nSamples);
            std::cout << "  Training accuracy: " << std::fixed << std::setprecision(4) << accGPU << std::endl;
        }

        std::cout << std::endl;
        std::cout << "OOB Error: " << std::fixed << std::setprecision(4) << rf->calculateOOBError() << std::endl;
        std::cout << std::endl;

        rf->printFeatureImportances();

        delete[] predictions;
        delete[] testData;
        delete[] testTargets;

    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        printUsage(argv[0]);
        delete rf;
        return 1;
    }

    delete rf;
    std::cout << std::endl;
    std::cout << "Done." << std::endl;

    return 0;
}
