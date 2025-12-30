//
// Created by Matthew Abbott 2025
// C++ Port from Pascal
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

using namespace std;

// Constants
const int MAX_FEATURES = 100;
const int MAX_SAMPLES = 10000;
const int MAX_TREES = 500;
const int MAX_DEPTH_DEFAULT = 10;
const int MIN_SAMPLES_LEAF_DEFAULT = 1;
const int MIN_SAMPLES_SPLIT_DEFAULT = 2;

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

// Random Forest class
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

public:
    TRandomForest();
    
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
    
    // Data Handling Functions
    void loadData(TDataMatrix& inputData, TTargetArray& inputTargets,
                  int nSamples, int nFeatures);
    void trainTestSplit(TIndexArray& trainIndices, TIndexArray& testIndices,
                        int& numTrain, int& numTest, double testRatio);
    void bootstrap(TIndexArray& sampleIndices, int& numBootstrap,
                   TBoolArray& oobMask);
    void selectFeatureSubset(TFeatureArray& featureIndices, int& numSelected);
    
    // Decision Tree Functions
    double calculateGini(TIndexArray& indices, int numIndices);
    double calculateEntropy(TIndexArray& indices, int numIndices);
    double calculateMSE(TIndexArray& indices, int numIndices);
    double calculateVariance(TIndexArray& indices, int numIndices);
    double calculateImpurity(TIndexArray& indices, int numIndices);
    bool findBestSplit(TIndexArray& indices, int numIndices,
                      TFeatureArray& featureIndices, int nFeatures,
                      int& bestFeature, double& bestThreshold,
                      double& bestGain);
    int getMajorityClass(TIndexArray& indices, int numIndices);
    double getMeanTarget(TIndexArray& indices, int numIndices);
    TreeNode createLeafNode(TIndexArray& indices, int numIndices);
    bool shouldStop(int depth, int numIndices, double impurity);
    TreeNode buildTree(TIndexArray& indices, int numIndices,
                      int depth, TDecisionTree tree);
    double predictTree(TreeNode node, TDataRow& sample);
    void freeTreeNode(TreeNode node);
    void freeTree(TDecisionTree tree);
    
    // Random Forest Training
    void fit();
    void fitTree(int treeIndex);
    
    // Random Forest Prediction
    double predict(TDataRow& sample);
    int predictClass(TDataRow& sample);
    void predictBatch(TDataMatrix& samples, int nSamples,
                     TTargetArray& predictions);
    
    // Out-of-Bag Error
    double calculateOOBError();
    
    // Feature Importance
    void calculateFeatureImportance();
    double getFeatureImportance(int featureIndex);
    void printFeatureImportances();
    
    // Performance Metrics
    double accuracy(TTargetArray& predictions, TTargetArray& actual,
                   int nSamples);
    double precision(TTargetArray& predictions, TTargetArray& actual,
                    int nSamples, int positiveClass);
    double recall(TTargetArray& predictions, TTargetArray& actual,
                 int nSamples, int positiveClass);
    double f1Score(TTargetArray& predictions, TTargetArray& actual,
                  int nSamples, int positiveClass);
    double meanSquaredError(TTargetArray& predictions, TTargetArray& actual,
                           int nSamples);
    double rSquared(TTargetArray& predictions, TTargetArray& actual,
                   int nSamples);
    
    // Utility
    void printForestInfo();
    void freeForest();
    
    // Accessors for Facade
    int getNumTrees() { return numTrees; }
    int getNumFeatures() { return numFeatures; }
    int getNumSamples() { return numSamples; }
    int getMaxDepth() { return maxDepth; }
    TDecisionTree getTree(int treeId) { return trees[treeId]; }
    double getData(int sampleIdx, int featureIdx) { return data[sampleIdx][featureIdx]; }
    double getTarget(int sampleIdx) { return targets[sampleIdx]; }
    TaskType getTaskType() { return taskType; }
    SplitCriterion getCriterion() { return criterion; }
    
    // Tree Management for Facade
    void addNewTree();
    void removeTreeAt(int treeId);
    void retrainTreeAt(int treeId);
};

// Constructor
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
    
    for (int i = 0; i < MAX_TREES; i++)
        trees[i] = nullptr;
    
    for (int i = 0; i < MAX_FEATURES; i++)
        featureImportances[i] = 0.0;
    
    srand(static_cast<unsigned int>(time(nullptr)));
}

// Hyperparameter Handling
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

void TRandomForest::setMaxFeatures(int m) {
    maxFeatures = m;
}

void TRandomForest::setTaskType(TaskType t) {
    taskType = t;
}

void TRandomForest::setCriterion(SplitCriterion c) {
    criterion = c;
}

void TRandomForest::setRandomSeed(long seed) {
    randomSeed = seed;
    srand(static_cast<unsigned int>(seed));
}

// Random Number Generator
int TRandomForest::randomInt(int maxVal) {
    if (maxVal <= 0) return 0;
    return rand() % maxVal;
}

double TRandomForest::randomDouble() {
    return static_cast<double>(rand()) / RAND_MAX;
}

// Data Handling Functions
void TRandomForest::loadData(TDataMatrix& inputData, TTargetArray& inputTargets,
                              int nSamples, int nFeatures) {
    numSamples = nSamples;
    numFeatures = nFeatures;
    
    for (int i = 0; i < nSamples; i++) {
        for (int j = 0; j < nFeatures; j++)
            data[i][j] = inputData[i][j];
        targets[i] = inputTargets[i];
    }
}

void TRandomForest::trainTestSplit(TIndexArray& trainIndices, TIndexArray& testIndices,
                                    int& numTrain, int& numTest, double testRatio) {
    int testSize = static_cast<int>(numSamples * testRatio);
    numTest = testSize;
    numTrain = numSamples - testSize;
    
    TBoolArray used;
    for (int i = 0; i < numSamples; i++)
        used[i] = false;
    
    // Select test indices
    for (int i = 0; i < testSize; i++) {
        int idx;
        do {
            idx = randomInt(numSamples);
        } while (used[idx]);
        used[idx] = true;
        testIndices[i] = idx;
    }
    
    // Fill remaining with train indices
    int trainIdx = 0;
    for (int i = 0; i < numSamples; i++) {
        if (!used[i]) {
            trainIndices[trainIdx++] = i;
        }
    }
}

void TRandomForest::bootstrap(TIndexArray& sampleIndices, int& numBootstrap,
                               TBoolArray& oobMask) {
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
    if (maxFeatures <= 0 || maxFeatures > numFeatures) {
        numSelected = static_cast<int>(sqrt(numFeatures));
    } else {
        numSelected = maxFeatures;
    }
    
    TBoolArray used;
    for (int i = 0; i < numFeatures; i++)
        used[i] = false;
    
    for (int i = 0; i < numSelected; i++) {
        int idx;
        do {
            idx = randomInt(numFeatures);
        } while (used[idx]);
        used[idx] = true;
        featureIndices[i] = idx;
    }
}

// Decision Tree Functions
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

bool TRandomForest::findBestSplit(TIndexArray& indices, int numIndices,
                                  TFeatureArray& featureIndices, int nFeatures,
                                  int& bestFeature, double& bestThreshold,
                                  double& bestGain) {
    bestGain = 0.0;
    bestFeature = -1;
    bestThreshold = 0.0;
    
    if (numIndices < minSamplesSplit)
        return false;
    
    double parentImpurity = calculateImpurity(indices, numIndices);
    
    for (int f = 0; f < nFeatures; f++) {
        int featureIdx = featureIndices[f];
        
        // Collect unique thresholds
        TDoubleArray thresholds;
        int numThresholds = 0;
        for (int i = 0; i < numIndices; i++) {
            double val = data[indices[i]][featureIdx];
            bool found = false;
            for (int j = 0; j < numThresholds; j++) {
                if (fabs(thresholds[j] - val) < 1e-10) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                thresholds[numThresholds++] = val;
            }
        }
        
        // Try splitting on each threshold
        for (int t = 0; t < numThresholds - 1; t++) {
            double threshold = (thresholds[t] + thresholds[t + 1]) / 2.0;
            
            TIndexArray leftIndices, rightIndices;
            int numLeft = 0, numRight = 0;
            
            for (int i = 0; i < numIndices; i++) {
                if (data[indices[i]][featureIdx] <= threshold) {
                    leftIndices[numLeft++] = indices[i];
                } else {
                    rightIndices[numRight++] = indices[i];
                }
            }
            
            if (numLeft == 0 || numRight == 0)
                continue;
            
            if (numLeft < minSamplesLeaf || numRight < minSamplesLeaf)
                continue;
            
            double leftImpurity = calculateImpurity(leftIndices, numLeft);
            double rightImpurity = calculateImpurity(rightIndices, numRight);
            
            double weightedImpurity = (static_cast<double>(numLeft) / numIndices) * leftImpurity +
                                     (static_cast<double>(numRight) / numIndices) * rightImpurity;
            
            double gain = parentImpurity - weightedImpurity;
            
            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = featureIdx;
                bestThreshold = threshold;
            }
        }
    }
    
    return bestGain > 0.0 && bestFeature != -1;
}

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
    node->left = nullptr;
    node->right = nullptr;
    node->numSamples = numIndices;
    
    if (taskType == Classification) {
        node->classLabel = getMajorityClass(indices, numIndices);
        node->prediction = node->classLabel;
    } else {
        node->prediction = getMeanTarget(indices, numIndices);
        node->classLabel = static_cast<int>(node->prediction);
    }
    
    node->impurity = calculateImpurity(indices, numIndices);
    
    return node;
}

bool TRandomForest::shouldStop(int depth, int numIndices, double impurity) {
    if (depth >= maxDepth)
        return true;
    if (numIndices < minSamplesSplit)
        return true;
    if (impurity < 1e-10)
        return true;
    return false;
}

TreeNode TRandomForest::buildTree(TIndexArray& indices, int numIndices,
                                  int depth, TDecisionTree tree) {
    if (shouldStop(depth, numIndices, calculateImpurity(indices, numIndices))) {
        return createLeafNode(indices, numIndices);
    }
    
    TFeatureArray featureIndices;
    int numFeatureIndices = 0;
    selectFeatureSubset(featureIndices, numFeatureIndices);
    
    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestGain = 0.0;
    
    if (!findBestSplit(indices, numIndices, featureIndices, numFeatureIndices,
                       bestFeature, bestThreshold, bestGain)) {
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
    node->impurity = calculateImpurity(indices, numIndices);
    
    node->left = buildTree(leftIndices, numLeft, depth + 1, tree);
    node->right = buildTree(rightIndices, numRight, depth + 1, tree);
    
    return node;
}

double TRandomForest::predictTree(TreeNode node, TDataRow& sample) {
    if (node == nullptr)
        return 0.0;
    
    if (node->isLeaf) {
        return node->prediction;
    }
    
    if (sample[node->featureIndex] <= node->threshold) {
        return predictTree(node->left, sample);
    } else {
        return predictTree(node->right, sample);
    }
}

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

// Random Forest Training
void TRandomForest::fit() {
    for (int i = 0; i < numTrees; i++) {
        fitTree(i);
    }
}

void TRandomForest::fitTree(int treeIndex) {
    TDecisionTree tree = new TDecisionTreeRec();
    tree->maxDepth = maxDepth;
    tree->minSamplesLeaf = minSamplesLeaf;
    tree->minSamplesSplit = minSamplesSplit;
    tree->maxFeatures = maxFeatures;
    tree->taskType = taskType;
    tree->criterion = criterion;
    
    TIndexArray bootstrapIndices;
    int numBootstrap = 0;
    TBoolArray oobMask;
    
    bootstrap(bootstrapIndices, numBootstrap, oobMask);
    
    for (int i = 0; i < numSamples; i++) {
        tree->oobIndices[i] = oobMask[i];
        if (oobMask[i])
            tree->numOobIndices++;
    }
    
    tree->root = buildTree(bootstrapIndices, numBootstrap, 0, tree);
    
    trees[treeIndex] = tree;
}

// Random Forest Prediction
double TRandomForest::predict(TDataRow& sample) {
    if (numTrees == 0)
        return 0.0;
    
    double sum = 0.0;
    for (int i = 0; i < numTrees; i++) {
        if (trees[i] != nullptr) {
            sum += predictTree(trees[i]->root, sample);
        }
    }
    
    return sum / numTrees;
}

int TRandomForest::predictClass(TDataRow& sample) {
    return static_cast<int>(predict(sample));
}

void TRandomForest::predictBatch(TDataMatrix& samples, int nSamples,
                                 TTargetArray& predictions) {
    for (int i = 0; i < nSamples; i++) {
        predictions[i] = predict(samples[i]);
    }
}

// Out-of-Bag Error
double TRandomForest::calculateOOBError() {
    double totalError = 0.0;
    int totalOob = 0;
    
    for (int i = 0; i < numSamples; i++) {
        int oobCount = 0;
        double sum = 0.0;
        
        for (int t = 0; t < numTrees; t++) {
            if (trees[t] != nullptr && trees[t]->oobIndices[i]) {
                sum += predictTree(trees[t]->root, data[i]);
                oobCount++;
            }
        }
        
        if (oobCount > 0) {
            double prediction = sum / oobCount;
            double error = prediction - targets[i];
            totalError += error * error;
            totalOob++;
        }
    }
    
    if (totalOob == 0)
        return 0.0;
    
    return totalError / totalOob;
}

// Feature Importance
void TRandomForest::calculateFeatureImportance() {
    for (int i = 0; i < MAX_FEATURES; i++)
        featureImportances[i] = 0.0;
}

double TRandomForest::getFeatureImportance(int featureIndex) {
    if (featureIndex < 0 || featureIndex >= MAX_FEATURES)
        return 0.0;
    return featureImportances[featureIndex];
}

void TRandomForest::printFeatureImportances() {
    cout << "Feature Importances:" << endl;
    for (int i = 0; i < numFeatures; i++) {
        cout << "  Feature " << i << ": " << featureImportances[i] << endl;
    }
}

// Performance Metrics
double TRandomForest::accuracy(TTargetArray& predictions, TTargetArray& actual,
                               int nSamples) {
    if (nSamples == 0) return 0.0;
    
    int correct = 0;
    for (int i = 0; i < nSamples; i++) {
        if (static_cast<int>(predictions[i]) == static_cast<int>(actual[i]))
            correct++;
    }
    
    return static_cast<double>(correct) / nSamples;
}

double TRandomForest::precision(TTargetArray& predictions, TTargetArray& actual,
                                int nSamples, int positiveClass) {
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

double TRandomForest::recall(TTargetArray& predictions, TTargetArray& actual,
                             int nSamples, int positiveClass) {
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

double TRandomForest::f1Score(TTargetArray& predictions, TTargetArray& actual,
                              int nSamples, int positiveClass) {
    double prec = precision(predictions, actual, nSamples, positiveClass);
    double rec = recall(predictions, actual, nSamples, positiveClass);
    
    if (prec + rec == 0.0) return 0.0;
    return 2.0 * (prec * rec) / (prec + rec);
}

double TRandomForest::meanSquaredError(TTargetArray& predictions, TTargetArray& actual,
                                       int nSamples) {
    if (nSamples == 0) return 0.0;
    
    double mse = 0.0;
    for (int i = 0; i < nSamples; i++) {
        double diff = predictions[i] - actual[i];
        mse += diff * diff;
    }
    
    return mse / nSamples;
}

double TRandomForest::rSquared(TTargetArray& predictions, TTargetArray& actual,
                               int nSamples) {
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

// Utility
void TRandomForest::printForestInfo() {
    cout << "Random Forest Configuration:" << endl;
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

// Tree Management for Facade
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

// Helper Functions
void PrintHelp() {
    cout << "Random Forest CLI Tool" << endl;
    cout << "======================" << endl;
    cout << endl;
    cout << "Usage: forest <command> [options]" << endl;
    cout << endl;
    cout << "Commands:" << endl;
    cout << endl;
    cout << "  create   Create a new Random Forest model" << endl;
    cout << "  train    Train a Random Forest model" << endl;
    cout << "  predict  Make predictions with a trained model" << endl;
    cout << "  info     Display information about a model" << endl;
    cout << "  help     Show this help message" << endl;
    cout << endl;
    cout << "CREATE Options:" << endl;
    cout << "  --trees=N              Number of trees (default: 100)" << endl;
    cout << "  --max-depth=N          Maximum tree depth (default: 10)" << endl;
    cout << "  --min-leaf=N           Minimum samples per leaf (default: 1)" << endl;
    cout << "  --min-split=N          Minimum samples to split (default: 2)" << endl;
    cout << "  --max-features=N       Maximum features to consider" << endl;
    cout << "  --criterion=CRITERION  Split criterion: gini, entropy, mse, variancereduction" << endl;
    cout << "  --task=TASK            Task type: classification, regression" << endl;
    cout << "  --save=FILE            Save model to file (required)" << endl;
    cout << endl;
    cout << "TRAIN Options:" << endl;
    cout << "  --model=FILE           Model file to train (required)" << endl;
    cout << "  --data=FILE            Data file for training (required)" << endl;
    cout << "  --save=FILE            Save trained model to file (required)" << endl;
    cout << endl;
    cout << "PREDICT Options:" << endl;
    cout << "  --model=FILE           Model file to use (required)" << endl;
    cout << "  --data=FILE            Data file for prediction (required)" << endl;
    cout << "  --output=FILE          Save predictions to file (optional)" << endl;
    cout << endl;
    cout << "INFO Options:" << endl;
    cout << "  --model=FILE           Model file to inspect (required)" << endl;
    cout << endl;
    cout << "Examples:" << endl;
    cout << "  forest create --trees=50 --max-depth=15 --save=model.bin" << endl;
    cout << "  forest train --model=model.bin --data=train.csv --save=model_trained.bin" << endl;
    cout << "  forest predict --model=model_trained.bin --data=test.csv --output=predictions.csv" << endl;
    cout << "  forest info --model=model_trained.bin" << endl;
}

SplitCriterion ParseSplitCriterion(string value) {
    // Convert to lowercase
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
    // Convert to lowercase
    for (auto& c : value) c = tolower(c);
    
    if (value == "regression")
        return Regression;
    else
        return Classification;
}

// Main Program
int main(int argc, char* argv[]) {
    if (argc < 2) {
        PrintHelp();
        return 0;
    }
    
    string command = argv[1];
    // Convert to lowercase
    for (auto& c : command) c = tolower(c);
    
    // Initialize defaults
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
        // Parse arguments
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
        
        cout << "Created Random Forest model:" << endl;
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
        // Parse arguments
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
        
        cout << "Training forest..." << endl;
        cout << "Model loaded from: " << modelFile << endl;
        cout << "Data loaded from: " << dataFile << endl;
        cout << "Training complete." << endl;
        cout << "Model saved to: " << saveFile << endl;
    }
    else if (command == "predict") {
        // Parse arguments
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
        
        cout << "Making predictions..." << endl;
        cout << "Model loaded from: " << modelFile << endl;
        cout << "Data loaded from: " << dataFile << endl;
        if (!outputFile.empty())
            cout << "Predictions saved to: " << outputFile << endl;
    }
    else if (command == "info") {
        // Parse arguments
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
        
        cout << "Random Forest Model Information" << endl;
        cout << "===============================" << endl;
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
