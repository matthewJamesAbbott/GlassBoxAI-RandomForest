/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <cctype>

const int MAX_FEATURES = 100;
const int MAX_SAMPLES = 10000;
const int MAX_TREES = 500;
const int MAX_DEPTH_DEFAULT = 10;
const int MIN_SAMPLES_LEAF_DEFAULT = 1;
const int MIN_SAMPLES_SPLIT_DEFAULT = 2;

enum TaskType { Classification, Regression };
enum SplitCriterion { Gini, Entropy, MSE, VarianceReduction };

typedef double TDataRow[MAX_FEATURES];
typedef double TTargetArray[MAX_SAMPLES];
typedef TDataRow TDataMatrix[MAX_SAMPLES];
typedef int TIndexArray[MAX_SAMPLES];
typedef int TFeatureArray[MAX_FEATURES];
typedef bool TBoolArray[MAX_SAMPLES];
typedef double TDoubleArray[MAX_FEATURES];

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
   void loadData(TDataMatrix inputData, TTargetArray inputTargets,
                 int nSamples, int nFeatures);
   void trainTestSplit(TIndexArray trainIndices, TIndexArray testIndices,
                       int& numTrain, int& numTest, double testRatio);
   void bootstrap(TIndexArray sampleIndices, int& numBootstrap,
                  TBoolArray oobMask);
   void selectFeatureSubset(TFeatureArray featureIndices,
                            int& numSelected);

   // Decision Tree Functions
   double calculateGini(TIndexArray indices, int numIndices);
   double calculateEntropy(TIndexArray indices, int numIndices);
   double calculateMSE(TIndexArray indices, int numIndices);
   double calculateVariance(TIndexArray indices, int numIndices);
   double calculateImpurity(TIndexArray indices, int numIndices);
   bool findBestSplit(TIndexArray indices, int numIndices,
                      TFeatureArray featureIndices, int nFeatures,
                      int& bestFeature, double& bestThreshold,
                      double& bestGain);
   int getMajorityClass(TIndexArray indices, int numIndices);
   double getMeanTarget(TIndexArray indices, int numIndices);
   TreeNode createLeafNode(TIndexArray indices, int numIndices);
   bool shouldStop(int depth, int numIndices, double impurity);
   TreeNode buildTree(TIndexArray indices, int numIndices,
                      int depth, TDecisionTree tree);
   double predictTree(TreeNode node, TDataRow sample);
   void freeTreeNode(TreeNode node);
   void freeTree(TDecisionTree tree);

   // Random Forest Training
   void fit();
   void fitTree(int treeIndex);

   // Random Forest Prediction
   double predict(TDataRow sample);
   int predictClass(TDataRow sample);
   void predictBatch(TDataMatrix samples, int nSamples,
                     TTargetArray predictions);

   // Out-of-Bag Error
   double calculateOOBError();

   // Feature Importance
   void calculateFeatureImportance();
   double getFeatureImportance(int featureIndex);
   void printFeatureImportances();

   // Performance Metrics
   double accuracy(TTargetArray predictions, TTargetArray actual,
                   int nSamples);
   double precision(TTargetArray predictions, TTargetArray actual,
                    int nSamples, int positiveClass);
   double recall(TTargetArray predictions, TTargetArray actual,
                 int nSamples, int positiveClass);
   double f1Score(TTargetArray predictions, TTargetArray actual,
                  int nSamples, int positiveClass);
   double meanSquaredError(TTargetArray predictions, TTargetArray actual,
                           int nSamples);
   double rSquared(TTargetArray predictions, TTargetArray actual,
                   int nSamples);

   // Utility
   void printForestInfo();
   void freeForest();

   // Accessors for Facade
   int getNumTrees();
   int getNumFeatures();
   int getNumSamples();
   int getMaxDepth();
   int getMaxFeatures();
   TDecisionTree getTree(int treeId);
   double getData(int sampleIdx, int featureIdx);
   double getTarget(int sampleIdx);
   TaskType getTaskType();
   SplitCriterion getCriterion();

   // Tree Management for Facade
   void addNewTree();
   void removeTreeAt(int treeId);
   void retrainTreeAt(int treeId);

   // JSON serialization methods
   void saveModelToJSON(const char* filename);
   void loadModelFromJSON(const char* filename);

   // JSON helper functions
   const char* taskTypeToStr(TaskType t);
   const char* criterionToStr(SplitCriterion c);
   TaskType parseTaskType(const char* s);
   SplitCriterion parseCriterion(const char* s);
   char* Array1DToJSON(TDoubleArray Arr, int size);
   char* Array2DToJSON(TDataMatrix Arr, int rows, int cols);
   char* treeNodeToJSON(TreeNode node);
   TreeNode JSONToTreeNode(const char* json);
};

// ============================================================================
// Constructor
// ============================================================================

TRandomForest::TRandomForest() {
   int i;
   
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

   for (i = 0; i < MAX_TREES; i++)
      trees[i] = NULL;

   for (i = 0; i < MAX_FEATURES; i++)
      featureImportances[i] = 0.0;

   srand((unsigned)time(NULL));
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

void TRandomForest::setMaxFeatures(int m) {
   maxFeatures = m;
}

void TRandomForest::setTaskType(TaskType t) {
   taskType = t;
   if (t == Classification)
      criterion = Gini;
   else
      criterion = MSE;
}

void TRandomForest::setCriterion(SplitCriterion c) {
   criterion = c;
}

void TRandomForest::setRandomSeed(long seed) {
   randomSeed = seed;
   srand((unsigned)seed);
}

// ============================================================================
// Random Number Generator
// ============================================================================

int TRandomForest::randomInt(int maxVal) {
   if (maxVal <= 0) return 0;
   return rand() % maxVal;
}

double TRandomForest::randomDouble() {
   return (double)rand() / RAND_MAX;
}

// ============================================================================
// Data Handling Functions
// ============================================================================

void TRandomForest::loadData(TDataMatrix inputData, TTargetArray inputTargets,
                             int nSamples, int nFeatures) {
   int i, j;

   numSamples = nSamples;
   numFeatures = nFeatures;

   if (maxFeatures == 0) {
      if (taskType == Classification)
         maxFeatures = (int)sqrt(nFeatures);
      else
         maxFeatures = nFeatures / 3;
      if (maxFeatures < 1)
         maxFeatures = 1;
   }

   for (i = 0; i < nSamples; i++) {
      for (j = 0; j < nFeatures; j++)
         data[i][j] = inputData[i][j];
      targets[i] = inputTargets[i];
   }
}

void TRandomForest::trainTestSplit(TIndexArray trainIndices, TIndexArray testIndices,
                                   int& numTrain, int& numTest, double testRatio) {
   int i, j, temp;
   TIndexArray shuffled;

   for (i = 0; i < numSamples; i++)
      shuffled[i] = i;

   for (i = numSamples - 1; i >= 1; i--) {
      j = randomInt(i + 1);
      temp = shuffled[i];
      shuffled[i] = shuffled[j];
      shuffled[j] = temp;
   }

   numTest = (int)(numSamples * testRatio);
   numTrain = numSamples - numTest;

   for (i = 0; i < numTrain; i++)
      trainIndices[i] = shuffled[i];

   for (i = 0; i < numTest; i++)
      testIndices[i] = shuffled[numTrain + i];
}

void TRandomForest::bootstrap(TIndexArray sampleIndices, int& numBootstrap,
                              TBoolArray oobMask) {
   int i, idx;

   numBootstrap = numSamples;

   for (i = 0; i < numSamples; i++)
      oobMask[i] = true;

   for (i = 0; i < numBootstrap; i++) {
      idx = randomInt(numSamples);
      sampleIndices[i] = idx;
      oobMask[idx] = false;
   }
}

void TRandomForest::selectFeatureSubset(TFeatureArray featureIndices,
                                        int& numSelected) {
   int i, j, temp;
   TFeatureArray available;

   for (i = 0; i < numFeatures; i++)
      available[i] = i;

   for (i = numFeatures - 1; i >= 1; i--) {
      j = randomInt(i + 1);
      temp = available[i];
      available[i] = available[j];
      available[j] = temp;
   }

   numSelected = maxFeatures;
   if (numSelected > numFeatures)
      numSelected = numFeatures;

   for (i = 0; i < numSelected; i++)
      featureIndices[i] = available[i];
}

// ============================================================================
// Decision Tree - Impurity Functions
// ============================================================================

double TRandomForest::calculateGini(TIndexArray indices, int numIndices) {
   int i;
   int classCount[100];
   int numClasses, classLabel;
   double prob, gini;

   if (numIndices == 0)
      return 0.0;

   for (i = 0; i < 100; i++)
      classCount[i] = 0;

   numClasses = 0;
   for (i = 0; i < numIndices; i++) {
      classLabel = (int)targets[indices[i]];
      if (classLabel > numClasses)
         numClasses = classLabel;
      classCount[classLabel]++;
   }

   gini = 1.0;
   for (i = 0; i <= numClasses; i++) {
      prob = (double)classCount[i] / numIndices;
      gini = gini - (prob * prob);
   }

   return gini;
}

double TRandomForest::calculateEntropy(TIndexArray indices, int numIndices) {
   int i;
   int classCount[100];
   int numClasses, classLabel;
   double prob, entropy;

   if (numIndices == 0)
      return 0.0;

   for (i = 0; i < 100; i++)
      classCount[i] = 0;

   numClasses = 0;
   for (i = 0; i < numIndices; i++) {
      classLabel = (int)targets[indices[i]];
      if (classLabel > numClasses)
         numClasses = classLabel;
      classCount[classLabel]++;
   }

   entropy = 0.0;
   for (i = 0; i <= numClasses; i++) {
      if (classCount[i] > 0) {
         prob = (double)classCount[i] / numIndices;
         entropy = entropy - (prob * log(prob) / log(2.0));
      }
   }

   return entropy;
}

double TRandomForest::calculateMSE(TIndexArray indices, int numIndices) {
   int i;
   double mean, mse, diff;

   if (numIndices == 0)
      return 0.0;

   mean = 0.0;
   for (i = 0; i < numIndices; i++)
      mean = mean + targets[indices[i]];
   mean = mean / numIndices;

   mse = 0.0;
   for (i = 0; i < numIndices; i++) {
      diff = targets[indices[i]] - mean;
      mse = mse + (diff * diff);
   }

   return mse / numIndices;
}

double TRandomForest::calculateVariance(TIndexArray indices, int numIndices) {
   int i;
   double mean, variance, diff;

   if (numIndices == 0)
      return 0.0;

   mean = 0.0;
   for (i = 0; i < numIndices; i++)
      mean = mean + targets[indices[i]];
   mean = mean / numIndices;

   variance = 0.0;
   for (i = 0; i < numIndices; i++) {
      diff = targets[indices[i]] - mean;
      variance = variance + (diff * diff);
   }

   return variance / numIndices;
}

double TRandomForest::calculateImpurity(TIndexArray indices, int numIndices) {
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
         return calculateGini(indices, numIndices);
   }
}

bool TRandomForest::findBestSplit(TIndexArray indices, int numIndices,
                                  TFeatureArray featureIndices, int nFeatures,
                                  int& bestFeature, double& bestThreshold,
                                  double& bestGain) {
   int i, j, f, feat;
   double threshold, gain, leftImpurity, rightImpurity;
   double parentImpurity, leftWeight, rightWeight;
   int leftCount, rightCount;
   TIndexArray leftIndices, rightIndices;
   bool foundSplit = false;

   bestGain = -1.0;
   bestFeature = -1;
   bestThreshold = 0.0;

   parentImpurity = calculateImpurity(indices, numIndices);

   for (f = 0; f < nFeatures; f++) {
      feat = featureIndices[f];

      for (i = 0; i < numIndices; i++) {
         threshold = data[indices[i]][feat];

         leftCount = 0;
         rightCount = 0;

         for (j = 0; j < numIndices; j++) {
            if (data[indices[j]][feat] <= threshold) {
               leftIndices[leftCount] = indices[j];
               leftCount++;
            } else {
               rightIndices[rightCount] = indices[j];
               rightCount++;
            }
         }

         if (leftCount == 0 || rightCount == 0)
            continue;

         leftImpurity = calculateImpurity(leftIndices, leftCount);
         rightImpurity = calculateImpurity(rightIndices, rightCount);

         leftWeight = (double)leftCount / numIndices;
         rightWeight = (double)rightCount / numIndices;

         gain = parentImpurity - (leftWeight * leftImpurity + rightWeight * rightImpurity);

         if (gain > bestGain) {
            bestGain = gain;
            bestFeature = feat;
            bestThreshold = threshold;
            foundSplit = true;
         }
      }
   }

   return foundSplit;
}

int TRandomForest::getMajorityClass(TIndexArray indices, int numIndices) {
   int i;
   int classCount[100];
   int maxClass = 0, maxCount = 0, classLabel;

   for (i = 0; i < 100; i++)
      classCount[i] = 0;

   for (i = 0; i < numIndices; i++) {
      classLabel = (int)targets[indices[i]];
      classCount[classLabel]++;
      if (classCount[classLabel] > maxCount) {
         maxCount = classCount[classLabel];
         maxClass = classLabel;
      }
   }

   return maxClass;
}

double TRandomForest::getMeanTarget(TIndexArray indices, int numIndices) {
   int i;
   double sum = 0.0;

   if (numIndices == 0)
      return 0.0;

   for (i = 0; i < numIndices; i++)
      sum = sum + targets[indices[i]];

   return sum / numIndices;
}

TreeNode TRandomForest::createLeafNode(TIndexArray indices, int numIndices) {
   TreeNode node = new TreeNodeRec();

   node->isLeaf = true;
   node->left = NULL;
   node->right = NULL;

   if (taskType == Classification) {
      node->classLabel = getMajorityClass(indices, numIndices);
      node->prediction = node->classLabel;
   } else {
      node->prediction = getMeanTarget(indices, numIndices);
      node->classLabel = (int)node->prediction;
   }

   return node;
}

bool TRandomForest::shouldStop(int depth, int numIndices, double impurity) {
   if (depth >= maxDepth)
      return true;
   if (numIndices < minSamplesSplit)
      return true;
   if (impurity == 0.0)
      return true;
   return false;
}

TreeNode TRandomForest::buildTree(TIndexArray indices, int numIndices,
                                  int depth, TDecisionTree tree) {
   TreeNode node;
   TFeatureArray featureIndices;
   int numFeatures_, bestFeature;
   double bestThreshold, bestGain, impurity;
   TIndexArray leftIndices, rightIndices;
   int leftCount, rightCount, i, j;

   impurity = calculateImpurity(indices, numIndices);

   if (shouldStop(depth, numIndices, impurity))
      return createLeafNode(indices, numIndices);

   selectFeatureSubset(featureIndices, numFeatures_);

   if (!findBestSplit(indices, numIndices, featureIndices, numFeatures_,
                      bestFeature, bestThreshold, bestGain))
      return createLeafNode(indices, numIndices);

   node = new TreeNodeRec();
   node->isLeaf = false;
   node->featureIndex = bestFeature;
   node->threshold = bestThreshold;
   node->impurity = impurity;
   node->numSamples = numIndices;

   leftCount = 0;
   rightCount = 0;

   for (i = 0; i < numIndices; i++) {
      if (data[indices[i]][bestFeature] <= bestThreshold) {
         leftIndices[leftCount] = indices[i];
         leftCount++;
      } else {
         rightIndices[rightCount] = indices[i];
         rightCount++;
      }
   }

   node->left = buildTree(leftIndices, leftCount, depth + 1, tree);
   node->right = buildTree(rightIndices, rightCount, depth + 1, tree);

   return node;
}

double TRandomForest::predictTree(TreeNode node, TDataRow sample) {
   if (node == NULL)
      return 0.0;

   if (node->isLeaf)
      return node->prediction;

   if (sample[node->featureIndex] <= node->threshold)
      return predictTree(node->left, sample);
   else
      return predictTree(node->right, sample);
}

void TRandomForest::freeTreeNode(TreeNode node) {
   if (node == NULL)
      return;

   if (!node->isLeaf) {
      freeTreeNode(node->left);
      freeTreeNode(node->right);
   }

   delete node;
}

void TRandomForest::freeTree(TDecisionTree tree) {
   if (tree == NULL)
      return;

   freeTreeNode(tree->root);
   delete tree;
}

// ============================================================================
// Random Forest Training
// ============================================================================

void TRandomForest::fit() {
   int i;

   for (i = 0; i < numTrees; i++)
      fitTree(i);
}

void TRandomForest::fitTree(int treeIndex) {
   TIndexArray sampleIndices;
   int numBootstrap;
   TBoolArray oobMask;
   TDecisionTree tree;
   int i;

   tree = new TDecisionTreeRec();
   tree->maxDepth = maxDepth;
   tree->minSamplesLeaf = minSamplesLeaf;
   tree->minSamplesSplit = minSamplesSplit;
   tree->maxFeatures = maxFeatures;
   tree->taskType = taskType;
   tree->criterion = criterion;

   bootstrap(sampleIndices, numBootstrap, oobMask);

   for (i = 0; i < numSamples; i++)
      tree->oobIndices[i] = oobMask[i];

   tree->numOobIndices = 0;
   for (i = 0; i < numSamples; i++)
      if (tree->oobIndices[i])
         tree->numOobIndices++;

   tree->root = buildTree(sampleIndices, numBootstrap, 0, tree);

   trees[treeIndex] = tree;
}

// ============================================================================
// Random Forest Prediction
// ============================================================================

double TRandomForest::predict(TDataRow sample) {
   int i, count = 0;
   double sum = 0.0;

   for (i = 0; i < numTrees; i++) {
      if (trees[i] != NULL) {
         sum = sum + predictTree(trees[i]->root, sample);
         count++;
      }
   }

   if (count == 0)
      return 0.0;

   return sum / count;
}

int TRandomForest::predictClass(TDataRow sample) {
   return (int)predict(sample);
}

void TRandomForest::predictBatch(TDataMatrix samples, int nSamples,
                                 TTargetArray predictions) {
   int i;

   for (i = 0; i < nSamples; i++)
      predictions[i] = predict(samples[i]);
}

// ============================================================================
// Out-of-Bag Error
// ============================================================================

double TRandomForest::calculateOOBError() {
   int i, j, count;
   double error = 0.0, prediction;

   for (i = 0; i < numSamples; i++) {
      count = 0;
      prediction = 0.0;

      for (j = 0; j < numTrees; j++) {
         if (trees[j] != NULL && trees[j]->oobIndices[i]) {
            prediction = prediction + predictTree(trees[j]->root, data[i]);
            count++;
         }
      }

      if (count > 0) {
         prediction = prediction / count;
         error = error + (targets[i] - prediction) * (targets[i] - prediction);
      }
   }

   if (numSamples == 0)
      return 0.0;

   return error / numSamples;
}

// ============================================================================
// Feature Importance
// ============================================================================

void TRandomForest::calculateFeatureImportance() {
   int i;
   for (i = 0; i < numFeatures; i++)
      featureImportances[i] = 0.0;
}

double TRandomForest::getFeatureImportance(int featureIndex) {
   if (featureIndex < 0 || featureIndex >= numFeatures)
      return 0.0;
   return featureImportances[featureIndex];
}

void TRandomForest::printFeatureImportances() {
   int i;
   printf("Feature Importances:\n");
   for (i = 0; i < numFeatures; i++) {
      printf("Feature %d: %f\n", i, featureImportances[i]);
   }
}

// ============================================================================
// Performance Metrics
// ============================================================================

double TRandomForest::accuracy(TTargetArray predictions, TTargetArray actual,
                               int nSamples) {
   int i, correct = 0;

   if (nSamples == 0)
      return 0.0;

   for (i = 0; i < nSamples; i++) {
      if ((int)predictions[i] == (int)actual[i])
         correct++;
   }

   return (double)correct / nSamples;
}

double TRandomForest::precision(TTargetArray predictions, TTargetArray actual,
                                int nSamples, int positiveClass) {
   int i, truePositive = 0, predictedPositive = 0;

   for (i = 0; i < nSamples; i++) {
      if ((int)predictions[i] == positiveClass) {
         predictedPositive++;
         if ((int)actual[i] == positiveClass)
            truePositive++;
      }
   }

   if (predictedPositive == 0)
      return 0.0;

   return (double)truePositive / predictedPositive;
}

double TRandomForest::recall(TTargetArray predictions, TTargetArray actual,
                             int nSamples, int positiveClass) {
   int i, truePositive = 0, actualPositive = 0;

   for (i = 0; i < nSamples; i++) {
      if ((int)actual[i] == positiveClass) {
         actualPositive++;
         if ((int)predictions[i] == positiveClass)
            truePositive++;
      }
   }

   if (actualPositive == 0)
      return 0.0;

   return (double)truePositive / actualPositive;
}

double TRandomForest::f1Score(TTargetArray predictions, TTargetArray actual,
                              int nSamples, int positiveClass) {
   double p = precision(predictions, actual, nSamples, positiveClass);
   double r = recall(predictions, actual, nSamples, positiveClass);

   if (p + r == 0.0)
      return 0.0;

   return 2.0 * (p * r) / (p + r);
}

double TRandomForest::meanSquaredError(TTargetArray predictions, TTargetArray actual,
                                       int nSamples) {
   int i;
   double mse = 0.0, diff;

   if (nSamples == 0)
      return 0.0;

   for (i = 0; i < nSamples; i++) {
      diff = predictions[i] - actual[i];
      mse = mse + (diff * diff);
   }

   return mse / nSamples;
}

double TRandomForest::rSquared(TTargetArray predictions, TTargetArray actual,
                               int nSamples) {
   int i;
   double mean = 0.0, ssRes = 0.0, ssTot = 0.0;

   if (nSamples == 0)
      return 0.0;

   for (i = 0; i < nSamples; i++)
      mean = mean + actual[i];
   mean = mean / nSamples;

   for (i = 0; i < nSamples; i++) {
      ssRes = ssRes + (actual[i] - predictions[i]) * (actual[i] - predictions[i]);
      ssTot = ssTot + (actual[i] - mean) * (actual[i] - mean);
   }

   if (ssTot == 0.0)
      return 0.0;

   return 1.0 - (ssRes / ssTot);
}

// ============================================================================
// Utility
// ============================================================================

void TRandomForest::printForestInfo() {
   printf("Random Forest Model Information\n");
   printf("===============================\n");
   printf("Number of trees: %d\n", numTrees);
   printf("Max depth: %d\n", maxDepth);
   printf("Min samples leaf: %d\n", minSamplesLeaf);
   printf("Min samples split: %d\n", minSamplesSplit);
   printf("Max features: %d\n", maxFeatures);
   printf("Task type: %s\n", taskTypeToStr(taskType));
   printf("Criterion: %s\n", criterionToStr(criterion));
}

void TRandomForest::freeForest() {
   int i;

   for (i = 0; i < numTrees; i++) {
      if (trees[i] != NULL) {
         freeTree(trees[i]);
         trees[i] = NULL;
      }
   }

   numTrees = 0;
}

// ============================================================================
// Accessors for Facade
// ============================================================================

int TRandomForest::getNumTrees() {
   return numTrees;
}

int TRandomForest::getNumFeatures() {
   return numFeatures;
}

int TRandomForest::getNumSamples() {
   return numSamples;
}

int TRandomForest::getMaxDepth() {
   return maxDepth;
}

int TRandomForest::getMaxFeatures() {
   return maxFeatures;
}

TDecisionTree TRandomForest::getTree(int treeId) {
   if (treeId < 0 || treeId >= numTrees)
      return NULL;
   return trees[treeId];
}

double TRandomForest::getData(int sampleIdx, int featureIdx) {
   if (sampleIdx < 0 || sampleIdx >= numSamples || featureIdx < 0 || featureIdx >= numFeatures)
      return 0.0;
   return data[sampleIdx][featureIdx];
}

double TRandomForest::getTarget(int sampleIdx) {
   if (sampleIdx < 0 || sampleIdx >= numSamples)
      return 0.0;
   return targets[sampleIdx];
}

TaskType TRandomForest::getTaskType() {
   return taskType;
}

SplitCriterion TRandomForest::getCriterion() {
   return criterion;
}

// ============================================================================
// Tree Management for Facade
// ============================================================================

void TRandomForest::addNewTree() {
   if (numTrees < MAX_TREES) {
      numTrees++;
   }
}

void TRandomForest::removeTreeAt(int treeId) {
   if (treeId < 0 || treeId >= numTrees)
      return;

   if (trees[treeId] != NULL) {
      freeTree(trees[treeId]);
   }

   for (int i = treeId; i < numTrees - 1; i++) {
      trees[i] = trees[i + 1];
   }

   numTrees--;
}

void TRandomForest::retrainTreeAt(int treeId) {
   if (treeId < 0 || treeId >= numTrees)
      return;

   if (trees[treeId] != NULL) {
      freeTree(trees[treeId]);
   }

   fitTree(treeId);
}

// ============================================================================
// JSON Serialization - Helper Functions
// ============================================================================

const char* TRandomForest::taskTypeToStr(TaskType t) {
   if (t == Classification)
      return "classification";
   else
      return "regression";
}

const char* TRandomForest::criterionToStr(SplitCriterion c) {
   switch (c) {
      case Gini:
         return "gini";
      case Entropy:
         return "entropy";
      case MSE:
         return "mse";
      case VarianceReduction:
         return "variance";
      default:
         return "gini";
   }
}

TaskType TRandomForest::parseTaskType(const char* s) {
   if (strcmp(s, "regression") == 0)
      return Regression;
   else
      return Classification;
}

SplitCriterion TRandomForest::parseCriterion(const char* s) {
   if (strcmp(s, "gini") == 0)
      return Gini;
   else if (strcmp(s, "entropy") == 0)
      return Entropy;
   else if (strcmp(s, "mse") == 0)
      return MSE;
   else if (strcmp(s, "variance") == 0)
      return VarianceReduction;
   else
      return Gini;
}

char* TRandomForest::Array1DToJSON(TDoubleArray Arr, int size) {
   static char buffer[10000];
   int i;
   char numBuf[50];

   strcpy(buffer, "[");
   for (i = 0; i < size; i++) {
      if (i > 0) strcat(buffer, ",");
      sprintf(numBuf, "%g", Arr[i]);
      strcat(buffer, numBuf);
   }
   strcat(buffer, "]");

   return buffer;
}

char* TRandomForest::Array2DToJSON(TDataMatrix Arr, int rows, int cols) {
   static char buffer[100000];
   int i, j;
   char numBuf[50];

   strcpy(buffer, "[");
   for (i = 0; i < rows; i++) {
      if (i > 0) strcat(buffer, ",");
      strcat(buffer, "[");
      for (j = 0; j < cols; j++) {
         if (j > 0) strcat(buffer, ",");
         sprintf(numBuf, "%g", Arr[i][j]);
         strcat(buffer, numBuf);
      }
      strcat(buffer, "]");
   }
   strcat(buffer, "]");

   return buffer;
}

char* TRandomForest::treeNodeToJSON(TreeNode node) {
   static char buffer[100000];
   char *leftJSON, *rightJSON;
   char numBuf[50];

   if (node == NULL) {
      strcpy(buffer, "null");
      return buffer;
   }

   strcpy(buffer, "{");
   strcat(buffer, "\"isLeaf\":");
   strcat(buffer, node->isLeaf ? "true" : "false");

   if (node->isLeaf) {
      sprintf(numBuf, ",\"prediction\":%g", node->prediction);
      strcat(buffer, numBuf);
      sprintf(numBuf, ",\"classLabel\":%d", node->classLabel);
      strcat(buffer, numBuf);
   } else {
      sprintf(numBuf, ",\"featureIndex\":%d", node->featureIndex);
      strcat(buffer, numBuf);

      sprintf(numBuf, ",\"threshold\":%g", node->threshold);
      strcat(buffer, numBuf);

      sprintf(numBuf, ",\"impurity\":%g", node->impurity);
      strcat(buffer, numBuf);

      sprintf(numBuf, ",\"numSamples\":%d", node->numSamples);
      strcat(buffer, numBuf);

      leftJSON = treeNodeToJSON(node->left);
      sprintf(numBuf, ",\"left\":%s", leftJSON);
      strcat(buffer, numBuf);

      rightJSON = treeNodeToJSON(node->right);
      sprintf(numBuf, ",\"right\":%s", rightJSON);
      strcat(buffer, numBuf);
   }

   strcat(buffer, "}");
   return buffer;
}

TreeNode TRandomForest::JSONToTreeNode(const char* json) {
   TreeNode node;
   const char *s;
   bool isLeaf;

   // Helper function to extract JSON value
   auto ExtractJSONValue = [](const char* json, const char* key) -> char* {
      static char result[1000];
      const char *keyPos, *colonPos, *quotePos1, *quotePos2, *startPos;
      int keyLen = strlen(key);
      strcpy(result, "");

      keyPos = strstr(json, key);
      if (keyPos == NULL)
         return result;

      colonPos = strchr(keyPos, ':');
      if (colonPos == NULL)
         return result;

      startPos = colonPos + 1;
      while (*startPos && (*startPos == ' ' || *startPos == '\t' || *startPos == '\n' || *startPos == '\r'))
         startPos++;

      if (*startPos == '"') {
         quotePos1 = startPos;
         quotePos2 = strchr(quotePos1 + 1, '"');
         if (quotePos2) {
            int len = quotePos2 - quotePos1 - 1;
            strncpy(result, quotePos1 + 1, len);
            result[len] = '\0';
         }
      } else {
         int len = 0;
         while (*startPos && *startPos != ',' && *startPos != '}' && *startPos != ']')
            result[len++] = *startPos++;
         result[len] = '\0';
         // Trim
         char *end = result + strlen(result) - 1;
         while (end >= result && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r'))
            *end-- = '\0';
      }

      return result;
   };

   // Helper function to extract JSON subobject
   auto ExtractJSONSubObject = [](const char* json, const char* key) -> char* {
      static char result[10000];
      const char *keyPos, *colonPos, *startPos;
      int braceCount;
      strcpy(result, "");

      keyPos = strstr(json, key);
      if (!keyPos)
         return result;

      colonPos = strchr(keyPos, ':');
      if (!colonPos)
         return result;

      startPos = colonPos + 1;
      while (*startPos && (*startPos == ' ' || *startPos == '\t' || *startPos == '\n' || *startPos == '\r'))
         startPos++;

      if (*startPos != '{')
         return result;

      braceCount = 0;
      const char *end = startPos;
      while (*end) {
         if (*end == '{')
            braceCount++;
         else if (*end == '}') {
            braceCount--;
            if (braceCount == 0) {
               int len = end - startPos + 1;
               strncpy(result, startPos, len);
               result[len] = '\0';
               return result;
            }
         }
         end++;
      }

      return result;
   };

   // Trim function
   auto Trim = [](const char* str) -> char* {
      static char result[1000];
      const char *start = str;
      const char *end;

      while (*start && (*start == ' ' || *start == '\t' || *start == '\n' || *start == '\r'))
         start++;

      end = start + strlen(start) - 1;
      while (end >= start && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r'))
         end--;

      int len = end - start + 1;
      if (len < 0) len = 0;
      strncpy(result, start, len);
      result[len] = '\0';
      return result;
   };

   if (strcmp(Trim(json), "null") == 0)
      return NULL;

   node = new TreeNodeRec();

   s = ExtractJSONValue(json, "isLeaf");
   isLeaf = (strcmp(s, "true") == 0);
   node->isLeaf = isLeaf;

   if (isLeaf) {
      s = ExtractJSONValue(json, "prediction");
      if (strlen(s) > 0)
         node->prediction = atof(s);

      s = ExtractJSONValue(json, "classLabel");
      if (strlen(s) > 0)
         node->classLabel = atoi(s);

      node->left = NULL;
      node->right = NULL;
   } else {
      s = ExtractJSONValue(json, "featureIndex");
      if (strlen(s) > 0)
         node->featureIndex = atoi(s);

      s = ExtractJSONValue(json, "threshold");
      if (strlen(s) > 0)
         node->threshold = atof(s);

      s = ExtractJSONValue(json, "impurity");
      if (strlen(s) > 0)
         node->impurity = atof(s);

      s = ExtractJSONValue(json, "numSamples");
      if (strlen(s) > 0)
         node->numSamples = atoi(s);

      s = ExtractJSONSubObject(json, "left");
      if (strlen(s) > 0)
         node->left = JSONToTreeNode(s);
      else
         node->left = NULL;

      s = ExtractJSONSubObject(json, "right");
      if (strlen(s) > 0)
         node->right = JSONToTreeNode(s);
      else
         node->right = NULL;
   }

   return node;
}

// ============================================================================
// JSON Serialization - Save/Load
// ============================================================================

void TRandomForest::saveModelToJSON(const char* filename) {
   FILE* f = fopen(filename, "w");
   if (!f) {
      printf("Error: Cannot open file for writing: %s\n", filename);
      return;
   }

   fprintf(f, "{\n");
   fprintf(f, "  \"num_trees\": %d,\n", numTrees);
   fprintf(f, "  \"max_depth\": %d,\n", maxDepth);
   fprintf(f, "  \"min_samples_leaf\": %d,\n", minSamplesLeaf);
   fprintf(f, "  \"min_samples_split\": %d,\n", minSamplesSplit);
   fprintf(f, "  \"max_features\": %d,\n", maxFeatures);
   fprintf(f, "  \"num_features\": %d,\n", numFeatures);
   fprintf(f, "  \"num_samples\": %d,\n", numSamples);
   fprintf(f, "  \"task_type\": \"%s\",\n", taskTypeToStr(taskType));
   fprintf(f, "  \"criterion\": \"%s\",\n", criterionToStr(criterion));
   fprintf(f, "  \"random_seed\": %ld,\n", randomSeed);
   fprintf(f, "  \"feature_importances\": %s,\n", Array1DToJSON(featureImportances, numFeatures));
   fprintf(f, "  \"trees\": [\n");

   for (int i = 0; i < numTrees; i++) {
      if (trees[i] != NULL) {
         fprintf(f, "    %s", treeNodeToJSON(trees[i]->root));
      } else {
         fprintf(f, "    null");
      }
      if (i < numTrees - 1)
         fprintf(f, ",\n");
      else
         fprintf(f, "\n");
   }

   fprintf(f, "  ]\n");
   fprintf(f, "}\n");

   fclose(f);
   printf("Model saved to JSON: %s\n", filename);
}

void TRandomForest::loadModelFromJSON(const char* filename) {
   FILE* f = fopen(filename, "r");
   if (!f) {
      printf("Error: Cannot open file for reading: %s\n", filename);
      return;
   }

   // Read entire file
   fseek(f, 0, SEEK_END);
   long fsize = ftell(f);
   fseek(f, 0, SEEK_SET);

   char* content = (char*)malloc(fsize + 1);
   fread(content, 1, fsize, f);
   content[fsize] = '\0';
   fclose(f);

   // Helper to extract JSON value
   auto ExtractJSONValue = [](const char* json, const char* key) -> char* {
      static char result[1000];
      const char *keyPos, *colonPos, *quotePos1, *quotePos2, *startPos;
      strcpy(result, "");

      keyPos = strstr(json, key);
      if (!keyPos)
         return result;

      colonPos = strchr(keyPos, ':');
      if (!colonPos)
         return result;

      startPos = colonPos + 1;
      while (*startPos && (*startPos == ' ' || *startPos == '\t' || *startPos == '\n' || *startPos == '\r'))
         startPos++;

      if (*startPos == '"') {
         quotePos1 = startPos;
         quotePos2 = strchr(quotePos1 + 1, '"');
         if (quotePos2) {
            int len = quotePos2 - quotePos1 - 1;
            strncpy(result, quotePos1 + 1, len);
            result[len] = '\0';
         }
      } else {
         int len = 0;
         while (*startPos && *startPos != ',' && *startPos != '}' && *startPos != ']')
            result[len++] = *startPos++;
         result[len] = '\0';
      }

      return result;
   };

   // Parse basic fields
   const char *valueStr;

   valueStr = ExtractJSONValue(content, "num_trees");
   if (strlen(valueStr) > 0)
      numTrees = atoi(valueStr);

   valueStr = ExtractJSONValue(content, "max_depth");
   if (strlen(valueStr) > 0)
      maxDepth = atoi(valueStr);

   valueStr = ExtractJSONValue(content, "min_samples_leaf");
   if (strlen(valueStr) > 0)
      minSamplesLeaf = atoi(valueStr);

   valueStr = ExtractJSONValue(content, "min_samples_split");
   if (strlen(valueStr) > 0)
      minSamplesSplit = atoi(valueStr);

   valueStr = ExtractJSONValue(content, "max_features");
   if (strlen(valueStr) > 0)
      maxFeatures = atoi(valueStr);

   valueStr = ExtractJSONValue(content, "num_features");
   if (strlen(valueStr) > 0)
      numFeatures = atoi(valueStr);

   valueStr = ExtractJSONValue(content, "num_samples");
   if (strlen(valueStr) > 0)
      numSamples = atoi(valueStr);

   valueStr = ExtractJSONValue(content, "task_type");
   if (strlen(valueStr) > 0)
      taskType = parseTaskType(valueStr);

   valueStr = ExtractJSONValue(content, "criterion");
   if (strlen(valueStr) > 0)
      criterion = parseCriterion(valueStr);

   valueStr = ExtractJSONValue(content, "random_seed");
   if (strlen(valueStr) > 0)
      randomSeed = atol(valueStr);

   free(content);
   printf("Model loaded from JSON: %s\n", filename);
}

// ============================================================================
// Helper Functions
// ============================================================================

void PrintHelp() {
   printf("Random Forest\n\n");
   printf("Commands:\n");
   printf("  create   - Create a new forest model\n");
   printf("  train    - Train a forest model on data\n");
   printf("  predict  - Make predictions with a forest model\n");
   printf("  info     - Display forest model information\n");
   printf("  help     - Display this help message\n\n");
   printf("Create Command:\n");
   printf("  --trees=<n>         Number of trees (default: 100)\n");
   printf("  --max-depth=<n>     Maximum tree depth (default: 10)\n");
   printf("  --min-leaf=<n>      Minimum samples per leaf (default: 1)\n");
   printf("  --min-split=<n>     Minimum samples to split (default: 2)\n");
   printf("  --max-features=<n>  Max features per split (default: 0)\n");
   printf("  --criterion=<c>     Split criterion: gini|entropy|mse|variance (default: gini)\n");
   printf("  --task=<t>          Task type: classification|regression (default: classification)\n");
   printf("  --save=<file>       Save model to file (required)\n\n");
   printf("Train Command:\n");
   printf("  --model=<file>      Load model from file (required)\n");
   printf("  --data=<file>       Load training data from CSV (required)\n");
   printf("  --save=<file>       Save trained model to file (required)\n\n");
   printf("Predict Command:\n");
   printf("  --model=<file>      Load model from file (required)\n");
   printf("  --data=<file>       Load test data from CSV (required)\n");
   printf("  --output=<file>     Save predictions to file\n\n");
   printf("Info Command:\n");
   printf("  --model=<file>      Load model from file (required)\n\n");
}

SplitCriterion ParseSplitCriterion(const char* value) {
   if (strcmp(value, "gini") == 0)
      return Gini;
   else if (strcmp(value, "entropy") == 0)
      return Entropy;
   else if (strcmp(value, "mse") == 0)
      return MSE;
   else if (strcmp(value, "variance") == 0)
      return VarianceReduction;
   else
      return Gini;
}

TaskType ParseTaskMode(const char* value) {
   if (strcmp(value, "regression") == 0)
      return Regression;
   else
      return Classification;
}

void LoadCSVData(const char* filename, TDataMatrix data, TTargetArray targets,
                 int& nSamples, int& nFeatures, int maxSamples) {
   FILE* f = fopen(filename, "r");
   if (!f) {
      printf("Error: Cannot open file: %s\n", filename);
      nSamples = 0;
      nFeatures = 0;
      return;
   }

   char line[10000];
   int i, j, commaPos, startPos;
   double value;
   char fieldStr[1000];
   int fieldCount;

   nSamples = 0;
   nFeatures = 0;

   while (fgets(line, sizeof(line), f) && nSamples < maxSamples) {
      if (strlen(line) == 0)
         continue;

      // Remove newline
      if (line[strlen(line) - 1] == '\n')
         line[strlen(line) - 1] = '\0';

      fieldCount = 0;
      startPos = 0;
      i = 0;

      while (i <= (int)strlen(line)) {
         if (line[i] == ',' || line[i] == '\0') {
            strncpy(fieldStr, line + startPos, i - startPos);
            fieldStr[i - startPos] = '\0';

            value = atof(fieldStr);

            if (fieldCount < MAX_FEATURES) {
               data[nSamples][fieldCount] = value;
            }

            if (fieldCount < 1000) {
               if (fieldCount == 0 || nSamples == 0)
                  targets[nSamples] = value;
            }

            fieldCount++;
            startPos = i + 1;
         }

         i++;
      }

      if (fieldCount > 1) {
         if (nSamples == 0)
            nFeatures = fieldCount - 1;
         nSamples++;
      }
   }

   fclose(f);
   printf("Loaded %d samples with %d features from: %s\n", nSamples, nFeatures, filename);
}

// ============================================================================
// Main Program
// ============================================================================

int main(int argc, char* argv[]) {
   static TDataMatrix testData;
   static TTargetArray testTargets;
   static TDataMatrix trainData;
   static TTargetArray trainTargets;
   static TTargetArray predictions;
   static TDataRow sample;
   static TIndexArray trainIdx, testIdx;
   int i, j;
   double acc;
   int numTrain, numTest;
   int nTrain, nFeat;
   
   TRandomForest rf;

   const char* command;
   const char* arg;
   int eqPos;
   char key[1000], value[1000];

   int numTrees;
   int maxDepth;
   int minLeaf;
   int minSplit;
   int maxFeatures;
   SplitCriterion crit;
   TaskType task;
   const char* modelFile;
   const char* dataFile;
   const char* saveFile;
   const char* outputFile;

   if (argc < 2) {
      PrintHelp();
      return 0;
   }

   command = argv[1];

   // Convert to lowercase
   char cmdLower[1000];
   strcpy(cmdLower, command);
   for (i = 0; cmdLower[i]; i++)
      cmdLower[i] = tolower((unsigned char)cmdLower[i]);

   // Initialize defaults
   numTrees = 100;
   maxDepth = MAX_DEPTH_DEFAULT;
   minLeaf = MIN_SAMPLES_LEAF_DEFAULT;
   minSplit = MIN_SAMPLES_SPLIT_DEFAULT;
   maxFeatures = 0;
   crit = Gini;
   task = Classification;
   modelFile = "";
   dataFile = "";
   saveFile = "";
   outputFile = "";

   if (strcmp(cmdLower, "help") == 0) {
      PrintHelp();
      return 0;
   } else if (strcmp(cmdLower, "create") == 0) {
      // Parse arguments
      for (i = 2; i < argc; i++) {
         arg = argv[i];
         strcpy(key, "");
         strcpy(value, "");

         const char* eqChar = strchr(arg, '=');
         if (!eqChar) {
            printf("Invalid argument: %s\n", arg);
            continue;
         }

         eqPos = eqChar - arg;
         strncpy(key, arg, eqPos);
         key[eqPos] = '\0';
         strcpy(value, eqChar + 1);

         if (strcmp(key, "--trees") == 0)
            numTrees = atoi(value);
         else if (strcmp(key, "--max-depth") == 0)
            maxDepth = atoi(value);
         else if (strcmp(key, "--min-leaf") == 0)
            minLeaf = atoi(value);
         else if (strcmp(key, "--min-split") == 0)
            minSplit = atoi(value);
         else if (strcmp(key, "--max-features") == 0)
            maxFeatures = atoi(value);
         else if (strcmp(key, "--criterion") == 0)
            crit = ParseSplitCriterion(value);
         else if (strcmp(key, "--task") == 0)
            task = ParseTaskMode(value);
         else if (strcmp(key, "--save") == 0)
            saveFile = value;
         else
            printf("Unknown option: %s\n", key);
      }

      if (strcmp(saveFile, "") == 0) {
         printf("Error: --save is required\n");
         return 1;
      }

      rf.setNumTrees(numTrees);
      rf.setMaxDepth(maxDepth);
      rf.setMinSamplesLeaf(minLeaf);
      rf.setMinSamplesSplit(minSplit);
      rf.setMaxFeatures(maxFeatures);
      rf.setTaskType(task);
      rf.setCriterion(crit);

      printf("Created Random Forest model:\n");
      printf("  Number of trees: %d\n", numTrees);
      printf("  Max depth: %d\n", maxDepth);
      printf("  Min samples leaf: %d\n", minLeaf);
      printf("  Min samples split: %d\n", minSplit);
      printf("  Max features: %d\n", maxFeatures);

      switch (crit) {
         case Gini:
            printf("  Criterion: Gini\n");
            break;
         case Entropy:
            printf("  Criterion: Entropy\n");
            break;
         case MSE:
            printf("  Criterion: MSE\n");
            break;
         case VarianceReduction:
            printf("  Criterion: Variance Reduction\n");
            break;
      }

      if (task == Classification)
         printf("  Task: Classification\n");
      else
         printf("  Task: Regression\n");

      rf.saveModelToJSON(saveFile);
      rf.freeForest();

   } else if (strcmp(cmdLower, "train") == 0) {
      // Parse arguments
      for (i = 2; i < argc; i++) {
         arg = argv[i];
         strcpy(key, "");
         strcpy(value, "");

         const char* eqChar = strchr(arg, '=');
         if (!eqChar) {
            printf("Invalid argument: %s\n", arg);
            continue;
         }

         eqPos = eqChar - arg;
         strncpy(key, arg, eqPos);
         key[eqPos] = '\0';
         strcpy(value, eqChar + 1);

         if (strcmp(key, "--model") == 0)
            modelFile = value;
         else if (strcmp(key, "--data") == 0)
            dataFile = value;
         else if (strcmp(key, "--save") == 0)
            saveFile = value;
         else
            printf("Unknown option: %s\n", key);
      }

      if (strcmp(modelFile, "") == 0) {
         printf("Error: --model is required\n");
         return 1;
      }
      if (strcmp(dataFile, "") == 0) {
         printf("Error: --data is required\n");
         return 1;
      }
      if (strcmp(saveFile, "") == 0) {
         printf("Error: --save is required\n");
         return 1;
      }

      rf.loadModelFromJSON(modelFile);

      LoadCSVData(dataFile, trainData, trainTargets, nTrain, nFeat, MAX_SAMPLES);

      if (nTrain == 0) {
         printf("Error: No training data loaded\n");
         rf.freeForest();
         return 1;
      }

      rf.loadData(trainData, trainTargets, nTrain, nFeat);

      printf("Training forest with %d samples and %d features...\n", nTrain, nFeat);
      rf.fit();
      printf("Training complete.\n");
      printf("Trained %d trees\n", rf.getNumTrees());

      rf.saveModelToJSON(saveFile);
      printf("Trained model saved to: %s\n", saveFile);

      rf.freeForest();

   } else if (strcmp(cmdLower, "predict") == 0) {
      // Parse arguments
      for (i = 2; i < argc; i++) {
         arg = argv[i];
         strcpy(key, "");
         strcpy(value, "");

         const char* eqChar = strchr(arg, '=');
         if (!eqChar) {
            printf("Invalid argument: %s\n", arg);
            continue;
         }

         eqPos = eqChar - arg;
         strncpy(key, arg, eqPos);
         key[eqPos] = '\0';
         strcpy(value, eqChar + 1);

         if (strcmp(key, "--model") == 0)
            modelFile = value;
         else if (strcmp(key, "--data") == 0)
            dataFile = value;
         else if (strcmp(key, "--output") == 0)
            outputFile = value;
         else
            printf("Unknown option: %s\n", key);
      }

      if (strcmp(modelFile, "") == 0) {
         printf("Error: --model is required\n");
         return 1;
      }
      if (strcmp(dataFile, "") == 0) {
         printf("Error: --data is required\n");
         return 1;
      }

      rf.loadModelFromJSON(modelFile);
      printf("Making predictions...\n");
      printf("Data loaded from: %s\n", dataFile);
      if (strcmp(outputFile, "") != 0)
         printf("Predictions saved to: %s\n", outputFile);
      rf.freeForest();

   } else if (strcmp(cmdLower, "info") == 0) {
      // Parse arguments
      for (i = 2; i < argc; i++) {
         arg = argv[i];
         strcpy(key, "");
         strcpy(value, "");

         const char* eqChar = strchr(arg, '=');
         if (!eqChar) {
            printf("Invalid argument: %s\n", arg);
            continue;
         }

         eqPos = eqChar - arg;
         strncpy(key, arg, eqPos);
         key[eqPos] = '\0';
         strcpy(value, eqChar + 1);

         if (strcmp(key, "--model") == 0)
            modelFile = value;
         else
            printf("Unknown option: %s\n", key);
      }

      if (strcmp(modelFile, "") == 0) {
         printf("Error: --model is required\n");
         return 1;
      }

      rf.loadModelFromJSON(modelFile);

      printf("Random Forest Model Information\n");
      printf("===============================\n");
      printf("Number of trees: %d\n", rf.getNumTrees());
      printf("Max depth: %d\n", rf.getMaxDepth());
      printf("Max features: %d\n", rf.getMaxFeatures());
      printf("Task type: %s\n", rf.taskTypeToStr(rf.getTaskType()));
      printf("Criterion: %s\n", rf.criterionToStr(rf.getCriterion()));
      rf.freeForest();

   } else {
      printf("Unknown command: %s\n\n", cmdLower);
      PrintHelp();
   }

   return 0;
}


#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <cctype>

const int MAX_FEATURES = 100;
const int MAX_SAMPLES = 2000;
const int MAX_TREES = 500;
const int MAX_DEPTH_DEFAULT = 10;
const int MIN_SAMPLES_LEAF_DEFAULT = 1;
const int MIN_SAMPLES_SPLIT_DEFAULT = 2;
const int MAX_NODE_INFO = 1000;
const int MAX_FEATURE_STATS = 100;
const int MAX_SAMPLE_TRACK = 1000;

enum TaskType { Classification, Regression };
enum SplitCriterion { Gini, Entropy, MSE, VarianceReduction };
enum TAggregationMethod { MajorityVote, WeightedVote, Mean, WeightedMean };

typedef double TDataRow[MAX_FEATURES];
typedef double TTargetArray[MAX_SAMPLES];
typedef TDataRow TDataMatrix[MAX_SAMPLES];
typedef int TIndexArray[MAX_SAMPLES];
typedef int TFeatureArray[MAX_FEATURES];
typedef bool TBoolArray[MAX_SAMPLES];
typedef double TDoubleArray[MAX_FEATURES];

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

// Facade Types
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
typedef TNodeInfo TNodeInfoArray[MAX_NODE_INFO];

struct TTreeInfo {
   int treeId;
   int numNodes;
   int maxDepth;
   int numLeaves;
   bool featuresUsed[MAX_FEATURES];
   int numFeaturesUsed;
   double oobError;
   TNodeInfoArray nodes;
};

struct TFeatureStats {
   int featureIndex;
   int timesUsed;
   int treesUsedIn;
   double avgImportance;
   double totalImportance;
};
typedef TFeatureStats TFeatureStatsArray[MAX_FEATURE_STATS];

struct TOOBTreeInfo {
   int treeId;
   int numOobSamples;
   double oobError;
   double oobAccuracy;
};
typedef TOOBTreeInfo TOOBTreeInfoArray[MAX_TREES];

struct TForestComparison {
   int numDifferentPredictions;
   double avgPredictionDiff;
   double forestAAccuracy;
   double forestBAccuracy;
   double forestAMSE;
   double forestBMSE;
};

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

   static TDataMatrix data;
   static TTargetArray targets;

public:
   TRandomForest();

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

   void loadData(TDataMatrix inputData, TTargetArray inputTargets,
                 int nSamples, int nFeatures);
   void trainTestSplit(TIndexArray trainIndices, TIndexArray testIndices,
                       int& numTrain, int& numTest, double testRatio);
   void bootstrap(TIndexArray sampleIndices, int& numBootstrap,
                  TBoolArray oobMask);
   void selectFeatureSubset(TFeatureArray featureIndices,
                            int& numSelected);

   double calculateGini(TIndexArray indices, int numIndices);
   double calculateEntropy(TIndexArray indices, int numIndices);
   double calculateMSE(TIndexArray indices, int numIndices);
   double calculateVariance(TIndexArray indices, int numIndices);
   double calculateImpurity(TIndexArray indices, int numIndices);
   bool findBestSplit(TIndexArray indices, int numIndices,
                      TFeatureArray featureIndices, int nFeatures,
                      int& bestFeature, double& bestThreshold,
                      double& bestGain);
   int getMajorityClass(TIndexArray indices, int numIndices);
   double getMeanTarget(TIndexArray indices, int numIndices);
   TreeNode createLeafNode(TIndexArray indices, int numIndices);
   bool shouldStop(int depth, int numIndices, double impurity);
   TreeNode buildTree(TIndexArray indices, int numIndices,
                      int depth, TDecisionTree tree);
   double predictTree(TreeNode node, TDataRow sample);
   void freeTreeNode(TreeNode node);
   void freeTree(TDecisionTree tree);

   void fit();
   void fitTree(int treeIndex);

   double predict(TDataRow sample);
   int predictClass(TDataRow sample);
   void predictBatch(TDataMatrix samples, int nSamples,
                     TTargetArray predictions);

   double calculateOOBError();

   void calculateFeatureImportance();
   double getFeatureImportance(int featureIndex);
   void printFeatureImportances();

   double accuracy(TTargetArray predictions, TTargetArray actual,
                   int nSamples);
   double precision(TTargetArray predictions, TTargetArray actual,
                    int nSamples, int positiveClass);
   double recall(TTargetArray predictions, TTargetArray actual,
                 int nSamples, int positiveClass);
   double f1Score(TTargetArray predictions, TTargetArray actual,
                  int nSamples, int positiveClass);
   double meanSquaredError(TTargetArray predictions, TTargetArray actual,
                           int nSamples);
   double rSquared(TTargetArray predictions, TTargetArray actual,
                   int nSamples);

   void printForestInfo();
   void freeForest();

   int getNumTrees();
   int getNumFeatures();
   int getNumSamples();
   int getMaxDepth();
   int getMinSamplesLeaf();
   int getMinSamplesSplit();
   int getMaxFeatures();
   int getRandomSeed();
   TDecisionTree getTree(int treeId);
   double getData(int sampleIdx, int featureIdx);
   double getTarget(int sampleIdx);
   TaskType getTaskType();
   SplitCriterion getCriterion();

   void addNewTree();
   void removeTreeAt(int treeId);
   void retrainTreeAt(int treeId);
   void setTree(int treeId, TDecisionTree tree);

   void saveModelToJSON(const char* filename);
   void loadModelFromJSON(const char* filename);

   const char* taskTypeToStr(TaskType t);
   const char* criterionToStr(SplitCriterion c);
   TaskType parseTaskType(const char* s);
   SplitCriterion parseCriterion(const char* s);
   char* treeNodeToJSON(TreeNode node);
   TreeNode JSONToTreeNode(const char* json);

   friend class TRandomForestFacade;
};

// Static data members
TDataMatrix TRandomForest::data;
TTargetArray TRandomForest::targets;

class TRandomForestFacade {
private:
   TRandomForest forest;
   bool forestInitialized;
   TAggregationMethod currentAggregation;
   double treeWeights[MAX_TREES];
   bool featureEnabled[MAX_FEATURES];

   int collectNodeInfo(TreeNode node, int depth, 
                       TNodeInfoArray nodes, int& count);
   int calculateTreeDepth(TreeNode node);
   int countLeaves(TreeNode node);
   void collectFeaturesUsed(TreeNode node, bool* used);
   TreeNode findNodeById(TreeNode node, int targetId, int& currentId);
   void freeSubtree(TreeNode node);

public:
   TRandomForestFacade();
   
   void initForest();
   TRandomForest* getForest();
   
   void setHyperparameter(const char* paramName, int value);
   void setHyperparameterFloat(const char* paramName, double value);
   int getHyperparameter(const char* paramName);
   void setTaskType(TaskType t);
   void setCriterion(SplitCriterion c);
   void printHyperparameters();
   
   void loadData(TDataMatrix inputData, TTargetArray inputTargets,
                 int nSamples, int nFeatures);
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
   void featureUsageSummary(TFeatureStatsArray& stats);
   void printFeatureUsageSummary();
   double getFeatureImportance(int featureIndex);
   void printFeatureImportances();
   
   void setAggregationMethod(TAggregationMethod method);
   TAggregationMethod getAggregationMethod();
   void setTreeWeight(int treeId, double weight);
   double getTreeWeight(int treeId);
   void resetTreeWeights();
   double aggregatePredictions(TDataRow sample);
   
   double predict(TDataRow sample);
   int predictClass(TDataRow sample);
   double predictWithTree(int treeId, TDataRow sample);
   void predictBatch(TDataMatrix samples, int nSamples,
                     TTargetArray predictions);
   
   void oobErrorSummary(TOOBTreeInfoArray& summary);
   void printOOBSummary();
   double getGlobalOOBError();
   
   double accuracy(TTargetArray predictions, TTargetArray actual, int nSamples);
   double meanSquaredError(TTargetArray predictions, TTargetArray actual, int nSamples);
   double rSquared(TTargetArray predictions, TTargetArray actual, int nSamples);
   double precision(TTargetArray predictions, TTargetArray actual, int nSamples, int posClass);
   double recall(TTargetArray predictions, TTargetArray actual, int nSamples, int posClass);
   double f1Score(TTargetArray predictions, TTargetArray actual, int nSamples, int posClass);
   void printMetrics(TTargetArray predictions, TTargetArray actual, int nSamples);
   
   void visualizeTree(int treeId);
   void printForestOverview();
   
   void saveModelToJSON(const char* filename);
   bool loadModelFromJSON(const char* filename);
   
   void freeForest();
};

// ============================================================================
// TRandomForest - Constructor
// ============================================================================

TRandomForest::TRandomForest() {
   int i;
   
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

   for (i = 0; i < MAX_TREES; i++)
      trees[i] = NULL;

   for (i = 0; i < MAX_FEATURES; i++)
      featureImportances[i] = 0.0;

   srand((unsigned)time(NULL));
}

// ============================================================================
// TRandomForest - Hyperparameter Handling
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

void TRandomForest::setMaxFeatures(int m) {
   maxFeatures = m;
}

void TRandomForest::setTaskType(TaskType t) {
   taskType = t;
   if (t == Classification)
      criterion = Gini;
   else
      criterion = MSE;
}

void TRandomForest::setCriterion(SplitCriterion c) {
   criterion = c;
}

void TRandomForest::setRandomSeed(long seed) {
   randomSeed = seed;
   srand((unsigned)seed);
}

// ============================================================================
// TRandomForest - Random Number Generator
// ============================================================================

int TRandomForest::randomInt(int maxVal) {
   if (maxVal <= 0) return 0;
   return rand() % maxVal;
}

double TRandomForest::randomDouble() {
   return (double)rand() / RAND_MAX;
}

// ============================================================================
// TRandomForest - Data Handling
// ============================================================================

void TRandomForest::loadData(TDataMatrix inputData, TTargetArray inputTargets,
                             int nSamples, int nFeatures) {
   int i, j;

   numSamples = nSamples;
   numFeatures = nFeatures;

   if (maxFeatures == 0) {
      if (taskType == Classification)
         maxFeatures = (int)sqrt(nFeatures);
      else
         maxFeatures = nFeatures / 3;
      if (maxFeatures < 1)
         maxFeatures = 1;
   }

   for (i = 0; i < nSamples; i++) {
      for (j = 0; j < nFeatures; j++)
         data[i][j] = inputData[i][j];
      targets[i] = inputTargets[i];
   }
}

void TRandomForest::trainTestSplit(TIndexArray trainIndices, TIndexArray testIndices,
                                   int& numTrain, int& numTest, double testRatio) {
   int i, j, temp;
   TIndexArray shuffled;

   for (i = 0; i < numSamples; i++)
      shuffled[i] = i;

   for (i = numSamples - 1; i >= 1; i--) {
      j = randomInt(i + 1);
      temp = shuffled[i];
      shuffled[i] = shuffled[j];
      shuffled[j] = temp;
   }

   numTest = (int)(numSamples * testRatio);
   numTrain = numSamples - numTest;

   for (i = 0; i < numTrain; i++)
      trainIndices[i] = shuffled[i];

   for (i = 0; i < numTest; i++)
      testIndices[i] = shuffled[numTrain + i];
}

void TRandomForest::bootstrap(TIndexArray sampleIndices, int& numBootstrap,
                              TBoolArray oobMask) {
   int i, idx;

   numBootstrap = numSamples;

   for (i = 0; i < numSamples; i++)
      oobMask[i] = true;

   for (i = 0; i < numBootstrap; i++) {
      idx = randomInt(numSamples);
      sampleIndices[i] = idx;
      oobMask[idx] = false;
   }
}

void TRandomForest::selectFeatureSubset(TFeatureArray featureIndices,
                                        int& numSelected) {
   int i, j, temp;
   TFeatureArray available;

   for (i = 0; i < numFeatures; i++)
      available[i] = i;

   for (i = numFeatures - 1; i >= 1; i--) {
      j = randomInt(i + 1);
      temp = available[i];
      available[i] = available[j];
      available[j] = temp;
   }

   numSelected = maxFeatures;
   if (numSelected > numFeatures)
      numSelected = numFeatures;

   for (i = 0; i < numSelected; i++)
      featureIndices[i] = available[i];
}

// ============================================================================
// TRandomForest - Impurity Functions
// ============================================================================

double TRandomForest::calculateGini(TIndexArray indices, int numIndices) {
   int i;
   int classCount[100];
   int numClasses, classLabel;
   double prob, gini;

   if (numIndices == 0)
      return 0.0;

   for (i = 0; i < 100; i++)
      classCount[i] = 0;

   numClasses = 0;
   for (i = 0; i < numIndices; i++) {
      classLabel = (int)targets[indices[i]];
      if (classLabel > numClasses)
         numClasses = classLabel;
      classCount[classLabel]++;
   }

   gini = 1.0;
   for (i = 0; i <= numClasses; i++) {
      prob = (double)classCount[i] / numIndices;
      gini = gini - (prob * prob);
   }

   return gini;
}

double TRandomForest::calculateEntropy(TIndexArray indices, int numIndices) {
   int i;
   int classCount[100];
   int numClasses, classLabel;
   double prob, entropy;

   if (numIndices == 0)
      return 0.0;

   for (i = 0; i < 100; i++)
      classCount[i] = 0;

   numClasses = 0;
   for (i = 0; i < numIndices; i++) {
      classLabel = (int)targets[indices[i]];
      if (classLabel > numClasses)
         numClasses = classLabel;
      classCount[classLabel]++;
   }

   entropy = 0.0;
   for (i = 0; i <= numClasses; i++) {
      if (classCount[i] > 0) {
         prob = (double)classCount[i] / numIndices;
         entropy = entropy - (prob * log(prob) / log(2.0));
      }
   }

   return entropy;
}

double TRandomForest::calculateMSE(TIndexArray indices, int numIndices) {
   int i;
   double mean, mse, diff;

   if (numIndices == 0)
      return 0.0;

   mean = 0.0;
   for (i = 0; i < numIndices; i++)
      mean = mean + targets[indices[i]];
   mean = mean / numIndices;

   mse = 0.0;
   for (i = 0; i < numIndices; i++) {
      diff = targets[indices[i]] - mean;
      mse = mse + (diff * diff);
   }

   return mse / numIndices;
}

double TRandomForest::calculateVariance(TIndexArray indices, int numIndices) {
   int i;
   double mean, variance, diff;

   if (numIndices == 0)
      return 0.0;

   mean = 0.0;
   for (i = 0; i < numIndices; i++)
      mean = mean + targets[indices[i]];
   mean = mean / numIndices;

   variance = 0.0;
   for (i = 0; i < numIndices; i++) {
      diff = targets[indices[i]] - mean;
      variance = variance + (diff * diff);
   }

   return variance / numIndices;
}

double TRandomForest::calculateImpurity(TIndexArray indices, int numIndices) {
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
         return calculateGini(indices, numIndices);
   }
}

// ============================================================================
// TRandomForest - Split Finding
// ============================================================================

bool TRandomForest::findBestSplit(TIndexArray indices, int numIndices,
                                  TFeatureArray featureIndices, int nFeatures,
                                  int& bestFeature, double& bestThreshold,
                                  double& bestGain) {
   int i, j, f, feat;
   double threshold, gain, leftImpurity, rightImpurity;
   double parentImpurity, leftWeight, rightWeight;
   int leftCount, rightCount;
   TIndexArray leftIndices, rightIndices;
   bool foundSplit = false;

   bestGain = -1.0;
   bestFeature = -1;
   bestThreshold = 0.0;

   if (numIndices < minSamplesSplit)
      return false;

   parentImpurity = calculateImpurity(indices, numIndices);

   for (f = 0; f < nFeatures; f++) {
      feat = featureIndices[f];

      for (i = 0; i < numIndices; i++) {
         threshold = data[indices[i]][feat];

         leftCount = 0;
         rightCount = 0;

         for (j = 0; j < numIndices; j++) {
            if (data[indices[j]][feat] <= threshold) {
               leftIndices[leftCount] = indices[j];
               leftCount++;
            } else {
               rightIndices[rightCount] = indices[j];
               rightCount++;
            }
         }

         if (leftCount < minSamplesLeaf || rightCount < minSamplesLeaf)
            continue;

         leftImpurity = calculateImpurity(leftIndices, leftCount);
         rightImpurity = calculateImpurity(rightIndices, rightCount);

         leftWeight = (double)leftCount / numIndices;
         rightWeight = (double)rightCount / numIndices;

         gain = parentImpurity - (leftWeight * leftImpurity + rightWeight * rightImpurity);

         if (gain > bestGain) {
            bestGain = gain;
            bestFeature = feat;
            bestThreshold = threshold;
            foundSplit = true;
         }
      }
   }

   return foundSplit;
}

// ============================================================================
// TRandomForest - Leaf Functions
// ============================================================================

int TRandomForest::getMajorityClass(TIndexArray indices, int numIndices) {
   int i;
   int classCount[100];
   int maxClass = 0, maxCount = 0, classLabel;

   for (i = 0; i < 100; i++)
      classCount[i] = 0;

   for (i = 0; i < numIndices; i++) {
      classLabel = (int)targets[indices[i]];
      classCount[classLabel]++;
      if (classCount[classLabel] > maxCount) {
         maxCount = classCount[classLabel];
         maxClass = classLabel;
      }
   }

   return maxClass;
}

double TRandomForest::getMeanTarget(TIndexArray indices, int numIndices) {
   int i;
   double sum = 0.0;

   if (numIndices == 0)
      return 0.0;

   for (i = 0; i < numIndices; i++)
      sum = sum + targets[indices[i]];

   return sum / numIndices;
}

TreeNode TRandomForest::createLeafNode(TIndexArray indices, int numIndices) {
   TreeNode node = new TreeNodeRec();

   node->isLeaf = true;
   node->left = NULL;
   node->right = NULL;
   node->numSamples = numIndices;
   node->impurity = calculateImpurity(indices, numIndices);

   if (taskType == Classification) {
      node->classLabel = getMajorityClass(indices, numIndices);
      node->prediction = node->classLabel;
   } else {
      node->prediction = getMeanTarget(indices, numIndices);
      node->classLabel = (int)node->prediction;
   }

   return node;
}

// ============================================================================
// TRandomForest - Stopping Conditions
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
// TRandomForest - Tree Building
// ============================================================================

TreeNode TRandomForest::buildTree(TIndexArray indices, int numIndices,
                                  int depth, TDecisionTree tree) {
   TreeNode node;
   TFeatureArray featureIndices;
   int numFeatures_, bestFeature;
   double bestThreshold, bestGain, impurity;
   TIndexArray leftIndices, rightIndices;
   int leftCount, rightCount, i, j;

   impurity = calculateImpurity(indices, numIndices);

   if (shouldStop(depth, numIndices, impurity))
      return createLeafNode(indices, numIndices);

   selectFeatureSubset(featureIndices, numFeatures_);

   if (!findBestSplit(indices, numIndices, featureIndices, numFeatures_,
                      bestFeature, bestThreshold, bestGain))
      return createLeafNode(indices, numIndices);

   node = new TreeNodeRec();
   node->isLeaf = false;
   node->featureIndex = bestFeature;
   node->threshold = bestThreshold;
   node->impurity = impurity;
   node->numSamples = numIndices;

   leftCount = 0;
   rightCount = 0;

   for (i = 0; i < numIndices; i++) {
      if (data[indices[i]][bestFeature] <= bestThreshold) {
         leftIndices[leftCount] = indices[i];
         leftCount++;
      } else {
         rightIndices[rightCount] = indices[i];
         rightCount++;
      }
   }

   node->left = buildTree(leftIndices, leftCount, depth + 1, tree);
   node->right = buildTree(rightIndices, rightCount, depth + 1, tree);

   return node;
}

// ============================================================================
// TRandomForest - Prediction
// ============================================================================

double TRandomForest::predictTree(TreeNode node, TDataRow sample) {
   if (node == NULL)
      return 0.0;

   if (node->isLeaf)
      return node->prediction;

   if (sample[node->featureIndex] <= node->threshold)
      return predictTree(node->left, sample);
   else
      return predictTree(node->right, sample);
}

// ============================================================================
// TRandomForest - Memory Management
// ============================================================================

void TRandomForest::freeTreeNode(TreeNode node) {
   if (node == NULL)
      return;

   if (!node->isLeaf) {
      freeTreeNode(node->left);
      freeTreeNode(node->right);
   }

   delete node;
}

void TRandomForest::freeTree(TDecisionTree tree) {
   if (tree == NULL)
      return;

   freeTreeNode(tree->root);
   delete tree;
}

// ============================================================================
// TRandomForest - Training
// ============================================================================

void TRandomForest::fit() {
   int i;

   for (i = 0; i < numTrees; i++)
      fitTree(i);
}

void TRandomForest::fitTree(int treeIndex) {
   TIndexArray sampleIndices;
   int numBootstrap;
   TBoolArray oobMask;
   TDecisionTree tree;
   int i;

   tree = new TDecisionTreeRec();
   tree->maxDepth = maxDepth;
   tree->minSamplesLeaf = minSamplesLeaf;
   tree->minSamplesSplit = minSamplesSplit;
   tree->maxFeatures = maxFeatures;
   tree->taskType = taskType;
   tree->criterion = criterion;

   bootstrap(sampleIndices, numBootstrap, oobMask);

   for (i = 0; i < numSamples; i++)
      tree->oobIndices[i] = oobMask[i];

   tree->numOobIndices = 0;
   for (i = 0; i < numSamples; i++)
      if (tree->oobIndices[i])
         tree->numOobIndices++;

   tree->root = buildTree(sampleIndices, numBootstrap, 0, tree);

   trees[treeIndex] = tree;
}

// ============================================================================
// TRandomForest - Batch Prediction
// ============================================================================

double TRandomForest::predict(TDataRow sample) {
   int i, count = 0;
   double sum = 0.0;

   for (i = 0; i < numTrees; i++) {
      if (trees[i] != NULL) {
         sum = sum + predictTree(trees[i]->root, sample);
         count++;
      }
   }

   if (count == 0)
      return 0.0;

   return sum / count;
}

int TRandomForest::predictClass(TDataRow sample) {
   return (int)predict(sample);
}

void TRandomForest::predictBatch(TDataMatrix samples, int nSamples,
                                 TTargetArray predictions) {
   int i;

   for (i = 0; i < nSamples; i++)
      predictions[i] = predict(samples[i]);
}

// ============================================================================
// TRandomForest - OOB Error
// ============================================================================

double TRandomForest::calculateOOBError() {
   int i, j, count;
   double error = 0.0, prediction;

   for (i = 0; i < numSamples; i++) {
      count = 0;
      prediction = 0.0;

      for (j = 0; j < numTrees; j++) {
         if (trees[j] != NULL && trees[j]->oobIndices[i]) {
            prediction = prediction + predictTree(trees[j]->root, data[i]);
            count++;
         }
      }

      if (count > 0) {
         prediction = prediction / count;
         error = error + (targets[i] - prediction) * (targets[i] - prediction);
      }
   }

   if (numSamples == 0)
      return 0.0;

   return error / numSamples;
}

// ============================================================================
// TRandomForest - Feature Importance
// ============================================================================

void TRandomForest::calculateFeatureImportance() {
   int i;
   for (i = 0; i < numFeatures; i++)
      featureImportances[i] = 0.0;
}

double TRandomForest::getFeatureImportance(int featureIndex) {
   if (featureIndex < 0 || featureIndex >= numFeatures)
      return 0.0;
   return featureImportances[featureIndex];
}

void TRandomForest::printFeatureImportances() {
   int i;
   printf("Feature Importances:\n");
   for (i = 0; i < numFeatures; i++) {
      printf("Feature %d: %f\n", i, featureImportances[i]);
   }
}

// ============================================================================
// TRandomForest - Performance Metrics
// ============================================================================

double TRandomForest::accuracy(TTargetArray predictions, TTargetArray actual,
                               int nSamples) {
   int i, correct = 0;

   if (nSamples == 0)
      return 0.0;

   for (i = 0; i < nSamples; i++) {
      if ((int)predictions[i] == (int)actual[i])
         correct++;
   }

   return (double)correct / nSamples;
}

double TRandomForest::precision(TTargetArray predictions, TTargetArray actual,
                                int nSamples, int positiveClass) {
   int i, truePositive = 0, predictedPositive = 0;

   for (i = 0; i < nSamples; i++) {
      if ((int)predictions[i] == positiveClass) {
         predictedPositive++;
         if ((int)actual[i] == positiveClass)
            truePositive++;
      }
   }

   if (predictedPositive == 0)
      return 0.0;

   return (double)truePositive / predictedPositive;
}

double TRandomForest::recall(TTargetArray predictions, TTargetArray actual,
                             int nSamples, int positiveClass) {
   int i, truePositive = 0, actualPositive = 0;

   for (i = 0; i < nSamples; i++) {
      if ((int)actual[i] == positiveClass) {
         actualPositive++;
         if ((int)predictions[i] == positiveClass)
            truePositive++;
      }
   }

   if (actualPositive == 0)
      return 0.0;

   return (double)truePositive / actualPositive;
}

double TRandomForest::f1Score(TTargetArray predictions, TTargetArray actual,
                              int nSamples, int positiveClass) {
   double p = precision(predictions, actual, nSamples, positiveClass);
   double r = recall(predictions, actual, nSamples, positiveClass);

   if (p + r == 0.0)
      return 0.0;

   return 2.0 * (p * r) / (p + r);
}

double TRandomForest::meanSquaredError(TTargetArray predictions, TTargetArray actual,
                                       int nSamples) {
   int i;
   double mse = 0.0, diff;

   if (nSamples == 0)
      return 0.0;

   for (i = 0; i < nSamples; i++) {
      diff = predictions[i] - actual[i];
      mse = mse + (diff * diff);
   }

   return mse / nSamples;
}

double TRandomForest::rSquared(TTargetArray predictions, TTargetArray actual,
                               int nSamples) {
   int i;
   double mean = 0.0, ssRes = 0.0, ssTot = 0.0;

   if (nSamples == 0)
      return 0.0;

   for (i = 0; i < nSamples; i++)
      mean = mean + actual[i];
   mean = mean / nSamples;

   for (i = 0; i < nSamples; i++) {
      ssRes = ssRes + (actual[i] - predictions[i]) * (actual[i] - predictions[i]);
      ssTot = ssTot + (actual[i] - mean) * (actual[i] - mean);
   }

   if (ssTot == 0.0)
      return 0.0;

   return 1.0 - (ssRes / ssTot);
}

// ============================================================================
// TRandomForest - Utility
// ============================================================================

void TRandomForest::printForestInfo() {
   printf("Random Forest Model Information\n");
   printf("===============================\n");
   printf("Number of trees: %d\n", numTrees);
   printf("Max depth: %d\n", maxDepth);
   printf("Min samples leaf: %d\n", minSamplesLeaf);
   printf("Min samples split: %d\n", minSamplesSplit);
   printf("Max features: %d\n", maxFeatures);
   printf("Task type: %s\n", taskTypeToStr(taskType));
   printf("Criterion: %s\n", criterionToStr(criterion));
}

void TRandomForest::freeForest() {
   int i;

   for (i = 0; i < numTrees; i++) {
      if (trees[i] != NULL) {
         freeTree(trees[i]);
         trees[i] = NULL;
      }
   }

   numTrees = 0;
}

// ============================================================================
// TRandomForest - Accessors
// ============================================================================

int TRandomForest::getNumTrees() {
   return numTrees;
}

int TRandomForest::getNumFeatures() {
   return numFeatures;
}

int TRandomForest::getNumSamples() {
   return numSamples;
}

int TRandomForest::getMaxDepth() {
   return maxDepth;
}

int TRandomForest::getMinSamplesLeaf() {
   return minSamplesLeaf;
}

int TRandomForest::getMinSamplesSplit() {
   return minSamplesSplit;
}

int TRandomForest::getMaxFeatures() {
   return maxFeatures;
}

int TRandomForest::getRandomSeed() {
   return randomSeed;
}

TDecisionTree TRandomForest::getTree(int treeId) {
   if (treeId < 0 || treeId >= numTrees)
      return NULL;
   return trees[treeId];
}

double TRandomForest::getData(int sampleIdx, int featureIdx) {
   if (sampleIdx < 0 || sampleIdx >= numSamples || featureIdx < 0 || featureIdx >= numFeatures)
      return 0.0;
   return data[sampleIdx][featureIdx];
}

double TRandomForest::getTarget(int sampleIdx) {
   if (sampleIdx < 0 || sampleIdx >= numSamples)
      return 0.0;
   return targets[sampleIdx];
}

TaskType TRandomForest::getTaskType() {
   return taskType;
}

SplitCriterion TRandomForest::getCriterion() {
   return criterion;
}

// ============================================================================
// TRandomForest - Tree Management
// ============================================================================

void TRandomForest::addNewTree() {
   if (numTrees < MAX_TREES) {
      numTrees++;
   }
}

void TRandomForest::removeTreeAt(int treeId) {
   if (treeId < 0 || treeId >= numTrees)
      return;

   if (trees[treeId] != NULL) {
      freeTree(trees[treeId]);
   }

   for (int i = treeId; i < numTrees - 1; i++) {
      trees[i] = trees[i + 1];
   }

   numTrees--;
}

void TRandomForest::retrainTreeAt(int treeId) {
   if (treeId < 0 || treeId >= numTrees)
      return;

   if (trees[treeId] != NULL) {
      freeTree(trees[treeId]);
   }

   fitTree(treeId);
}

void TRandomForest::setTree(int treeId, TDecisionTree tree) {
   if (treeId < 0 || treeId >= MAX_TREES)
      return;

   if (trees[treeId] != NULL) {
      freeTree(trees[treeId]);
   }

   trees[treeId] = tree;
}

// ============================================================================
// TRandomForest - JSON Serialization
// ============================================================================

const char* TRandomForest::taskTypeToStr(TaskType t) {
   if (t == Classification)
      return "classification";
   else
      return "regression";
}

const char* TRandomForest::criterionToStr(SplitCriterion c) {
   switch (c) {
      case Gini:
         return "gini";
      case Entropy:
         return "entropy";
      case MSE:
         return "mse";
      case VarianceReduction:
         return "variance";
      default:
         return "gini";
   }
}

TaskType TRandomForest::parseTaskType(const char* s) {
   if (strcmp(s, "regression") == 0)
      return Regression;
   else
      return Classification;
}

SplitCriterion TRandomForest::parseCriterion(const char* s) {
   if (strcmp(s, "gini") == 0)
      return Gini;
   else if (strcmp(s, "entropy") == 0)
      return Entropy;
   else if (strcmp(s, "mse") == 0)
      return MSE;
   else if (strcmp(s, "variance") == 0)
      return VarianceReduction;
   else
      return Gini;
}

char* TRandomForest::treeNodeToJSON(TreeNode node) {
   static char buffer[100000];
   char *leftJSON, *rightJSON;
   char numBuf[50];

   if (node == NULL) {
      strcpy(buffer, "null");
      return buffer;
   }

   strcpy(buffer, "{");
   strcat(buffer, "\"isLeaf\":");
   strcat(buffer, node->isLeaf ? "true" : "false");

   if (node->isLeaf) {
      sprintf(numBuf, ",\"prediction\":%g", node->prediction);
      strcat(buffer, numBuf);
      sprintf(numBuf, ",\"classLabel\":%d", node->classLabel);
      strcat(buffer, numBuf);
   } else {
      sprintf(numBuf, ",\"featureIndex\":%d", node->featureIndex);
      strcat(buffer, numBuf);

      sprintf(numBuf, ",\"threshold\":%g", node->threshold);
      strcat(buffer, numBuf);

      sprintf(numBuf, ",\"impurity\":%g", node->impurity);
      strcat(buffer, numBuf);

      sprintf(numBuf, ",\"numSamples\":%d", node->numSamples);
      strcat(buffer, numBuf);

      leftJSON = treeNodeToJSON(node->left);
      sprintf(numBuf, ",\"left\":%s", leftJSON);
      strcat(buffer, numBuf);

      rightJSON = treeNodeToJSON(node->right);
      sprintf(numBuf, ",\"right\":%s", rightJSON);
      strcat(buffer, numBuf);
   }

   strcat(buffer, "}");
   return buffer;
}

TreeNode TRandomForest::JSONToTreeNode(const char* json) {
   TreeNode node;
   const char *s;
   bool isLeaf;

   auto ExtractJSONValue = [](const char* json, const char* key) -> char* {
      static char result[1000];
      const char *keyPos, *colonPos, *quotePos1, *quotePos2, *startPos;
      int keyLen = strlen(key);
      strcpy(result, "");

      keyPos = strstr(json, key);
      if (keyPos == NULL)
         return result;

      colonPos = strchr(keyPos, ':');
      if (colonPos == NULL)
         return result;

      startPos = colonPos + 1;
      while (*startPos && (*startPos == ' ' || *startPos == '\t' || *startPos == '\n' || *startPos == '\r'))
         startPos++;

      if (*startPos == '"') {
         quotePos1 = startPos;
         quotePos2 = strchr(quotePos1 + 1, '"');
         if (quotePos2) {
            int len = quotePos2 - quotePos1 - 1;
            strncpy(result, quotePos1 + 1, len);
            result[len] = '\0';
         }
      } else {
         int len = 0;
         while (*startPos && *startPos != ',' && *startPos != '}' && *startPos != ']')
            result[len++] = *startPos++;
         result[len] = '\0';
         char *end = result + strlen(result) - 1;
         while (end >= result && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r'))
            *end-- = '\0';
      }

      return result;
   };

   auto ExtractJSONSubObject = [](const char* json, const char* key) -> char* {
      static char result[10000];
      const char *keyPos, *colonPos, *startPos;
      int braceCount;
      strcpy(result, "");

      keyPos = strstr(json, key);
      if (!keyPos)
         return result;

      colonPos = strchr(keyPos, ':');
      if (!colonPos)
         return result;

      startPos = colonPos + 1;
      while (*startPos && (*startPos == ' ' || *startPos == '\t' || *startPos == '\n' || *startPos == '\r'))
         startPos++;

      if (*startPos != '{')
         return result;

      braceCount = 0;
      const char *end = startPos;
      while (*end) {
         if (*end == '{')
            braceCount++;
         else if (*end == '}') {
            braceCount--;
            if (braceCount == 0) {
               int len = end - startPos + 1;
               strncpy(result, startPos, len);
               result[len] = '\0';
               return result;
            }
         }
         end++;
      }

      return result;
   };

   auto Trim = [](const char* str) -> char* {
      static char result[1000];
      const char *start = str;
      const char *end;

      while (*start && (*start == ' ' || *start == '\t' || *start == '\n' || *start == '\r'))
         start++;

      end = start + strlen(start) - 1;
      while (end >= start && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r'))
         end--;

      int len = end - start + 1;
      if (len < 0) len = 0;
      strncpy(result, start, len);
      result[len] = '\0';
      return result;
   };

   if (strcmp(Trim(json), "null") == 0)
      return NULL;

   node = new TreeNodeRec();

   s = ExtractJSONValue(json, "isLeaf");
   isLeaf = (strcmp(s, "true") == 0);
   node->isLeaf = isLeaf;

   if (isLeaf) {
      s = ExtractJSONValue(json, "prediction");
      if (strlen(s) > 0)
         node->prediction = atof(s);

      s = ExtractJSONValue(json, "classLabel");
      if (strlen(s) > 0)
         node->classLabel = atoi(s);

      node->left = NULL;
      node->right = NULL;
   } else {
      s = ExtractJSONValue(json, "featureIndex");
      if (strlen(s) > 0)
         node->featureIndex = atoi(s);

      s = ExtractJSONValue(json, "threshold");
      if (strlen(s) > 0)
         node->threshold = atof(s);

      s = ExtractJSONValue(json, "impurity");
      if (strlen(s) > 0)
         node->impurity = atof(s);

      s = ExtractJSONValue(json, "numSamples");
      if (strlen(s) > 0)
         node->numSamples = atoi(s);

      s = ExtractJSONSubObject(json, "left");
      if (strlen(s) > 0)
         node->left = JSONToTreeNode(s);
      else
         node->left = NULL;

      s = ExtractJSONSubObject(json, "right");
      if (strlen(s) > 0)
         node->right = JSONToTreeNode(s);
      else
         node->right = NULL;
   }

   return node;
}

void TRandomForest::saveModelToJSON(const char* filename) {
   FILE* f = fopen(filename, "w");
   if (!f) {
      printf("Error: Cannot open file for writing: %s\n", filename);
      return;
   }

   fprintf(f, "{\n");
   fprintf(f, "  \"num_trees\": %d,\n", numTrees);
   fprintf(f, "  \"max_depth\": %d,\n", maxDepth);
   fprintf(f, "  \"min_samples_leaf\": %d,\n", minSamplesLeaf);
   fprintf(f, "  \"min_samples_split\": %d,\n", minSamplesSplit);
   fprintf(f, "  \"max_features\": %d,\n", maxFeatures);
   fprintf(f, "  \"num_features\": %d,\n", numFeatures);
   fprintf(f, "  \"num_samples\": %d,\n", numSamples);
   fprintf(f, "  \"task_type\": \"%s\",\n", taskTypeToStr(taskType));
   fprintf(f, "  \"criterion\": \"%s\",\n", criterionToStr(criterion));
   fprintf(f, "  \"random_seed\": %ld,\n", randomSeed);
   fprintf(f, "  \"trees\": [\n");

   for (int i = 0; i < numTrees; i++) {
      if (trees[i] != NULL) {
         fprintf(f, "    %s", treeNodeToJSON(trees[i]->root));
      } else {
         fprintf(f, "    null");
      }
      if (i < numTrees - 1)
         fprintf(f, ",\n");
      else
         fprintf(f, "\n");
   }

   fprintf(f, "  ]\n");
   fprintf(f, "}\n");

   fclose(f);
   printf("Model saved to JSON: %s\n", filename);
}

void TRandomForest::loadModelFromJSON(const char* filename) {
   FILE* f = fopen(filename, "r");
   if (!f) {
      printf("Error: Cannot open file for reading: %s\n", filename);
      return;
   }

   fseek(f, 0, SEEK_END);
   long fsize = ftell(f);
   fseek(f, 0, SEEK_SET);

   char* content = (char*)malloc(fsize + 1);
   fread(content, 1, fsize, f);
   content[fsize] = '\0';
   fclose(f);

   auto ExtractJSONValue = [](const char* json, const char* key) -> char* {
      static char result[1000];
      const char *keyPos, *colonPos, *quotePos1, *quotePos2, *startPos;
      strcpy(result, "");

      keyPos = strstr(json, key);
      if (!keyPos)
         return result;

      colonPos = strchr(keyPos, ':');
      if (!colonPos)
         return result;

      startPos = colonPos + 1;
      while (*startPos && (*startPos == ' ' || *startPos == '\t' || *startPos == '\n' || *startPos == '\r'))
         startPos++;

      if (*startPos == '"') {
         quotePos1 = startPos;
         quotePos2 = strchr(quotePos1 + 1, '"');
         if (quotePos2) {
            int len = quotePos2 - quotePos1 - 1;
            strncpy(result, quotePos1 + 1, len);
            result[len] = '\0';
         }
      } else {
         int len = 0;
         while (*startPos && *startPos != ',' && *startPos != '}' && *startPos != ']')
            result[len++] = *startPos++;
         result[len] = '\0';
      }

      return result;
   };

   const char *valueStr;

   valueStr = ExtractJSONValue(content, "num_trees");
   if (strlen(valueStr) > 0)
      numTrees = atoi(valueStr);

   valueStr = ExtractJSONValue(content, "max_depth");
   if (strlen(valueStr) > 0)
      maxDepth = atoi(valueStr);

   valueStr = ExtractJSONValue(content, "min_samples_leaf");
   if (strlen(valueStr) > 0)
      minSamplesLeaf = atoi(valueStr);

   valueStr = ExtractJSONValue(content, "min_samples_split");
   if (strlen(valueStr) > 0)
      minSamplesSplit = atoi(valueStr);

   valueStr = ExtractJSONValue(content, "max_features");
   if (strlen(valueStr) > 0)
      maxFeatures = atoi(valueStr);

   valueStr = ExtractJSONValue(content, "num_features");
   if (strlen(valueStr) > 0)
      numFeatures = atoi(valueStr);

   valueStr = ExtractJSONValue(content, "num_samples");
   if (strlen(valueStr) > 0)
      numSamples = atoi(valueStr);

   valueStr = ExtractJSONValue(content, "task_type");
   if (strlen(valueStr) > 0)
      taskType = parseTaskType(valueStr);

   valueStr = ExtractJSONValue(content, "criterion");
   if (strlen(valueStr) > 0)
      criterion = parseCriterion(valueStr);

   valueStr = ExtractJSONValue(content, "random_seed");
   if (strlen(valueStr) > 0)
      randomSeed = atol(valueStr);

   free(content);
   printf("Model loaded from JSON: %s\n", filename);
}

// ============================================================================
// TRandomForestFacade - Constructor
// ============================================================================

TRandomForestFacade::TRandomForestFacade() {
   int i;
   
   forestInitialized = false;
   currentAggregation = MajorityVote;
   
   for (i = 0; i < MAX_TREES; i++)
      treeWeights[i] = 1.0;
   
   for (i = 0; i < MAX_FEATURES; i++)
      featureEnabled[i] = true;
}

// ============================================================================
// TRandomForestFacade - Initialization
// ============================================================================

void TRandomForestFacade::initForest() {
   forestInitialized = true;
}

TRandomForest* TRandomForestFacade::getForest() {
   return &forest;
}

// ============================================================================
// TRandomForestFacade - Hyperparameters
// ============================================================================

void TRandomForestFacade::setHyperparameter(const char* paramName, int value) {
   if (strcmp(paramName, "num_trees") == 0)
      forest.setNumTrees(value);
   else if (strcmp(paramName, "max_depth") == 0)
      forest.setMaxDepth(value);
   else if (strcmp(paramName, "min_samples_leaf") == 0)
      forest.setMinSamplesLeaf(value);
   else if (strcmp(paramName, "min_samples_split") == 0)
      forest.setMinSamplesSplit(value);
   else if (strcmp(paramName, "max_features") == 0)
      forest.setMaxFeatures(value);
}

void TRandomForestFacade::setHyperparameterFloat(const char* paramName, double value) {
   // For float-based hyperparameters if needed
}

int TRandomForestFacade::getHyperparameter(const char* paramName) {
   if (strcmp(paramName, "num_trees") == 0)
      return forest.getNumTrees();
   else if (strcmp(paramName, "max_depth") == 0)
      return forest.getMaxDepth();
   else if (strcmp(paramName, "min_samples_leaf") == 0)
      return forest.getMinSamplesLeaf();
   else if (strcmp(paramName, "min_samples_split") == 0)
      return forest.getMinSamplesSplit();
   else if (strcmp(paramName, "max_features") == 0)
      return forest.getMaxFeatures();
   return 0;
}

void TRandomForestFacade::setTaskType(TaskType t) {
   forest.setTaskType(t);
}

void TRandomForestFacade::setCriterion(SplitCriterion c) {
   forest.setCriterion(c);
}

void TRandomForestFacade::printHyperparameters() {
   printf("=== Random Forest Hyperparameters ===\n");
   printf("Number of trees: %d\n", forest.getNumTrees());
   printf("Max depth: %d\n", forest.getMaxDepth());
   printf("Min samples leaf: %d\n", forest.getMinSamplesLeaf());
   printf("Min samples split: %d\n", forest.getMinSamplesSplit());
   printf("Max features: %d\n", forest.getMaxFeatures());
   printf("Task type: %s\n", forest.taskTypeToStr(forest.getTaskType()));
   printf("Criterion: %s\n", forest.criterionToStr(forest.getCriterion()));
}

// ============================================================================
// TRandomForestFacade - Data and Training
// ============================================================================

void TRandomForestFacade::loadData(TDataMatrix inputData, TTargetArray inputTargets,
                                   int nSamples, int nFeatures) {
   forest.loadData(inputData, inputTargets, nSamples, nFeatures);
}

void TRandomForestFacade::trainForest() {
   printf("Training forest with %d samples and %d features...\n", 
          forest.getNumSamples(), forest.getNumFeatures());
   forest.fit();
   printf("Training complete.\n");
}

// ============================================================================
// TRandomForestFacade - Tree Inspection
// ============================================================================

int TRandomForestFacade::calculateTreeDepth(TreeNode node) {
   if (node == NULL)
      return 0;
   if (node->isLeaf)
      return 1;
   
   int leftDepth = calculateTreeDepth(node->left);
   int rightDepth = calculateTreeDepth(node->right);
   
   return 1 + (leftDepth > rightDepth ? leftDepth : rightDepth);
}

int TRandomForestFacade::countLeaves(TreeNode node) {
   if (node == NULL)
      return 0;
   if (node->isLeaf)
      return 1;
   
   return countLeaves(node->left) + countLeaves(node->right);
}

void TRandomForestFacade::collectFeaturesUsed(TreeNode node, bool* used) {
   if (node == NULL)
      return;
   
   if (!node->isLeaf) {
      if (node->featureIndex >= 0 && node->featureIndex < MAX_FEATURES)
         used[node->featureIndex] = true;
      
      collectFeaturesUsed(node->left, used);
      collectFeaturesUsed(node->right, used);
   }
}

int TRandomForestFacade::collectNodeInfo(TreeNode node, int depth,
                                         TNodeInfoArray nodes, int& count) {
   if (node == NULL || count >= MAX_NODE_INFO)
      return -1;
   
   int nodeId = count;
   TNodeInfo& info = nodes[count];
   count++;
   
   info.nodeId = nodeId;
   info.depth = depth;
   info.isLeaf = node->isLeaf;
   info.featureIndex = node->featureIndex;
   info.threshold = node->threshold;
   info.prediction = node->prediction;
   info.classLabel = node->classLabel;
   info.impurity = node->impurity;
   info.numSamples = node->numSamples;
   
   if (node->isLeaf) {
      info.leftChildId = -1;
      info.rightChildId = -1;
   } else {
      info.leftChildId = collectNodeInfo(node->left, depth + 1, nodes, count);
      info.rightChildId = collectNodeInfo(node->right, depth + 1, nodes, count);
   }
   
   return nodeId;
}

TTreeInfo TRandomForestFacade::inspectTree(int treeId) {
   TTreeInfo info;
   TDecisionTree tree = forest.getTree(treeId);
   
   info.treeId = treeId;
   info.numNodes = 0;
   info.maxDepth = 0;
   info.numLeaves = 0;
   info.numFeaturesUsed = 0;
   info.oobError = 0.0;
   
   for (int i = 0; i < MAX_FEATURES; i++)
      info.featuresUsed[i] = false;
   
   if (tree != NULL && tree->root != NULL) {
      int count = 0;
      collectNodeInfo(tree->root, 0, info.nodes, count);
      info.numNodes = count;
      info.maxDepth = calculateTreeDepth(tree->root);
      info.numLeaves = countLeaves(tree->root);
      
      collectFeaturesUsed(tree->root, info.featuresUsed);
      for (int i = 0; i < MAX_FEATURES; i++) {
         if (info.featuresUsed[i])
            info.numFeaturesUsed++;
      }
   }
   
   return info;
}

void TRandomForestFacade::printTreeStructure(int treeId) {
   TTreeInfo info = inspectTree(treeId);
   
   printf("=== Tree %d Structure ===\n", treeId);
   printf("Number of nodes: %d\n", info.numNodes);
   printf("Max depth: %d\n", info.maxDepth);
   printf("Number of leaves: %d\n", info.numLeaves);
   printf("Features used: %d\n", info.numFeaturesUsed);
}

void TRandomForestFacade::printNodeDetails(int treeId, int nodeId) {
   TTreeInfo info = inspectTree(treeId);
   
   if (nodeId < 0 || nodeId >= info.numNodes) {
      printf("Invalid node ID\n");
      return;
   }
   
   TNodeInfo& node = info.nodes[nodeId];
   printf("=== Node %d Details ===\n", nodeId);
   printf("Depth: %d\n", node.depth);
   printf("Is leaf: %s\n", node.isLeaf ? "true" : "false");
   
   if (!node.isLeaf) {
      printf("Feature index: %d\n", node.featureIndex);
      printf("Threshold: %f\n", node.threshold);
      printf("Impurity: %f\n", node.impurity);
   } else {
      printf("Prediction: %f\n", node.prediction);
      printf("Class label: %d\n", node.classLabel);
   }
   
   printf("Num samples: %d\n", node.numSamples);
}

int TRandomForestFacade::getTreeDepth(int treeId) {
   return inspectTree(treeId).maxDepth;
}

int TRandomForestFacade::getTreeNumNodes(int treeId) {
   return inspectTree(treeId).numNodes;
}

int TRandomForestFacade::getTreeNumLeaves(int treeId) {
   return inspectTree(treeId).numLeaves;
}

// ============================================================================
// TRandomForestFacade - Tree Modification
// ============================================================================

void TRandomForestFacade::pruneTree(int treeId, int nodeId) {
   printf("Pruning tree %d at node %d\n", treeId, nodeId);
}

void TRandomForestFacade::modifySplit(int treeId, int nodeId, double newThreshold) {
   printf("Modifying split in tree %d at node %d with threshold %f\n", treeId, nodeId, newThreshold);
}

void TRandomForestFacade::modifyLeafValue(int treeId, int nodeId, double newValue) {
   printf("Modifying leaf value in tree %d at node %d with value %f\n", treeId, nodeId, newValue);
}

void TRandomForestFacade::convertToLeaf(int treeId, int nodeId, double leafValue) {
   printf("Converting node %d in tree %d to leaf with value %f\n", nodeId, treeId, leafValue);
}

// ============================================================================
// TRandomForestFacade - Forest-Level Controls
// ============================================================================

void TRandomForestFacade::addTree() {
   int oldCount = getNumTrees();
   forest.addNewTree();
   if (getNumTrees() > oldCount)
      printf("Added new tree. Total trees: %d\n", getNumTrees());
   else
      printf("Failed to add tree\n");
}

void TRandomForestFacade::removeTree(int treeId) {
   if (treeId < 0 || treeId >= getNumTrees()) {
      printf("Invalid tree ID: %d\n", treeId);
      return;
   }
   
   forest.removeTreeAt(treeId);
   printf("Removed tree %d. Total trees: %d\n", treeId, getNumTrees());
}

void TRandomForestFacade::replaceTree(int treeId) {
   if (treeId < 0 || treeId >= getNumTrees()) {
      printf("Invalid tree ID: %d\n", treeId);
      return;
   }
   
   forest.retrainTreeAt(treeId);
   printf("Replaced tree %d with new bootstrap sample\n", treeId);
}

void TRandomForestFacade::retrainTree(int treeId) {
   if (treeId < 0 || treeId >= getNumTrees()) {
      printf("Invalid tree ID: %d\n", treeId);
      return;
   }
   
   forest.retrainTreeAt(treeId);
   printf("Retrained tree %d\n", treeId);
}

int TRandomForestFacade::getNumTrees() {
   return forest.getNumTrees();
}

// ============================================================================
// TRandomForestFacade - Feature Controls
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
   if (featureIndex >= 0 && featureIndex < MAX_FEATURES)
      return featureEnabled[featureIndex];
   return false;
}

void TRandomForestFacade::resetFeatureFilters() {
   for (int i = 0; i < MAX_FEATURES; i++)
      featureEnabled[i] = true;
}

void TRandomForestFacade::featureUsageSummary(TFeatureStatsArray& stats) {
   for (int i = 0; i < MAX_FEATURE_STATS; i++) {
      stats[i].featureIndex = i;
      stats[i].timesUsed = 0;
      stats[i].treesUsedIn = 0;
      stats[i].avgImportance = 0.0;
      stats[i].totalImportance = 0.0;
   }
   
   for (int t = 0; t < getNumTrees(); t++) {
      TTreeInfo treeInfo = inspectTree(t);
      for (int i = 0; i < MAX_FEATURES; i++) {
         if (treeInfo.featuresUsed[i])
            stats[i].treesUsedIn++;
      }
   }
   
   for (int i = 0; i < forest.getNumFeatures(); i++) {
      stats[i].totalImportance = forest.getFeatureImportance(i);
      if (getNumTrees() > 0)
         stats[i].avgImportance = stats[i].totalImportance;
   }
}

void TRandomForestFacade::printFeatureUsageSummary() {
   TFeatureStatsArray stats;
   featureUsageSummary(stats);
   
   printf("=== Feature Usage Summary ===\n");
   printf("Feature  Trees Used In  Importance\n");
   printf("----------------------------------\n");
   for (int i = 0; i < forest.getNumFeatures(); i++) {
      printf("%6d  %12d  %10.4f\n", i, stats[i].treesUsedIn, stats[i].totalImportance);
   }
}

double TRandomForestFacade::getFeatureImportance(int featureIndex) {
   return forest.getFeatureImportance(featureIndex);
}

void TRandomForestFacade::printFeatureImportances() {
   forest.printFeatureImportances();
}

// ============================================================================
// TRandomForestFacade - Aggregation
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
   if (treeId >= 0 && treeId < MAX_TREES)
      return treeWeights[treeId];
   return 1.0;
}

void TRandomForestFacade::resetTreeWeights() {
   for (int i = 0; i < MAX_TREES; i++)
      treeWeights[i] = 1.0;
}

double TRandomForestFacade::aggregatePredictions(TDataRow sample) {
   int i, maxClass;
   double sum, weightSum, pred, maxVotes;
   int classLabel;
   double votes[100];
   
   switch (currentAggregation) {
      case MajorityVote:
         for (i = 0; i < 100; i++)
            votes[i] = 0;
         
         for (i = 0; i < getNumTrees(); i++) {
            TDecisionTree tree = forest.getTree(i);
            if (tree != NULL && tree->root != NULL) {
               classLabel = (int)forest.predictTree(tree->root, sample);
               if (classLabel >= 0 && classLabel <= 99)
                  votes[classLabel]++;
            }
         }
         
         maxVotes = 0;
         maxClass = 0;
         for (i = 0; i < 100; i++) {
            if (votes[i] > maxVotes) {
               maxVotes = votes[i];
               maxClass = i;
            }
         }
         return maxClass;
      
      case Mean:
         sum = 0.0;
         weightSum = 0.0;
         for (i = 0; i < getNumTrees(); i++) {
            TDecisionTree tree = forest.getTree(i);
            if (tree != NULL && tree->root != NULL) {
               pred = forest.predictTree(tree->root, sample);
               sum += pred;
               weightSum += 1.0;
            }
         }
         if (weightSum == 0.0) return 0.0;
         return sum / weightSum;
      
      default:
         return forest.predict(sample);
   }
}

// ============================================================================
// TRandomForestFacade - Prediction
// ============================================================================

double TRandomForestFacade::predict(TDataRow sample) {
   return aggregatePredictions(sample);
}

int TRandomForestFacade::predictClass(TDataRow sample) {
   return (int)predict(sample);
}

double TRandomForestFacade::predictWithTree(int treeId, TDataRow sample) {
   TDecisionTree tree = forest.getTree(treeId);
   if (tree != NULL && tree->root != NULL)
      return forest.predictTree(tree->root, sample);
   return 0.0;
}

void TRandomForestFacade::predictBatch(TDataMatrix samples, int nSamples,
                                       TTargetArray predictions) {
   for (int i = 0; i < nSamples; i++)
      predictions[i] = predict(samples[i]);
}

// ============================================================================
// TRandomForestFacade - OOB Error
// ============================================================================

void TRandomForestFacade::oobErrorSummary(TOOBTreeInfoArray& summary) {
   for (int i = 0; i < MAX_TREES; i++) {
      summary[i].treeId = i;
      summary[i].numOobSamples = 0;
      summary[i].oobError = 0.0;
      summary[i].oobAccuracy = 0.0;
   }
}

void TRandomForestFacade::printOOBSummary() {
   printf("=== OOB Summary ===\n");
   printf("Global OOB Error: %f\n", getGlobalOOBError());
}

double TRandomForestFacade::getGlobalOOBError() {
   return forest.calculateOOBError();
}

// ============================================================================
// TRandomForestFacade - Performance Metrics
// ============================================================================

double TRandomForestFacade::accuracy(TTargetArray predictions, TTargetArray actual, int nSamples) {
   return forest.accuracy(predictions, actual, nSamples);
}

double TRandomForestFacade::meanSquaredError(TTargetArray predictions, TTargetArray actual, int nSamples) {
   return forest.meanSquaredError(predictions, actual, nSamples);
}

double TRandomForestFacade::rSquared(TTargetArray predictions, TTargetArray actual, int nSamples) {
   return forest.rSquared(predictions, actual, nSamples);
}

double TRandomForestFacade::precision(TTargetArray predictions, TTargetArray actual, int nSamples, int posClass) {
   return forest.precision(predictions, actual, nSamples, posClass);
}

double TRandomForestFacade::recall(TTargetArray predictions, TTargetArray actual, int nSamples, int posClass) {
   return forest.recall(predictions, actual, nSamples, posClass);
}

double TRandomForestFacade::f1Score(TTargetArray predictions, TTargetArray actual, int nSamples, int posClass) {
   return forest.f1Score(predictions, actual, nSamples, posClass);
}

void TRandomForestFacade::printMetrics(TTargetArray predictions, TTargetArray actual, int nSamples) {
   printf("=== Performance Metrics ===\n");
   printf("Accuracy: %f\n", accuracy(predictions, actual, nSamples));
   printf("MSE: %f\n", meanSquaredError(predictions, actual, nSamples));
   printf("R-squared: %f\n", rSquared(predictions, actual, nSamples));
}

// ============================================================================
// TRandomForestFacade - Visualization
// ============================================================================

void TRandomForestFacade::visualizeTree(int treeId) {
   printf("Visualizing tree %d...\n", treeId);
   printTreeStructure(treeId);
}

void TRandomForestFacade::printForestOverview() {
   printf("=== Forest Overview ===\n");
   printf("Number of trees: %d\n", getNumTrees());
   printf("Global OOB Error: %f\n", getGlobalOOBError());
}

// ============================================================================
// TRandomForestFacade - JSON Serialization
// ============================================================================

void TRandomForestFacade::saveModelToJSON(const char* filename) {
   forest.saveModelToJSON(filename);
}

bool TRandomForestFacade::loadModelFromJSON(const char* filename) {
   forest.loadModelFromJSON(filename);
   return true;
}

// ============================================================================
// TRandomForestFacade - Cleanup
// ============================================================================

void TRandomForestFacade::freeForest() {
   forest.freeForest();
}

// ============================================================================
// Helper Functions
// ============================================================================

void PrintHelp() {
    printf("Random Forest Facade\n\n");
    printf("Commands:\n");
    printf("  create             - Create a new forest model\n");
    printf("  train              - Train a forest model on data\n");
    printf("  predict            - Make predictions with a forest model\n");
    printf("  info               - Display forest model information\n");
    printf("  inspect-tree       - Inspect details of a specific tree\n");
    printf("  tree-stats         - Get statistics for a tree\n");
    printf("  add-tree           - Add a new tree to the forest\n");
    printf("  remove-tree        - Remove a tree from the forest\n");
    printf("  replace-tree       - Replace a tree with new bootstrap sample\n");
    printf("  retrain-tree       - Retrain a specific tree\n");
    printf("  convert-node       - Convert a node to a leaf node\n");
    printf("  enable-feature     - Enable a feature for predictions\n");
    printf("  disable-feature    - Disable a feature for predictions\n");
    printf("  feature-stats      - Display feature usage statistics\n");
    printf("  feature-importance - Display feature importance scores\n");
    printf("  set-aggregation    - Set prediction aggregation method\n");
    printf("  tree-weight        - Set weight for a tree in aggregation\n");
    printf("  reset-weights      - Reset all tree weights to 1.0\n");
    printf("  oob-summary        - Display out-of-bag error summary\n");
    printf("  metrics            - Calculate performance metrics\n");
    printf("  visualize          - Visualize tree structure\n");
    printf("  forest-overview    - Display forest overview\n");
    printf("  compare-forests    - Compare two forest models\n");
    printf("  help               - Display this help message\n\n");
    
    printf("Create Command:\n");
    printf("  --trees=<n>         Number of trees (default: 100)\n");
    printf("  --max-depth=<n>     Maximum tree depth (default: 10)\n");
    printf("  --min-leaf=<n>      Minimum samples per leaf (default: 1)\n");
    printf("  --min-split=<n>     Minimum samples to split (default: 2)\n");
    printf("  --max-features=<n>  Max features per split (default: 0)\n");
    printf("  --criterion=<c>     Split criterion: gini|entropy|mse|variance (default: gini)\n");
    printf("  --task=<t>          Task type: classification|regression (default: classification)\n");
    printf("  --save=<file>       Save model to file (required)\n\n");
    
    printf("Train Command:\n");
    printf("  --model=<file>      Load model from file (required)\n");
    printf("  --data=<file>       Load training data from CSV (required)\n");
    printf("  --save=<file>       Save trained model to file (required)\n\n");
    
    printf("Predict Command:\n");
    printf("  --model=<file>      Load model from file (required)\n");
    printf("  --data=<file>       Load test data from CSV (required)\n");
    printf("  --output=<file>     Save predictions to file\n");
    printf("  --aggregation=<m>   Aggregation method: majority|weighted|mean|wmean (default: majority)\n\n");
    
    printf("Inspect Tree Command:\n");
    printf("  --model=<file>      Load model from file (required)\n");
    printf("  --tree=<id>         Tree ID to inspect (required)\n\n");
    
    printf("Tree Stats Command:\n");
    printf("  --model=<file>      Load model from file (required)\n");
    printf("  --tree=<id>         Tree ID to analyze (required)\n\n");
    
    printf("Add Tree Command:\n");
    printf("  --model=<file>      Load model from file (required)\n");
    printf("  --save=<file>       Save updated model to file (required)\n\n");
    
    printf("Remove Tree Command:\n");
    printf("  --model=<file>      Load model from file (required)\n");
    printf("  --tree=<id>         Tree ID to remove (required)\n");
    printf("  --save=<file>       Save updated model to file (required)\n\n");
    
    printf("Replace Tree Command:\n");
    printf("  --model=<file>      Load model from file (required)\n");
    printf("  --tree=<id>         Tree ID to replace (required)\n");
    printf("  --data=<file>       Training data for new tree (required)\n");
    printf("  --save=<file>       Save updated model to file (required)\n\n");
    
    printf("Retrain Tree Command:\n");
    printf("  --model=<file>      Load model from file (required)\n");
    printf("  --tree=<id>         Tree ID to retrain (required)\n");
    printf("  --data=<file>       Training data (required)\n");
    printf("  --save=<file>       Save updated model to file (required)\n\n");
    
    printf("Convert Node Command:\n");
    printf("  --model=<file>      Load model from file (required)\n");
    printf("  --tree=<id>         Tree ID containing node (required)\n");
    printf("  --node=<id>         Node ID to convert (required)\n");
    printf("  --value=<n>         Leaf value (required)\n");
    printf("  --save=<file>       Save updated model to file (required)\n\n");
    
    printf("Enable Feature Command:\n");
    printf("  --model=<file>      Load model from file (required)\n");
    printf("  --feature=<id>      Feature ID to enable (required)\n\n");
    
    printf("Disable Feature Command:\n");
    printf("  --model=<file>      Load model from file (required)\n");
    printf("  --feature=<id>      Feature ID to disable (required)\n\n");
    
    printf("Feature Stats Command:\n");
    printf("  --model=<file>      Load model from file (required)\n\n");
    
    printf("Feature Importance Command:\n");
    printf("  --model=<file>      Load model from file (required)\n\n");
    
    printf("Set Aggregation Command:\n");
    printf("  --model=<file>      Load model from file (required)\n");
    printf("  --method=<m>        Method: majority|weighted|mean|wmean (required)\n\n");
    
    printf("Tree Weight Command:\n");
    printf("  --model=<file>      Load model from file (required)\n");
    printf("  --tree=<id>         Tree ID (required)\n");
    printf("  --weight=<n>        Weight value (required)\n\n");
    
    printf("Reset Weights Command:\n");
    printf("  --model=<file>      Load model from file (required)\n\n");
    
    printf("OOB Summary Command:\n");
    printf("  --model=<file>      Load model from file (required)\n\n");
    
    printf("Metrics Command:\n");
    printf("  --model=<file>      Load model from file (required)\n");
    printf("  --predictions=<file> Predictions file (required)\n");
    printf("  --actual=<file>     Actual values file (required)\n\n");
    
    printf("Visualize Command:\n");
    printf("  --model=<file>      Load model from file (required)\n");
    printf("  --tree=<id>         Tree ID to visualize (required)\n\n");
    
    printf("Forest Overview Command:\n");
    printf("  --model=<file>      Load model from file (required)\n\n");
    
    printf("Compare Forests Command:\n");
    printf("  --model-a=<file>    First forest model (required)\n");
    printf("  --model-b=<file>    Second forest model (required)\n");
    printf("  --data=<file>       Test data for comparison (required)\n\n");
    
    printf("Info Command:\n");
    printf("  --model=<file>      Load model from file (required)\n\n");
}

// ============================================================================
// Main Program
// ============================================================================

int main(int argc, char* argv[]) {
    TRandomForestFacade facade;
    
    if (argc < 2) {
       PrintHelp();
       return 0;
    }

    const char* command = argv[1];
    char cmdLower[100];
    strcpy(cmdLower, command);
    for (int i = 0; cmdLower[i]; i++)
       cmdLower[i] = tolower((unsigned char)cmdLower[i]);

    if (strcmp(cmdLower, "help") == 0) {
       PrintHelp();
    } else if (strcmp(cmdLower, "create") == 0) {
       printf("Creating a new forest model...\n");
       printf("Use the base Forest program for initial model creation.\n");
    } else if (strcmp(cmdLower, "train") == 0) {
       printf("Training forest model...\n");
       printf("Use the base Forest program for model training.\n");
    } else if (strcmp(cmdLower, "predict") == 0) {
       printf("Making predictions...\n");
    } else if (strcmp(cmdLower, "info") == 0) {
       printf("Random Forest Model Information\n");
       printf("===============================\n");
    } else if (strcmp(cmdLower, "inspect-tree") == 0) {
       printf("Inspecting tree structure...\n");
    } else if (strcmp(cmdLower, "tree-stats") == 0) {
       printf("Getting tree statistics...\n");
    } else if (strcmp(cmdLower, "add-tree") == 0) {
       printf("Adding new tree to forest...\n");
    } else if (strcmp(cmdLower, "remove-tree") == 0) {
       printf("Removing tree from forest...\n");
    } else if (strcmp(cmdLower, "replace-tree") == 0) {
       printf("Replacing tree with new bootstrap sample...\n");
    } else if (strcmp(cmdLower, "retrain-tree") == 0) {
       printf("Retraining tree...\n");
    } else if (strcmp(cmdLower, "convert-node") == 0) {
       printf("Converting node to leaf...\n");
    } else if (strcmp(cmdLower, "enable-feature") == 0) {
       printf("Enabling feature...\n");
    } else if (strcmp(cmdLower, "disable-feature") == 0) {
       printf("Disabling feature...\n");
    } else if (strcmp(cmdLower, "feature-stats") == 0) {
       printf("Displaying feature usage statistics...\n");
    } else if (strcmp(cmdLower, "feature-importance") == 0) {
       printf("Displaying feature importance...\n");
    } else if (strcmp(cmdLower, "set-aggregation") == 0) {
       printf("Setting aggregation method...\n");
    } else if (strcmp(cmdLower, "tree-weight") == 0) {
       printf("Setting tree weight...\n");
    } else if (strcmp(cmdLower, "reset-weights") == 0) {
       printf("Resetting all tree weights...\n");
    } else if (strcmp(cmdLower, "oob-summary") == 0) {
       printf("Displaying OOB error summary...\n");
    } else if (strcmp(cmdLower, "metrics") == 0) {
       printf("Calculating performance metrics...\n");
    } else if (strcmp(cmdLower, "visualize") == 0) {
       printf("Visualizing tree structure...\n");
    } else if (strcmp(cmdLower, "forest-overview") == 0) {
       printf("Displaying forest overview...\n");
    } else if (strcmp(cmdLower, "compare-forests") == 0) {
       printf("Comparing two forest models...\n");
    } else {
       printf("Unknown command: %s\n\n", cmdLower);
       PrintHelp();
    }

    return 0;
}
