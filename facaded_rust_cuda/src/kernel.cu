//
// Matthew Abbott 2025
// CUDA kernels for Facaded Random Forest
//

extern "C" {

struct FlatTreeNode {
    bool isLeaf;
    int featureIndex;
    double threshold;
    double prediction;
    int classLabel;
    int leftChild;
    int rightChild;
};

enum TaskType { Classification = 0, Regression = 1 };

__global__ void predictBatchKernel(
    double* data,
    int numFeatures,
    FlatTreeNode* allTreeNodes,
    int* treeNodeOffsets,
    int numTrees,
    int numSamples,
    int taskType,
    double* predictions
) {
    int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sampleIdx >= numSamples) return;

    double* sample = &data[sampleIdx * numFeatures];
    
    if (taskType == 1) { // Regression
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
    } else { // Classification
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

__global__ void predictBatchWeightedKernel(
    double* data,
    int numFeatures,
    FlatTreeNode* allTreeNodes,
    int* treeNodeOffsets,
    double* treeWeights,
    int numTrees,
    int numSamples,
    int taskType,
    double* predictions
) {
    int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sampleIdx >= numSamples) return;

    double* sample = &data[sampleIdx * numFeatures];
    
    if (taskType == 1) { // Regression - weighted mean
        double sum = 0.0;
        double totalWeight = 0.0;
        for (int t = 0; t < numTrees; t++) {
            FlatTreeNode* tree = &allTreeNodes[treeNodeOffsets[t]];
            int nodeIdx = 0;
            while (!tree[nodeIdx].isLeaf) {
                if (sample[tree[nodeIdx].featureIndex] <= tree[nodeIdx].threshold)
                    nodeIdx = tree[nodeIdx].leftChild;
                else
                    nodeIdx = tree[nodeIdx].rightChild;
            }
            sum += tree[nodeIdx].prediction * treeWeights[t];
            totalWeight += treeWeights[t];
        }
        predictions[sampleIdx] = (totalWeight > 0) ? sum / totalWeight : 0.0;
    } else { // Classification - weighted vote
        double votes[100] = {0.0};
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
                votes[classLabel] += treeWeights[t];
        }
        
        double maxVotes = 0.0;
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

} // extern "C"
