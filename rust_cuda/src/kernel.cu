//
// Created by Matthew Abbott 2025
// CUDA kernels for Random Forest
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

} // extern "C"
