//
// Matthew Abbott 2025
// Combined Random Forest + Facade Rust CUDA (cudarc)
//

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::Arc;

#[cfg(kani)]
mod kani;

pub const MAX_FEATURES: usize = 100;
pub const MAX_SAMPLES: usize = 10000;
pub const MAX_TREES: usize = 500;
pub const MAX_DEPTH_DEFAULT: i32 = 10;
pub const MIN_SAMPLES_LEAF_DEFAULT: i32 = 1;
pub const MIN_SAMPLES_SPLIT_DEFAULT: i32 = 2;
pub const MAX_NODES: usize = 4096;

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(i32)]
pub enum TaskType {
    Classification = 0,
    Regression = 1,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SplitCriterion {
    Gini,
    Entropy,
    MSE,
    VarianceReduction,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum AggregationMethod {
    MajorityVote,
    WeightedVote,
    Mean,
    WeightedMean,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct FlatTreeNode {
    pub is_leaf: i32,
    pub feature_index: i32,
    pub threshold: f64,
    pub prediction: f64,
    pub class_label: i32,
    pub left_child: i32,
    pub right_child: i32,
}

unsafe impl DeviceRepr for FlatTreeNode {}

#[derive(Clone)]
pub struct FlatTree {
    pub nodes: Vec<FlatTreeNode>,
    pub num_nodes: usize,
    pub oob_indices: Vec<bool>,
    pub num_oob_indices: usize,
}

impl FlatTree {
    fn new() -> Self {
        Self {
            nodes: vec![FlatTreeNode::default(); MAX_NODES],
            num_nodes: 0,
            oob_indices: vec![false; MAX_SAMPLES],
            num_oob_indices: 0,
        }
    }
}

#[allow(dead_code)]
struct TreeNode {
    is_leaf: bool,
    feature_index: i32,
    threshold: f64,
    prediction: f64,
    class_label: i32,
    impurity: f64,
    num_samples: usize,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

impl TreeNode {
    fn new_leaf(prediction: f64, class_label: i32, impurity: f64, num_samples: usize) -> Self {
        Self {
            is_leaf: true,
            feature_index: -1,
            threshold: 0.0,
            prediction,
            class_label,
            impurity,
            num_samples,
            left: None,
            right: None,
        }
    }

    fn new_split(
        feature_index: i32,
        threshold: f64,
        prediction: f64,
        class_label: i32,
        impurity: f64,
        num_samples: usize,
    ) -> Self {
        Self {
            is_leaf: false,
            feature_index,
            threshold,
            prediction,
            class_label,
            impurity,
            num_samples,
            left: None,
            right: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NodeInfo {
    pub node_id: i32,
    pub depth: i32,
    pub is_leaf: bool,
    pub feature_index: i32,
    pub threshold: f64,
    pub prediction: f64,
    pub class_label: i32,
    pub impurity: f64,
    pub num_samples: i32,
    pub left_child_id: i32,
    pub right_child_id: i32,
}

#[derive(Clone, Debug, Default)]
pub struct TreeInfo {
    pub tree_id: i32,
    pub num_nodes: i32,
    pub max_depth: i32,
    pub num_leaves: i32,
    pub features_used: Vec<bool>,
    pub num_features_used: i32,
    pub oob_error: f64,
    pub nodes: Vec<NodeInfo>,
}

#[derive(Clone, Debug, Default)]
pub struct FeatureStats {
    pub feature_index: i32,
    pub times_used: i32,
    pub trees_used_in: i32,
    pub avg_importance: f64,
    pub total_importance: f64,
}

#[derive(Clone, Debug, Default)]
pub struct SampleTrackInfo {
    pub sample_index: i32,
    pub trees_influenced: Vec<bool>,
    pub num_trees_influenced: i32,
    pub oob_trees: Vec<bool>,
    pub num_oob_trees: i32,
    pub predictions: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct OOBTreeInfo {
    pub tree_id: i32,
    pub num_oob_samples: i32,
    pub oob_error: f64,
    pub oob_accuracy: f64,
}

pub struct GpuResources {
    device: Arc<CudaDevice>,
    d_all_tree_nodes: CudaSlice<FlatTreeNode>,
    d_tree_node_offsets: CudaSlice<i32>,
    total_gpu_nodes: usize,
}

pub struct TRandomForest {
    pub trees: Vec<Option<FlatTree>>,
    pub num_trees: usize,
    pub max_depth: i32,
    pub min_samples_leaf: i32,
    pub min_samples_split: i32,
    pub max_features: i32,
    pub num_features: i32,
    pub num_samples: usize,
    pub task_type: TaskType,
    pub criterion: SplitCriterion,
    pub feature_importances: Vec<f64>,
    pub random_seed: u64,
    pub rng_state: u64,

    pub data: Vec<f64>,
    pub targets: Vec<f64>,

    pub gpu_resources: Option<GpuResources>,
}

impl TRandomForest {
    #[cfg(not(kani))]
    pub fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Self::new_with_seed(seed)
    }

    #[cfg(kani)]
    pub fn new() -> Self {
        Self::new_with_seed(42)
    }

    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            trees: vec![None; MAX_TREES],
            num_trees: 100,
            max_depth: MAX_DEPTH_DEFAULT,
            min_samples_leaf: MIN_SAMPLES_LEAF_DEFAULT,
            min_samples_split: MIN_SAMPLES_SPLIT_DEFAULT,
            max_features: 0,
            num_features: 0,
            num_samples: 0,
            task_type: TaskType::Classification,
            criterion: SplitCriterion::Gini,
            feature_importances: vec![0.0; MAX_FEATURES],
            random_seed: 42,
            rng_state: seed,
            data: vec![0.0; MAX_SAMPLES * MAX_FEATURES],
            targets: vec![0.0; MAX_SAMPLES],
            gpu_resources: None,
        }
    }

    pub fn set_num_trees(&mut self, n: usize) {
        self.num_trees = n.clamp(1, MAX_TREES);
    }

    pub fn set_max_depth(&mut self, d: i32) {
        self.max_depth = d.max(1);
    }

    pub fn set_min_samples_leaf(&mut self, m: i32) {
        self.min_samples_leaf = m.max(1);
    }

    pub fn set_min_samples_split(&mut self, m: i32) {
        self.min_samples_split = m.max(2);
    }

    pub fn set_max_features(&mut self, m: i32) {
        self.max_features = m;
    }

    pub fn set_task_type(&mut self, t: TaskType) {
        self.task_type = t;
        self.criterion = if t == TaskType::Classification {
            SplitCriterion::Gini
        } else {
            SplitCriterion::MSE
        };
    }

    pub fn set_criterion(&mut self, c: SplitCriterion) {
        self.criterion = c;
    }

    pub fn set_random_seed(&mut self, seed: u64) {
        self.random_seed = seed;
        self.rng_state = seed;
    }

    fn random_int(&mut self, max_val: usize) -> usize {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.rng_state >> 33) as usize) % max_val
    }

    #[allow(dead_code)]
    fn random_double(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.rng_state >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn load_data(&mut self, input_data: &[f64], input_targets: &[f64], n_samples: usize, n_features: usize) {
        self.num_samples = n_samples;
        self.num_features = n_features as i32;

        if self.max_features == 0 {
            if self.task_type == TaskType::Classification {
                self.max_features = (n_features as f64).sqrt().round() as i32;
            } else {
                self.max_features = (n_features / 3) as i32;
            }
            if self.max_features < 1 {
                self.max_features = 1;
            }
        }

        for i in 0..n_samples {
            for j in 0..n_features {
                self.data[i * MAX_FEATURES + j] = input_data[i * n_features + j];
            }
            self.targets[i] = input_targets[i];
        }
    }

    fn bootstrap(&mut self, sample_indices: &mut Vec<usize>, oob_mask: &mut Vec<bool>) {
        sample_indices.clear();
        sample_indices.resize(self.num_samples, 0);
        oob_mask.clear();
        oob_mask.resize(self.num_samples, true);

        for i in 0..self.num_samples {
            let idx = self.random_int(self.num_samples);
            sample_indices[i] = idx;
            oob_mask[idx] = false;
        }
    }

    fn select_feature_subset(&mut self) -> Vec<usize> {
        let mut available: Vec<usize> = (0..self.num_features as usize).collect();

        for i in (1..self.num_features as usize).rev() {
            let j = self.random_int(i + 1);
            available.swap(i, j);
        }

        let num_selected = (self.max_features as usize).min(self.num_features as usize);
        available.truncate(num_selected);
        available
    }

    fn calculate_gini(&self, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }

        let mut class_count = [0i32; 100];

        for &idx in indices {
            let class_label = self.targets[idx].round() as i32;
            if class_label >= 0 && class_label < 100 {
                class_count[class_label as usize] += 1;
            }
        }

        let n = indices.len() as f64;
        let mut gini = 1.0;
        for count in &class_count {
            let prob = *count as f64 / n;
            gini -= prob * prob;
        }
        gini
    }

    fn calculate_entropy(&self, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }

        let mut class_count = [0i32; 100];

        for &idx in indices {
            let class_label = self.targets[idx].round() as i32;
            if class_label >= 0 && class_label < 100 {
                class_count[class_label as usize] += 1;
            }
        }

        let n = indices.len() as f64;
        let mut entropy = 0.0;
        for count in &class_count {
            if *count > 0 {
                let prob = *count as f64 / n;
                entropy -= prob * prob.log2();
            }
        }
        entropy
    }

    fn calculate_mse(&self, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }

        let mean: f64 = indices.iter().map(|&i| self.targets[i]).sum::<f64>() / indices.len() as f64;

        indices.iter().map(|&i| {
            let diff = self.targets[i] - mean;
            diff * diff
        }).sum::<f64>() / indices.len() as f64
    }

    fn calculate_impurity(&self, indices: &[usize]) -> f64 {
        match self.criterion {
            SplitCriterion::Gini => self.calculate_gini(indices),
            SplitCriterion::Entropy => self.calculate_entropy(indices),
            SplitCriterion::MSE | SplitCriterion::VarianceReduction => self.calculate_mse(indices),
        }
    }

    fn find_best_split(
        &self,
        indices: &[usize],
        feature_indices: &[usize],
    ) -> Option<(usize, f64, f64)> {
        if indices.len() < self.min_samples_split as usize {
            return None;
        }

        let parent_impurity = self.calculate_impurity(indices);
        let mut best_gain = 0.0;
        let mut best_feature = None;
        let mut best_threshold = 0.0;

        for &feat in feature_indices {
            let mut indexed_values: Vec<(usize, f64)> = indices
                .iter()
                .map(|&i| (i, self.data[i * MAX_FEATURES + feat]))
                .collect();
            indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for i in 0..indexed_values.len() - 1 {
                if indexed_values[i].1 == indexed_values[i + 1].1 {
                    continue;
                }

                let threshold = (indexed_values[i].1 + indexed_values[i + 1].1) / 2.0;

                let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
                    .iter()
                    .partition(|&&idx| self.data[idx * MAX_FEATURES + feat] <= threshold);

                if left_indices.len() < self.min_samples_leaf as usize
                    || right_indices.len() < self.min_samples_leaf as usize
                {
                    continue;
                }

                let left_impurity = self.calculate_impurity(&left_indices);
                let right_impurity = self.calculate_impurity(&right_indices);

                let gain = parent_impurity
                    - (left_indices.len() as f64 / indices.len() as f64) * left_impurity
                    - (right_indices.len() as f64 / indices.len() as f64) * right_impurity;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = Some(feat);
                    best_threshold = threshold;
                }
            }
        }

        best_feature.map(|f| (f, best_threshold, best_gain))
    }

    fn get_majority_class(&self, indices: &[usize]) -> i32 {
        let mut class_count = [0i32; 100];

        for &idx in indices {
            let class_label = self.targets[idx].round() as i32;
            if class_label >= 0 && class_label < 100 {
                class_count[class_label as usize] += 1;
            }
        }

        class_count
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(i, _)| i as i32)
            .unwrap_or(0)
    }

    fn get_mean_target(&self, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }
        indices.iter().map(|&i| self.targets[i]).sum::<f64>() / indices.len() as f64
    }

    fn create_leaf_node(&self, indices: &[usize]) -> TreeNode {
        let impurity = self.calculate_impurity(indices);
        
        if self.task_type == TaskType::Classification {
            let class_label = self.get_majority_class(indices);
            TreeNode::new_leaf(class_label as f64, class_label, impurity, indices.len())
        } else {
            let prediction = self.get_mean_target(indices);
            TreeNode::new_leaf(prediction, prediction.round() as i32, impurity, indices.len())
        }
    }

    fn should_stop(&self, depth: i32, num_indices: usize, impurity: f64) -> bool {
        depth >= self.max_depth
            || (num_indices as i32) < self.min_samples_split
            || num_indices <= self.min_samples_leaf as usize
            || impurity < 1e-10
    }

    fn build_tree(&mut self, indices: &[usize], depth: i32) -> TreeNode {
        let current_impurity = self.calculate_impurity(indices);

        if self.should_stop(depth, indices.len(), current_impurity) {
            return self.create_leaf_node(indices);
        }

        let feature_indices = self.select_feature_subset();

        let split = self.find_best_split(indices, &feature_indices);
        if split.is_none() {
            return self.create_leaf_node(indices);
        }

        let (best_feature, best_threshold, _) = split.unwrap();

        let (prediction, class_label) = if self.task_type == TaskType::Classification {
            let cl = self.get_majority_class(indices);
            (cl as f64, cl)
        } else {
            let pred = self.get_mean_target(indices);
            (pred, pred.round() as i32)
        };

        let mut node = TreeNode::new_split(
            best_feature as i32,
            best_threshold,
            prediction,
            class_label,
            current_impurity,
            indices.len(),
        );

        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&idx| self.data[idx * MAX_FEATURES + best_feature] <= best_threshold);

        let left_impurity = self.calculate_impurity(&left_indices);
        let right_impurity = self.calculate_impurity(&right_indices);

        self.feature_importances[best_feature] += indices.len() as f64 * current_impurity
            - left_indices.len() as f64 * left_impurity
            - right_indices.len() as f64 * right_impurity;

        node.left = Some(Box::new(self.build_tree(&left_indices, depth + 1)));
        node.right = Some(Box::new(self.build_tree(&right_indices, depth + 1)));

        node
    }

    fn flatten_tree(node: &TreeNode, flat: &mut FlatTree, node_idx: &mut usize) {
        if *node_idx >= MAX_NODES {
            return;
        }

        let current_idx = *node_idx;
        *node_idx += 1;

        flat.nodes[current_idx] = FlatTreeNode {
            is_leaf: if node.is_leaf { 1 } else { 0 },
            feature_index: node.feature_index,
            threshold: node.threshold,
            prediction: node.prediction,
            class_label: node.class_label,
            left_child: -1,
            right_child: -1,
        };

        if !node.is_leaf {
            if let Some(ref left) = node.left {
                flat.nodes[current_idx].left_child = *node_idx as i32;
                Self::flatten_tree(left, flat, node_idx);
            }
            if let Some(ref right) = node.right {
                flat.nodes[current_idx].right_child = *node_idx as i32;
                Self::flatten_tree(right, flat, node_idx);
            }
        }
    }

    pub fn fit(&mut self) {
        for i in 0..MAX_FEATURES {
            self.feature_importances[i] = 0.0;
        }

        for i in 0..self.num_trees {
            self.fit_tree(i);
        }

        self.calculate_feature_importance();
        self.init_gpu();
    }

    fn fit_tree(&mut self, tree_index: usize) {
        let mut flat = FlatTree::new();

        let mut sample_indices = Vec::new();
        let mut oob_mask = Vec::new();
        self.bootstrap(&mut sample_indices, &mut oob_mask);

        flat.oob_indices = oob_mask.clone();
        flat.num_oob_indices = oob_mask.iter().filter(|&&x| x).count();

        let root = self.build_tree(&sample_indices, 0);

        let mut node_idx = 0;
        Self::flatten_tree(&root, &mut flat, &mut node_idx);
        flat.num_nodes = node_idx;

        self.trees[tree_index] = Some(flat);
    }

    fn init_gpu(&mut self) {
        self.free_gpu();

        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Warning: Failed to initialize CUDA device: {}", e);
                return;
            }
        };

        let mut total_gpu_nodes = 0;
        for t in 0..self.num_trees {
            if let Some(ref tree) = self.trees[t] {
                total_gpu_nodes += tree.num_nodes;
            }
        }

        let mut h_all_nodes = Vec::with_capacity(total_gpu_nodes);
        let mut h_offsets = Vec::with_capacity(self.num_trees);

        let mut offset = 0i32;
        for t in 0..self.num_trees {
            if let Some(ref tree) = self.trees[t] {
                h_offsets.push(offset);
                for n in 0..tree.num_nodes {
                    h_all_nodes.push(tree.nodes[n]);
                }
                offset += tree.num_nodes as i32;
            }
        }

        let d_all_tree_nodes = match device.htod_sync_copy(&h_all_nodes) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Warning: Failed to copy tree nodes to GPU: {}", e);
                return;
            }
        };

        let d_tree_node_offsets = match device.htod_sync_copy(&h_offsets) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Warning: Failed to copy offsets to GPU: {}", e);
                return;
            }
        };

        self.gpu_resources = Some(GpuResources {
            device,
            d_all_tree_nodes,
            d_tree_node_offsets,
            total_gpu_nodes,
        });
    }

    fn free_gpu(&mut self) {
        self.gpu_resources = None;
    }

    pub fn predict(&self, sample: &[f64]) -> f64 {
        if self.task_type == TaskType::Regression {
            let mut sum = 0.0;
            for t in 0..self.num_trees {
                if let Some(ref tree) = self.trees[t] {
                    let mut node_idx = 0;
                    while tree.nodes[node_idx].is_leaf == 0 {
                        if sample[tree.nodes[node_idx].feature_index as usize]
                            <= tree.nodes[node_idx].threshold
                        {
                            node_idx = tree.nodes[node_idx].left_child as usize;
                        } else {
                            node_idx = tree.nodes[node_idx].right_child as usize;
                        }
                    }
                    sum += tree.nodes[node_idx].prediction;
                }
            }
            sum / self.num_trees as f64
        } else {
            let mut votes = [0i32; 100];
            for t in 0..self.num_trees {
                if let Some(ref tree) = self.trees[t] {
                    let mut node_idx = 0;
                    while tree.nodes[node_idx].is_leaf == 0 {
                        if sample[tree.nodes[node_idx].feature_index as usize]
                            <= tree.nodes[node_idx].threshold
                        {
                            node_idx = tree.nodes[node_idx].left_child as usize;
                        } else {
                            node_idx = tree.nodes[node_idx].right_child as usize;
                        }
                    }
                    let class_label = tree.nodes[node_idx].class_label;
                    if class_label >= 0 && class_label < 100 {
                        votes[class_label as usize] += 1;
                    }
                }
            }

            votes
                .iter()
                .enumerate()
                .max_by_key(|(_, &v)| v)
                .map(|(i, _)| i as f64)
                .unwrap_or(0.0)
        }
    }

    pub fn predict_class(&self, sample: &[f64]) -> i32 {
        self.predict(sample).round() as i32
    }

    pub fn predict_batch(&self, samples: &[f64], n_samples: usize) -> Vec<f64> {
        let mut predictions = vec![0.0; n_samples];
        for i in 0..n_samples {
            let start = i * self.num_features as usize;
            let end = start + self.num_features as usize;
            predictions[i] = self.predict(&samples[start..end]);
        }
        predictions
    }

    pub fn predict_batch_gpu(&self, samples: &[f64], n_samples: usize) -> Vec<f64> {
        let gpu = match &self.gpu_resources {
            Some(g) => g,
            None => return self.predict_batch(samples, n_samples),
        };

        let kernel_src = include_str!("kernel.cu");
        let ptx = match cudarc::nvrtc::compile_ptx(kernel_src) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Warning: Failed to compile kernel: {}", e);
                return self.predict_batch(samples, n_samples);
            }
        };

        if let Err(e) = gpu.device.load_ptx(ptx, "random_forest", &["predictBatchKernel"]) {
            eprintln!("Warning: Failed to load PTX: {}", e);
            return self.predict_batch(samples, n_samples);
        }

        let func = match gpu.device.get_func("random_forest", "predictBatchKernel") {
            Some(f) => f,
            None => {
                eprintln!("Warning: Failed to get kernel function");
                return self.predict_batch(samples, n_samples);
            }
        };

        let d_samples = match gpu.device.htod_sync_copy(samples) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Warning: Failed to copy samples to GPU: {}", e);
                return self.predict_batch(samples, n_samples);
            }
        };

        let d_predictions: CudaSlice<f64> = match gpu.device.alloc_zeros(n_samples) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Warning: Failed to allocate predictions on GPU: {}", e);
                return self.predict_batch(samples, n_samples);
            }
        };

        let block_size = 256u32;
        let num_blocks = ((n_samples as u32) + block_size - 1) / block_size;

        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let num_features = self.num_features;
        let num_trees = self.num_trees as i32;
        let task_type = self.task_type as i32;

        unsafe {
            if let Err(e) = func.launch(
                config,
                (
                    &d_samples,
                    num_features,
                    &gpu.d_all_tree_nodes,
                    &gpu.d_tree_node_offsets,
                    num_trees,
                    n_samples as i32,
                    task_type,
                    &d_predictions,
                ),
            ) {
                eprintln!("Warning: Kernel launch failed: {}", e);
                return self.predict_batch(samples, n_samples);
            }
        }

        match gpu.device.dtoh_sync_copy(&d_predictions) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Warning: Failed to copy predictions from GPU: {}", e);
                self.predict_batch(samples, n_samples)
            }
        }
    }

    pub fn calculate_oob_error(&self) -> f64 {
        let mut predictions = vec![0.0; MAX_SAMPLES];
        let mut pred_counts = vec![0i32; MAX_SAMPLES];
        let mut votes = vec![[0i32; 100]; MAX_SAMPLES];

        for t in 0..self.num_trees {
            if let Some(ref tree) = self.trees[t] {
                for i in 0..self.num_samples {
                    if tree.oob_indices[i] {
                        let mut sample = vec![0.0; self.num_features as usize];
                        for j in 0..self.num_features as usize {
                            sample[j] = self.data[i * MAX_FEATURES + j];
                        }

                        let pred = self.predict(&sample);
                        if self.task_type == TaskType::Regression {
                            predictions[i] += pred;
                        } else {
                            let j = pred.round() as i32;
                            if j >= 0 && j < 100 {
                                votes[i][j as usize] += 1;
                            }
                        }
                        pred_counts[i] += 1;
                    }
                }
            }
        }

        let mut error = 0.0;
        let mut count = 0;

        for i in 0..self.num_samples {
            if pred_counts[i] > 0 {
                if self.task_type == TaskType::Regression {
                    let pred = predictions[i] / pred_counts[i] as f64;
                    let diff = pred - self.targets[i];
                    error += diff * diff;
                } else {
                    let max_class = votes[i]
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, &v)| v)
                        .map(|(i, _)| i as i32)
                        .unwrap_or(0);
                    if max_class != self.targets[i].round() as i32 {
                        error += 1.0;
                    }
                }
                count += 1;
            }
        }

        if count > 0 { error / count as f64 } else { 0.0 }
    }

    fn calculate_feature_importance(&mut self) {
        let total: f64 = self.feature_importances[..self.num_features as usize].iter().sum();

        if total > 0.0 {
            for i in 0..self.num_features as usize {
                self.feature_importances[i] /= total;
            }
        }
    }

    pub fn get_feature_importance(&self, feature_index: usize) -> f64 {
        if feature_index < self.num_features as usize {
            self.feature_importances[feature_index]
        } else {
            0.0
        }
    }

    pub fn print_feature_importances(&self) {
        println!("Feature Importances:");
        for i in 0..self.num_features as usize {
            println!("  Feature {}: {:.4}", i, self.feature_importances[i]);
        }
    }

    pub fn accuracy(predictions: &[f64], actual: &[f64]) -> f64 {
        let correct = predictions
            .iter()
            .zip(actual.iter())
            .filter(|(&p, &a)| p.round() as i32 == a.round() as i32)
            .count();
        correct as f64 / predictions.len() as f64
    }

    pub fn precision(predictions: &[f64], actual: &[f64], positive_class: i32) -> f64 {
        let (tp, fp) = predictions.iter().zip(actual.iter()).fold((0, 0), |(tp, fp), (&p, &a)| {
            if p.round() as i32 == positive_class {
                if a.round() as i32 == positive_class { (tp + 1, fp) } else { (tp, fp + 1) }
            } else {
                (tp, fp)
            }
        });
        if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 }
    }

    pub fn recall(predictions: &[f64], actual: &[f64], positive_class: i32) -> f64 {
        let (tp, fn_count) = predictions.iter().zip(actual.iter()).fold((0, 0), |(tp, fn_c), (&p, &a)| {
            if a.round() as i32 == positive_class {
                if p.round() as i32 == positive_class { (tp + 1, fn_c) } else { (tp, fn_c + 1) }
            } else {
                (tp, fn_c)
            }
        });
        if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 }
    }

    pub fn f1_score(predictions: &[f64], actual: &[f64], positive_class: i32) -> f64 {
        let p = Self::precision(predictions, actual, positive_class);
        let r = Self::recall(predictions, actual, positive_class);
        if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 }
    }

    pub fn mean_squared_error(predictions: &[f64], actual: &[f64]) -> f64 {
        predictions
            .iter()
            .zip(actual.iter())
            .map(|(&p, &a)| (p - a).powi(2))
            .sum::<f64>()
            / predictions.len() as f64
    }

    pub fn r_squared(predictions: &[f64], actual: &[f64]) -> f64 {
        let mean: f64 = actual.iter().sum::<f64>() / actual.len() as f64;

        let ss_res: f64 = predictions.iter().zip(actual.iter()).map(|(&p, &a)| (p - a).powi(2)).sum();
        let ss_tot: f64 = actual.iter().map(|&a| (a - mean).powi(2)).sum();

        if ss_tot > 0.0 { 1.0 - (ss_res / ss_tot) } else { 0.0 }
    }

    pub fn print_forest_info(&self) {
        println!("Random Forest Configuration (CUDA/cudarc):");
        println!("  Number of Trees: {}", self.num_trees);
        println!("  Max Depth: {}", self.max_depth);
        println!("  Min Samples Leaf: {}", self.min_samples_leaf);
        println!("  Min Samples Split: {}", self.min_samples_split);
        println!("  Max Features: {}", self.max_features);
        println!("  Number of Features: {}", self.num_features);
        println!("  Number of Samples: {}", self.num_samples);
        println!(
            "  Task Type: {}",
            if self.task_type == TaskType::Classification { "Classification" } else { "Regression" }
        );
        println!(
            "  Criterion: {}",
            match self.criterion {
                SplitCriterion::Gini => "Gini",
                SplitCriterion::Entropy => "Entropy",
                SplitCriterion::MSE => "MSE",
                SplitCriterion::VarianceReduction => "Variance Reduction",
            }
        );
        println!("  GPU Initialized: {}", self.gpu_resources.is_some());
        if let Some(ref gpu) = self.gpu_resources {
            println!("  Total GPU Nodes: {}", gpu.total_gpu_nodes);
        }
    }

    pub fn free_forest(&mut self) {
        for i in 0..MAX_TREES {
            self.trees[i] = None;
        }
    }

    pub fn get_num_trees(&self) -> usize { self.num_trees }
    pub fn get_num_features(&self) -> i32 { self.num_features }
    pub fn get_num_samples(&self) -> usize { self.num_samples }
    pub fn get_max_depth_val(&self) -> i32 { self.max_depth }
    pub fn get_task_type(&self) -> TaskType { self.task_type }
    pub fn get_criterion(&self) -> SplitCriterion { self.criterion }

    pub fn load_csv(&mut self, filename: &str, target_column: i32, has_header: bool) -> bool {
        let file = match File::open(filename) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error: Cannot open file {}: {}", filename, e);
                return false;
            }
        };

        let reader = BufReader::new(file);
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut num_cols = 0;

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = match line_result {
                Ok(l) => l,
                Err(_) => continue,
            };

            if has_header && line_num == 0 {
                continue;
            }
            if line.is_empty() {
                continue;
            }

            let row: Vec<f64> = line
                .split(',')
                .map(|cell| cell.trim().parse::<f64>().unwrap_or(0.0))
                .collect();

            if num_cols == 0 {
                num_cols = row.len();
            }
            if row.len() == num_cols {
                rows.push(row);
            }
        }

        if rows.is_empty() {
            eprintln!("Error: No data loaded from {}", filename);
            return false;
        }

        let n_samples = rows.len().min(MAX_SAMPLES);
        let n_features = (num_cols - 1).min(MAX_FEATURES);
        let target_col = if target_column < 0 { num_cols as i32 - 1 } else { target_column };

        self.num_samples = n_samples;
        self.num_features = n_features as i32;

        if self.max_features == 0 {
            if self.task_type == TaskType::Classification {
                self.max_features = (n_features as f64).sqrt().round() as i32;
            } else {
                self.max_features = (n_features / 3) as i32;
            }
            if self.max_features < 1 {
                self.max_features = 1;
            }
        }

        for i in 0..n_samples {
            let mut feat_idx = 0;
            for j in 0..num_cols {
                if j == target_col as usize {
                    self.targets[i] = rows[i][j];
                } else if feat_idx < n_features {
                    self.data[i * MAX_FEATURES + feat_idx] = rows[i][j];
                    feat_idx += 1;
                }
            }
        }

        println!(
            "Loaded {} samples with {} features from {}",
            n_samples, n_features, filename
        );
        true
    }

    pub fn save_model(&self, filename: &str) -> bool {
        let mut file = match File::create(filename) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error: Cannot open file {} for writing: {}", filename, e);
                return false;
            }
        };

        let magic = b"RFRS";
        let version: i32 = 1;

        macro_rules! write_val {
            ($val:expr) => {
                if file.write_all(&$val.to_le_bytes()).is_err() {
                    return false;
                }
            };
        }

        if file.write_all(magic).is_err() { return false; }
        write_val!(version);
        write_val!(self.num_trees as i32);
        write_val!(self.max_depth);
        write_val!(self.min_samples_leaf);
        write_val!(self.min_samples_split);
        write_val!(self.max_features);
        write_val!(self.num_features);
        write_val!(self.num_samples as i32);
        write_val!(self.task_type as i32);
        write_val!(self.criterion as i32);

        for i in 0..MAX_FEATURES {
            write_val!(self.feature_importances[i]);
        }

        for t in 0..self.num_trees {
            if let Some(ref tree) = self.trees[t] {
                write_val!(tree.num_nodes as i32);
                write_val!(tree.num_oob_indices as i32);
                
                for n in 0..tree.num_nodes {
                    let node = &tree.nodes[n];
                    write_val!(node.is_leaf);
                    write_val!(node.feature_index);
                    write_val!(node.threshold);
                    write_val!(node.prediction);
                    write_val!(node.class_label);
                    write_val!(node.left_child);
                    write_val!(node.right_child);
                }

                for i in 0..MAX_SAMPLES {
                    let b: u8 = if tree.oob_indices[i] { 1 } else { 0 };
                    if file.write_all(&[b]).is_err() { return false; }
                }
            }
        }

        println!("Model saved to {}", filename);
        true
    }

    pub fn load_model(&mut self, filename: &str) -> bool {
        let mut file = match File::open(filename) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error: Cannot open file {} for reading: {}", filename, e);
                return false;
            }
        };

        let mut magic = [0u8; 4];
        if file.read_exact(&mut magic).is_err() { return false; }
        if &magic != b"RFRS" {
            eprintln!("Error: Invalid model file format");
            return false;
        }

        self.free_forest();
        self.free_gpu();

        macro_rules! read_i32 {
            () => {{
                let mut buf = [0u8; 4];
                if file.read_exact(&mut buf).is_err() { return false; }
                i32::from_le_bytes(buf)
            }};
        }

        macro_rules! read_f64 {
            () => {{
                let mut buf = [0u8; 8];
                if file.read_exact(&mut buf).is_err() { return false; }
                f64::from_le_bytes(buf)
            }};
        }

        let _version = read_i32!();
        self.num_trees = read_i32!() as usize;
        self.max_depth = read_i32!();
        self.min_samples_leaf = read_i32!();
        self.min_samples_split = read_i32!();
        self.max_features = read_i32!();
        self.num_features = read_i32!();
        self.num_samples = read_i32!() as usize;
        self.task_type = match read_i32!() {
            1 => TaskType::Regression,
            _ => TaskType::Classification,
        };
        self.criterion = match read_i32!() {
            1 => SplitCriterion::Entropy,
            2 => SplitCriterion::MSE,
            3 => SplitCriterion::VarianceReduction,
            _ => SplitCriterion::Gini,
        };

        for i in 0..MAX_FEATURES {
            self.feature_importances[i] = read_f64!();
        }

        for t in 0..self.num_trees {
            let mut tree = FlatTree::new();
            tree.num_nodes = read_i32!() as usize;
            tree.num_oob_indices = read_i32!() as usize;

            for n in 0..tree.num_nodes {
                tree.nodes[n].is_leaf = read_i32!();
                tree.nodes[n].feature_index = read_i32!();
                tree.nodes[n].threshold = read_f64!();
                tree.nodes[n].prediction = read_f64!();
                tree.nodes[n].class_label = read_i32!();
                tree.nodes[n].left_child = read_i32!();
                tree.nodes[n].right_child = read_i32!();
            }

            for i in 0..MAX_SAMPLES {
                let mut b = [0u8; 1];
                if file.read_exact(&mut b).is_err() { return false; }
                tree.oob_indices[i] = b[0] != 0;
            }

            self.trees[t] = Some(tree);
        }

        println!("Model loaded from {}", filename);
        self.init_gpu();
        true
    }

    pub fn predict_csv(&self, input_file: &str, output_file: &str, has_header: bool) -> bool {
        let in_file = match File::open(input_file) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error: Cannot open file {}: {}", input_file, e);
                return false;
            }
        };

        let mut out_file = match File::create(output_file) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error: Cannot open file {} for writing: {}", output_file, e);
                return false;
            }
        };

        let reader = BufReader::new(in_file);

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = match line_result {
                Ok(l) => l,
                Err(_) => continue,
            };

            if has_header && line_num == 0 {
                writeln!(out_file, "{},prediction", line).ok();
                continue;
            }
            if line.is_empty() {
                continue;
            }

            let row: Vec<f64> = line
                .split(',')
                .map(|cell| cell.trim().parse::<f64>().unwrap_or(0.0))
                .collect();

            let mut sample = vec![0.0; self.num_features as usize];
            for j in 0..self.num_features as usize {
                if j < row.len() {
                    sample[j] = row[j];
                }
            }

            let pred = self.predict(&sample);
            writeln!(out_file, "{},{:.4}", line, pred).ok();
        }

        println!("Predictions saved to {}", output_file);
        true
    }

    pub fn add_new_tree(&mut self) {
        if self.num_trees < MAX_TREES {
            self.fit_tree(self.num_trees);
            self.num_trees += 1;
            self.init_gpu();
        }
    }

    pub fn remove_tree_at(&mut self, tree_id: usize) {
        if tree_id < self.num_trees && self.num_trees > 1 {
            for i in tree_id..self.num_trees - 1 {
                self.trees[i] = self.trees[i + 1].take();
            }
            self.trees[self.num_trees - 1] = None;
            self.num_trees -= 1;
            self.init_gpu();
        }
    }

    pub fn retrain_tree_at(&mut self, tree_id: usize) {
        if tree_id < self.num_trees {
            self.fit_tree(tree_id);
            self.init_gpu();
        }
    }
}

pub struct TRandomForestFacade {
    pub forest: TRandomForest,
    pub forest_initialized: bool,
    pub current_aggregation: AggregationMethod,
    pub tree_weights: Vec<f64>,
    pub feature_enabled: Vec<bool>,
}

impl TRandomForestFacade {
    pub fn new() -> Self {
        Self {
            forest: TRandomForest::new(),
            forest_initialized: false,
            current_aggregation: AggregationMethod::MajorityVote,
            tree_weights: vec![1.0; MAX_TREES],
            feature_enabled: vec![true; MAX_FEATURES],
        }
    }

    pub fn init_forest(&mut self) {
        self.forest_initialized = true;
    }

    pub fn set_hyperparameter(&mut self, param_name: &str, value: i32) {
        match param_name {
            "n_estimators" => self.forest.set_num_trees(value as usize),
            "max_depth" => self.forest.set_max_depth(value),
            "min_samples_leaf" => self.forest.set_min_samples_leaf(value),
            "min_samples_split" => self.forest.set_min_samples_split(value),
            "max_features" => self.forest.set_max_features(value),
            _ => {}
        }
    }

    #[allow(dead_code)]
    pub fn set_hyperparameter_float(&mut self, _param_name: &str, _value: f64) {}

    #[allow(dead_code)]
    pub fn get_hyperparameter(&self, _param_name: &str) -> i32 { 0 }

    pub fn set_task_type(&mut self, t: TaskType) { self.forest.set_task_type(t); }
    pub fn set_criterion(&mut self, c: SplitCriterion) { self.forest.set_criterion(c); }
    pub fn print_hyperparameters(&self) { self.forest.print_forest_info(); }

    pub fn load_csv(&mut self, filename: &str) -> bool { self.forest.load_csv(filename, -1, true) }
    pub fn train(&mut self) { self.forest.fit(); }
    pub fn train_gpu(&mut self) { self.forest.init_gpu(); self.forest.fit(); }

    pub fn inspect_tree(&self, tree_id: usize) -> TreeInfo {
        let mut info = TreeInfo {
            tree_id: tree_id as i32,
            features_used: vec![false; MAX_FEATURES],
            ..Default::default()
        };

        if let Some(ref tree) = self.forest.trees.get(tree_id).and_then(|t| t.as_ref()) {
            info.num_nodes = tree.num_nodes as i32;
            
            let mut num_leaves = 0;

            for n in 0..tree.num_nodes {
                let node = &tree.nodes[n];
                if node.is_leaf != 0 {
                    num_leaves += 1;
                } else {
                    if node.feature_index >= 0 && (node.feature_index as usize) < MAX_FEATURES {
                        info.features_used[node.feature_index as usize] = true;
                    }
                }
            }

            fn calc_depth(nodes: &[FlatTreeNode], idx: usize, depth: i32) -> i32 {
                if idx >= nodes.len() || nodes[idx].is_leaf != 0 {
                    return depth;
                }
                let left_depth = if nodes[idx].left_child >= 0 {
                    calc_depth(nodes, nodes[idx].left_child as usize, depth + 1)
                } else { depth };
                let right_depth = if nodes[idx].right_child >= 0 {
                    calc_depth(nodes, nodes[idx].right_child as usize, depth + 1)
                } else { depth };
                left_depth.max(right_depth)
            }

            let max_depth = calc_depth(&tree.nodes, 0, 0);

            info.max_depth = max_depth;
            info.num_leaves = num_leaves;
            info.num_features_used = info.features_used.iter().filter(|&&x| x).count() as i32;
        }

        info
    }

    pub fn print_tree_info(&self, tree_id: usize) {
        let info = self.inspect_tree(tree_id);
        println!("Tree {}: {} nodes, max depth: {}, leaves: {}", 
            tree_id, info.num_nodes, info.max_depth, info.num_leaves);
    }

    pub fn print_tree_structure(&self, tree_id: usize) {
        println!("Tree {} structure:", tree_id);
        if let Some(ref tree) = self.forest.trees.get(tree_id).and_then(|t| t.as_ref()) {
            fn print_node(nodes: &[FlatTreeNode], idx: usize, indent: usize) {
                if idx >= nodes.len() { return; }
                let node = &nodes[idx];
                let prefix = "  ".repeat(indent);
                if node.is_leaf != 0 {
                    println!("{}[Leaf] prediction={:.4}, class={}", prefix, node.prediction, node.class_label);
                } else {
                    println!("{}[Node] feature={}, threshold={:.4}", prefix, node.feature_index, node.threshold);
                    if node.left_child >= 0 {
                        println!("{}  Left:", prefix);
                        print_node(nodes, node.left_child as usize, indent + 2);
                    }
                    if node.right_child >= 0 {
                        println!("{}  Right:", prefix);
                        print_node(nodes, node.right_child as usize, indent + 2);
                    }
                }
            }
            print_node(&tree.nodes, 0, 0);
        }
    }

    pub fn add_tree(&mut self) { self.forest.add_new_tree(); }
    pub fn remove_tree(&mut self, tree_id: usize) { self.forest.remove_tree_at(tree_id); }
    pub fn replace_tree(&mut self, tree_id: usize) { self.forest.retrain_tree_at(tree_id); }
    pub fn retrain_tree(&mut self, tree_id: usize) { self.forest.retrain_tree_at(tree_id); }
    pub fn get_num_trees(&self) -> usize { self.forest.get_num_trees() }

    pub fn enable_feature(&mut self, feature_index: usize) {
        if feature_index < MAX_FEATURES {
            self.feature_enabled[feature_index] = true;
        }
    }

    pub fn disable_feature(&mut self, feature_index: usize) {
        if feature_index < MAX_FEATURES {
            self.feature_enabled[feature_index] = false;
        }
    }

    pub fn reset_features(&mut self) {
        for i in 0..MAX_FEATURES {
            self.feature_enabled[i] = true;
        }
    }

    pub fn print_feature_usage(&self) {
        println!("Feature Usage Summary:");
        let mut usage = vec![0i32; MAX_FEATURES];
        
        for t in 0..self.forest.num_trees {
            if let Some(ref tree) = self.forest.trees[t] {
                for n in 0..tree.num_nodes {
                    let node = &tree.nodes[n];
                    if node.is_leaf == 0 && node.feature_index >= 0 && (node.feature_index as usize) < MAX_FEATURES {
                        usage[node.feature_index as usize] += 1;
                    }
                }
            }
        }

        for i in 0..self.forest.num_features as usize {
            let enabled = if self.feature_enabled[i] { "enabled" } else { "disabled" };
            println!("  Feature {}: {} splits ({})", i, usage[i], enabled);
        }
    }

    pub fn print_feature_importances(&self) { self.forest.print_feature_importances(); }

    pub fn set_aggregation_method(&mut self, method: AggregationMethod) {
        self.current_aggregation = method;
    }

    pub fn get_aggregation_method(&self) -> AggregationMethod { self.current_aggregation }

    pub fn set_tree_weight(&mut self, tree_id: usize, weight: f64) {
        if tree_id < MAX_TREES {
            self.tree_weights[tree_id] = weight;
        }
    }

    pub fn get_tree_weight(&self, tree_id: usize) -> f64 {
        if tree_id < MAX_TREES { self.tree_weights[tree_id] } else { 1.0 }
    }

    pub fn reset_tree_weights(&mut self) {
        for i in 0..MAX_TREES {
            self.tree_weights[i] = 1.0;
        }
    }

    pub fn predict(&self, sample: &[f64]) -> f64 {
        match self.current_aggregation {
            AggregationMethod::MajorityVote | AggregationMethod::Mean => {
                self.forest.predict(sample)
            }
            AggregationMethod::WeightedVote | AggregationMethod::WeightedMean => {
                self.predict_weighted(sample)
            }
        }
    }

    fn predict_weighted(&self, sample: &[f64]) -> f64 {
        if self.forest.task_type == TaskType::Regression {
            let mut sum = 0.0;
            let mut total_weight = 0.0;
            for t in 0..self.forest.num_trees {
                if let Some(ref tree) = self.forest.trees[t] {
                    let mut node_idx = 0;
                    while tree.nodes[node_idx].is_leaf == 0 {
                        if sample[tree.nodes[node_idx].feature_index as usize]
                            <= tree.nodes[node_idx].threshold
                        {
                            node_idx = tree.nodes[node_idx].left_child as usize;
                        } else {
                            node_idx = tree.nodes[node_idx].right_child as usize;
                        }
                    }
                    sum += tree.nodes[node_idx].prediction * self.tree_weights[t];
                    total_weight += self.tree_weights[t];
                }
            }
            if total_weight > 0.0 { sum / total_weight } else { 0.0 }
        } else {
            let mut votes = [0.0f64; 100];
            for t in 0..self.forest.num_trees {
                if let Some(ref tree) = self.forest.trees[t] {
                    let mut node_idx = 0;
                    while tree.nodes[node_idx].is_leaf == 0 {
                        if sample[tree.nodes[node_idx].feature_index as usize]
                            <= tree.nodes[node_idx].threshold
                        {
                            node_idx = tree.nodes[node_idx].left_child as usize;
                        } else {
                            node_idx = tree.nodes[node_idx].right_child as usize;
                        }
                    }
                    let class_label = tree.nodes[node_idx].class_label;
                    if class_label >= 0 && class_label < 100 {
                        votes[class_label as usize] += self.tree_weights[t];
                    }
                }
            }

            votes
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as f64)
                .unwrap_or(0.0)
        }
    }

    pub fn predict_class(&self, sample: &[f64]) -> i32 {
        self.predict(sample).round() as i32
    }

    pub fn predict_batch(&self, samples: &[f64], n_samples: usize) -> Vec<f64> {
        let mut predictions = vec![0.0; n_samples];
        for i in 0..n_samples {
            let start = i * self.forest.num_features as usize;
            let end = start + self.forest.num_features as usize;
            predictions[i] = self.predict(&samples[start..end]);
        }
        predictions
    }

    pub fn track_sample(&self, sample_index: usize) -> SampleTrackInfo {
        let mut info = SampleTrackInfo {
            sample_index: sample_index as i32,
            trees_influenced: vec![false; MAX_TREES],
            oob_trees: vec![false; MAX_TREES],
            predictions: vec![0.0; MAX_TREES],
            ..Default::default()
        };

        for t in 0..self.forest.num_trees {
            if let Some(ref tree) = self.forest.trees[t] {
                if sample_index < tree.oob_indices.len() {
                    if tree.oob_indices[sample_index] {
                        info.oob_trees[t] = true;
                        info.num_oob_trees += 1;
                    } else {
                        info.trees_influenced[t] = true;
                        info.num_trees_influenced += 1;
                    }
                }
            }
        }

        info
    }

    pub fn print_sample_tracking(&self, sample_index: usize) {
        let info = self.track_sample(sample_index);
        println!("Sample {} tracking:", sample_index);
        println!("  Trees influenced: {}", info.num_trees_influenced);
        println!("  OOB trees: {}", info.num_oob_trees);
    }

    pub fn oob_error_summary(&self) -> Vec<OOBTreeInfo> {
        let mut summary = Vec::new();
        for t in 0..self.forest.num_trees {
            if let Some(ref tree) = self.forest.trees[t] {
                let num_oob = tree.oob_indices.iter().take(self.forest.num_samples).filter(|&&x| x).count();
                summary.push(OOBTreeInfo {
                    tree_id: t as i32,
                    num_oob_samples: num_oob as i32,
                    oob_error: 0.0,
                    oob_accuracy: 0.0,
                });
            }
        }
        summary
    }

    pub fn print_oob_summary(&self) {
        println!("OOB Error Summary:");
        let summary = self.oob_error_summary();
        for info in &summary {
            println!("  Tree {}: {} OOB samples", info.tree_id, info.num_oob_samples);
        }
        println!("Global OOB Error: {:.4}", self.forest.calculate_oob_error());
    }

    pub fn get_global_oob_error(&self) -> f64 { self.forest.calculate_oob_error() }

    pub fn accuracy(&self, predictions: &[f64], actual: &[f64]) -> f64 {
        TRandomForest::accuracy(predictions, actual)
    }

    pub fn mean_squared_error(&self, predictions: &[f64], actual: &[f64]) -> f64 {
        TRandomForest::mean_squared_error(predictions, actual)
    }

    pub fn highlight_misclassified(&self, predictions: &[f64], actual: &[f64]) {
        println!("Misclassified Samples:");
        for (i, (&p, &a)) in predictions.iter().zip(actual.iter()).enumerate() {
            if p.round() as i32 != a.round() as i32 {
                println!("  Sample {}: predicted={}, actual={}", i, p.round() as i32, a.round() as i32);
            }
        }
    }

    pub fn save_model(&self, filename: &str) -> bool { self.forest.save_model(filename) }
    pub fn load_model(&mut self, filename: &str) -> bool { self.forest.load_model(filename) }
    pub fn print_forest_info(&self) { self.forest.print_forest_info(); }
}

fn print_help() {
    println!("Random Forest Facade CLI (CUDA/cudarc)");
    println!("Matthew Abbott 2025");
    println!("Advanced Random Forest with Introspection, Tree Manipulation, and Feature Control");
    println!();
    println!("Usage: forest_facade <command> [options]");
    println!();
    println!("=== Core Commands ===");
    println!("  create              Create a new empty forest model");
    println!("  train               Train a random forest model");
    println!("  predict             Make predictions using a trained model");
    println!("  evaluate            Evaluate model on test data");
    println!("  save                Save model to file");
    println!("  load                Load model from file");
    println!("  info                Show forest hyperparameters");
    println!("  gpu-info            Show GPU device information");
    println!("  help                Show this help message");
    println!();
    println!("=== Tree Inspection & Manipulation ===");
    println!("  inspect-tree        Inspect tree structure and nodes");
    println!("  tree-depth          Get depth of a specific tree");
    println!("  tree-nodes          Get node count of a specific tree");
    println!("  tree-leaves         Get leaf count of a specific tree");
    println!("  node-details        Get details of a specific node");
    println!("  prune-tree          Prune subtree at specified node");
    println!("  modify-split        Modify split threshold at node");
    println!("  modify-leaf         Modify leaf prediction value");
    println!("  convert-to-leaf     Convert node to leaf");
    println!();
    println!("=== Tree Management ===");
    println!("  add-tree            Add a new tree to the forest");
    println!("  remove-tree         Remove a tree from the forest");
    println!("  replace-tree        Replace a tree with new bootstrap sample");
    println!("  retrain-tree        Retrain a specific tree");
    println!();
    println!("=== Feature Control ===");
    println!("  enable-feature      Enable a feature for predictions");
    println!("  disable-feature     Disable a feature for predictions");
    println!("  reset-features      Reset all feature filters");
    println!("  feature-usage       Show feature usage summary");
    println!("  importance          Show feature importances");
    println!();
    println!("=== Aggregation Control ===");
    println!("  set-aggregation     Set prediction aggregation method");
    println!("  get-aggregation     Get current aggregation method");
    println!("  set-weight          Set weight for specific tree");
    println!("  get-weight          Get weight of specific tree");
    println!("  reset-weights       Reset all tree weights to 1.0");
    println!();
    println!("=== Performance Analysis ===");
    println!("  oob-summary         Show OOB error summary per tree");
    println!("  track-sample        Track which trees influence a sample");
    println!("  metrics             Calculate accuracy/MSE/F1 etc.");
    println!("  misclassified       Highlight misclassified samples");
    println!("  worst-trees         Find trees with highest error");
    println!();
    println!("=== Options ===");
    println!();
    println!("Data & Model:");
    println!("  --input=<file>          Training input data (CSV)");
    println!("  --target=<file>         Training targets (CSV)");
    println!("  --data=<file>           Test/prediction data (CSV)");
    println!("  --model=<file>          Model file (default: forest.bin)");
    println!("  --output=<file>         Output predictions file");
    println!();
    println!("Hyperparameters:");
    println!("  --trees=<n>             Number of trees (default: 100)");
    println!("  --depth=<n>             Max tree depth (default: 10)");
    println!("  --min-leaf=<n>          Min samples per leaf (default: 1)");
    println!("  --min-split=<n>         Min samples to split node (default: 2)");
    println!("  --max-features=<n>      Max features per split (0=auto)");
    println!("  --task=<class|reg>      Task type (default: class)");
    println!("  --criterion=<c>         Split criterion: gini/entropy/mse/var");
    println!();
    println!("Tree Manipulation:");
    println!("  --tree=<id>             Tree ID for operations");
    println!("  --node=<id>             Node ID for operations");
    println!("  --threshold=<val>       New split threshold");
    println!("  --value=<val>           New leaf value");
    println!();
    println!("Feature/Weight Control:");
    println!("  --feature=<id>          Feature ID for operations");
    println!("  --weight=<val>          Tree weight (0.0-1.0)");
    println!("  --aggregation=<method>  majority|weighted|mean|weighted-mean");
    println!("  --sample=<id>           Sample ID for tracking");
    println!();
    println!("=== Examples ===");
    println!("  # Create and train forest");
    println!("  forest_facade create --trees=100 --depth=10 --model=rf.bin");
    println!("  forest_facade train --input=data.csv --target=labels.csv --model=rf.bin");
    println!();
    println!("  # Make predictions and evaluate");
    println!("  forest_facade predict --data=test.csv --model=rf.bin --output=preds.csv");
    println!("  forest_facade evaluate --data=test.csv --model=rf.bin");
    println!();
    println!("  # Tree inspection");
    println!("  forest_facade inspect-tree --tree=5 --model=rf.bin");
    println!("  forest_facade tree-depth --tree=5 --model=rf.bin");
    println!();
    println!("  # Feature analysis");
    println!("  forest_facade feature-usage --model=rf.bin");
    println!("  forest_facade importance --model=rf.bin");
    println!();
    println!("  # Tree manipulation");
    println!("  forest_facade add-tree --model=rf.bin");
    println!("  forest_facade remove-tree --tree=5 --model=rf.bin");
    println!("  forest_facade disable-feature --feature=3 --model=rf.bin");
    println!();
    println!("  # Aggregation control");
    println!("  forest_facade set-aggregation --aggregation=weighted-mean --model=rf.bin");
    println!("  forest_facade set-weight --tree=5 --weight=1.5 --model=rf.bin");
}

fn get_arg(args: &[String], name: &str) -> Option<String> {
    let prefix = format!("{}=", name);
    for arg in args {
        if arg.starts_with(&prefix) {
            return Some(arg[prefix.len()..].to_string());
        }
    }
    None
}

fn get_arg_int(args: &[String], name: &str, default: i32) -> i32 {
    get_arg(args, name).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn get_arg_float(args: &[String], name: &str, default: f64) -> f64 {
    get_arg(args, name).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_help();
        return;
    }

    let command = args[1].to_lowercase();
    let mut facade = TRandomForestFacade::new();

    match command.as_str() {
        "help" | "--help" | "-h" => {
            print_help();
        }

        "create" => {
            facade.init_forest();
            let trees = get_arg_int(&args, "--trees", 100);
            let depth = get_arg_int(&args, "--depth", 10);
            facade.set_hyperparameter("n_estimators", trees);
            facade.set_hyperparameter("max_depth", depth);
            println!("Created Random Forest (CUDA/cudarc): {} trees, depth {}", trees, depth);
            
            if let Some(model_file) = get_arg(&args, "--model") {
                facade.save_model(&model_file);
            }
        }

        "train" => {
            facade.init_forest();
            let input_file = get_arg(&args, "--input");
            let model_file = get_arg(&args, "--model");
            
            if input_file.is_none() {
                eprintln!("Error: --input is required");
                std::process::exit(1);
            }
            
            println!("Training forest from: {}", input_file.as_ref().unwrap());
            if facade.load_csv(input_file.as_ref().unwrap()) {
                facade.train();
                println!("Training complete");
                
                if let Some(model) = model_file {
                    facade.save_model(&model);
                }
            }
        }

        "predict" => {
            let model_file = get_arg(&args, "--model");
            let data_file = get_arg(&args, "--data");
            let output_file = get_arg(&args, "--output").unwrap_or_else(|| "predictions.csv".to_string());

            if model_file.is_none() || data_file.is_none() {
                eprintln!("Error: --model and --data are required");
                std::process::exit(1);
            }

            facade.load_model(model_file.as_ref().unwrap());
            println!("Making predictions on: {}", data_file.as_ref().unwrap());
            facade.forest.predict_csv(data_file.as_ref().unwrap(), &output_file, true);
        }

        "info" => {
            let model_file = get_arg(&args, "--model");
            if model_file.is_none() {
                eprintln!("Error: --model is required");
                std::process::exit(1);
            }
            facade.load_model(model_file.as_ref().unwrap());
            facade.print_forest_info();
        }

        "gpu-info" => {
            match CudaDevice::new(0) {
                Ok(_device) => {
                    println!("GPU Device 0 available");
                    println!("  cudarc initialized successfully");
                }
                Err(e) => {
                    println!("No GPU available: {}", e);
                }
            }
        }

        "add-tree" => {
            if let Some(model_file) = get_arg(&args, "--model") {
                facade.load_model(&model_file);
                facade.add_tree();
                println!("Added tree. Total trees: {}", facade.get_num_trees());
                facade.save_model(&model_file);
            }
        }

        "remove-tree" => {
            let tree_id = get_arg_int(&args, "--tree", 0) as usize;
            if let Some(model_file) = get_arg(&args, "--model") {
                facade.load_model(&model_file);
                facade.remove_tree(tree_id);
                println!("Removed tree {}", tree_id);
                facade.save_model(&model_file);
            }
        }

        "retrain-tree" => {
            let tree_id = get_arg_int(&args, "--tree", 0) as usize;
            if let Some(model_file) = get_arg(&args, "--model") {
                facade.load_model(&model_file);
                facade.retrain_tree(tree_id);
                println!("Retrained tree {}", tree_id);
                facade.save_model(&model_file);
            }
        }

        "inspect-tree" => {
            let tree_id = get_arg_int(&args, "--tree", 0) as usize;
            if let Some(model_file) = get_arg(&args, "--model") {
                facade.load_model(&model_file);
                facade.print_tree_info(tree_id);
                facade.print_tree_structure(tree_id);
            }
        }

        "feature-usage" => {
            if let Some(model_file) = get_arg(&args, "--model") {
                facade.load_model(&model_file);
                facade.print_feature_usage();
            }
        }

        "importance" => {
            if let Some(model_file) = get_arg(&args, "--model") {
                facade.load_model(&model_file);
                facade.print_feature_importances();
            }
        }

        "set-aggregation" => {
            let method = get_arg(&args, "--aggregation").unwrap_or_default();
            let agg = match method.to_lowercase().as_str() {
                "weighted" | "weighted-vote" => AggregationMethod::WeightedVote,
                "mean" => AggregationMethod::Mean,
                "weighted-mean" => AggregationMethod::WeightedMean,
                _ => AggregationMethod::MajorityVote,
            };
            facade.set_aggregation_method(agg);
            println!("Set aggregation to: {:?}", agg);
        }

        "set-weight" => {
            let tree_id = get_arg_int(&args, "--tree", 0) as usize;
            let weight = get_arg_float(&args, "--weight", 1.0);
            facade.set_tree_weight(tree_id, weight);
            println!("Set weight for tree {} to {:.2}", tree_id, weight);
        }

        "oob-summary" => {
            if let Some(model_file) = get_arg(&args, "--model") {
                facade.load_model(&model_file);
                facade.print_oob_summary();
            }
        }

        "track-sample" => {
            let sample_id = get_arg_int(&args, "--sample", 0) as usize;
            if let Some(model_file) = get_arg(&args, "--model") {
                facade.load_model(&model_file);
                facade.print_sample_tracking(sample_id);
            }
        }

        _ => {
            eprintln!("Unknown command: {}", command);
            println!();
            print_help();
            std::process::exit(1);
        }
    }
}
