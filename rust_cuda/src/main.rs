//
// Created by Matthew Abbott 2025
// CUDA port of Random Forest using cudarc
//

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::Arc;

const MAX_FEATURES: usize = 100;
const MAX_SAMPLES: usize = 10000;
const MAX_TREES: usize = 500;
const MAX_DEPTH_DEFAULT: i32 = 10;
const MIN_SAMPLES_LEAF_DEFAULT: i32 = 1;
const MIN_SAMPLES_SPLIT_DEFAULT: i32 = 2;
const MAX_NODES: usize = 4096;

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

struct GpuResources {
    device: Arc<CudaDevice>,
    d_all_tree_nodes: CudaSlice<FlatTreeNode>,
    d_tree_node_offsets: CudaSlice<i32>,
    total_gpu_nodes: usize,
}

pub struct TRandomForest {
    trees: Vec<Option<FlatTree>>,
    num_trees: usize,
    max_depth: i32,
    min_samples_leaf: i32,
    min_samples_split: i32,
    max_features: i32,
    num_features: i32,
    num_samples: usize,
    task_type: TaskType,
    criterion: SplitCriterion,
    feature_importances: Vec<f64>,
    random_seed: u64,
    rng_state: u64,

    data: Vec<f64>,
    targets: Vec<f64>,

    gpu_resources: Option<GpuResources>,
}

impl TRandomForest {
    pub fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

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
}

fn print_help() {
    println!("Random Forest CLI Tool (CUDA/cudarc)");
    println!("Matthew Abbott 2025");
    println!();
    println!("Usage: forest_cuda <command> [options]");
    println!();
    println!("Commands:");
    println!("  create   Create a new random forest model");
    println!("  train    Train a random forest model");
    println!("  predict  Make predictions with a trained model");
    println!("  info     Display model information");
    println!("  help     Show this help message");
    println!();
    println!("CREATE Options:");
    println!("  --trees=N          Number of trees (default: 100)");
    println!("  --max-depth=N      Maximum tree depth (default: 10)");
    println!("  --min-leaf=N       Minimum samples per leaf (default: 1)");
    println!("  --min-split=N      Minimum samples to split (default: 2)");
    println!("  --max-features=N   Maximum features per split (default: sqrt(n))");
    println!("  --criterion=C      Split criterion: gini, entropy, mse, variancereduction (default: gini)");
    println!("  --task=T           Task type: classification, regression (default: classification)");
    println!("  --save=FILE        Save model to file (required)");
    println!();
    println!("TRAIN Options:");
    println!("  --model=FILE       Model file to load (required)");
    println!("  --data=FILE        Training data CSV file (required)");
    println!("  --save=FILE        Save trained model to file (required)");
    println!();
    println!("PREDICT Options:");
    println!("  --model=FILE       Model file to load (required)");
    println!("  --data=FILE        Data file for predictions (required)");
    println!("  --output=FILE      Save predictions to file (optional)");
    println!();
    println!("INFO Options:");
    println!("  --model=FILE       Model file to inspect (required)");
    println!();
    println!("Examples:");
    println!("  forest_cuda create --trees=50 --max-depth=15 --save=model.bin");
    println!("  forest_cuda train --model=model.bin --data=train.csv --save=model_trained.bin");
    println!("  forest_cuda predict --model=model_trained.bin --data=test.csv --output=predictions.csv");
    println!("  forest_cuda info --model=model_trained.bin");
}

fn parse_split_criterion(value: &str) -> SplitCriterion {
    match value.to_lowercase().as_str() {
        "entropy" => SplitCriterion::Entropy,
        "mse" => SplitCriterion::MSE,
        "variancereduction" => SplitCriterion::VarianceReduction,
        _ => SplitCriterion::Gini,
    }
}

fn parse_task_mode(value: &str) -> TaskType {
    match value.to_lowercase().as_str() {
        "regression" => TaskType::Regression,
        _ => TaskType::Classification,
    }
}

fn parse_args(args: &[String]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for arg in args {
        if let Some(eq_pos) = arg.find('=') {
            let key = arg[..eq_pos].to_string();
            let value = arg[eq_pos + 1..].to_string();
            map.insert(key, value);
        }
    }
    map
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_help();
        return;
    }

    let command = args[1].to_lowercase();

    match command.as_str() {
        "help" | "--help" | "-h" => {
            print_help();
        }
        "create" => {
            let params = parse_args(&args[2..]);

            let num_trees: usize = params.get("--trees").and_then(|v| v.parse().ok()).unwrap_or(100);
            let max_depth: i32 = params.get("--max-depth").and_then(|v| v.parse().ok()).unwrap_or(MAX_DEPTH_DEFAULT);
            let min_leaf: i32 = params.get("--min-leaf").and_then(|v| v.parse().ok()).unwrap_or(MIN_SAMPLES_LEAF_DEFAULT);
            let min_split: i32 = params.get("--min-split").and_then(|v| v.parse().ok()).unwrap_or(MIN_SAMPLES_SPLIT_DEFAULT);
            let max_features: i32 = params.get("--max-features").and_then(|v| v.parse().ok()).unwrap_or(0);
            let criterion = params.get("--criterion").map(|v| parse_split_criterion(v)).unwrap_or(SplitCriterion::Gini);
            let task = params.get("--task").map(|v| parse_task_mode(v)).unwrap_or(TaskType::Classification);
            let save_file = params.get("--save");

            if save_file.is_none() {
                eprintln!("Error: --save is required");
                std::process::exit(1);
            }

            let mut rf = TRandomForest::new();
            rf.set_num_trees(num_trees);
            rf.set_max_depth(max_depth);
            rf.set_min_samples_leaf(min_leaf);
            rf.set_min_samples_split(min_split);
            rf.set_max_features(max_features);
            rf.set_criterion(criterion);
            rf.set_task_type(task);

            println!("Created Random Forest model (CUDA/cudarc):");
            println!("  Number of trees: {}", num_trees);
            println!("  Max depth: {}", max_depth);
            println!("  Min samples leaf: {}", min_leaf);
            println!("  Min samples split: {}", min_split);
            println!("  Max features: {}", max_features);
            println!("  Criterion: {:?}", criterion);
            println!("  Task: {:?}", task);
            println!("  Saved to: {}", save_file.unwrap());

            rf.save_model(save_file.unwrap());
        }
        "train" => {
            let params = parse_args(&args[2..]);

            let model_file = params.get("--model");
            let data_file = params.get("--data");
            let save_file = params.get("--save");

            if model_file.is_none() {
                eprintln!("Error: --model is required");
                std::process::exit(1);
            }
            if data_file.is_none() {
                eprintln!("Error: --data is required");
                std::process::exit(1);
            }
            if save_file.is_none() {
                eprintln!("Error: --save is required");
                std::process::exit(1);
            }

            let mut rf = TRandomForest::new();
            if !rf.load_model(model_file.unwrap()) {
                eprintln!("Failed to load model");
                std::process::exit(1);
            }

            if !rf.load_csv(data_file.unwrap(), -1, true) {
                eprintln!("Failed to load data");
                std::process::exit(1);
            }

            println!("Training forest (CUDA/cudarc)...");
            rf.fit();
            println!("Training complete.");

            rf.save_model(save_file.unwrap());
        }
        "predict" => {
            let params = parse_args(&args[2..]);

            let model_file = params.get("--model");
            let data_file = params.get("--data");
            let output_file = params.get("--output");

            if model_file.is_none() {
                eprintln!("Error: --model is required");
                std::process::exit(1);
            }
            if data_file.is_none() {
                eprintln!("Error: --data is required");
                std::process::exit(1);
            }

            let mut rf = TRandomForest::new();
            if !rf.load_model(model_file.unwrap()) {
                eprintln!("Failed to load model");
                std::process::exit(1);
            }

            println!("Making predictions (CUDA/cudarc)...");

            if let Some(out) = output_file {
                rf.predict_csv(data_file.unwrap(), out, true);
            } else {
                rf.predict_csv(data_file.unwrap(), "predictions.csv", true);
            }
        }
        "info" => {
            let params = parse_args(&args[2..]);

            let model_file = params.get("--model");

            if model_file.is_none() {
                eprintln!("Error: --model is required");
                std::process::exit(1);
            }

            let mut rf = TRandomForest::new();
            if !rf.load_model(model_file.unwrap()) {
                eprintln!("Failed to load model");
                std::process::exit(1);
            }

            println!();
            println!("Random Forest Model Information (CUDA/cudarc)");
            println!("=============================================");
            rf.print_forest_info();
        }
        _ => {
            eprintln!("Unknown command: {}", command);
            println!();
            print_help();
            std::process::exit(1);
        }
    }
}
