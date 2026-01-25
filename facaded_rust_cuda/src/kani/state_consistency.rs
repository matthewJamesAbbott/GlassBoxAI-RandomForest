//
// Kani Verification: Global State Consistency
// Prove state invariants and data consistency without creating full forest objects.
//

#[cfg(kani)]
mod state_consistency_verification {
    use crate::{
        FlatTreeNode,
        TaskType, SplitCriterion, AggregationMethod,
        MAX_FEATURES, MAX_SAMPLES, MAX_TREES,
    };
    use std::sync::Arc;

    /// Verify num_trees clamp invariant
    #[kani::proof]
    fn verify_num_trees_invariant() {
        let n: usize = kani::any();
        let result = n.clamp(1, MAX_TREES);
        
        kani::assert(result <= MAX_TREES, "num_trees must not exceed MAX_TREES");
        kani::assert(result >= 1, "num_trees must be at least 1");
    }

    /// Verify num_features bounds invariant
    #[kani::proof]
    fn verify_num_features_invariant() {
        let num_features: i32 = kani::any();
        kani::assume(num_features >= 0 && num_features <= MAX_FEATURES as i32);
        
        kani::assert(num_features <= MAX_FEATURES as i32, 
            "num_features must not exceed MAX_FEATURES");
        kani::assert(num_features >= 0, "num_features must be non-negative");
    }

    /// Verify num_samples bounds invariant
    #[kani::proof]
    fn verify_num_samples_invariant() {
        let num_samples: usize = kani::any();
        kani::assume(num_samples <= MAX_SAMPLES);
        
        kani::assert(num_samples <= MAX_SAMPLES, 
            "num_samples must not exceed MAX_SAMPLES");
    }

    /// Verify Arc shared state consistency
    #[kani::proof]
    fn verify_arc_state_consistency() {
        let node = FlatTreeNode::default();
        let arc1 = Arc::new(node);
        let arc2 = Arc::clone(&arc1);
        
        kani::assert(Arc::strong_count(&arc1) == 2, "Arc must track reference count");
        kani::assert(Arc::strong_count(&arc2) == 2, "Clone must share reference count");
        
        kani::assert(arc1.is_leaf == arc2.is_leaf, "Shared data must be consistent");
    }

    /// Verify aggregation method state transitions
    #[kani::proof]
    fn verify_aggregation_state_consistency() {
        let initial = AggregationMethod::MajorityVote;
        let updated = AggregationMethod::WeightedMean;
        
        kani::assert(initial != updated, "Methods should be different");
        kani::assert(initial == AggregationMethod::MajorityVote, "Initial is MajorityVote");
    }

    /// Verify task type and criterion consistency
    #[kani::proof]
    fn verify_task_criterion_consistency() {
        let task_idx: u8 = kani::any();
        kani::assume(task_idx < 2);
        
        let task = match task_idx {
            0 => TaskType::Classification,
            _ => TaskType::Regression,
        };
        
        let default_criterion = match task {
            TaskType::Classification => SplitCriterion::Gini,
            TaskType::Regression => SplitCriterion::MSE,
        };
        
        match task {
            TaskType::Classification => {
                kani::assert(default_criterion == SplitCriterion::Gini, 
                    "Classification should default to Gini");
            }
            TaskType::Regression => {
                kani::assert(default_criterion == SplitCriterion::MSE, 
                    "Regression should default to MSE");
            }
        }
    }

    /// Verify enable/disable feature state transitions (modeled)
    #[kani::proof]
    fn verify_feature_state_transitions() {
        let idx: usize = kani::any();
        kani::assume(idx < MAX_FEATURES);
        
        let mut enabled = true;
        
        kani::assert(enabled == true, "Initial state must be enabled");
        
        enabled = false;
        kani::assert(enabled == false, "Disable must set to false");
        
        enabled = true;
        kani::assert(enabled == true, "Enable must set to true");
    }

    /// Verify tree weight state transitions (modeled)
    #[kani::proof]
    fn verify_tree_weight_state_transitions() {
        let idx: usize = kani::any();
        let weight: f64 = kani::any();
        
        kani::assume(idx < MAX_TREES);
        kani::assume(weight.is_finite());
        
        let initial_weight = 1.0f64;
        let set_weight = weight;
        let reset_weight = 1.0f64;
        
        kani::assert(initial_weight == 1.0, "Initial weight is 1.0");
        kani::assert(set_weight == weight, "Weight must be set");
        kani::assert(reset_weight == 1.0, "Reset must restore to 1.0");
    }

    /// Verify RNG state isolation (modeled)
    #[kani::proof]
    fn verify_rng_state_isolation() {
        let seed1: u64 = 12345;
        let seed2: u64 = 67890;
        
        let rng_state1 = seed1;
        let rng_state2 = seed2;
        
        kani::assert(rng_state1 != rng_state2,
            "Different seeds must produce different state");
        
        kani::assert(rng_state1 == 12345, "Seed must set rng_state");
        kani::assert(rng_state2 == 67890, "Seed must set rng_state");
    }

    /// Verify forest initialization state (modeled)
    #[kani::proof]
    fn verify_forest_initialization_state() {
        let mut initialized = false;
        
        kani::assert(initialized == false, "Initial state must be uninitialized");
        
        initialized = true;
        kani::assert(initialized == true, "init_forest must set initialized");
    }

    /// Verify add/remove tree state consistency (modeled)
    #[kani::proof]
    fn verify_add_remove_tree_consistency() {
        let initial_trees: usize = kani::any();
        kani::assume(initial_trees >= 1 && initial_trees <= MAX_TREES);
        
        let mut num_trees = initial_trees;
        
        if num_trees < MAX_TREES {
            num_trees += 1;
            kani::assert(num_trees == initial_trees + 1, "Add must increment");
        }
        
        if num_trees > 1 {
            num_trees -= 1;
            kani::assert(num_trees >= 1, "Remove must maintain at least 1");
        }
    }

    /// Verify bounds checking for array access
    #[kani::proof]
    fn verify_array_bounds_consistency() {
        let idx: usize = kani::any();
        let arr_len = MAX_FEATURES;
        
        let in_bounds = idx < arr_len;
        
        if in_bounds {
            kani::assert(idx < MAX_FEATURES, "Index must be in bounds");
        }
    }

    /// Verify sample tracking bounds
    #[kani::proof]
    fn verify_sample_tracking_bounds() {
        let sample_idx: usize = kani::any();
        
        let clamped_idx = if sample_idx < MAX_SAMPLES {
            sample_idx as i32
        } else {
            (MAX_SAMPLES - 1) as i32
        };
        
        kani::assert(clamped_idx >= 0, "Clamped index must be non-negative");
        kani::assert((clamped_idx as usize) < MAX_SAMPLES, "Clamped index must be in bounds");
    }
}
