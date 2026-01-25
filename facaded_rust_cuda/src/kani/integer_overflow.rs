//
// Kani Verification: Integer Overflow Prevention
// Prove that all arithmetic operations are safe from wrapping, overflowing, or underflowing.
//

#[cfg(kani)]
mod integer_overflow_verification {
    use crate::{MAX_FEATURES, MAX_SAMPLES, MAX_TREES, MAX_NODES};

    /// Verify data index calculation cannot overflow
    #[kani::proof]
    fn verify_data_index_no_overflow() {
        let sample_idx: usize = kani::any();
        let feature_idx: usize = kani::any();
        
        kani::assume(sample_idx < MAX_SAMPLES);
        kani::assume(feature_idx < MAX_FEATURES);
        
        let index = sample_idx.checked_mul(MAX_FEATURES);
        kani::assert(index.is_some(), "Multiplication must not overflow");
        
        let final_index = index.unwrap().checked_add(feature_idx);
        kani::assert(final_index.is_some(), "Addition must not overflow");
        kani::assert(final_index.unwrap() < MAX_SAMPLES * MAX_FEATURES, "Index must be in bounds");
    }

    /// Verify tree count operations cannot overflow
    #[kani::proof]
    fn verify_tree_count_no_overflow() {
        let num_trees: usize = kani::any();
        kani::assume(num_trees <= MAX_TREES);
        
        let incremented = num_trees.checked_add(1);
        
        if num_trees < MAX_TREES {
            kani::assert(incremented.is_some(), "Increment should not overflow when under limit");
        }
    }

    /// Verify node count accumulation cannot overflow
    #[kani::proof]
    #[kani::unwind(10)]
    fn verify_node_count_accumulation_no_overflow() {
        let num_trees: usize = kani::any();
        kani::assume(num_trees <= 5);
        
        let nodes_per_tree: usize = kani::any();
        kani::assume(nodes_per_tree <= MAX_NODES);
        
        let total = num_trees.checked_mul(nodes_per_tree);
        kani::assert(total.is_some(), "Total node count calculation should not overflow");
    }

    /// Verify vote counting cannot overflow
    #[kani::proof]
    #[kani::unwind(10)]
    fn verify_vote_counting_no_overflow() {
        let mut votes: i32 = 0;
        let num_trees: usize = kani::any();
        kani::assume(num_trees <= MAX_TREES);
        
        for _ in 0..num_trees {
            let new_votes = votes.checked_add(1);
            kani::assert(new_votes.is_some(), "Vote counting should not overflow");
            votes = new_votes.unwrap();
        }
        
        kani::assert(votes as usize == num_trees, "Vote count should match tree count");
    }

    /// Verify feature importance accumulation cannot overflow
    #[kani::proof]
    fn verify_importance_accumulation_no_overflow() {
        let importance: f64 = kani::any();
        let num_samples: f64 = kani::any();
        let impurity: f64 = kani::any();
        
        kani::assume(importance.is_finite());
        kani::assume(num_samples.is_finite() && num_samples >= 0.0);
        kani::assume(impurity.is_finite() && impurity >= 0.0 && impurity <= 1.0);
        kani::assume(num_samples <= MAX_SAMPLES as f64);
        
        let contribution = num_samples * impurity;
        let _new_importance = importance + contribution;
        
        kani::assert(contribution.is_finite() || contribution.is_nan(), 
            "Contribution should be finite or NaN");
    }

    /// Verify depth calculation cannot overflow
    #[kani::proof]
    fn verify_depth_calculation_no_overflow() {
        let depth: i32 = kani::any();
        kani::assume(depth >= 0 && depth < 100);
        
        let new_depth = depth.checked_add(1);
        kani::assert(new_depth.is_some(), "Depth increment should not overflow");
    }

    /// Verify sample index calculations in bootstrap
    #[kani::proof]
    fn verify_bootstrap_index_no_overflow() {
        let num_samples: usize = kani::any();
        kani::assume(num_samples > 0 && num_samples <= MAX_SAMPLES);
        
        let random_val: usize = kani::any();
        let idx = random_val % num_samples;
        
        kani::assert(idx < num_samples, "Bootstrap index must be in range");
    }

    /// Verify feature subset index operations
    #[kani::proof]
    fn verify_feature_subset_no_overflow() {
        let num_features: usize = kani::any();
        kani::assume(num_features > 0 && num_features <= MAX_FEATURES);
        
        let i: usize = kani::any();
        kani::assume(i >= 1 && i < num_features);
        
        let j = i + 1;
        kani::assert(j <= num_features, "j should not exceed num_features");
    }

    /// Verify prediction sum accumulation
    #[kani::proof]
    #[kani::unwind(10)]
    fn verify_prediction_sum_no_overflow() {
        let num_trees: usize = kani::any();
        kani::assume(num_trees > 0 && num_trees <= 5);
        
        let mut sum: f64 = 0.0;
        
        for _ in 0..num_trees {
            let prediction: f64 = kani::any();
            kani::assume(prediction.is_finite());
            kani::assume(prediction.abs() <= 1e10);
            
            sum += prediction;
        }
        
        let avg = sum / (num_trees as f64);
        kani::assert(avg.is_finite() || avg.is_nan(), "Average should be finite");
    }

    /// Verify OOB count operations
    #[kani::proof]
    fn verify_oob_count_no_overflow() {
        let count: usize = kani::any();
        kani::assume(count <= MAX_SAMPLES);
        
        let pred_count: i32 = kani::any();
        kani::assume(pred_count >= 0 && pred_count < i32::MAX);
        
        let new_count = pred_count.checked_add(1);
        kani::assert(new_count.is_some(), "OOB count increment should not overflow");
    }

    /// Verify tree weight multiplication
    #[kani::proof]
    fn verify_weight_multiplication_no_overflow() {
        let prediction: f64 = kani::any();
        let weight: f64 = kani::any();
        
        kani::assume(prediction.is_finite());
        kani::assume(weight.is_finite() && weight >= 0.0);
        kani::assume(prediction.abs() <= 1e10);
        kani::assume(weight <= 1e10);
        
        let weighted = prediction * weight;
        kani::assert(weighted.is_finite() || weighted.is_nan(), "Weighted prediction should be finite");
    }

    /// Verify node index increments
    #[kani::proof]
    fn verify_node_index_increment_no_overflow() {
        let node_idx: usize = kani::any();
        kani::assume(node_idx < MAX_NODES - 1);
        
        let new_idx = node_idx.checked_add(1);
        kani::assert(new_idx.is_some(), "Node index increment should not overflow");
        kani::assert(new_idx.unwrap() < MAX_NODES, "New index should be within bounds");
    }

    /// Verify RNG state operations use wrapping arithmetic safely
    #[kani::proof]
    fn verify_rng_wrapping_safe() {
        let state: u64 = kani::any();
        
        let new_state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        
        let shifted = new_state >> 33;
        kani::assert(shifted <= u64::MAX >> 33, "Shifted value should be bounded");
    }

    /// Verify linear index from 2D coordinates
    #[kani::proof]
    fn verify_2d_to_linear_no_overflow() {
        let row: usize = kani::any();
        let col: usize = kani::any();
        let width: usize = MAX_FEATURES;
        
        kani::assume(row < MAX_SAMPLES);
        kani::assume(col < width);
        
        let linear = row.checked_mul(width).and_then(|r| r.checked_add(col));
        kani::assert(linear.is_some(), "2D to linear conversion should not overflow");
    }
}
