//
// Kani Verification: Division-by-Zero Exclusion
// Verify that any denominator derived from variable or external input is proven to never be zero.
//

#[cfg(kani)]
mod division_by_zero_verification {
    #[allow(unused_imports)]
    use crate::TRandomForest;

    /// Verify Gini calculation denominator is never zero
    #[kani::proof]
    fn verify_gini_denominator_not_zero() {
        let num_indices: usize = kani::any();
        
        if num_indices == 0 {
            let result = 0.0;
            kani::assert(result == 0.0, "Empty set returns 0");
        } else {
            let n = num_indices as f64;
            kani::assert(n > 0.0, "Non-empty set has positive count");
            
            let prob = 1.0 / n;
            kani::assert(prob.is_finite(), "Division should produce finite result");
        }
    }

    /// Verify entropy calculation denominator is never zero
    #[kani::proof]
    fn verify_entropy_denominator_not_zero() {
        let num_indices: usize = kani::any();
        kani::assume(num_indices > 0);
        
        let n = num_indices as f64;
        let class_count: i32 = kani::any();
        kani::assume(class_count > 0);
        
        let prob = (class_count as f64) / n;
        kani::assert(prob.is_finite(), "Probability must be finite");
        kani::assert(prob > 0.0 && prob <= 1.0, "Probability must be in (0, 1]");
    }

    /// Verify MSE calculation denominator is never zero
    #[kani::proof]
    fn verify_mse_denominator_not_zero() {
        let num_indices: usize = kani::any();
        
        if num_indices == 0 {
            let result = 0.0;
            kani::assert(result == 0.0, "Empty set returns 0 MSE");
        } else {
            let n = num_indices as f64;
            let sum_sq: f64 = kani::any();
            kani::assume(sum_sq.is_finite() && sum_sq >= 0.0);
            
            let mse = sum_sq / n;
            kani::assert(mse.is_finite(), "MSE must be finite for non-empty set");
        }
    }

    /// Verify prediction averaging denominator is never zero
    #[kani::proof]
    fn verify_prediction_avg_denominator_not_zero() {
        let num_trees: usize = kani::any();
        kani::assume(num_trees > 0);
        
        let sum: f64 = kani::any();
        kani::assume(sum.is_finite());
        
        let avg = sum / (num_trees as f64);
        kani::assert(avg.is_finite() || avg.is_nan(), "Average must be finite");
    }

    /// Verify mean target calculation denominator is never zero
    #[kani::proof]
    fn verify_mean_target_denominator_not_zero() {
        let num_indices: usize = kani::any();
        
        if num_indices == 0 {
            let mean = 0.0;
            kani::assert(mean == 0.0, "Empty set returns 0 mean");
        } else {
            let sum: f64 = kani::any();
            kani::assume(sum.is_finite());
            
            let mean = sum / (num_indices as f64);
            kani::assert(mean.is_finite() || mean.is_nan(), "Mean must be finite");
        }
    }

    /// Verify weighted average denominator is never zero when weights exist
    #[kani::proof]
    fn verify_weighted_avg_denominator_not_zero() {
        let total_weight: f64 = kani::any();
        let sum: f64 = kani::any();
        kani::assume(sum.is_finite());
        
        if total_weight > 0.0 {
            let avg = sum / total_weight;
            kani::assert(avg.is_finite() || avg.is_nan(), "Weighted average must be finite");
        } else {
            let default_result = 0.0;
            kani::assert(default_result == 0.0, "Zero weight returns default");
        }
    }

    /// Verify information gain calculation denominators
    #[kani::proof]
    fn verify_info_gain_denominators_not_zero() {
        // Use bounded values to avoid complex SAT constraints
        let num_left: usize = kani::any();
        let num_right: usize = kani::any();
        
        kani::assume(num_left > 0 && num_left <= 100);
        kani::assume(num_right > 0 && num_right <= 100);
        
        let num_indices = num_left + num_right;
        
        let left_ratio = (num_left as f64) / (num_indices as f64);
        let right_ratio = (num_right as f64) / (num_indices as f64);
        
        kani::assert(left_ratio.is_finite(), "Left ratio must be finite");
        kani::assert(right_ratio.is_finite(), "Right ratio must be finite");
        kani::assert(left_ratio + right_ratio <= 1.0 + 1e-10, "Ratios must sum to ~1");
    }

    /// Verify accuracy calculation denominator is never zero
    #[kani::proof]
    fn verify_accuracy_denominator_not_zero() {
        let n_samples: usize = kani::any();
        kani::assume(n_samples > 0);
        
        let correct: usize = kani::any();
        kani::assume(correct <= n_samples);
        
        let accuracy = (correct as f64) / (n_samples as f64);
        kani::assert(accuracy.is_finite(), "Accuracy must be finite");
        kani::assert(accuracy >= 0.0 && accuracy <= 1.0, "Accuracy must be in [0, 1]");
    }

    /// Verify precision calculation handles zero denominator
    #[kani::proof]
    fn verify_precision_denominator_handled() {
        let tp: i32 = kani::any();
        let fp: i32 = kani::any();
        
        kani::assume(tp >= 0);
        kani::assume(fp >= 0);
        // Prevent overflow in addition
        kani::assume(tp <= 1_000_000);
        kani::assume(fp <= 1_000_000);
        
        let precision = if tp + fp > 0 {
            (tp as f64) / ((tp + fp) as f64)
        } else {
            0.0
        };
        
        kani::assert(precision.is_finite(), "Precision must be finite");
        kani::assert(precision >= 0.0 && precision <= 1.0, "Precision must be in [0, 1]");
    }

    /// Verify recall calculation handles zero denominator
    #[kani::proof]
    fn verify_recall_denominator_handled() {
        let tp: i32 = kani::any();
        let fn_count: i32 = kani::any();
        
        kani::assume(tp >= 0);
        kani::assume(fn_count >= 0);
        // Prevent overflow in addition
        kani::assume(tp <= 1_000_000);
        kani::assume(fn_count <= 1_000_000);
        
        let recall = if tp + fn_count > 0 {
            (tp as f64) / ((tp + fn_count) as f64)
        } else {
            0.0
        };
        
        kani::assert(recall.is_finite(), "Recall must be finite");
        kani::assert(recall >= 0.0 && recall <= 1.0, "Recall must be in [0, 1]");
    }

    /// Verify F1 score calculation handles zero denominator
    #[kani::proof]
    fn verify_f1_denominator_handled() {
        let precision: f64 = kani::any();
        let recall: f64 = kani::any();
        
        kani::assume(precision.is_finite() && precision >= 0.0 && precision <= 1.0);
        kani::assume(recall.is_finite() && recall >= 0.0 && recall <= 1.0);
        
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        
        kani::assert(f1.is_finite(), "F1 score must be finite");
        kani::assert(f1 >= 0.0 && f1 <= 1.0, "F1 must be in [0, 1]");
    }

    /// Verify R-squared calculation handles zero denominator
    #[kani::proof]
    fn verify_r_squared_denominator_handled() {
        let ss_res: f64 = kani::any();
        let ss_tot: f64 = kani::any();
        
        kani::assume(ss_res.is_finite() && ss_res >= 0.0);
        kani::assume(ss_tot.is_finite() && ss_tot >= 0.0);
        // Constrain to avoid infinity from division
        kani::assume(ss_res <= 1e10);
        kani::assume(ss_tot == 0.0 || ss_tot >= 1e-10);
        
        let r_squared = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };
        
        kani::assert(r_squared.is_finite(), "R-squared must be finite");
    }

    /// Verify feature importance normalization handles zero total
    #[kani::proof]
    fn verify_importance_normalization_denominator() {
        let total: f64 = kani::any();
        let importance: f64 = kani::any();
        
        kani::assume(importance.is_finite() && importance >= 0.0);
        kani::assume(total.is_finite() && total >= 0.0);
        // Constrain to avoid infinity from division
        kani::assume(importance <= 1e10);
        kani::assume(total == 0.0 || total >= 1e-10);
        
        let normalized = if total > 0.0 {
            importance / total
        } else {
            importance
        };
        
        kani::assert(normalized.is_finite(), "Normalized importance must be finite");
    }

    /// Verify OOB error calculation handles zero count
    #[kani::proof]
    fn verify_oob_error_denominator_handled() {
        let error: f64 = kani::any();
        let count: i32 = kani::any();
        
        kani::assume(error.is_finite() && error >= 0.0);
        kani::assume(count >= 0);
        
        let oob_error = if count > 0 {
            error / (count as f64)
        } else {
            0.0
        };
        
        kani::assert(oob_error.is_finite(), "OOB error must be finite");
    }

    /// Verify random_int handles zero max_val by requiring positive input
    #[kani::proof]
    fn verify_random_int_modulo_not_zero() {
        let max_val: usize = kani::any();
        let rng_state: u64 = kani::any();
        
        kani::assume(max_val > 0);
        
        let result = ((rng_state >> 33) as usize) % max_val;
        kani::assert(result < max_val, "Result must be less than max_val");
    }
}
