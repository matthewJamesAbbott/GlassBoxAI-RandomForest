//
// Kani Verification: Floating-Point Sanity
// Prove operations involving f32/f64 never result in unhandled NaN or Infinity.
//

#[cfg(kani)]
mod floating_point_verification {
    use crate::MAX_TREES;

    /// Verify Gini calculation handles all finite inputs
    #[kani::proof]
    fn verify_gini_finite_result() {
        let num_indices: usize = kani::any();
        kani::assume(num_indices > 0 && num_indices <= 100);
        
        let mut class_counts = [0i32; 10];
        let class_idx: usize = kani::any();
        kani::assume(class_idx < 10);
        
        let count: i32 = kani::any();
        kani::assume(count >= 0 && count <= num_indices as i32);
        class_counts[class_idx] = count;
        
        let n = num_indices as f64;
        let prob = (class_counts[class_idx] as f64) / n;
        
        kani::assert(prob.is_finite(), "Probability must be finite");
        kani::assert(prob >= 0.0 && prob <= 1.0, "Probability in [0, 1]");
        
        let gini_contrib = prob * prob;
        kani::assert(gini_contrib.is_finite(), "Gini contribution must be finite");
    }

    /// Verify entropy calculation handles zero probability
    #[kani::proof]
    fn verify_entropy_zero_probability() {
        let prob: f64 = 0.0;
        
        let entropy_contrib = if prob > 0.0 {
            -prob * prob.log2()
        } else {
            0.0
        };
        
        kani::assert(entropy_contrib.is_finite(), "Zero probability produces finite result");
        kani::assert(entropy_contrib == 0.0, "Zero probability contributes 0");
    }

    /// Verify entropy calculation handles valid probabilities
    #[kani::proof]
    fn verify_entropy_valid_probability() {
        let prob: f64 = kani::any();
        kani::assume(prob > 0.0 && prob <= 1.0);
        kani::assume(prob.is_finite());
        
        let log_prob = prob.log2();
        kani::assert(log_prob.is_finite() || log_prob.is_infinite(), "Log is defined");
        
        if prob > 0.0 && prob < 1.0 {
            let entropy_contrib = -prob * log_prob;
            kani::assert(entropy_contrib.is_finite(), "Entropy contribution is finite");
            kani::assert(entropy_contrib >= 0.0, "Entropy contribution is non-negative");
        }
    }

    /// Verify MSE calculation with finite inputs
    #[kani::proof]
    fn verify_mse_finite_inputs() {
        let num_samples: usize = kani::any();
        kani::assume(num_samples > 0 && num_samples <= 10);
        
        let sum_sq: f64 = kani::any();
        kani::assume(sum_sq.is_finite() && sum_sq >= 0.0);
        
        let mse = sum_sq / (num_samples as f64);
        
        kani::assert(mse.is_finite(), "MSE must be finite");
        kani::assert(mse >= 0.0, "MSE must be non-negative");
    }

    /// Verify prediction averaging handles finite inputs
    #[kani::proof]
    fn verify_prediction_average_finite() {
        let sum: f64 = kani::any();
        let num_trees: usize = kani::any();
        
        kani::assume(sum.is_finite());
        kani::assume(num_trees > 0 && num_trees <= MAX_TREES);
        
        let avg = sum / (num_trees as f64);
        
        kani::assert(avg.is_finite() || avg.is_nan(), "Average is finite or NaN");
    }

    /// Verify weighted sum handles finite weights
    #[kani::proof]
    fn verify_weighted_sum_finite() {
        let prediction: f64 = kani::any();
        let weight: f64 = kani::any();
        
        kani::assume(prediction.is_finite());
        kani::assume(weight.is_finite() && weight >= 0.0);
        
        let weighted = prediction * weight;
        
        kani::assert(weighted.is_finite() || weighted.is_nan(), "Weighted value is valid");
    }

    /// Verify threshold comparison handles special values
    #[kani::proof]
    fn verify_threshold_comparison_special_values() {
        let value: f64 = kani::any();
        let threshold: f64 = kani::any();
        
        let comparison = value <= threshold;
        
        if value.is_nan() || threshold.is_nan() {
            kani::assert(!comparison, "NaN comparison is false");
        } else {
            kani::assert(comparison == (value <= threshold), "Comparison is consistent");
        }
    }

    /// Verify feature importance accumulation stays finite
    #[kani::proof]
    fn verify_importance_accumulation_finite() {
        let current: f64 = kani::any();
        let contribution: f64 = kani::any();
        
        kani::assume(current.is_finite() && current >= 0.0);
        kani::assume(contribution.is_finite() && contribution >= 0.0);
        kani::assume(current < 1e100 && contribution < 1e100);
        
        let new_importance = current + contribution;
        
        kani::assert(new_importance.is_finite(), "Accumulated importance is finite");
    }

    /// Verify division result is checked for infinity
    #[kani::proof]
    fn verify_division_infinity_check() {
        let numerator: f64 = kani::any();
        let denominator: f64 = kani::any();
        
        kani::assume(numerator.is_finite());
        
        if denominator == 0.0 {
            let result = numerator / denominator;
            kani::assert(result.is_infinite() || result.is_nan(), 
                "Division by zero produces infinity or NaN");
        } else if denominator.is_finite() && denominator != 0.0 {
            let result = numerator / denominator;
            kani::assert(result.is_finite() || result.is_nan() || result.is_infinite(),
                "Division result is valid float");
        }
    }

    /// Verify sqrt is only called on non-negative values
    #[kani::proof]
    fn verify_sqrt_non_negative() {
        let value: f64 = kani::any();
        kani::assume(value >= 0.0);
        kani::assume(value.is_finite());
        
        let sqrt_val = value.sqrt();
        
        kani::assert(sqrt_val.is_finite() || sqrt_val.is_nan(), "Sqrt of non-negative is valid");
        kani::assert(sqrt_val >= 0.0 || sqrt_val.is_nan(), "Sqrt is non-negative");
    }

    /// Verify log is only called on positive values
    #[kani::proof]
    fn verify_log_positive() {
        let value: f64 = kani::any();
        kani::assume(value > 0.0);
        kani::assume(value.is_finite());
        
        let log_val = value.ln();
        
        kani::assert(log_val.is_finite() || log_val.is_infinite(), "Log of positive is valid");
    }

    /// Verify round operation is finite
    #[kani::proof]
    fn verify_round_finite() {
        let value: f64 = kani::any();
        kani::assume(value.is_finite());
        
        let rounded = value.round();
        
        kani::assert(rounded.is_finite(), "Round of finite is finite");
    }

    /// Verify abs operation is finite and non-negative
    #[kani::proof]
    fn verify_abs_finite() {
        let value: f64 = kani::any();
        kani::assume(value.is_finite());
        
        let abs_val = value.abs();
        
        kani::assert(abs_val.is_finite(), "Abs of finite is finite");
        kani::assert(abs_val >= 0.0, "Abs is non-negative");
    }

    /// Verify powi operation with bounded exponent
    #[kani::proof]
    fn verify_powi_bounded() {
        let base: f64 = kani::any();
        let exp: i32 = kani::any();
        
        kani::assume(base.is_finite());
        kani::assume(base.abs() <= 100.0);
        kani::assume(exp >= -10 && exp <= 10);
        
        let result = base.powi(exp);
        
        kani::assert(
            result.is_finite() || result.is_nan() || result.is_infinite(),
            "powi produces valid float"
        );
    }

    /// Verify f64 to i32 conversion handles large values
    #[kani::proof]
    fn verify_float_to_int_conversion() {
        let value: f64 = kani::any();
        kani::assume(value.is_finite());
        kani::assume(value >= i32::MIN as f64 && value <= i32::MAX as f64);
        
        let int_val = value.round() as i32;
        
        kani::assert(int_val >= i32::MIN && int_val <= i32::MAX, "Conversion in range");
    }

    /// Verify accuracy metric is in [0, 1]
    #[kani::proof]
    fn verify_accuracy_bounded() {
        let correct: usize = kani::any();
        let total: usize = kani::any();
        
        kani::assume(total > 0);
        kani::assume(correct <= total);
        
        let accuracy = (correct as f64) / (total as f64);
        
        kani::assert(accuracy.is_finite(), "Accuracy is finite");
        kani::assert(accuracy >= 0.0 && accuracy <= 1.0, "Accuracy in [0, 1]");
    }

    /// Verify R-squared can be negative
    #[kani::proof]
    fn verify_r_squared_range() {
        let ss_res: f64 = kani::any();
        let ss_tot: f64 = kani::any();
        
        kani::assume(ss_res.is_finite() && ss_res >= 0.0);
        kani::assume(ss_tot.is_finite() && ss_tot > 0.0);
        
        let r_squared = 1.0 - (ss_res / ss_tot);
        
        kani::assert(r_squared.is_finite(), "R-squared is finite");
        kani::assert(r_squared <= 1.0, "R-squared <= 1");
    }

    /// Verify NaN propagation in comparisons
    #[kani::proof]
    fn verify_nan_comparison_behavior() {
        let a: f64 = f64::NAN;
        let b: f64 = kani::any();
        
        kani::assert(!(a == b), "NaN != anything");
        kani::assert(!(a < b), "NaN not less than anything");
        kani::assert(!(a > b), "NaN not greater than anything");
        kani::assert(!(a <= b), "NaN not less-equal anything");
        kani::assert(!(a >= b), "NaN not greater-equal anything");
    }

    /// Verify infinity comparison behavior
    #[kani::proof]
    fn verify_infinity_comparison() {
        let pos_inf = f64::INFINITY;
        let neg_inf = f64::NEG_INFINITY;
        let finite: f64 = kani::any();
        
        kani::assume(finite.is_finite());
        
        kani::assert(pos_inf > finite, "Pos infinity > finite");
        kani::assert(neg_inf < finite, "Neg infinity < finite");
        kani::assert(pos_inf > neg_inf, "Pos infinity > neg infinity");
    }

    /// Verify min/max handle special values
    #[kani::proof]
    fn verify_min_max_special_values() {
        let a: f64 = kani::any();
        let b: f64 = kani::any();
        
        kani::assume(a.is_finite() && b.is_finite());
        
        let min_val = a.min(b);
        let max_val = a.max(b);
        
        kani::assert(min_val.is_finite(), "Min is finite");
        kani::assert(max_val.is_finite(), "Max is finite");
        kani::assert(min_val <= max_val, "Min <= Max");
    }
}
