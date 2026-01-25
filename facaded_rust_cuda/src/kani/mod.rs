//
// Kani Verification Test Suite for Facaded Random Forest (CISA Hardening)
// Matthew Abbott 2025
//
// This module contains formal verification harnesses using the Kani Rust Verifier
// to prove memory safety, arithmetic integrity, and security properties following
// CISA "Secure by Design" standards.
//

pub mod bounds_checks;
pub mod pointer_validity;
pub mod no_panic;
pub mod integer_overflow;
pub mod division_by_zero;
pub mod state_consistency;
pub mod deadlock_free;
pub mod input_sanitization;
pub mod result_coverage;
pub mod memory_leaks;
pub mod constant_time;
pub mod state_machine;
pub mod enum_exhaustion;
pub mod floating_point;
pub mod resource_limits;
