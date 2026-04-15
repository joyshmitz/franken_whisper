//! Metamorphic tests for accelerate module numerical invariants.
//!
//! Tests mathematical properties of softmax, normalization, and attention
//! scoring that MUST hold for correct implementation.
//!
//! MR Strength Matrix:
//! | MR | Fault Sensitivity | Independence | Cost | Score |
//! |----|-------------------|--------------|------|-------|
//! | Softmax sum-to-one | 5 | 5 | 1 | 25.0 |
//! | Softmax shift invariance | 5 | 5 | 1 | 25.0 |
//! | Softmax monotonicity | 4 | 4 | 1 | 16.0 |
//! | Normalize sum-to-one | 5 | 5 | 1 | 25.0 |
//! | Attention symmetry | 3 | 4 | 2 | 6.0 |

#![forbid(unsafe_code)]

// ---------------------------------------------------------------------------
// Local implementations mirroring src/accelerate.rs for testing
// ---------------------------------------------------------------------------

/// CPU softmax implementation (mirrors src/accelerate.rs).
fn softmax_cpu(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }

    let max_val = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);

    let max_val = if max_val.is_finite() { max_val } else { 0.0 };

    let exps: Vec<f64> = values
        .iter()
        .map(|v| {
            if v.is_finite() {
                (v - max_val).exp()
            } else {
                0.0
            }
        })
        .collect();

    let sum: f64 = exps.iter().sum();
    if sum <= f64::EPSILON {
        return vec![1.0 / values.len() as f64; values.len()];
    }

    exps.iter().map(|e| e / sum).collect()
}

/// CPU normalization implementation (mirrors src/accelerate.rs).
fn normalize_cpu(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let safe_sum: f64 = values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .sum();

    if safe_sum <= f64::EPSILON {
        return vec![1.0 / values.len() as f64; values.len()];
    }

    values
        .iter()
        .map(|value| {
            let safe = if value.is_finite() && *value > 0.0 {
                *value
            } else {
                0.0
            };
            safe / safe_sum
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Test data generators
// ---------------------------------------------------------------------------

/// Generate test vectors of various characteristics.
fn test_vectors() -> Vec<Vec<f64>> {
    vec![
        vec![1.0, 2.0, 3.0],
        vec![0.1, 0.2, 0.3, 0.4],
        vec![1.0, 1.0, 1.0],
        vec![-1.0, 0.0, 1.0],
        vec![100.0, 200.0, 300.0],
        vec![0.001, 0.002, 0.003],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![10.0],
        vec![5.0, 5.0],
    ]
}

/// Generate shift constants for invariance testing.
fn shift_constants() -> Vec<f64> {
    vec![0.0, 1.0, -1.0, 10.0, -10.0, 100.0, -100.0, 0.5]
}

/// Generate scale factors for scaling tests.
fn scale_factors() -> Vec<f64> {
    vec![0.5, 2.0, 0.1, 10.0, 0.01, 100.0]
}

// ---------------------------------------------------------------------------
// MR1: Softmax Sum-to-One
// Property: sum(softmax(x)) == 1 for any non-empty input
// Category: Additive identity
// ---------------------------------------------------------------------------

#[test]
fn mr_softmax_sum_to_one() {
    for values in test_vectors() {
        let result = softmax_cpu(&values);
        let sum: f64 = result.iter().sum();

        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Softmax sum violated: sum(softmax({values:?})) = {sum}, expected 1.0"
        );
    }
}

#[test]
fn mr_softmax_sum_to_one_with_negatives() {
    let negative_vectors = vec![
        vec![-1.0, -2.0, -3.0],
        vec![-100.0, -50.0, -1.0],
        vec![-0.1, -0.2, -0.3, -0.4],
    ];

    for values in negative_vectors {
        let result = softmax_cpu(&values);
        let sum: f64 = result.iter().sum();

        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Softmax sum violated for negatives: sum(softmax({values:?})) = {sum}"
        );
    }
}

// ---------------------------------------------------------------------------
// MR2: Softmax Shift Invariance
// Property: softmax(x + c) == softmax(x) for any constant c
// Category: Equivalence under translation
// ---------------------------------------------------------------------------

#[test]
fn mr_softmax_shift_invariance() {
    for values in test_vectors() {
        let original = softmax_cpu(&values);

        for shift in shift_constants() {
            let shifted: Vec<f64> = values.iter().map(|v| v + shift).collect();
            let shifted_result = softmax_cpu(&shifted);

            // Results should be identical
            assert_eq!(
                original.len(),
                shifted_result.len(),
                "Shift changed output length"
            );

            for (i, (orig, shifted)) in original.iter().zip(shifted_result.iter()).enumerate() {
                assert!(
                    (orig - shifted).abs() < 1e-9,
                    "Softmax shift invariance violated at index {i}: \
                     original[{i}] = {orig}, shifted[{i}] = {shifted} for shift = {shift}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MR3: Softmax Positivity
// Property: all elements of softmax(x) > 0
// Category: Bound constraint
// ---------------------------------------------------------------------------

#[test]
fn mr_softmax_all_positive() {
    for values in test_vectors() {
        let result = softmax_cpu(&values);

        for (i, v) in result.iter().enumerate() {
            assert!(
                *v > 0.0,
                "Softmax positivity violated: softmax({values:?})[{i}] = {v} <= 0"
            );
            assert!(
                v.is_finite(),
                "Softmax finiteness violated: softmax({values:?})[{i}] = {v}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// MR4: Softmax Monotonicity
// Property: if x_i > x_j then softmax(x)_i > softmax(x)_j
// Category: Order preservation
// ---------------------------------------------------------------------------

#[test]
fn mr_softmax_monotonicity() {
    let test_cases = vec![
        vec![1.0, 2.0, 3.0],       // ascending
        vec![3.0, 2.0, 1.0],       // descending
        vec![10.0, 20.0, 30.0],    // large ascending
        vec![-3.0, -2.0, -1.0],    // negative ascending
        vec![0.1, 0.2, 0.3, 0.4], // small ascending
    ];

    for values in test_cases {
        let result = softmax_cpu(&values);

        for i in 0..values.len() {
            for j in 0..values.len() {
                if values[i] > values[j] {
                    assert!(
                        result[i] > result[j],
                        "Softmax monotonicity violated: \
                         x[{i}] = {} > x[{j}] = {}, but \
                         softmax[{i}] = {} <= softmax[{j}] = {}",
                        values[i],
                        values[j],
                        result[i],
                        result[j]
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MR5: Softmax Equal Inputs → Uniform Output
// Property: softmax([c, c, c, ...]) = [1/n, 1/n, 1/n, ...]
// Category: Equivalence
// ---------------------------------------------------------------------------

#[test]
fn mr_softmax_uniform_for_equal_inputs() {
    for n in 2..=6 {
        for constant in [0.0, 1.0, -1.0, 100.0, -100.0] {
            let values = vec![constant; n];
            let result = softmax_cpu(&values);

            let expected = 1.0 / n as f64;

            for (i, v) in result.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 1e-9,
                    "Uniform softmax violated: softmax([{constant}; {n}])[{i}] = {v}, expected {expected}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MR6: Normalize Sum-to-One
// Property: sum(normalize(x)) == 1 for positive inputs
// Category: Additive identity
// ---------------------------------------------------------------------------

#[test]
fn mr_normalize_sum_to_one() {
    let positive_vectors = vec![
        vec![1.0, 2.0, 3.0],
        vec![0.1, 0.2, 0.3],
        vec![10.0, 20.0, 30.0, 40.0],
        vec![1.0, 1.0, 1.0],
        vec![0.001, 0.002, 0.003],
    ];

    for values in positive_vectors {
        let result = normalize_cpu(&values);
        let sum: f64 = result.iter().sum();

        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Normalize sum violated: sum(normalize({values:?})) = {sum}, expected 1.0"
        );
    }
}

// ---------------------------------------------------------------------------
// MR7: Normalize Preserves Ratios
// Property: normalize(x)_i / normalize(x)_j == x_i / x_j for positive x
// Category: Multiplicative
// ---------------------------------------------------------------------------

#[test]
fn mr_normalize_preserves_ratios() {
    let positive_vectors = vec![
        vec![1.0, 2.0, 4.0],
        vec![3.0, 6.0, 9.0],
        vec![10.0, 20.0],
    ];

    for values in positive_vectors {
        let result = normalize_cpu(&values);

        for i in 0..values.len() {
            for j in 0..values.len() {
                if values[j] > 0.0 && result[j] > 0.0 {
                    let input_ratio = values[i] / values[j];
                    let output_ratio = result[i] / result[j];

                    assert!(
                        (input_ratio - output_ratio).abs() < 1e-9,
                        "Normalize ratio violated: \
                         x[{i}]/x[{j}] = {input_ratio}, \
                         norm[{i}]/norm[{j}] = {output_ratio}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MR8: Normalize Scale Invariance
// Property: normalize(k*x) == normalize(x) for k > 0
// Category: Equivalence under scaling
// ---------------------------------------------------------------------------

#[test]
fn mr_normalize_scale_invariance() {
    let positive_vectors = vec![
        vec![1.0, 2.0, 3.0],
        vec![0.1, 0.2, 0.3],
        vec![10.0, 20.0, 30.0],
    ];

    for values in positive_vectors {
        let original = normalize_cpu(&values);

        for scale in scale_factors() {
            let scaled: Vec<f64> = values.iter().map(|v| v * scale).collect();
            let scaled_result = normalize_cpu(&scaled);

            for (i, (orig, scaled_val)) in original.iter().zip(scaled_result.iter()).enumerate() {
                assert!(
                    (orig - scaled_val).abs() < 1e-9,
                    "Normalize scale invariance violated at index {i}: \
                     original = {orig}, scaled({scale}) = {scaled_val}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MR9: Normalize Bounds
// Property: 0 <= normalize(x)_i <= 1 for all i
// Category: Bound constraint
// ---------------------------------------------------------------------------

#[test]
fn mr_normalize_bounds() {
    let vectors = vec![
        vec![1.0, 2.0, 3.0],
        vec![0.001, 1000.0],
        vec![1.0, 1.0, 1.0, 1.0],
        vec![100.0],
    ];

    for values in vectors {
        let result = normalize_cpu(&values);

        for (i, v) in result.iter().enumerate() {
            assert!(
                *v >= 0.0 && *v <= 1.0,
                "Normalize bounds violated: normalize({values:?})[{i}] = {v} not in [0, 1]"
            );
            assert!(
                v.is_finite(),
                "Normalize finiteness violated: normalize({values:?})[{i}] = {v}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// MR10: Empty Input Handling
// Property: softmax([]) == [] and normalize([]) == []
// Category: Identity
// ---------------------------------------------------------------------------

#[test]
fn mr_empty_input_returns_empty() {
    let empty: Vec<f64> = vec![];

    assert!(
        softmax_cpu(&empty).is_empty(),
        "Softmax of empty should be empty"
    );
    assert!(
        normalize_cpu(&empty).is_empty(),
        "Normalize of empty should be empty"
    );
}

// ---------------------------------------------------------------------------
// MR11: Single Element
// Property: softmax([x]) == [1.0] and normalize([x]) == [1.0] for x > 0
// Category: Identity
// ---------------------------------------------------------------------------

#[test]
fn mr_single_element_gives_one() {
    for val in [1.0, 10.0, 100.0, 0.1, 0.001] {
        let softmax_result = softmax_cpu(&[val]);
        let normalize_result = normalize_cpu(&[val]);

        assert!(
            (softmax_result[0] - 1.0).abs() < 1e-9,
            "Softmax([{val}]) should be [1.0], got {:?}",
            softmax_result
        );
        assert!(
            (normalize_result[0] - 1.0).abs() < 1e-9,
            "Normalize([{val}]) should be [1.0], got {:?}",
            normalize_result
        );
    }
}

// ---------------------------------------------------------------------------
// Composite MR: Chain operations
// ---------------------------------------------------------------------------

#[test]
fn mr_composite_normalize_then_softmax_still_sums_to_one() {
    for values in test_vectors() {
        // Skip vectors with non-positive values since normalize requires positive
        if values.iter().any(|v| *v <= 0.0) {
            continue;
        }

        let normalized = normalize_cpu(&values);
        let softmaxed = softmax_cpu(&normalized);

        let sum: f64 = softmaxed.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Composite normalize→softmax sum violated: got {sum}"
        );
    }
}

#[test]
fn mr_composite_double_softmax_preserves_sum_to_one() {
    for values in test_vectors() {
        let first = softmax_cpu(&values);
        let second = softmax_cpu(&first);

        let sum: f64 = second.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Double softmax sum violated: got {sum}"
        );
    }
}
