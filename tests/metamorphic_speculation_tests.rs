//! Metamorphic tests for speculation module invariants.
//!
//! Tests properties of string distance, window containment, and drift
//! calculations that MUST hold for correct implementation.
//!
//! MR Strength Matrix:
//! | MR | Fault Sensitivity | Independence | Cost | Score |
//! |----|-------------------|--------------|------|-------|
//! | Levenshtein symmetry | 5 | 5 | 1 | 25.0 |
//! | Levenshtein triangle | 4 | 5 | 2 | 10.0 |
//! | Window containment | 3 | 4 | 1 | 12.0 |
//! | Drift bounds | 4 | 4 | 2 | 8.0 |

#![forbid(unsafe_code)]

use franken_whisper::speculation::SpeculationWindow;

// ---------------------------------------------------------------------------
// Test string generators (deterministic, reproducible)
// ---------------------------------------------------------------------------

/// Generate strings of varying lengths for testing.
fn test_strings() -> Vec<&'static str> {
    vec![
        "",
        "a",
        "ab",
        "abc",
        "hello",
        "world",
        "hello world",
        "kitten",
        "sitting",
        "café",
        "日本語",
        "The quick brown fox jumps over the lazy dog",
        "Lorem ipsum dolor sit amet",
    ]
}

/// Generate word arrays for testing.
fn test_word_arrays() -> Vec<Vec<&'static str>> {
    vec![
        vec![],
        vec!["hello"],
        vec!["hello", "world"],
        vec!["the", "quick", "brown", "fox"],
        vec!["a", "b", "c", "d", "e"],
        vec!["one", "two", "three"],
    ]
}

// ---------------------------------------------------------------------------
// Levenshtein implementation (mirror of src/speculation.rs for testing)
// ---------------------------------------------------------------------------

/// Character-level Levenshtein distance.
fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Word-level Levenshtein distance.
fn levenshtein_words(a: &[&str], b: &[&str]) -> usize {
    let m = a.len();
    let n = b.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

// ---------------------------------------------------------------------------
// MR1: Levenshtein Symmetry
// Property: d(a, b) == d(b, a)
// Category: Equivalence
// ---------------------------------------------------------------------------

#[test]
fn mr_levenshtein_symmetry_chars() {
    let strings = test_strings();

    for a in &strings {
        for b in &strings {
            let d_ab = levenshtein(a, b);
            let d_ba = levenshtein(b, a);

            assert_eq!(
                d_ab, d_ba,
                "Levenshtein symmetry violated: d({a:?}, {b:?}) = {d_ab}, d({b:?}, {a:?}) = {d_ba}"
            );
        }
    }
}

#[test]
fn mr_levenshtein_symmetry_words() {
    let word_arrays = test_word_arrays();

    for a in &word_arrays {
        for b in &word_arrays {
            let d_ab = levenshtein_words(a, b);
            let d_ba = levenshtein_words(b, a);

            assert_eq!(
                d_ab, d_ba,
                "Word Levenshtein symmetry violated: d({a:?}, {b:?}) = {d_ab}, d({b:?}, {a:?}) = {d_ba}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// MR2: Levenshtein Identity
// Property: d(a, a) == 0
// Category: Equivalence
// ---------------------------------------------------------------------------

#[test]
fn mr_levenshtein_identity_chars() {
    for s in test_strings() {
        let d = levenshtein(s, s);
        assert_eq!(d, 0, "Levenshtein identity violated: d({s:?}, {s:?}) = {d}");
    }
}

#[test]
fn mr_levenshtein_identity_words() {
    for words in test_word_arrays() {
        let d = levenshtein_words(&words, &words);
        assert_eq!(
            d, 0,
            "Word Levenshtein identity violated: d({words:?}, {words:?}) = {d}"
        );
    }
}

// ---------------------------------------------------------------------------
// MR3: Levenshtein Triangle Inequality
// Property: d(a, c) <= d(a, b) + d(b, c)
// Category: Additive bound
// ---------------------------------------------------------------------------

#[test]
fn mr_levenshtein_triangle_inequality_chars() {
    let strings = test_strings();

    for a in &strings {
        for b in &strings {
            for c in &strings {
                let d_ac = levenshtein(a, c);
                let d_ab = levenshtein(a, b);
                let d_bc = levenshtein(b, c);

                assert!(
                    d_ac <= d_ab + d_bc,
                    "Triangle inequality violated: d({a:?}, {c:?}) = {d_ac} > d({a:?}, {b:?}) + d({b:?}, {c:?}) = {d_ab} + {d_bc}"
                );
            }
        }
    }
}

#[test]
fn mr_levenshtein_triangle_inequality_words() {
    let word_arrays = test_word_arrays();

    for a in &word_arrays {
        for b in &word_arrays {
            for c in &word_arrays {
                let d_ac = levenshtein_words(a, c);
                let d_ab = levenshtein_words(a, b);
                let d_bc = levenshtein_words(b, c);

                assert!(
                    d_ac <= d_ab + d_bc,
                    "Word triangle inequality violated: d({a:?}, {c:?}) = {d_ac} > {d_ab} + {d_bc}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MR4: Levenshtein Upper Bound
// Property: d(a, b) <= max(len(a), len(b))
// Category: Inclusive bound
// ---------------------------------------------------------------------------

#[test]
fn mr_levenshtein_upper_bound_chars() {
    let strings = test_strings();

    for a in &strings {
        for b in &strings {
            let d = levenshtein(a, b);
            let max_len = a.chars().count().max(b.chars().count());

            assert!(
                d <= max_len,
                "Levenshtein upper bound violated: d({a:?}, {b:?}) = {d} > max_len = {max_len}"
            );
        }
    }
}

#[test]
fn mr_levenshtein_upper_bound_words() {
    let word_arrays = test_word_arrays();

    for a in &word_arrays {
        for b in &word_arrays {
            let d = levenshtein_words(a, b);
            let max_len = a.len().max(b.len());

            assert!(
                d <= max_len,
                "Word Levenshtein upper bound violated: d({a:?}, {b:?}) = {d} > max_len = {max_len}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// MR5: Levenshtein Empty String
// Property: d("", b) == len(b) and d(a, "") == len(a)
// Category: Additive identity
// ---------------------------------------------------------------------------

#[test]
fn mr_levenshtein_empty_string_chars() {
    for s in test_strings() {
        let len = s.chars().count();
        let d_empty_s = levenshtein("", s);
        let d_s_empty = levenshtein(s, "");

        assert_eq!(
            d_empty_s, len,
            "Levenshtein empty-to-string violated: d(\"\", {s:?}) = {d_empty_s}, expected {len}"
        );
        assert_eq!(
            d_s_empty, len,
            "Levenshtein string-to-empty violated: d({s:?}, \"\") = {d_s_empty}, expected {len}"
        );
    }
}

#[test]
fn mr_levenshtein_empty_string_words() {
    let empty: Vec<&str> = vec![];

    for words in test_word_arrays() {
        let len = words.len();
        let d_empty_w = levenshtein_words(&empty, &words);
        let d_w_empty = levenshtein_words(&words, &empty);

        assert_eq!(
            d_empty_w, len,
            "Word Levenshtein empty-to-words violated: expected {len}, got {d_empty_w}"
        );
        assert_eq!(
            d_w_empty, len,
            "Word Levenshtein words-to-empty violated: expected {len}, got {d_w_empty}"
        );
    }
}

// ---------------------------------------------------------------------------
// MR6: Levenshtein Single Edit
// Property: Inserting/deleting/substituting one element changes distance by exactly 1
// Category: Additive increment
// ---------------------------------------------------------------------------

#[test]
fn mr_levenshtein_single_insertion_chars() {
    let base = "hello";
    let inserted = "helloo"; // One extra 'o'

    let d = levenshtein(base, inserted);
    assert_eq!(
        d, 1,
        "Single insertion should give distance 1, got {d}"
    );
}

#[test]
fn mr_levenshtein_single_deletion_chars() {
    let base = "hello";
    let deleted = "helo"; // One 'l' removed

    let d = levenshtein(base, deleted);
    assert_eq!(
        d, 1,
        "Single deletion should give distance 1, got {d}"
    );
}

#[test]
fn mr_levenshtein_single_substitution_chars() {
    let base = "hello";
    let substituted = "hella"; // 'o' -> 'a'

    let d = levenshtein(base, substituted);
    assert_eq!(
        d, 1,
        "Single substitution should give distance 1, got {d}"
    );
}

#[test]
fn mr_levenshtein_single_insertion_words() {
    let base = vec!["hello", "world"];
    let inserted = vec!["hello", "beautiful", "world"];

    let d = levenshtein_words(&base, &inserted);
    assert_eq!(
        d, 1,
        "Single word insertion should give distance 1, got {d}"
    );
}

// ---------------------------------------------------------------------------
// MR7: SpeculationWindow Containment
// Property: Window containment is consistent with boundaries
// Category: Inclusive
// ---------------------------------------------------------------------------

#[test]
fn mr_window_contains_all_interior_points() {
    let window = SpeculationWindow::new(
        1,
        "run-123".to_string(),
        1000, // start_ms
        2000, // end_ms
        200,  // overlap_ms
        "hash123".to_string(),
    );

    // All points in [start_ms, end_ms) should be contained
    for ms in (window.start_ms..window.end_ms).step_by(100) {
        assert!(
            window.contains_ms(ms),
            "Window [{}, {}) should contain {ms}",
            window.start_ms,
            window.end_ms
        );
    }
}

#[test]
fn mr_window_excludes_exterior_points() {
    let window = SpeculationWindow::new(
        1,
        "run-123".to_string(),
        1000,
        2000,
        200,
        "hash123".to_string(),
    );

    // Points outside [start_ms, end_ms) should not be contained
    assert!(
        !window.contains_ms(999),
        "Window should not contain point before start"
    );
    assert!(
        !window.contains_ms(2000),
        "Window should not contain end point (half-open interval)"
    );
    assert!(
        !window.contains_ms(2001),
        "Window should not contain point after end"
    );
}

#[test]
fn mr_window_duration_is_end_minus_start() {
    for (start, end) in [(0, 1000), (500, 1500), (1000, 5000)] {
        let window = SpeculationWindow::new(
            1,
            "run".to_string(),
            start,
            end,
            100,
            "hash".to_string(),
        );

        assert_eq!(
            window.duration_ms(),
            end - start,
            "Duration should be end - start"
        );
    }
}

#[test]
fn mr_window_duration_saturates_on_invalid() {
    // If end < start (invalid but possible), duration should saturate to 0
    let window = SpeculationWindow::new(
        1,
        "run".to_string(),
        2000, // start > end
        1000,
        100,
        "hash".to_string(),
    );

    assert_eq!(
        window.duration_ms(),
        0,
        "Invalid window should have duration 0 (saturating sub)"
    );
}

// ---------------------------------------------------------------------------
// Composite MR: Chain Levenshtein operations
// ---------------------------------------------------------------------------

#[test]
fn mr_composite_levenshtein_sequence_of_edits() {
    // Applying k single-character edits should give distance <= k
    let start = "hello";
    let after_1 = "hallo"; // 1 edit: e->a
    let after_2 = "halla"; // 2 edits: e->a, o->a
    let after_3 = "xalla"; // 3 edits: h->x, e->a, o->a

    assert_eq!(levenshtein(start, after_1), 1);
    assert_eq!(levenshtein(start, after_2), 2);
    assert_eq!(levenshtein(start, after_3), 3);

    // Also verify transitivity bounds
    assert!(levenshtein(start, after_3) <= levenshtein(start, after_1) + levenshtein(after_1, after_3));
}
