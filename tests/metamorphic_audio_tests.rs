//! Metamorphic tests for audio processing invariants.
//!
//! These tests verify properties that MUST hold for correct audio processing
//! without requiring an oracle for "correct" output. Instead, we verify
//! relationships between transformed inputs and their outputs.
//!
//! MR Strength Matrix:
//! | MR | Fault Sensitivity | Independence | Cost | Score |
//! |----|-------------------|--------------|------|-------|
//! | Volume scaling | 4 | 5 | 2 | 10.0 |
//! | Silence padding | 3 | 4 | 1 | 12.0 |
//! | Resample round-trip | 5 | 5 | 3 | 8.3 |
//! | Concatenation bounds | 3 | 4 | 2 | 6.0 |

#![forbid(unsafe_code)]

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Test audio generators (deterministic, reproducible)
// ---------------------------------------------------------------------------

/// Generate a sine wave at the given frequency and sample rate.
fn generate_sine_wave(freq_hz: f32, sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * PI * freq_hz * t).sin()
        })
        .collect()
}

/// Generate silence (zeros) for the given duration.
fn generate_silence(sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    vec![0.0; num_samples]
}

/// Generate white noise (deterministic PRNG for reproducibility).
fn generate_noise(sample_rate: u32, duration_secs: f32, seed: u64) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let mut state = seed;
    (0..num_samples)
        .map(|_| {
            // Simple LCG for reproducibility
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to [-1, 1]
            ((state >> 33) as f32 / (u32::MAX >> 1) as f32) * 2.0 - 1.0
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Audio transformations (the "T" in metamorphic relations)
// ---------------------------------------------------------------------------

/// Scale audio volume by a constant factor.
fn scale_volume(samples: &[f32], factor: f32) -> Vec<f32> {
    samples.iter().map(|s| (s * factor).clamp(-1.0, 1.0)).collect()
}

/// Prepend silence to audio.
fn prepend_silence(samples: &[f32], silence_samples: usize) -> Vec<f32> {
    let mut result = vec![0.0; silence_samples];
    result.extend_from_slice(samples);
    result
}

/// Append silence to audio.
fn append_silence(samples: &[f32], silence_samples: usize) -> Vec<f32> {
    let mut result = samples.to_vec();
    result.extend(std::iter::repeat(0.0).take(silence_samples));
    result
}

/// Linear interpolation resample (matches src/audio.rs implementation).
fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    if src_rate == dst_rate {
        return input.to_vec();
    }

    let ratio = src_rate as f64 / dst_rate as f64;
    let output_len = (((input.len() as f64) * dst_rate as f64) / src_rate as f64).ceil() as usize;
    let mut output = Vec::with_capacity(output_len.max(1));

    for idx in 0..output_len.max(1) {
        let src_pos = idx as f64 * ratio;
        let left_idx = src_pos.floor() as usize;
        let right_idx = left_idx.saturating_add(1);

        let left = input[left_idx.min(input.len().saturating_sub(1))];
        let right = input[right_idx.min(input.len().saturating_sub(1))];
        let frac = (src_pos - left_idx as f64) as f32;
        output.push(left + (right - left) * frac);
    }

    output
}

/// Concatenate two audio buffers.
fn concat_audio(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut result = a.to_vec();
    result.extend_from_slice(b);
    result
}

// ---------------------------------------------------------------------------
// Comparison utilities
// ---------------------------------------------------------------------------

/// Compute RMS (root mean square) energy of audio.
fn rms_energy(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|s| (*s as f64).powi(2)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

/// Compute peak amplitude.
fn peak_amplitude(samples: &[f32]) -> f32 {
    samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
}

/// Compute mean squared error between two signals.
fn mse(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    sum / a.len() as f64
}

// ---------------------------------------------------------------------------
// MR1: Volume Scaling Invariance
// Property: Scaling audio by factor k should scale RMS energy by |k|
// Category: Multiplicative
// ---------------------------------------------------------------------------

#[test]
fn mr_volume_scaling_preserves_rms_ratio() {
    let sample_rate = 16_000;
    let duration = 0.5;

    for freq in [440.0, 880.0, 220.0] {
        let original = generate_sine_wave(freq, sample_rate, duration);
        let rms_original = rms_energy(&original);

        // Only test factors that won't cause clipping (factor <= 1.0)
        // since clipping breaks the linear scaling relationship
        for factor in [0.5, 0.25, 0.75, 0.1] {
            let scaled = scale_volume(&original, factor);
            let rms_scaled = rms_energy(&scaled);

            // For unclipped audio, RMS should scale linearly with volume factor
            // Allow 1% tolerance for floating-point precision
            let expected_rms = rms_original * factor.abs() as f64;
            let ratio = rms_scaled / expected_rms;

            assert!(
                (ratio - 1.0).abs() < 0.01,
                "RMS scaling violated for factor {factor}: expected {expected_rms:.6}, got {rms_scaled:.6} (ratio {ratio:.4})"
            );
        }
    }
}

#[test]
fn mr_volume_scaling_peak_amplitude_scales() {
    let sample_rate = 16_000;
    let original = generate_sine_wave(440.0, sample_rate, 0.1);
    let peak_original = peak_amplitude(&original);

    for factor in [0.5, 0.25, 0.75] {
        let scaled = scale_volume(&original, factor);
        let peak_scaled = peak_amplitude(&scaled);

        // Peak should scale linearly (for non-clipping factors)
        let expected_peak = peak_original * factor;
        let diff = (peak_scaled - expected_peak).abs();

        assert!(
            diff < 0.001,
            "Peak amplitude scaling violated for factor {factor}: expected {expected_peak:.6}, got {peak_scaled:.6}"
        );
    }
}

// ---------------------------------------------------------------------------
// MR2: Silence Padding Invariance
// Property: Adding silence should not change non-silent content
// Category: Additive
// ---------------------------------------------------------------------------

#[test]
fn mr_prepend_silence_preserves_original_content() {
    let sample_rate = 16_000;
    let original = generate_sine_wave(440.0, sample_rate, 0.2);
    let silence_samples = 1600; // 0.1 seconds of silence

    let padded = prepend_silence(&original, silence_samples);

    // The padded audio should be longer by exactly silence_samples
    assert_eq!(padded.len(), original.len() + silence_samples);

    // The first silence_samples should be zero
    let silence_portion = &padded[..silence_samples];
    assert!(
        silence_portion.iter().all(|&s| s == 0.0),
        "Prepended portion should be silence"
    );

    // The remaining portion should match original exactly
    let content_portion = &padded[silence_samples..];
    assert_eq!(
        content_portion, &original[..],
        "Content after prepended silence should match original"
    );
}

#[test]
fn mr_append_silence_preserves_original_content() {
    let sample_rate = 16_000;
    let original = generate_sine_wave(440.0, sample_rate, 0.2);
    let silence_samples = 1600;

    let padded = append_silence(&original, silence_samples);

    // Length check
    assert_eq!(padded.len(), original.len() + silence_samples);

    // Original content preserved at start
    let content_portion = &padded[..original.len()];
    assert_eq!(content_portion, &original[..]);

    // Trailing portion is silence
    let silence_portion = &padded[original.len()..];
    assert!(silence_portion.iter().all(|&s| s == 0.0));
}

#[test]
fn mr_silence_padding_rms_relation() {
    // RMS of padded signal should relate predictably to original RMS
    // RMS(padded) = RMS(original) * sqrt(original_len / padded_len)
    let sample_rate = 16_000;
    let original = generate_sine_wave(440.0, sample_rate, 0.2);
    let silence_samples = original.len(); // Double the length with silence

    let padded = append_silence(&original, silence_samples);

    let rms_original = rms_energy(&original);
    let rms_padded = rms_energy(&padded);

    // Expected: RMS decreases by sqrt(2) when we double length with zeros
    let expected_rms = rms_original / 2.0_f64.sqrt();
    let ratio = rms_padded / expected_rms;

    assert!(
        (ratio - 1.0).abs() < 0.01,
        "Silence padding RMS relation violated: expected {expected_rms:.6}, got {rms_padded:.6}"
    );
}

// ---------------------------------------------------------------------------
// MR3: Resampling Round-Trip
// Property: resample(resample(x, A, B), B, A) ≈ x
// Category: Invertive
// ---------------------------------------------------------------------------

#[test]
fn mr_resample_roundtrip_recovers_signal() {
    let src_rate = 16_000;
    let dst_rate = 8_000;

    let original = generate_sine_wave(440.0, src_rate, 0.1);

    // Downsample then upsample
    let downsampled = resample_linear(&original, src_rate, dst_rate);
    let recovered = resample_linear(&downsampled, dst_rate, src_rate);

    // Length should be approximately preserved
    let len_diff = (recovered.len() as i64 - original.len() as i64).abs();
    assert!(
        len_diff <= 2,
        "Round-trip length drift too large: original {}, recovered {}",
        original.len(),
        recovered.len()
    );

    // Truncate to common length for comparison
    let common_len = original.len().min(recovered.len());
    let orig_slice = &original[..common_len];
    let rec_slice = &recovered[..common_len];

    // MSE should be small (linear interpolation introduces some error)
    let error = mse(orig_slice, rec_slice);
    assert!(
        error < 0.01,
        "Resample round-trip MSE too high: {error:.6} (threshold 0.01)"
    );
}

#[test]
fn mr_resample_identity_when_same_rate() {
    let rate = 16_000;
    let original = generate_sine_wave(440.0, rate, 0.1);

    let resampled = resample_linear(&original, rate, rate);

    assert_eq!(
        original.len(),
        resampled.len(),
        "Same-rate resample should preserve length"
    );

    // Should be identical
    let error = mse(&original, &resampled);
    assert!(
        error < 1e-10,
        "Same-rate resample should be identity, got MSE {error}"
    );
}

#[test]
fn mr_resample_preserves_frequency_content() {
    // A 440Hz tone resampled should still have dominant frequency at 440Hz
    // We verify this by checking that zero-crossing RATE (crossings per second)
    // remains approximately constant regardless of sample rate
    let src_rate = 16_000;
    let dst_rate = 32_000; // Upsample 2x
    let duration = 0.1;

    let original = generate_sine_wave(440.0, src_rate, duration);
    let upsampled = resample_linear(&original, src_rate, dst_rate);

    // Count zero crossings
    fn count_zero_crossings(samples: &[f32]) -> usize {
        samples
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count()
    }

    let orig_crossings = count_zero_crossings(&original);
    let up_crossings = count_zero_crossings(&upsampled);

    // Zero crossing count should be approximately the same (same frequency content)
    // Linear interpolation may create slight differences at boundaries
    let crossing_ratio = up_crossings as f64 / orig_crossings as f64;

    assert!(
        (crossing_ratio - 1.0).abs() < 0.1,
        "Zero crossing count changed unexpectedly: original {orig_crossings}, upsampled {up_crossings}"
    );
}

// ---------------------------------------------------------------------------
// MR4: Concatenation Bounds
// Property: Properties of concat(A, B) bound properties of A and B
// Category: Inclusive
// ---------------------------------------------------------------------------

#[test]
fn mr_concat_length_is_sum() {
    let sample_rate = 16_000;
    let a = generate_sine_wave(440.0, sample_rate, 0.1);
    let b = generate_sine_wave(880.0, sample_rate, 0.15);

    let concatenated = concat_audio(&a, &b);

    assert_eq!(
        concatenated.len(),
        a.len() + b.len(),
        "Concatenated length should be sum of parts"
    );
}

#[test]
fn mr_concat_peak_is_max_of_parts() {
    let sample_rate = 16_000;
    let a = scale_volume(&generate_sine_wave(440.0, sample_rate, 0.1), 0.5);
    let b = scale_volume(&generate_sine_wave(880.0, sample_rate, 0.1), 0.8);

    let concatenated = concat_audio(&a, &b);

    let peak_a = peak_amplitude(&a);
    let peak_b = peak_amplitude(&b);
    let peak_concat = peak_amplitude(&concatenated);

    let expected_peak = peak_a.max(peak_b);

    assert!(
        (peak_concat - expected_peak).abs() < 1e-6,
        "Concatenated peak should be max of parts: expected {expected_peak}, got {peak_concat}"
    );
}

#[test]
fn mr_concat_rms_bounds() {
    // RMS of concat should be bounded by weighted combination of parts
    let sample_rate = 16_000;
    let a = generate_sine_wave(440.0, sample_rate, 0.1);
    let b = generate_sine_wave(880.0, sample_rate, 0.2);

    let concatenated = concat_audio(&a, &b);

    let rms_a = rms_energy(&a);
    let rms_b = rms_energy(&b);
    let rms_concat = rms_energy(&concatenated);

    // RMS of concatenation is sqrt((n_a * rms_a^2 + n_b * rms_b^2) / (n_a + n_b))
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;
    let expected_rms = ((n_a * rms_a.powi(2) + n_b * rms_b.powi(2)) / (n_a + n_b)).sqrt();

    let ratio = rms_concat / expected_rms;
    assert!(
        (ratio - 1.0).abs() < 0.001,
        "Concatenation RMS formula violated: expected {expected_rms:.6}, got {rms_concat:.6}"
    );
}

#[test]
fn mr_concat_preserves_content_at_boundaries() {
    let sample_rate = 16_000;
    let a = generate_sine_wave(440.0, sample_rate, 0.1);
    let b = generate_sine_wave(880.0, sample_rate, 0.1);

    let concatenated = concat_audio(&a, &b);

    // First part should match exactly
    assert_eq!(&concatenated[..a.len()], &a[..]);

    // Second part should match exactly
    assert_eq!(&concatenated[a.len()..], &b[..]);
}

// ---------------------------------------------------------------------------
// MR5: Noise Addition Bounds (for robustness testing)
// Property: Adding low-level noise shouldn't dramatically change signal properties
// Category: Additive with tolerance
// ---------------------------------------------------------------------------

#[test]
fn mr_low_noise_preserves_rms_approximately() {
    let sample_rate = 16_000;
    let signal = generate_sine_wave(440.0, sample_rate, 0.2);
    let noise = generate_noise(sample_rate, 0.2, 42);

    // Scale noise to be 1% of signal level
    let noise_level = 0.01;
    let scaled_noise: Vec<f32> = noise.iter().map(|n| n * noise_level).collect();

    // Add noise to signal
    let noisy: Vec<f32> = signal
        .iter()
        .zip(scaled_noise.iter())
        .map(|(s, n)| (s + n).clamp(-1.0, 1.0))
        .collect();

    let rms_signal = rms_energy(&signal);
    let rms_noisy = rms_energy(&noisy);

    // RMS should change by less than 2% for 1% noise
    let ratio = rms_noisy / rms_signal;
    assert!(
        (ratio - 1.0).abs() < 0.02,
        "Low noise changed RMS more than expected: ratio {ratio:.4}"
    );
}

// ---------------------------------------------------------------------------
// Composite MR: Chain multiple transformations
// ---------------------------------------------------------------------------

#[test]
fn mr_composite_scale_then_pad_then_concat() {
    let sample_rate = 16_000;
    let a = generate_sine_wave(440.0, sample_rate, 0.1);
    let b = generate_sine_wave(880.0, sample_rate, 0.1);

    // Apply transformations in sequence
    let a_scaled = scale_volume(&a, 0.5);
    let a_padded = append_silence(&a_scaled, 800);

    let b_scaled = scale_volume(&b, 0.5);
    let b_padded = prepend_silence(&b_scaled, 800);

    let concatenated = concat_audio(&a_padded, &b_padded);

    // Verify composite properties
    let expected_len = a.len() + 800 + 800 + b.len();
    assert_eq!(concatenated.len(), expected_len);

    // Peak should be 0.5x original peak (from scaling)
    let original_peak = peak_amplitude(&a).max(peak_amplitude(&b));
    let concat_peak = peak_amplitude(&concatenated);
    let expected_peak = original_peak * 0.5;

    assert!(
        (concat_peak - expected_peak).abs() < 0.001,
        "Composite transformation peak check failed"
    );
}
