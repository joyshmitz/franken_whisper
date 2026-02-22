//! bd-3pf.18: Cross-engine conformance comparator and shadow-run CI gate.
//!
//! This test file implements a conformance comparator harness that runs bridge
//! and native engines on a fixture corpus, computes drift metrics, enforces
//! tolerance gates, and emits reproducible artifacts for CI/shadow rollout
//! decisions.
//!
//! All structs are test-only and defined within this file.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use franken_whisper::backend::{
    DiarizationPilot, InsanelyFastPilot, TranscriptSegment, WhisperCppPilot,
};

// ---------------------------------------------------------------------------
// Mock engine abstraction (test-only)
// ---------------------------------------------------------------------------

/// Identifies the category of a mock engine for conformance testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum MockEngineKind {
    /// Bridge-style engine wrapping an external process.
    Bridge,
    /// Native pilot engine with in-process mock inference.
    Pilot,
}

impl MockEngineKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Bridge => "bridge",
            Self::Pilot => "pilot",
        }
    }
}

/// A mock engine that produces deterministic `TranscriptSegment` output for a
/// given audio duration. Each engine has a name, kind, and a closure that
/// produces segments.
struct MockEngine {
    name: String,
    kind: MockEngineKind,
    /// Closure that takes audio duration in ms and returns transcript segments.
    produce: Box<dyn Fn(u64) -> Vec<TranscriptSegment>>,
}

impl MockEngine {
    fn run(&self, duration_ms: u64) -> Vec<TranscriptSegment> {
        (self.produce)(duration_ms)
    }
}

// ---------------------------------------------------------------------------
// DriftMetrics — per-segment WER approximation and confidence delta
// ---------------------------------------------------------------------------

/// Drift metrics computed between two engine outputs on the same input.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DriftMetrics {
    /// Name of the reference engine.
    reference_engine: String,
    /// Name of the candidate engine.
    candidate_engine: String,
    /// Approximate word error rate between the two outputs (0.0 = identical).
    wer_approx: f64,
    /// Mean absolute difference in confidence scores across aligned segments.
    confidence_delta: f64,
    /// Number of segments in the reference output.
    reference_segment_count: usize,
    /// Number of segments in the candidate output.
    candidate_segment_count: usize,
    /// Maximum per-segment WER observed.
    max_segment_wer: f64,
}

impl DriftMetrics {
    /// Compute drift metrics between reference and candidate segment lists.
    ///
    /// Segments are compared positionally: the i-th reference segment is
    /// compared against the i-th candidate segment. If segment counts differ,
    /// surplus segments contribute a WER of 1.0 and a confidence delta of 1.0.
    fn compute(
        reference_engine: &str,
        candidate_engine: &str,
        reference: &[TranscriptSegment],
        candidate: &[TranscriptSegment],
    ) -> Self {
        let ref_count = reference.len();
        let cand_count = candidate.len();
        let max_count = ref_count.max(cand_count);

        if max_count == 0 {
            return Self {
                reference_engine: reference_engine.to_owned(),
                candidate_engine: candidate_engine.to_owned(),
                wer_approx: 0.0,
                confidence_delta: 0.0,
                reference_segment_count: 0,
                candidate_segment_count: 0,
                max_segment_wer: 0.0,
            };
        }

        let mut total_wer = 0.0_f64;
        let mut total_conf_delta = 0.0_f64;
        let mut max_seg_wer = 0.0_f64;

        for i in 0..max_count {
            let seg_wer = if i < ref_count && i < cand_count {
                word_error_rate_approx(&reference[i].text, &candidate[i].text)
            } else {
                // Surplus segment counts as complete mismatch.
                1.0
            };

            let seg_conf_delta = if i < ref_count && i < cand_count {
                (reference[i].confidence - candidate[i].confidence).abs()
            } else {
                1.0
            };

            total_wer += seg_wer;
            total_conf_delta += seg_conf_delta;
            if seg_wer > max_seg_wer {
                max_seg_wer = seg_wer;
            }
        }

        Self {
            reference_engine: reference_engine.to_owned(),
            candidate_engine: candidate_engine.to_owned(),
            wer_approx: total_wer / max_count as f64,
            confidence_delta: total_conf_delta / max_count as f64,
            reference_segment_count: ref_count,
            candidate_segment_count: cand_count,
            max_segment_wer: max_seg_wer,
        }
    }
}

/// Approximate word error rate between two strings, computed as the fraction
/// of words that differ when comparing token-by-token. This is a simplified
/// Levenshtein-at-word-level approximation using the longer string's word
/// count as the denominator.
fn word_error_rate_approx(reference: &str, candidate: &str) -> f64 {
    let ref_words: Vec<&str> = reference.split_whitespace().collect();
    let cand_words: Vec<&str> = candidate.split_whitespace().collect();

    let max_len = ref_words.len().max(cand_words.len());
    if max_len == 0 {
        return 0.0;
    }

    let mut mismatches = 0_usize;
    for i in 0..max_len {
        let r = ref_words.get(i).copied().unwrap_or("");
        let c = cand_words.get(i).copied().unwrap_or("");
        if r != c {
            mismatches += 1;
        }
    }

    mismatches as f64 / max_len as f64
}

// ---------------------------------------------------------------------------
// ToleranceGate — enforces max drift thresholds
// ---------------------------------------------------------------------------

/// A gate that enforces maximum acceptable drift between engine outputs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ToleranceGate {
    /// Maximum acceptable mean WER across segments.
    max_wer: f64,
    /// Maximum acceptable mean confidence delta across segments.
    max_confidence_delta: f64,
    /// Maximum acceptable WER for any single segment.
    max_single_segment_wer: f64,
}

/// Result of applying a tolerance gate to drift metrics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct GateResult {
    /// Whether the gate passed.
    passed: bool,
    /// Human-readable reasons for failure (empty if passed).
    violations: Vec<String>,
}

impl ToleranceGate {
    /// Evaluate drift metrics against this gate's thresholds.
    fn evaluate(&self, metrics: &DriftMetrics) -> GateResult {
        let mut violations = Vec::new();

        if metrics.wer_approx > self.max_wer {
            violations.push(format!(
                "WER {:.4} exceeds max {:.4}",
                metrics.wer_approx, self.max_wer
            ));
        }
        if metrics.confidence_delta > self.max_confidence_delta {
            violations.push(format!(
                "confidence delta {:.4} exceeds max {:.4}",
                metrics.confidence_delta, self.max_confidence_delta
            ));
        }
        if metrics.max_segment_wer > self.max_single_segment_wer {
            violations.push(format!(
                "max segment WER {:.4} exceeds max {:.4}",
                metrics.max_segment_wer, self.max_single_segment_wer
            ));
        }

        GateResult {
            passed: violations.is_empty(),
            violations,
        }
    }
}

// ---------------------------------------------------------------------------
// ConformanceReport — per-engine results with reproducibility hash
// ---------------------------------------------------------------------------

/// Per-engine result in the conformance report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct EngineResult {
    engine_name: String,
    engine_kind: String,
    segment_count: usize,
    avg_confidence: f64,
    segments_json: String,
}

/// Full conformance report emitted as a CI artifact.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ConformanceReport {
    /// Identifier for this report run.
    run_id: String,
    /// Duration of the fixture corpus input in milliseconds.
    fixture_duration_ms: u64,
    /// Per-engine results.
    engine_results: Vec<EngineResult>,
    /// Pairwise drift metrics between all engine pairs.
    drift_metrics: Vec<DriftMetrics>,
    /// Gate evaluation results for each pair.
    gate_results: Vec<PairGateResult>,
    /// Overall pass/fail for the conformance run.
    overall_passed: bool,
    /// Whether this was a shadow run (informational only, never fails CI).
    shadow_mode: bool,
    /// SHA-256 hash of the deterministic report content for reproducibility.
    reproducibility_hash: String,
}

/// Gate result for a specific engine pair.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct PairGateResult {
    reference_engine: String,
    candidate_engine: String,
    gate_result: GateResult,
}

impl ConformanceReport {
    /// Compute the reproducibility hash over the deterministic content.
    ///
    /// The hash covers engine results and drift metrics (excluding the hash
    /// field itself) to ensure identical inputs always produce the same hash.
    fn compute_hash(
        fixture_duration_ms: u64,
        engine_results: &[EngineResult],
        drift_metrics: &[DriftMetrics],
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(fixture_duration_ms.to_le_bytes());
        for er in engine_results {
            hasher.update(er.engine_name.as_bytes());
            hasher.update(er.engine_kind.as_bytes());
            hasher.update(er.segment_count.to_le_bytes());
            hasher.update(er.avg_confidence.to_le_bytes());
            hasher.update(er.segments_json.as_bytes());
        }
        for dm in drift_metrics {
            hasher.update(dm.reference_engine.as_bytes());
            hasher.update(dm.candidate_engine.as_bytes());
            hasher.update(dm.wer_approx.to_le_bytes());
            hasher.update(dm.confidence_delta.to_le_bytes());
            hasher.update(dm.reference_segment_count.to_le_bytes());
            hasher.update(dm.candidate_segment_count.to_le_bytes());
            hasher.update(dm.max_segment_wer.to_le_bytes());
        }
        let digest = hasher.finalize();
        format!("{digest:x}")
    }
}

// ---------------------------------------------------------------------------
// ConformanceHarness — runs multiple engines on the same mock input
// ---------------------------------------------------------------------------

/// Harness that runs multiple mock engines on a fixture corpus and produces a
/// conformance report with drift metrics and tolerance gate evaluation.
struct ConformanceHarness {
    engines: Vec<MockEngine>,
    gate: ToleranceGate,
    shadow_mode: bool,
}

impl ConformanceHarness {
    fn new(gate: ToleranceGate, shadow_mode: bool) -> Self {
        Self {
            engines: Vec::new(),
            gate,
            shadow_mode,
        }
    }

    fn add_engine(&mut self, engine: MockEngine) {
        self.engines.push(engine);
    }

    /// Run all engines on the fixture corpus (single duration) and produce a
    /// conformance report.
    fn run(&self, fixture_duration_ms: u64) -> ConformanceReport {
        // Use BTreeMap for deterministic iteration order by engine name.
        let mut results: BTreeMap<String, (MockEngineKind, Vec<TranscriptSegment>)> =
            BTreeMap::new();

        for engine in &self.engines {
            let segments = engine.run(fixture_duration_ms);
            results.insert(engine.name.clone(), (engine.kind, segments));
        }

        // Build per-engine results.
        let engine_results: Vec<EngineResult> = results
            .iter()
            .map(|(name, (kind, segments))| {
                let avg_conf = if segments.is_empty() {
                    0.0
                } else {
                    segments.iter().map(|s| s.confidence).sum::<f64>() / segments.len() as f64
                };
                let segments_json =
                    serde_json::to_string(segments).expect("segments should serialize");
                EngineResult {
                    engine_name: name.clone(),
                    engine_kind: kind.as_str().to_owned(),
                    segment_count: segments.len(),
                    avg_confidence: avg_conf,
                    segments_json,
                }
            })
            .collect();

        // Compute pairwise drift metrics.
        let names: Vec<&String> = results.keys().collect();
        let mut drift_metrics = Vec::new();
        let mut gate_results = Vec::new();

        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                let ref_name = names[i];
                let cand_name = names[j];
                let ref_segs = &results[ref_name].1;
                let cand_segs = &results[cand_name].1;

                let metrics = DriftMetrics::compute(ref_name, cand_name, ref_segs, cand_segs);
                let gr = self.gate.evaluate(&metrics);

                gate_results.push(PairGateResult {
                    reference_engine: ref_name.clone(),
                    candidate_engine: cand_name.clone(),
                    gate_result: gr,
                });
                drift_metrics.push(metrics);
            }
        }

        let overall_passed = if self.shadow_mode {
            // Shadow mode never fails the pipeline.
            true
        } else {
            gate_results.iter().all(|pgr| pgr.gate_result.passed)
        };

        let reproducibility_hash =
            ConformanceReport::compute_hash(fixture_duration_ms, &engine_results, &drift_metrics);

        ConformanceReport {
            run_id: "conformance-test-run".to_owned(),
            fixture_duration_ms,
            engine_results,
            drift_metrics,
            gate_results,
            overall_passed,
            shadow_mode: self.shadow_mode,
            reproducibility_hash,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: build standard mock engines using the pilot structs
// ---------------------------------------------------------------------------

/// Create a WhisperCppPilot-based mock engine.
fn whisper_cpp_pilot_engine() -> MockEngine {
    MockEngine {
        name: "whisper-cpp-pilot".to_owned(),
        kind: MockEngineKind::Pilot,
        produce: Box::new(|duration_ms| {
            let pilot = WhisperCppPilot::new(
                "models/ggml-base.bin".to_owned(),
                4,
                Some("en".to_owned()),
                false,
            );
            pilot.transcribe(duration_ms)
        }),
    }
}

/// Create an InsanelyFastPilot-based mock engine (single-file batch).
fn insanely_fast_pilot_engine() -> MockEngine {
    MockEngine {
        name: "insanely-fast-pilot".to_owned(),
        kind: MockEngineKind::Pilot,
        produce: Box::new(|duration_ms| {
            let pilot = InsanelyFastPilot::new(
                "openai/whisper-large-v3".to_owned(),
                1,
                "cpu".to_owned(),
                "float32".to_owned(),
            );
            pilot
                .transcribe_batch(&[duration_ms])
                .into_iter()
                .next()
                .unwrap_or_default()
        }),
    }
}

/// Create a DiarizationPilot-based mock engine (flattened to TranscriptSegment).
fn diarization_pilot_engine() -> MockEngine {
    MockEngine {
        name: "diarization-pilot".to_owned(),
        kind: MockEngineKind::Pilot,
        produce: Box::new(|duration_ms| {
            let pilot = DiarizationPilot::new(
                "whisper-large-v3".to_owned(),
                "WAV2VEC2_ASR_LARGE_LV60K_960H".to_owned(),
                Some(2),
                "en".to_owned(),
            );
            let diarized = pilot.process(duration_ms);
            diarized
                .segments
                .into_iter()
                .map(|ds| TranscriptSegment {
                    start_ms: ds.start_ms,
                    end_ms: ds.end_ms,
                    text: ds.text,
                    confidence: ds.confidence,
                })
                .collect()
        }),
    }
}

/// Create a simulated bridge engine that mimics WhisperCpp output via
/// deterministic generation (test-only, no external processes).
fn whisper_cpp_bridge_engine() -> MockEngine {
    MockEngine {
        name: "whisper-cpp-bridge".to_owned(),
        kind: MockEngineKind::Bridge,
        produce: Box::new(|duration_ms| {
            // Simulate bridge output matching the WhisperCppPilot format
            // with identical content (zero drift by design).
            let pilot = WhisperCppPilot::new(
                "models/ggml-base.bin".to_owned(),
                4,
                Some("en".to_owned()),
                false,
            );
            pilot.transcribe(duration_ms)
        }),
    }
}

/// Create a simulated bridge engine that introduces controlled drift
/// (different text, slightly different confidence) to test tolerance gates.
fn drifted_bridge_engine() -> MockEngine {
    MockEngine {
        name: "drifted-bridge".to_owned(),
        kind: MockEngineKind::Bridge,
        produce: Box::new(|duration_ms| {
            let segment_duration_ms: u64 = 5000;
            let n_segments = if duration_ms == 0 {
                0
            } else {
                duration_ms.div_ceil(segment_duration_ms) as usize
            };

            let phrases = [
                "The fast brown fox leaps over the lazy dog.",
                "Hello world, this is a test transcription.",
                "Speech recognition is a fascinating field.",
                "Machine learning continues to advance.",
            ];

            (0..n_segments)
                .map(|i| {
                    let start_ms = (i as u64) * segment_duration_ms;
                    let end_ms = std::cmp::min(start_ms + segment_duration_ms, duration_ms);
                    let text = phrases[i % phrases.len()].to_owned();
                    let confidence = 0.90 - (i as f64 * 0.01);

                    TranscriptSegment {
                        start_ms,
                        end_ms,
                        text,
                        confidence,
                    }
                })
                .collect()
        }),
    }
}

/// Create a heavily drifted engine that will fail tight tolerance gates.
fn heavily_drifted_engine() -> MockEngine {
    MockEngine {
        name: "heavily-drifted".to_owned(),
        kind: MockEngineKind::Bridge,
        produce: Box::new(|duration_ms| {
            let segment_duration_ms: u64 = 5000;
            let n_segments = if duration_ms == 0 {
                0
            } else {
                duration_ms.div_ceil(segment_duration_ms) as usize
            };

            let phrases = [
                "Completely different text here.",
                "Nothing like the original at all.",
                "This output is totally wrong.",
                "No match whatsoever expected.",
            ];

            (0..n_segments)
                .map(|i| {
                    let start_ms = (i as u64) * segment_duration_ms;
                    let end_ms = std::cmp::min(start_ms + segment_duration_ms, duration_ms);
                    TranscriptSegment {
                        start_ms,
                        end_ms,
                        text: phrases[i % phrases.len()].to_owned(),
                        confidence: 0.50 - (i as f64 * 0.05),
                    }
                })
                .collect()
        }),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[test]
fn harness_runs_both_bridge_and_pilot_engines() {
    let gate = ToleranceGate {
        max_wer: 1.0,
        max_confidence_delta: 1.0,
        max_single_segment_wer: 1.0,
    };
    let mut harness = ConformanceHarness::new(gate, false);
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(insanely_fast_pilot_engine());
    harness.add_engine(diarization_pilot_engine());
    harness.add_engine(whisper_cpp_bridge_engine());

    let report = harness.run(15_000);

    // All four engines should be present in results.
    assert_eq!(report.engine_results.len(), 4);

    // Verify we have both bridge and pilot engines.
    let kinds: Vec<&str> = report
        .engine_results
        .iter()
        .map(|er| er.engine_kind.as_str())
        .collect();
    assert!(kinds.contains(&"bridge"), "should have bridge engine");
    assert!(kinds.contains(&"pilot"), "should have pilot engine");

    // Each engine should produce non-empty segments for 15s input.
    for er in &report.engine_results {
        assert!(
            er.segment_count > 0,
            "engine {} should produce segments for 15s input",
            er.engine_name
        );
    }

    // Pairwise drift metrics: C(4,2) = 6 pairs.
    assert_eq!(report.drift_metrics.len(), 6);
    assert_eq!(report.gate_results.len(), 6);
}

#[test]
fn drift_metrics_correctly_computed_identical_engines() {
    // whisper-cpp-pilot and whisper-cpp-bridge produce identical output.
    let pilot = WhisperCppPilot::new(
        "models/ggml-base.bin".to_owned(),
        4,
        Some("en".to_owned()),
        false,
    );
    let segments = pilot.transcribe(10_000);
    let metrics = DriftMetrics::compute("engine-a", "engine-b", &segments, &segments);

    assert_eq!(
        metrics.wer_approx, 0.0,
        "identical outputs should have zero WER"
    );
    assert_eq!(
        metrics.confidence_delta, 0.0,
        "identical outputs should have zero confidence delta"
    );
    assert_eq!(metrics.max_segment_wer, 0.0);
    assert_eq!(metrics.reference_segment_count, segments.len());
    assert_eq!(metrics.candidate_segment_count, segments.len());
}

#[test]
fn drift_metrics_correctly_computed_different_engines() {
    let pilot = WhisperCppPilot::new(
        "models/ggml-base.bin".to_owned(),
        4,
        Some("en".to_owned()),
        false,
    );
    let reference = pilot.transcribe(10_000);

    // Drifted engine uses different phrases.
    let candidate: Vec<TranscriptSegment> = vec![
        TranscriptSegment {
            start_ms: 0,
            end_ms: 5000,
            text: "The fast brown fox leaps over the lazy dog.".to_owned(),
            confidence: 0.90,
        },
        TranscriptSegment {
            start_ms: 5000,
            end_ms: 10_000,
            text: "Hello world, this is a test transcription.".to_owned(),
            confidence: 0.89,
        },
    ];

    let metrics = DriftMetrics::compute("ref", "cand", &reference, &candidate);

    // First segment: "quick" vs "fast" and "jumps" vs "leaps" = 2 diffs out of 9 words.
    // Second segment: identical text = 0 diffs.
    assert!(
        metrics.wer_approx > 0.0,
        "different text should produce non-zero WER"
    );
    assert!(
        metrics.wer_approx < 1.0,
        "partially different text should produce WER < 1.0"
    );
    assert!(
        metrics.confidence_delta > 0.0,
        "different confidence should produce non-zero delta"
    );
}

#[test]
fn drift_metrics_handles_empty_inputs() {
    let metrics = DriftMetrics::compute("a", "b", &[], &[]);
    assert_eq!(metrics.wer_approx, 0.0);
    assert_eq!(metrics.confidence_delta, 0.0);
    assert_eq!(metrics.max_segment_wer, 0.0);
}

#[test]
fn drift_metrics_handles_mismatched_segment_counts() {
    let reference = vec![
        TranscriptSegment {
            start_ms: 0,
            end_ms: 5000,
            text: "Hello world.".to_owned(),
            confidence: 0.9,
        },
        TranscriptSegment {
            start_ms: 5000,
            end_ms: 10_000,
            text: "Extra segment.".to_owned(),
            confidence: 0.8,
        },
    ];
    let candidate = vec![TranscriptSegment {
        start_ms: 0,
        end_ms: 5000,
        text: "Hello world.".to_owned(),
        confidence: 0.9,
    }];

    let metrics = DriftMetrics::compute("ref", "cand", &reference, &candidate);

    // 2 segments max: first is identical (WER=0, delta=0), second has no match (WER=1, delta=1).
    assert_eq!(metrics.reference_segment_count, 2);
    assert_eq!(metrics.candidate_segment_count, 1);
    assert!(
        (metrics.wer_approx - 0.5).abs() < 1e-10,
        "one matching + one surplus = 0.5 mean WER"
    );
    assert_eq!(metrics.max_segment_wer, 1.0);
}

#[test]
fn tolerance_gate_passes_within_threshold() {
    let gate = ToleranceGate {
        max_wer: 0.3,
        max_confidence_delta: 0.1,
        max_single_segment_wer: 0.5,
    };

    let metrics = DriftMetrics {
        reference_engine: "a".to_owned(),
        candidate_engine: "b".to_owned(),
        wer_approx: 0.1,
        confidence_delta: 0.02,
        reference_segment_count: 2,
        candidate_segment_count: 2,
        max_segment_wer: 0.2,
    };

    let result = gate.evaluate(&metrics);
    assert!(result.passed, "metrics within tolerance should pass");
    assert!(result.violations.is_empty());
}

#[test]
fn tolerance_gate_fails_above_threshold() {
    let gate = ToleranceGate {
        max_wer: 0.1,
        max_confidence_delta: 0.05,
        max_single_segment_wer: 0.2,
    };

    let metrics = DriftMetrics {
        reference_engine: "a".to_owned(),
        candidate_engine: "b".to_owned(),
        wer_approx: 0.5,
        confidence_delta: 0.3,
        reference_segment_count: 2,
        candidate_segment_count: 2,
        max_segment_wer: 0.8,
    };

    let result = gate.evaluate(&metrics);
    assert!(!result.passed, "metrics above tolerance should fail");
    assert_eq!(
        result.violations.len(),
        3,
        "all three thresholds should be violated"
    );
}

#[test]
fn tolerance_gate_partial_failure() {
    let gate = ToleranceGate {
        max_wer: 0.5,
        max_confidence_delta: 0.01,
        max_single_segment_wer: 1.0,
    };

    let metrics = DriftMetrics {
        reference_engine: "a".to_owned(),
        candidate_engine: "b".to_owned(),
        wer_approx: 0.2,
        confidence_delta: 0.1,
        reference_segment_count: 2,
        candidate_segment_count: 2,
        max_segment_wer: 0.3,
    };

    let result = gate.evaluate(&metrics);
    assert!(!result.passed, "should fail on confidence delta");
    assert_eq!(result.violations.len(), 1);
    assert!(result.violations[0].contains("confidence delta"));
}

#[test]
fn report_serializes_to_json_for_ci() {
    let gate = ToleranceGate {
        max_wer: 1.0,
        max_confidence_delta: 1.0,
        max_single_segment_wer: 1.0,
    };
    let mut harness = ConformanceHarness::new(gate, false);
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(whisper_cpp_bridge_engine());

    let report = harness.run(10_000);

    // Serialize to JSON.
    let json = serde_json::to_string_pretty(&report).expect("report should serialize to JSON");

    // Deserialize back and verify round-trip.
    let deserialized: ConformanceReport =
        serde_json::from_str(&json).expect("JSON should deserialize back to ConformanceReport");

    assert_eq!(deserialized.run_id, report.run_id);
    assert_eq!(deserialized.fixture_duration_ms, report.fixture_duration_ms);
    assert_eq!(
        deserialized.engine_results.len(),
        report.engine_results.len()
    );
    assert_eq!(deserialized.drift_metrics.len(), report.drift_metrics.len());
    assert_eq!(deserialized.overall_passed, report.overall_passed);
    assert_eq!(deserialized.shadow_mode, report.shadow_mode);
    assert_eq!(
        deserialized.reproducibility_hash,
        report.reproducibility_hash
    );

    // Verify key fields are present in the JSON.
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed.get("run_id").is_some());
    assert!(parsed.get("engine_results").is_some());
    assert!(parsed.get("drift_metrics").is_some());
    assert!(parsed.get("gate_results").is_some());
    assert!(parsed.get("overall_passed").is_some());
    assert!(parsed.get("reproducibility_hash").is_some());
}

#[test]
fn shadow_run_mode_never_fails_pipeline() {
    // Use a very tight gate so drift would normally cause failure.
    let gate = ToleranceGate {
        max_wer: 0.0,
        max_confidence_delta: 0.0,
        max_single_segment_wer: 0.0,
    };
    let mut harness = ConformanceHarness::new(gate, true); // shadow_mode = true
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(insanely_fast_pilot_engine());

    let report = harness.run(10_000);

    // Shadow mode should always report overall_passed = true.
    assert!(
        report.overall_passed,
        "shadow mode should never fail the pipeline"
    );
    assert!(report.shadow_mode, "should be marked as shadow mode");

    // But individual gate results should still reflect actual drift.
    let any_failed = report
        .gate_results
        .iter()
        .any(|pgr| !pgr.gate_result.passed);
    assert!(
        any_failed,
        "individual gates should still detect drift in shadow mode"
    );
}

#[test]
fn shadow_run_vs_enforced_mode_differ_on_failure() {
    // Tight gate with engines that produce different output.
    let gate = ToleranceGate {
        max_wer: 0.0,
        max_confidence_delta: 0.0,
        max_single_segment_wer: 0.0,
    };

    // Enforced mode.
    let mut enforced = ConformanceHarness::new(gate.clone(), false);
    enforced.add_engine(whisper_cpp_pilot_engine());
    enforced.add_engine(insanely_fast_pilot_engine());
    let enforced_report = enforced.run(10_000);

    // Shadow mode.
    let gate2 = ToleranceGate {
        max_wer: 0.0,
        max_confidence_delta: 0.0,
        max_single_segment_wer: 0.0,
    };
    let mut shadow = ConformanceHarness::new(gate2, true);
    shadow.add_engine(whisper_cpp_pilot_engine());
    shadow.add_engine(insanely_fast_pilot_engine());
    let shadow_report = shadow.run(10_000);

    // Enforced should fail, shadow should pass.
    assert!(
        !enforced_report.overall_passed,
        "enforced mode should fail with tight gate"
    );
    assert!(
        shadow_report.overall_passed,
        "shadow mode should pass regardless"
    );

    // But drift metrics should be identical.
    assert_eq!(
        enforced_report.drift_metrics.len(),
        shadow_report.drift_metrics.len()
    );
    for (e, s) in enforced_report
        .drift_metrics
        .iter()
        .zip(shadow_report.drift_metrics.iter())
    {
        assert_eq!(e.wer_approx, s.wer_approx);
        assert_eq!(e.confidence_delta, s.confidence_delta);
    }
}

#[test]
fn reproducibility_same_inputs_produce_same_hash() {
    let gate = ToleranceGate {
        max_wer: 1.0,
        max_confidence_delta: 1.0,
        max_single_segment_wer: 1.0,
    };

    // Run the harness twice with identical configuration.
    let mut harness1 = ConformanceHarness::new(gate.clone(), false);
    harness1.add_engine(whisper_cpp_pilot_engine());
    harness1.add_engine(insanely_fast_pilot_engine());
    harness1.add_engine(diarization_pilot_engine());
    let report1 = harness1.run(15_000);

    let gate2 = ToleranceGate {
        max_wer: 1.0,
        max_confidence_delta: 1.0,
        max_single_segment_wer: 1.0,
    };
    let mut harness2 = ConformanceHarness::new(gate2, false);
    harness2.add_engine(whisper_cpp_pilot_engine());
    harness2.add_engine(insanely_fast_pilot_engine());
    harness2.add_engine(diarization_pilot_engine());
    let report2 = harness2.run(15_000);

    assert_eq!(
        report1.reproducibility_hash, report2.reproducibility_hash,
        "identical inputs must produce identical reproducibility hash"
    );

    // Verify the hash is a non-empty hex string.
    assert!(!report1.reproducibility_hash.is_empty());
    assert!(
        report1
            .reproducibility_hash
            .chars()
            .all(|c| c.is_ascii_hexdigit()),
        "hash should be hex"
    );
}

#[test]
fn reproducibility_different_inputs_produce_different_hash() {
    let gate = ToleranceGate {
        max_wer: 1.0,
        max_confidence_delta: 1.0,
        max_single_segment_wer: 1.0,
    };

    let mut harness1 = ConformanceHarness::new(gate.clone(), false);
    harness1.add_engine(whisper_cpp_pilot_engine());
    let report1 = harness1.run(10_000);

    let gate2 = ToleranceGate {
        max_wer: 1.0,
        max_confidence_delta: 1.0,
        max_single_segment_wer: 1.0,
    };
    let mut harness2 = ConformanceHarness::new(gate2, false);
    harness2.add_engine(whisper_cpp_pilot_engine());
    let report2 = harness2.run(20_000);

    assert_ne!(
        report1.reproducibility_hash, report2.reproducibility_hash,
        "different input durations must produce different hashes"
    );
}

#[test]
fn harness_with_identical_bridge_and_pilot_shows_zero_drift() {
    let gate = ToleranceGate {
        max_wer: 0.001,
        max_confidence_delta: 0.001,
        max_single_segment_wer: 0.001,
    };
    let mut harness = ConformanceHarness::new(gate, false);
    // Both use WhisperCppPilot internally, so output is identical.
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(whisper_cpp_bridge_engine());

    let report = harness.run(10_000);

    assert!(
        report.overall_passed,
        "identical engines should pass even tight gates"
    );
    assert_eq!(report.drift_metrics.len(), 1);
    assert_eq!(report.drift_metrics[0].wer_approx, 0.0);
    assert_eq!(report.drift_metrics[0].confidence_delta, 0.0);
}

#[test]
fn harness_with_drifted_engine_fails_tight_gate() {
    let gate = ToleranceGate {
        max_wer: 0.05,
        max_confidence_delta: 0.01,
        max_single_segment_wer: 0.1,
    };
    let mut harness = ConformanceHarness::new(gate, false);
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(drifted_bridge_engine());

    let report = harness.run(10_000);

    assert!(
        !report.overall_passed,
        "drifted engine should fail tight gate"
    );
}

#[test]
fn harness_with_drifted_engine_passes_loose_gate() {
    let gate = ToleranceGate {
        max_wer: 0.5,
        max_confidence_delta: 0.5,
        max_single_segment_wer: 0.5,
    };
    let mut harness = ConformanceHarness::new(gate, false);
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(drifted_bridge_engine());

    let report = harness.run(10_000);

    assert!(
        report.overall_passed,
        "slightly drifted engine should pass loose gate"
    );
}

#[test]
fn harness_with_heavily_drifted_engine_fails_even_loose_gate() {
    let gate = ToleranceGate {
        max_wer: 0.5,
        max_confidence_delta: 0.3,
        max_single_segment_wer: 0.8,
    };
    let mut harness = ConformanceHarness::new(gate, false);
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(heavily_drifted_engine());

    let report = harness.run(10_000);

    assert!(
        !report.overall_passed,
        "heavily drifted engine should fail even a moderately loose gate"
    );
}

#[test]
fn word_error_rate_approx_identical_strings() {
    assert_eq!(word_error_rate_approx("hello world", "hello world"), 0.0);
}

#[test]
fn word_error_rate_approx_completely_different() {
    assert_eq!(word_error_rate_approx("hello world", "foo bar"), 1.0);
}

#[test]
fn word_error_rate_approx_partial_mismatch() {
    // "quick" vs "fast" = 1 mismatch out of 3 words.
    let wer = word_error_rate_approx("the quick fox", "the fast fox");
    assert!((wer - 1.0 / 3.0).abs() < 1e-10);
}

#[test]
fn word_error_rate_approx_empty_strings() {
    assert_eq!(word_error_rate_approx("", ""), 0.0);
}

#[test]
fn word_error_rate_approx_one_empty() {
    assert_eq!(word_error_rate_approx("hello world", ""), 1.0);
    assert_eq!(word_error_rate_approx("", "hello world"), 1.0);
}

#[test]
fn word_error_rate_approx_different_lengths() {
    // "hello world foo" vs "hello world" => 3 words max, 1 mismatch (missing "foo").
    let wer = word_error_rate_approx("hello world foo", "hello world");
    assert!((wer - 1.0 / 3.0).abs() < 1e-10);
}

#[test]
fn report_contains_all_engine_names() {
    let gate = ToleranceGate {
        max_wer: 1.0,
        max_confidence_delta: 1.0,
        max_single_segment_wer: 1.0,
    };
    let mut harness = ConformanceHarness::new(gate, false);
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(insanely_fast_pilot_engine());
    harness.add_engine(diarization_pilot_engine());

    let report = harness.run(10_000);
    let names: Vec<&str> = report
        .engine_results
        .iter()
        .map(|er| er.engine_name.as_str())
        .collect();

    assert!(names.contains(&"whisper-cpp-pilot"));
    assert!(names.contains(&"insanely-fast-pilot"));
    assert!(names.contains(&"diarization-pilot"));
}

#[test]
fn zero_duration_input_produces_empty_segments() {
    let gate = ToleranceGate {
        max_wer: 1.0,
        max_confidence_delta: 1.0,
        max_single_segment_wer: 1.0,
    };
    let mut harness = ConformanceHarness::new(gate, false);
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(insanely_fast_pilot_engine());

    let report = harness.run(0);

    for er in &report.engine_results {
        assert_eq!(
            er.segment_count, 0,
            "zero duration should produce zero segments for {}",
            er.engine_name
        );
    }
    assert!(report.overall_passed);
}
