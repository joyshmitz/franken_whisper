//! Tests for SpeculationWindowController and CorrectionEvidenceLedger (bd-qlt.15).

use franken_whisper::model::TranscriptionSegment;
use franken_whisper::speculation::{
    BetaPosterior, ControllerAction, CorrectionDecision, CorrectionDrift, CorrectionEvent,
    CorrectionEvidenceEntry, CorrectionEvidenceLedger, SpeculationWindowController,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_seg(text: &str, conf: f64) -> TranscriptionSegment {
    TranscriptionSegment {
        start_sec: Some(0.0),
        end_sec: Some(1.0),
        text: text.to_owned(),
        speaker: None,
        confidence: Some(conf),
    }
}

/// Create a controller with sensible defaults.
///
/// initial=3000, min=1000, max=30000, step=500
fn default_controller() -> SpeculationWindowController {
    SpeculationWindowController::new(3000, 1000, 30_000, 500)
}

/// Create a ledger with capacity 500.
fn default_ledger() -> CorrectionEvidenceLedger {
    CorrectionEvidenceLedger::new(500)
}

/// Build a `CorrectionDrift` for testing with the given approximate WER.
fn make_drift(wer: f64) -> CorrectionDrift {
    CorrectionDrift {
        wer_approx: wer,
        confidence_delta: 0.0,
        segment_count_delta: 0,
        text_edit_distance: 0,
    }
}

/// Build a `CorrectionDecision::Confirm` for testing.
fn confirm_decision(seq: u64, wer: f64) -> CorrectionDecision {
    CorrectionDecision::Confirm {
        seq,
        drift: make_drift(wer),
    }
}

/// Build a `CorrectionDecision::Correct` with a minimal CorrectionEvent.
fn correct_decision(correction_id: u64, window_id: u64, _wer: f64) -> CorrectionDecision {
    let fast_segs = vec![make_seg("hello", 0.75)];
    let quality_segs = vec![make_seg("world", 0.75)];
    let event = CorrectionEvent::new(
        correction_id,
        0,
        window_id,
        "whisper-large".to_owned(),
        quality_segs,
        200,
        "2026-01-01T00:00:00Z".to_owned(),
        &fast_segs,
    );
    CorrectionDecision::Correct { correction: event }
}

/// Feed `n` confirmations into the controller with the given WER and apply.
fn feed_confirmations(ctrl: &mut SpeculationWindowController, n: usize, wer: f64) {
    for i in 0..n {
        let decision = confirm_decision(i as u64, wer);
        let drift = make_drift(wer);
        ctrl.observe(&decision, &drift);
        ctrl.apply();
    }
}

/// Feed `n` corrections into the controller with the given WER and apply.
fn feed_corrections(ctrl: &mut SpeculationWindowController, n: usize, wer: f64) {
    for i in 0..n {
        let decision = correct_decision(i as u64, i as u64, wer);
        let drift = make_drift(wer);
        ctrl.observe(&decision, &drift);
        ctrl.apply();
    }
}

/// Build a `CorrectionEvidenceEntry` for ledger tests.
fn make_ledger_entry(
    entry_id: u64,
    window_id: u64,
    decision: &str,
    wer: f64,
    fast_latency_ms: u64,
    quality_latency_ms: u64,
    window_size_ms: u64,
) -> CorrectionEvidenceEntry {
    CorrectionEvidenceEntry {
        entry_id,
        window_id,
        run_id: "run-1".to_owned(),
        timestamp_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
        fast_model_id: "whisper-tiny".to_owned(),
        fast_latency_ms,
        fast_confidence_mean: 0.75,
        fast_segment_count: 1,
        quality_model_id: "whisper-large".to_owned(),
        quality_latency_ms,
        quality_confidence_mean: 0.75,
        quality_segment_count: 1,
        drift: CorrectionDrift {
            wer_approx: wer,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        },
        decision: decision.to_owned(),
        window_size_ms,
        correction_rate_at_decision: 0.0,
        controller_confidence: 0.5,
        fallback_active: false,
        fallback_reason: None,
    }
}

// ===========================================================================
// SpeculationWindowController tests
// ===========================================================================

#[test]
fn controller_initial_state() {
    let ctrl = default_controller();
    assert_eq!(ctrl.current_window_ms(), 3000);
    assert!(!ctrl.is_fallback_active());
    assert_eq!(ctrl.state().window_count, 0);
    assert!(ctrl.evidence().is_empty());
    assert!((ctrl.state().correction_rate - 0.0).abs() < 1e-9);
    assert!((ctrl.state().mean_wer - 0.0).abs() < 1e-9);
}

#[test]
fn controller_observe_confirm_holds_window() {
    let mut ctrl = default_controller();
    // Feed 6 confirmations with near-zero WER (past MIN_WINDOWS_FOR_ADAPT=5).
    // With low correction rate and < 10 windows, controller should Hold.
    feed_confirmations(&mut ctrl, 6, 0.0);
    // Window should remain at initial 3000 (Hold actions don't change size).
    assert_eq!(ctrl.current_window_ms(), 3000);
}

#[test]
fn controller_observe_correction_may_grow() {
    let mut ctrl = default_controller();
    // Feed many corrections with high WER to exceed HIGH_CORRECTION_RATE (0.25)
    // and HIGH_WER_THRESHOLD (0.125). We need > 5 observations and > 50%
    // confidence (window_count >= 10).
    // First get past the min window count with a mix:
    feed_corrections(&mut ctrl, 12, 0.5);
    // With many corrections, correction_rate should be high, WER high,
    // so the controller should have grown the window.
    assert!(ctrl.current_window_ms() > 3000);
}

#[test]
fn controller_window_clamped_to_max() {
    // Start near the max.
    let mut ctrl = SpeculationWindowController::new(29_500, 1000, 30_000, 1000);
    // Feed many corrections to force growth.
    feed_corrections(&mut ctrl, 20, 0.5);
    // Must never exceed max_window_ms.
    assert!(ctrl.current_window_ms() <= 30_000);
}

#[test]
fn controller_window_clamped_to_min() {
    // Start near the minimum.
    let mut ctrl = SpeculationWindowController::new(1500, 1000, 30_000, 500);
    // Feed many confirmations with zero WER. After enough (> 20 consecutive),
    // ZERO_CORRECTION_FORCE_SHRINK kicks in.
    feed_confirmations(&mut ctrl, 25, 0.0);
    // Must never go below min_window_ms.
    assert!(ctrl.current_window_ms() >= 1000);
}

#[test]
fn controller_fallback_on_high_correction_rate() {
    let mut ctrl = default_controller();
    // Feed enough corrections to push correction_rate > 75% (RUNAWAY_CORRECTION_RATE).
    // All corrections, no confirmations.
    feed_corrections(&mut ctrl, 10, 0.5);
    // After sustained corrections, fallback should activate.
    assert!(ctrl.is_fallback_active());
}

#[test]
fn controller_fallback_recoverable() {
    let mut ctrl = default_controller();
    // First, trigger fallback with many corrections.
    feed_corrections(&mut ctrl, 10, 0.5);
    assert!(ctrl.is_fallback_active());

    // Now feed enough confirmations to bring correction rate below the threshold
    // and clear the fallback. The RUNAWAY_CORRECTION_RATE is 0.75. We need
    // the Bayesian correction_rate = (alpha - 2) / (alpha + beta - 4) to drop
    // below 0.75. After 10 corrections: alpha=12, beta=2. We need to add
    // enough confirmations to get (12-2)/(12-2+beta_new-2) < 0.75.
    // 10/(10 + (beta_new - 2)) < 0.75 => 10 < 0.75*(8 + beta_new) =>
    // 10 < 6 + 0.75*beta_new => 4 < 0.75*beta_new => beta_new > 5.33
    // So we need beta to go from 2 to at least 8, meaning at least 6 confirms.
    // But since the controller also needs confidence and window_count checks,
    // let's be generous.
    feed_confirmations(&mut ctrl, 30, 0.0);
    // Fallback should now be cleared.
    assert!(!ctrl.is_fallback_active());
}

#[test]
fn controller_insufficient_data_no_adaptation() {
    let mut ctrl = default_controller();
    // Only 3 observations: below MIN_WINDOWS_FOR_ADAPT (5).
    for i in 0..3 {
        let decision = confirm_decision(i, 0.0);
        let drift = make_drift(0.0);
        ctrl.observe(&decision, &drift);
    }
    // recommend() should return Hold because too few observations.
    assert_eq!(ctrl.recommend(), ControllerAction::Hold);
    // apply() should keep the window unchanged.
    let new_size = ctrl.apply();
    assert_eq!(new_size, 3000);
}

#[test]
fn controller_posterior_starts_weakly_informative() {
    let ctrl = default_controller();
    let post = ctrl.posterior();
    // Beta(2,2) — weakly informative prior.
    assert!((post.alpha - 2.0).abs() < 1e-9);
    assert!((post.beta - 2.0).abs() < 1e-9);
    // Mean should be 0.5.
    assert!((post.mean() - 0.5).abs() < 1e-9);
}

#[test]
fn controller_posterior_updates_after_observations() {
    let mut ctrl = default_controller();
    // Feed 20 confirms, 0 corrections.
    feed_confirmations(&mut ctrl, 20, 0.0);
    let post = ctrl.posterior();
    // alpha = 2 (prior), beta = 2 + 20 = 22.
    assert!((post.alpha - 2.0).abs() < 1e-9);
    assert!((post.beta - 22.0).abs() < 1e-9);
    // Mean should be 2/24 ≈ 0.0833...
    let expected_mean = 2.0 / 24.0;
    assert!((post.mean() - expected_mean).abs() < 1e-6);
}

#[test]
fn controller_evidence_recorded_on_observe() {
    let mut ctrl = default_controller();
    assert!(ctrl.evidence().is_empty());

    let decision = confirm_decision(0, 0.0);
    let drift = make_drift(0.0);
    ctrl.observe(&decision, &drift);
    // observe alone does not record evidence; apply does.
    ctrl.apply();
    assert_eq!(ctrl.evidence().len(), 1);

    let decision2 = correct_decision(1, 1, 0.25);
    let drift2 = make_drift(0.25);
    ctrl.observe(&decision2, &drift2);
    ctrl.apply();
    assert_eq!(ctrl.evidence().len(), 2);
}

#[test]
fn controller_shrink_after_sustained_confirms() {
    let mut ctrl = default_controller();
    // ZERO_CORRECTION_FORCE_SHRINK = 20 consecutive confirms.
    // After > 20 consecutive zero-correction confirms, it should shrink.
    // We need window_count >= MIN_WINDOWS_FOR_ADAPT (5) and also
    // current_window_ms > min_window_ms.
    feed_confirmations(&mut ctrl, 25, 0.0);
    // After 25 consecutive confirms with zero WER, the window should have
    // shrunk at least once from the initial 3000.
    assert!(ctrl.current_window_ms() < 3000);
}

#[test]
fn controller_hold_in_stable_state() {
    let mut ctrl = default_controller();
    // Feed a mix: 3 corrections, 7 confirmations. Correction rate ~30%.
    // This is above LOW_CORRECTION_RATE (0.0625) but may not meet the
    // HIGH_CORRECTION_RATE + HIGH_WER condition firmly enough to grow.
    // Let's target correction_rate ~0.3 with moderate WER.
    for i in 0..3_u64 {
        let d = correct_decision(i, i, 0.125);
        let drift = make_drift(0.125);
        ctrl.observe(&d, &drift);
        ctrl.apply();
    }
    for i in 3..10_u64 {
        let d = confirm_decision(i, 0.0);
        let drift = make_drift(0.0);
        ctrl.observe(&d, &drift);
        ctrl.apply();
    }
    // With moderate correction rate and low WER, the controller should Hold.
    // The window should remain at or near initial.
    let action = ctrl.recommend();
    // We expect Hold because correction_rate is moderate but mean_wer is
    // relatively low (only corrections have wer=0.125, diluted by 7 confirms).
    assert_eq!(action, ControllerAction::Hold);
}

#[test]
fn controller_decision_count_increments() {
    let mut ctrl = default_controller();
    for i in 0..5_u64 {
        let d = confirm_decision(i, 0.0);
        let drift = make_drift(0.0);
        ctrl.observe(&d, &drift);
        ctrl.apply();
    }
    // Each apply() records one evidence entry with incrementing decision_id.
    let evidence = ctrl.evidence();
    assert_eq!(evidence.len(), 5);
    for (idx, entry) in evidence.iter().enumerate() {
        assert_eq!(entry.decision_id, idx as u64);
    }
}

// ===========================================================================
// CorrectionEvidenceLedger tests
// ===========================================================================

#[test]
fn ledger_new_is_empty() {
    let ledger = default_ledger();
    assert_eq!(ledger.entries().len(), 0);
    assert_eq!(ledger.total_recorded(), 0);
}

#[test]
fn ledger_zero_capacity_retains_none_but_counts_total() {
    let mut ledger = CorrectionEvidenceLedger::new(0);
    ledger.record(make_ledger_entry(0, 0, "corrected", 0.25, 50, 200, 3000));
    assert_eq!(ledger.entries().len(), 0);
    assert_eq!(ledger.total_recorded(), 1);
}

#[test]
fn ledger_record_adds_entry() {
    let mut ledger = default_ledger();
    let entry = make_ledger_entry(0, 0, "confirmed", 0.0, 50, 200, 3000);
    ledger.record(entry);
    assert_eq!(ledger.entries().len(), 1);
    assert_eq!(ledger.total_recorded(), 1);
}

#[test]
fn ledger_capacity_bounds_respected() {
    let mut ledger = CorrectionEvidenceLedger::new(3);
    for i in 0..5 {
        let entry = make_ledger_entry(i, i, "confirmed", 0.0, 50, 200, 3000);
        ledger.record(entry);
    }
    // Capacity is 3, so only 3 entries retained.
    assert_eq!(ledger.entries().len(), 3);
    // Total recorded includes evicted.
    assert_eq!(ledger.total_recorded(), 5);
    // Oldest retained should be entry_id=2 (0 and 1 evicted).
    assert_eq!(ledger.entries().front().unwrap().entry_id, 2);
}

#[test]
fn ledger_entries_returns_recent() {
    let mut ledger = CorrectionEvidenceLedger::new(5);
    for i in 0..5 {
        let entry = make_ledger_entry(i, i, "confirmed", 0.0, 50, 200, 3000);
        ledger.record(entry);
    }
    let entries = ledger.entries();
    // Should be in insertion order (oldest first).
    for (idx, entry) in entries.iter().enumerate() {
        assert_eq!(entry.entry_id, idx as u64);
    }
}

#[test]
fn ledger_total_recorded_counts_all() {
    let mut ledger = CorrectionEvidenceLedger::new(2);
    for i in 0..10 {
        let entry = make_ledger_entry(i, i, "confirmed", 0.0, 50, 200, 3000);
        ledger.record(entry);
    }
    assert_eq!(ledger.total_recorded(), 10);
    assert_eq!(ledger.entries().len(), 2);
}

#[test]
fn ledger_correction_rate_computed() {
    let mut ledger = default_ledger();
    // 2 corrections out of 4 total = 0.5
    ledger.record(make_ledger_entry(0, 0, "corrected", 0.25, 50, 200, 3000));
    ledger.record(make_ledger_entry(1, 1, "confirmed", 0.0, 50, 200, 3000));
    ledger.record(make_ledger_entry(2, 2, "corrected", 0.25, 50, 200, 3000));
    ledger.record(make_ledger_entry(3, 3, "confirmed", 0.0, 50, 200, 3000));
    let rate = ledger.correction_rate();
    assert!((rate - 0.5).abs() < 1e-9);
}

#[test]
fn ledger_mean_fast_latency() {
    let mut ledger = default_ledger();
    // fast_latency_ms: 100, 200, 300 => mean = 200.0
    ledger.record(make_ledger_entry(0, 0, "confirmed", 0.0, 100, 400, 3000));
    ledger.record(make_ledger_entry(1, 1, "confirmed", 0.0, 200, 400, 3000));
    ledger.record(make_ledger_entry(2, 2, "confirmed", 0.0, 300, 400, 3000));
    let mean = ledger.mean_fast_latency();
    assert!((mean - 200.0).abs() < 1e-9);
}

#[test]
fn ledger_mean_quality_latency() {
    let mut ledger = default_ledger();
    // quality_latency_ms: 400, 600, 800 => mean = 600.0
    ledger.record(make_ledger_entry(0, 0, "confirmed", 0.0, 50, 400, 3000));
    ledger.record(make_ledger_entry(1, 1, "confirmed", 0.0, 50, 600, 3000));
    ledger.record(make_ledger_entry(2, 2, "confirmed", 0.0, 50, 800, 3000));
    let mean = ledger.mean_quality_latency();
    assert!((mean - 600.0).abs() < 1e-9);
}

#[test]
fn ledger_mean_wer() {
    let mut ledger = default_ledger();
    // wer: 0.0, 0.25, 0.5, 0.25 => mean = 0.25
    ledger.record(make_ledger_entry(0, 0, "confirmed", 0.0, 50, 200, 3000));
    ledger.record(make_ledger_entry(1, 1, "corrected", 0.25, 50, 200, 3000));
    ledger.record(make_ledger_entry(2, 2, "corrected", 0.5, 50, 200, 3000));
    ledger.record(make_ledger_entry(3, 3, "corrected", 0.25, 50, 200, 3000));
    let mean = ledger.mean_wer();
    assert!((mean - 0.25).abs() < 1e-9);
}

#[test]
fn ledger_latency_savings_pct() {
    let mut ledger = default_ledger();
    // fast=100, quality=400 => savings = (400-100)/400 * 100 = 75.0
    ledger.record(make_ledger_entry(0, 0, "confirmed", 0.0, 100, 400, 3000));
    let savings = ledger.latency_savings_pct();
    assert!((savings - 75.0).abs() < 1e-9);
}

#[test]
fn ledger_to_evidence_json_valid() {
    let mut ledger = default_ledger();
    ledger.record(make_ledger_entry(0, 0, "confirmed", 0.0, 50, 200, 3000));
    ledger.record(make_ledger_entry(1, 1, "corrected", 0.25, 50, 200, 3000));

    let json = ledger.to_evidence_json();
    assert!(json.is_object());
    assert_eq!(json["type"], "correction_evidence_ledger");
    assert_eq!(json["total_recorded"], 2);
    assert_eq!(json["retained"], 2);
    assert_eq!(json["capacity"], 500);
    assert!(json["diagnostics"].is_object());
    assert!(json["entries"].is_array());
    assert_eq!(json["entries"].as_array().unwrap().len(), 2);
}

#[test]
fn ledger_diagnostics_summary() {
    let mut ledger = default_ledger();
    ledger.record(make_ledger_entry(0, 0, "corrected", 0.25, 100, 400, 3000));
    ledger.record(make_ledger_entry(1, 1, "confirmed", 0.0, 100, 400, 3000));

    let diag = ledger.diagnostics();
    assert!(diag.is_object());
    // correction_rate = 1/2 = 0.5
    let cr = diag["correction_rate"].as_f64().unwrap();
    assert!((cr - 0.5).abs() < 1e-9);
    // mean_fast_latency = 100
    let mfl = diag["mean_fast_latency_ms"].as_f64().unwrap();
    assert!((mfl - 100.0).abs() < 1e-9);
    // mean_quality_latency = 400
    let mql = diag["mean_quality_latency_ms"].as_f64().unwrap();
    assert!((mql - 400.0).abs() < 1e-9);
    // mean_wer = (0.25 + 0.0) / 2 = 0.125
    let mw = diag["mean_wer"].as_f64().unwrap();
    assert!((mw - 0.125).abs() < 1e-9);
    // latency_savings_pct = (400-100)/400 * 100 = 75.0
    let lsp = diag["latency_savings_pct"].as_f64().unwrap();
    assert!((lsp - 75.0).abs() < 1e-9);
}

#[test]
fn ledger_empty_analysis_returns_zero() {
    let ledger = default_ledger();
    assert!((ledger.correction_rate() - 0.0).abs() < 1e-9);
    assert!((ledger.mean_fast_latency() - 0.0).abs() < 1e-9);
    assert!((ledger.mean_quality_latency() - 0.0).abs() < 1e-9);
    assert!((ledger.mean_wer() - 0.0).abs() < 1e-9);
    assert!((ledger.latency_savings_pct() - 0.0).abs() < 1e-9);
    assert!(ledger.window_size_trend().is_empty());
}

#[test]
fn ledger_window_size_trend_direction() {
    let mut ledger = default_ledger();
    // Record entries with increasing window sizes.
    ledger.record(make_ledger_entry(0, 0, "confirmed", 0.0, 50, 200, 2000));
    ledger.record(make_ledger_entry(1, 1, "confirmed", 0.0, 50, 200, 3000));
    ledger.record(make_ledger_entry(2, 2, "confirmed", 0.0, 50, 200, 4000));

    let trend = ledger.window_size_trend();
    assert_eq!(trend.len(), 3);
    // Verify each (window_id, window_size_ms) pair.
    assert_eq!(trend[0], (0, 2000));
    assert_eq!(trend[1], (1, 3000));
    assert_eq!(trend[2], (2, 4000));

    // Trend is increasing.
    assert!(trend[2].1 > trend[0].1);

    // Now test decreasing trend.
    let mut ledger2 = default_ledger();
    ledger2.record(make_ledger_entry(0, 0, "confirmed", 0.0, 50, 200, 5000));
    ledger2.record(make_ledger_entry(1, 1, "confirmed", 0.0, 50, 200, 3000));
    ledger2.record(make_ledger_entry(2, 2, "confirmed", 0.0, 50, 200, 1000));
    let trend2 = ledger2.window_size_trend();
    assert!(trend2[2].1 < trend2[0].1);

    // Flat trend.
    let mut ledger3 = default_ledger();
    ledger3.record(make_ledger_entry(0, 0, "confirmed", 0.0, 50, 200, 3000));
    ledger3.record(make_ledger_entry(1, 1, "confirmed", 0.0, 50, 200, 3000));
    let trend3 = ledger3.window_size_trend();
    assert_eq!(trend3[0].1, trend3[1].1);
}

// ===========================================================================
// BetaPosterior unit tests (supplemental)
// ===========================================================================

#[test]
fn beta_posterior_weakly_informative() {
    let p = BetaPosterior::weakly_informative();
    assert!((p.alpha - 2.0).abs() < 1e-9);
    assert!((p.beta - 2.0).abs() < 1e-9);
    assert!((p.mean() - 0.5).abs() < 1e-9);
}

#[test]
fn beta_posterior_variance_decreases_with_data() {
    let mut p = BetaPosterior::weakly_informative();
    let initial_var = p.variance();
    for _ in 0..20 {
        p.observe_confirmation();
    }
    assert!(p.variance() < initial_var);
}

#[test]
fn beta_posterior_observe_updates() {
    let mut p = BetaPosterior::weakly_informative();
    p.observe_correction();
    assert!((p.alpha - 3.0).abs() < 1e-9);
    assert!((p.beta - 2.0).abs() < 1e-9);
    p.observe_confirmation();
    assert!((p.alpha - 3.0).abs() < 1e-9);
    assert!((p.beta - 3.0).abs() < 1e-9);
    // Mean should be 3/6 = 0.5
    assert!((p.mean() - 0.5).abs() < 1e-9);
}
