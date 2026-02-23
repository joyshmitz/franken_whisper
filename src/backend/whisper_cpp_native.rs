//! Native whisper.cpp engine pilot (bd-1rj.9).
//!
//! This module implements an in-process, deterministic native pilot:
//! no subprocess execution, no external binary probes, and stable segment
//! schema parity with the bridge adapter.

use std::fs;
use std::path::Path;
use std::time::Duration;

use serde_json::{Value, json};

use crate::error::FwResult;
use crate::model::{BackendKind, TranscribeRequest, TranscriptionResult, TranscriptionSegment};

use super::native_audio::analyze_wav;

const WAV_HEADER_BYTES: u64 = 44;
const PCM16_MONO_16KHZ_BYTES_PER_SECOND: u64 = 32_000;
const MIN_DURATION_MS: u64 = 1_000;
const MAX_DURATION_MS: u64 = 30 * 60 * 1_000;
const DEFAULT_DURATION_MS: u64 = 15_000;
const MIN_REGION_DURATION_MS: u64 = 250;

#[derive(Debug, Clone)]
struct NativeSegmentation {
    duration_ms: u64,
    pilot_segments: Vec<super::TranscriptSegment>,
    analysis_provenance: Value,
}

/// In-process pilot is always available at compile/runtime.
pub fn is_available() -> bool {
    true
}

pub fn run(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    _work_dir: &Path,
    _timeout: Duration,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<TranscriptionResult> {
    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    let threads = usize::try_from(request.backend_params.threads.unwrap_or(4)).unwrap_or(4);
    let pilot = super::WhisperCppPilot::new(
        request
            .model
            .clone()
            .unwrap_or_else(|| "ggml-base.en".to_owned()),
        threads,
        request.language.clone(),
        request.translate,
    );
    let native = build_native_segmentation(request, normalized_wav, &pilot, token)?;
    let segments = to_transcription_segments(&native.pilot_segments, token)?;
    let transcript = super::transcript_from_segments(&segments);

    Ok(TranscriptionResult {
        backend: BackendKind::WhisperCpp,
        transcript,
        language: request.language.clone(),
        segments: segments.clone(),
        acceleration: None,
        raw_output: json!({
            "engine": "whisper.cpp-native",
            "schema_version": "native-pilot-v1",
            "in_process": true,
            "model": request.model.clone().unwrap_or_else(|| "ggml-base.en".to_owned()),
            "threads": request.backend_params.threads.unwrap_or(4),
            "duration_ms": native.duration_ms,
            "streaming_supported": true,
            "analysis": native.analysis_provenance,
            "segments": segments,
        }),
        artifact_paths: Vec::new(),
    })
}

pub fn run_streaming(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    _work_dir: &Path,
    _timeout: Duration,
    token: Option<&crate::orchestrator::CancellationToken>,
    on_segment: &dyn Fn(TranscriptionSegment),
) -> FwResult<TranscriptionResult> {
    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    let threads = usize::try_from(request.backend_params.threads.unwrap_or(4)).unwrap_or(4);
    let pilot = super::WhisperCppPilot::new(
        request
            .model
            .clone()
            .unwrap_or_else(|| "ggml-base.en".to_owned()),
        threads,
        request.language.clone(),
        request.translate,
    );
    let native = build_native_segmentation(request, normalized_wav, &pilot, token)?;
    let segments = to_transcription_segments(&native.pilot_segments, token)?;
    for segment in &segments {
        if let Some(tok) = token {
            tok.checkpoint()?;
        }
        on_segment(segment.clone());
    }

    Ok(TranscriptionResult {
        backend: BackendKind::WhisperCpp,
        transcript: super::transcript_from_segments(&segments),
        language: request.language.clone(),
        segments: segments.clone(),
        acceleration: None,
        raw_output: json!({
            "engine": "whisper.cpp-native",
            "schema_version": "native-pilot-v1",
            "in_process": true,
            "model": request.model.clone().unwrap_or_else(|| "ggml-base.en".to_owned()),
            "threads": request.backend_params.threads.unwrap_or(4),
            "duration_ms": native.duration_ms,
            "streaming_supported": true,
            "streaming_emitted_segments": segments.len(),
            "analysis": native.analysis_provenance,
            "segments": segments,
        }),
        artifact_paths: Vec::new(),
    })
}

fn build_native_segmentation(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    pilot: &super::WhisperCppPilot,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<NativeSegmentation> {
    let duration_hint = request.backend_params.duration_ms;
    let analysis = match analyze_wav(normalized_wav, duration_hint) {
        Ok(analysis) => analysis,
        Err(error) => {
            let duration_ms = estimate_duration_ms(request, normalized_wav);
            let pilot_segments = pilot.transcribe(duration_ms);
            return Ok(NativeSegmentation {
                duration_ms,
                pilot_segments,
                analysis_provenance: json!({
                    "mode": "duration_fallback",
                    "reason": error,
                    "duration_ms": duration_ms,
                }),
            });
        }
    };

    let mut pilot_segments = Vec::new();
    for (region_index, region) in analysis.active_regions.iter().enumerate() {
        if let Some(tok) = token {
            tok.checkpoint()?;
        }
        let region_duration_ms = region
            .end_ms
            .saturating_sub(region.start_ms)
            .max(MIN_REGION_DURATION_MS);
        let mut region_segments = pilot.transcribe(region_duration_ms);
        for (segment_index, mut segment) in region_segments.drain(..).enumerate() {
            if let Some(tok) = token {
                tok.checkpoint()?;
            }
            let start_ms = region.start_ms.saturating_add(segment.start_ms);
            let mut end_ms = region.start_ms.saturating_add(segment.end_ms);
            if end_ms > region.end_ms {
                end_ms = region.end_ms;
            }
            if end_ms <= start_ms {
                continue;
            }

            let energy_bonus = (region.avg_rms as f64 * 4.0).clamp(0.0, 0.08);
            let continuity_penalty = ((region_index + segment_index) as f64) * 0.003;
            segment.start_ms = start_ms;
            segment.end_ms = end_ms;
            segment.confidence =
                (segment.confidence + energy_bonus - continuity_penalty).clamp(0.45, 0.99);
            pilot_segments.push(segment);
        }
    }

    pilot_segments.sort_by_key(|segment| (segment.start_ms, segment.end_ms));

    let segment_count = pilot_segments.len();
    Ok(NativeSegmentation {
        duration_ms: analysis.duration_ms,
        pilot_segments,
        analysis_provenance: json!({
            "mode": "waveform",
            "duration_hint_ms": duration_hint,
            "analysis": analysis.as_json(),
            "segments_from_active_regions": segment_count,
        }),
    })
}

fn estimate_duration_ms(request: &TranscribeRequest, normalized_wav: &Path) -> u64 {
    if let Some(duration_ms) = request.backend_params.duration_ms {
        return duration_ms.clamp(MIN_DURATION_MS, MAX_DURATION_MS);
    }

    let Ok(metadata) = fs::metadata(normalized_wav) else {
        return DEFAULT_DURATION_MS;
    };
    let audio_bytes = metadata.len().saturating_sub(WAV_HEADER_BYTES);
    let estimated =
        ((audio_bytes as f64 / PCM16_MONO_16KHZ_BYTES_PER_SECOND as f64) * 1_000.0).round() as u64;
    estimated.clamp(MIN_DURATION_MS, MAX_DURATION_MS)
}

fn to_transcription_segments(
    pilot_segments: &[super::TranscriptSegment],
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<Vec<TranscriptionSegment>> {
    let mut segments = Vec::with_capacity(pilot_segments.len());
    for seg in pilot_segments {
        if let Some(tok) = token {
            tok.checkpoint()?;
        }
        segments.push(TranscriptionSegment {
            start_sec: Some(seg.start_ms as f64 / 1_000.0),
            end_sec: Some(seg.end_ms as f64 / 1_000.0),
            text: seg.text.trim().to_owned(),
            speaker: None,
            confidence: Some(seg.confidence.clamp(0.0, 1.0)),
        });
    }
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::time::Duration;

    use crate::backend::{Engine, TranscriptSegment, WhisperCppPilot};
    use crate::model::{BackendKind, BackendParams, InputSource, TranscribeRequest};
    use crate::orchestrator::CancellationToken;

    use super::*;

    fn native_request() -> TranscribeRequest {
        TranscribeRequest {
            input: InputSource::File {
                path: PathBuf::from("input.wav"),
            },
            backend: BackendKind::WhisperCpp,
            model: Some("ggml-base.en".to_owned()),
            language: Some("en".to_owned()),
            translate: false,
            diarize: false,
            persist: false,
            db_path: PathBuf::from("state.sqlite3"),
            timeout_ms: None,
            backend_params: BackendParams::default(),
        }
    }

    #[test]
    fn native_engine_name_follows_naming_convention() {
        // Per native_engine_contract.md §1.2: native names follow
        // "<backend>-native" pattern, distinct from bridge adapter.
        let engine = super::super::WhisperCppNativeEngine;
        assert_eq!(engine.name(), "whisper.cpp-native");
    }

    #[test]
    fn native_engine_kind_matches_bridge_adapter() {
        // Per native_engine_contract.md §1: native engine returns the
        // SAME BackendKind variant as the bridge adapter it replaces.
        let native = super::super::WhisperCppNativeEngine;
        let bridge = super::super::WhisperCppEngine;
        assert_eq!(native.kind(), bridge.kind());
        assert_eq!(native.kind(), BackendKind::WhisperCpp);
    }

    #[test]
    fn native_engine_capabilities_superset_of_bridge() {
        // Per native_engine_contract.md §1: native engine must declare
        // at least the capabilities the bridge adapter declares.
        let native = super::super::WhisperCppNativeEngine;
        let bridge = super::super::WhisperCppEngine;
        let native_caps = native.capabilities();
        let bridge_caps = bridge.capabilities();

        // Every capability the bridge declares, native must also declare.
        if bridge_caps.supports_diarization {
            assert!(native_caps.supports_diarization);
        }
        if bridge_caps.supports_translation {
            assert!(native_caps.supports_translation);
        }
        if bridge_caps.supports_word_timestamps {
            assert!(native_caps.supports_word_timestamps);
        }
        if bridge_caps.supports_gpu {
            assert!(native_caps.supports_gpu);
        }
        if bridge_caps.supports_streaming {
            assert!(native_caps.supports_streaming);
        }
    }

    #[test]
    fn native_engine_name_distinct_from_bridge() {
        // Shadow-run comparison needs distinct names to distinguish
        // engine identity in ReplayEnvelope.backend_identity.
        let native = super::super::WhisperCppNativeEngine;
        let bridge = super::super::WhisperCppEngine;
        assert_ne!(native.name(), bridge.name());
        assert!(native.name().contains("native"));
    }

    #[test]
    fn native_engine_availability_is_in_process_true() {
        assert!(is_available());
    }

    #[test]
    fn native_run_produces_non_empty_segments_without_subprocess_artifacts() {
        let request = native_request();
        let result = run(
            &request,
            Path::new("does-not-need-to-exist.wav"),
            Path::new("."),
            Duration::from_secs(1),
            None,
        )
        .expect("native pilot run should succeed");

        assert_eq!(result.backend, BackendKind::WhisperCpp);
        assert!(!result.segments.is_empty());
        assert!(result.artifact_paths.is_empty());
        assert_eq!(
            result.raw_output["engine"].as_str(),
            Some("whisper.cpp-native")
        );
        assert_eq!(
            result.raw_output["analysis"]["mode"].as_str(),
            Some("duration_fallback")
        );
    }

    #[test]
    fn streaming_run_emits_same_number_of_segments_as_final_result() {
        let request = native_request();
        let streamed = Mutex::new(Vec::new());

        let result = run_streaming(
            &request,
            Path::new("does-not-need-to-exist.wav"),
            Path::new("."),
            Duration::from_secs(1),
            None,
            &|segment| streamed.lock().expect("lock").push(segment),
        )
        .expect("streaming pilot run should succeed");

        let streamed = streamed.lock().expect("lock");
        assert_eq!(streamed.len(), result.segments.len());
        for (streamed_segment, final_segment) in streamed.iter().zip(result.segments.iter()) {
            assert_eq!(streamed_segment.start_sec, final_segment.start_sec);
            assert_eq!(streamed_segment.end_sec, final_segment.end_sec);
            assert_eq!(streamed_segment.text, final_segment.text);
            assert_eq!(streamed_segment.speaker, final_segment.speaker);
            assert_eq!(streamed_segment.confidence, final_segment.confidence);
        }
    }

    fn write_pcm16_mono_wav(path: &Path, sample_rate: u32, samples: &[i16]) {
        let data_len = (samples.len() * 2) as u32;
        let mut bytes = Vec::with_capacity(44 + data_len as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        std::fs::write(path, bytes).expect("write wav");
    }

    #[test]
    fn native_run_is_content_sensitive_for_silence_vs_tone() {
        let dir = tempfile::tempdir().expect("tempdir");
        let silence_path = dir.path().join("silence.wav");
        let tone_path = dir.path().join("tone.wav");

        write_pcm16_mono_wav(&silence_path, 16_000, &vec![0i16; 16_000]);
        let mut tone = vec![0i16; 8_000];
        tone.extend((0..16_000).map(|i| if i % 2 == 0 { 9_000 } else { -9_000 }));
        tone.extend(vec![0i16; 8_000]);
        write_pcm16_mono_wav(&tone_path, 16_000, &tone);

        let req = native_request();
        let silence = run(
            &req,
            &silence_path,
            dir.path(),
            Duration::from_secs(1),
            None,
        )
        .expect("silence run");
        let tone =
            run(&req, &tone_path, dir.path(), Duration::from_secs(1), None).expect("tone run");

        assert!(
            silence.segments.len() < tone.segments.len(),
            "tone should produce richer segmentation than silence"
        );
        assert_eq!(
            tone.raw_output["analysis"]["mode"].as_str(),
            Some("waveform")
        );
    }

    #[test]
    fn native_run_is_deterministic_and_keeps_segment_invariants() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("deterministic.wav");
        let mut samples = vec![0i16; 4_000];
        samples.extend((0..16_000).map(|i| if i % 2 == 0 { 8_500 } else { -8_500 }));
        samples.extend(vec![0i16; 4_000]);
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let req = native_request();
        let first = run(&req, &wav, dir.path(), Duration::from_secs(1), None)
            .expect("first deterministic run");
        let second = run(&req, &wav, dir.path(), Duration::from_secs(1), None)
            .expect("second deterministic run");

        assert_eq!(first.segments.len(), second.segments.len());
        for (left, right) in first.segments.iter().zip(second.segments.iter()) {
            assert_eq!(left.start_sec, right.start_sec);
            assert_eq!(left.end_sec, right.end_sec);
            assert_eq!(left.text, right.text);
            assert_eq!(left.speaker, right.speaker);
            assert_eq!(left.confidence, right.confidence);
        }
        assert_eq!(first.transcript, second.transcript);
        assert_eq!(first.raw_output["analysis"], second.raw_output["analysis"]);

        assert!(
            first
                .segments
                .windows(2)
                .all(|pair| pair[0].end_sec <= pair[1].start_sec),
            "segments should be monotonic"
        );
        assert!(first.segments.iter().all(|segment| {
            segment
                .confidence
                .is_none_or(|confidence| (0.0..=1.0).contains(&confidence))
        }));
    }

    #[test]
    fn native_run_honors_cancellation_token_checkpoints() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("tone.wav");
        let mut samples = vec![0i16; 4_000];
        samples.extend((0..16_000).map(|i| if i % 2 == 0 { 8_000 } else { -8_000 }));
        samples.extend(vec![0i16; 4_000]);
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let request = native_request();
        let pilot =
            WhisperCppPilot::new("ggml-base.en".to_owned(), 4, Some("en".to_owned()), false);
        let token = CancellationToken::with_deadline_from_now(Duration::from_millis(0));
        let result = build_native_segmentation(&request, &wav, &pilot, Some(&token));
        assert!(
            result.is_err(),
            "expired cancellation token should fail waveform segmentation"
        );
    }

    #[test]
    fn estimate_duration_ms_clamps_hint_and_defaults() {
        let mut req = native_request();

        // Below minimum → clamp to MIN_DURATION_MS (1_000)
        req.backend_params.duration_ms = Some(100);
        assert_eq!(estimate_duration_ms(&req, Path::new("no.wav")), 1_000);

        // Above maximum → clamp to MAX_DURATION_MS (1_800_000)
        req.backend_params.duration_ms = Some(10_000_000);
        assert_eq!(estimate_duration_ms(&req, Path::new("no.wav")), 1_800_000);

        // Valid hint → pass through
        req.backend_params.duration_ms = Some(42_000);
        assert_eq!(estimate_duration_ms(&req, Path::new("no.wav")), 42_000);

        // No hint + no file → DEFAULT_DURATION_MS (15_000)
        req.backend_params.duration_ms = None;
        assert_eq!(estimate_duration_ms(&req, Path::new("no_such.wav")), 15_000);
    }

    #[test]
    fn estimate_duration_ms_from_wav_file_size() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("test.wav");
        // 1 second of PCM16 mono 16kHz = 32000 bytes + 44 header
        let samples = vec![0i16; 16_000];
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let mut req = native_request();
        req.backend_params.duration_ms = None;
        let estimated = estimate_duration_ms(&req, &wav);
        assert!(
            (estimated as i64 - 1_000).unsigned_abs() < 50,
            "1s of PCM should estimate ~1000ms, got {estimated}"
        );
    }

    #[test]
    fn to_transcription_segments_trims_and_clamps() {
        let pilot_segments = vec![
            TranscriptSegment {
                start_ms: 500,
                end_ms: 1500,
                text: "  hello world  ".to_owned(),
                confidence: 1.5, // above 1.0 → clamp
            },
            TranscriptSegment {
                start_ms: 2000,
                end_ms: 3000,
                text: "ok".to_owned(),
                confidence: -0.2, // below 0.0 → clamp
            },
        ];

        let result = to_transcription_segments(&pilot_segments, None).expect("segments");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "hello world");
        assert!((result[0].start_sec.unwrap() - 0.5).abs() < 0.001);
        assert!((result[0].end_sec.unwrap() - 1.5).abs() < 0.001);
        assert_eq!(result[0].confidence, Some(1.0));
        assert!(result[0].speaker.is_none());

        assert_eq!(result[1].confidence, Some(0.0));
    }

    #[test]
    fn streaming_run_cancellation_aborts_emission() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("tone.wav");
        let mut samples = vec![0i16; 4_000];
        samples.extend((0..16_000).map(|i| if i % 2 == 0 { 8_000 } else { -8_000 }));
        samples.extend(vec![0i16; 4_000]);
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let request = native_request();
        let token = CancellationToken::with_deadline_from_now(Duration::from_millis(0));
        let emitted = Mutex::new(Vec::new());

        let result = run_streaming(
            &request,
            &wav,
            dir.path(),
            Duration::from_secs(1),
            Some(&token),
            &|seg| emitted.lock().expect("lock").push(seg),
        );

        assert!(result.is_err(), "expired token should abort streaming run");
        assert!(
            emitted.lock().expect("lock").is_empty(),
            "no segments should be emitted before cancellation"
        );
    }

    #[test]
    fn run_with_translate_flag_populates_raw_output() {
        let mut req = native_request();
        req.translate = true;
        req.model = None; // defaults to "ggml-base.en"

        let result = run(
            &req,
            Path::new("missing.wav"),
            Path::new("."),
            Duration::from_secs(1),
            None,
        )
        .expect("run should succeed");

        assert_eq!(result.raw_output["model"].as_str(), Some("ggml-base.en"));
        assert_eq!(result.raw_output["in_process"].as_bool(), Some(true));
        assert!(result.raw_output["streaming_supported"].as_bool().unwrap());
    }

    // ── Task #210 — whisper_cpp_native edge-case tests ──────────────

    #[test]
    fn estimate_duration_ms_sub_header_file_clamps_to_min() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let tiny = tmp.path().join("tiny.wav");
        // 10 bytes < WAV_HEADER_BYTES (44) → audio_bytes saturates to 0
        // → estimated = 0 → clamped to MIN_DURATION_MS (1_000).
        std::fs::write(&tiny, b"0123456789").unwrap();

        let mut req = native_request();
        req.backend_params.duration_ms = None;
        assert_eq!(
            estimate_duration_ms(&req, &tiny),
            1_000,
            "sub-header-size file should clamp to MIN_DURATION_MS"
        );
    }

    #[test]
    fn to_transcription_segments_empty_input_returns_empty() {
        let result = to_transcription_segments(&[], None).unwrap();
        assert!(
            result.is_empty(),
            "empty pilot segments should produce empty output"
        );
    }

    #[test]
    fn to_transcription_segments_cancellation_propagates() {
        let pilot_segs = vec![TranscriptSegment {
            text: "hello".to_owned(),
            start_ms: 0,
            end_ms: 1000,
            confidence: 0.9,
        }];

        let token = CancellationToken::with_deadline_from_now(Duration::from_millis(0));
        std::thread::sleep(Duration::from_millis(5));

        let result = to_transcription_segments(&pilot_segs, Some(&token));
        assert!(
            result.is_err(),
            "expired cancellation token should propagate error"
        );
    }

    #[test]
    fn run_with_large_threads_value_does_not_panic() {
        let mut req = native_request();
        req.backend_params.threads = Some(u32::MAX);

        let result = run(
            &req,
            Path::new("missing.wav"),
            Path::new("."),
            Duration::from_secs(1),
            None,
        )
        .expect("large threads value should not panic");

        assert_eq!(
            result.raw_output["threads"].as_u64(),
            Some(u32::MAX as u64),
            "raw_output should record the exact requested threads value"
        );
    }

    #[test]
    fn run_silence_produces_empty_transcript() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let wav = tmp.path().join("silence.wav");
        // Write a valid WAV header + 16000 samples of silence (1 second at 16kHz).
        let sample_rate: u32 = 16_000;
        let num_samples: u32 = 16_000;
        let data_size = num_samples * 2; // 16-bit = 2 bytes per sample
        let file_size = 36 + data_size;

        let mut header = Vec::with_capacity(44);
        header.extend_from_slice(b"RIFF");
        header.extend_from_slice(&file_size.to_le_bytes());
        header.extend_from_slice(b"WAVEfmt ");
        header.extend_from_slice(&16u32.to_le_bytes()); // subchunk1 size
        header.extend_from_slice(&1u16.to_le_bytes()); // PCM format
        header.extend_from_slice(&1u16.to_le_bytes()); // mono
        header.extend_from_slice(&sample_rate.to_le_bytes());
        header.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        header.extend_from_slice(&2u16.to_le_bytes()); // block align
        header.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        header.extend_from_slice(b"data");
        header.extend_from_slice(&data_size.to_le_bytes());
        // All zeros = silence.
        header.extend(vec![0u8; data_size as usize]);
        std::fs::write(&wav, &header).unwrap();

        let req = native_request();
        let result = run(&req, &wav, tmp.path(), Duration::from_secs(5), None)
            .expect("silence run should succeed");

        // Silence should produce an empty or minimal transcript.
        assert!(
            result.segments.is_empty()
                || result.transcript.is_empty()
                || result.transcript.trim().is_empty(),
            "silence should produce empty/minimal transcript, got: {:?}",
            result.transcript
        );
    }
}
