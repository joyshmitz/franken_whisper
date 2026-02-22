//! Native insanely-fast engine pilot (bd-1rj.10).
//!
//! In-process deterministic implementation that mirrors the bridge adapter's
//! normalized output contract including diarization token readiness checks,
//! batching knobs, and canonical segment normalization.

use std::fs;
use std::path::Path;
use std::time::Duration;

use serde_json::{Value, json};

use crate::error::{FwError, FwResult};
use crate::model::{BackendKind, TranscribeRequest, TranscriptionResult, TranscriptionSegment};

use super::native_audio::analyze_wav;

const WAV_HEADER_BYTES: u64 = 44;
const PCM16_MONO_16KHZ_BYTES_PER_SECOND: u64 = 32_000;
const MIN_DURATION_MS: u64 = 1_000;
const MAX_DURATION_MS: u64 = 30 * 60 * 1_000;
const DEFAULT_DURATION_MS: u64 = 20_000;
const MIN_REGION_DURATION_MS: u64 = 250;

#[derive(Debug, Clone)]
struct NativeBatchSegmentation {
    duration_ms: u64,
    pilot_segments: Vec<super::TranscriptSegment>,
    analysis_provenance: Value,
}

/// In-process pilot is always available.
pub fn is_available() -> bool {
    true
}

/// Whether a HuggingFace token is present for the given request.
pub(crate) fn hf_token_present_for_request(request: &TranscribeRequest) -> bool {
    super::insanely_fast::hf_token_present_for_request(request)
}

pub fn run(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    work_dir: &Path,
    _timeout: Duration,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<TranscriptionResult> {
    if request.diarize && !hf_token_present_for_request(request) {
        return Err(FwError::BackendUnavailable(
            "diarization requires HF token (`--hf-token` or env `FRANKEN_WHISPER_HF_TOKEN` / `HF_TOKEN`)"
                .to_owned(),
        ));
    }
    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    let batch_size = request
        .backend_params
        .batch_size
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(8)
        .max(1);
    let device = request
        .backend_params
        .gpu_device
        .clone()
        .unwrap_or_else(|| {
            if request.backend_params.no_gpu {
                "cpu".to_owned()
            } else {
                "cuda:0".to_owned()
            }
        });
    let dtype = if device.starts_with("cuda") || device.starts_with("mps") {
        "float16".to_owned()
    } else {
        "float32".to_owned()
    };
    let pilot = super::InsanelyFastPilot::new(
        request
            .model
            .clone()
            .unwrap_or_else(|| "openai/whisper-large-v3".to_owned()),
        batch_size,
        device.clone(),
        dtype.clone(),
    );
    let native = build_native_segmentation(request, normalized_wav, &pilot, token)?;
    let segments = to_transcription_segments(&native.pilot_segments, request.diarize, token)?;
    let transcript = super::transcript_from_segments(&segments);

    let output_path = super::insanely_fast::output_path_for(request, work_dir);
    if let Some(parent) = output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let raw_output = json!({
        "engine": "insanely-fast-native",
        "schema_version": "native-pilot-v1",
        "in_process": true,
        "model": request.model.clone().unwrap_or_else(|| "openai/whisper-large-v3".to_owned()),
        "text": transcript,
        "language": request.language.clone(),
        "chunks": segments.iter().map(|seg| {
            json!({
                "text": seg.text,
                "timestamp": [seg.start_sec, seg.end_sec],
                "speaker": seg.speaker,
                "confidence": seg.confidence,
            })
        }).collect::<Vec<_>>(),
        "telemetry": {
            "batch_size_requested": request.backend_params.batch_size,
            "batch_size_effective": batch_size,
            "batch_count": 1,
            "device": device,
            "dtype": dtype,
            "flash_attention_requested": request.backend_params.flash_attention,
            "duration_ms": native.duration_ms,
            "analysis": native.analysis_provenance,
        }
    });
    fs::write(&output_path, serde_json::to_vec_pretty(&raw_output)?)?;

    Ok(TranscriptionResult {
        backend: BackendKind::InsanelyFast,
        transcript: super::transcript_from_segments(&segments),
        language: request.language.clone(),
        segments,
        acceleration: None,
        raw_output,
        artifact_paths: vec![output_path.display().to_string()],
    })
}

fn build_native_segmentation(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    pilot: &super::InsanelyFastPilot,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<NativeBatchSegmentation> {
    let duration_hint = request.backend_params.duration_ms;
    let analysis = match analyze_wav(normalized_wav, duration_hint) {
        Ok(analysis) => analysis,
        Err(error) => {
            let duration_ms = estimate_duration_ms(request, normalized_wav);
            let pilot_segments = pilot
                .transcribe_batch(&[duration_ms])
                .into_iter()
                .next()
                .unwrap_or_default();
            return Ok(NativeBatchSegmentation {
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
        let mut region_segments = pilot
            .transcribe_batch(&[region_duration_ms])
            .into_iter()
            .next()
            .unwrap_or_default();
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

            let energy_bonus = (region.avg_rms as f64 * 3.0).clamp(0.0, 0.09);
            let continuity_penalty = ((region_index + segment_index) as f64) * 0.0025;
            segment.start_ms = start_ms;
            segment.end_ms = end_ms;
            segment.confidence =
                (segment.confidence + energy_bonus - continuity_penalty).clamp(0.5, 0.995);
            pilot_segments.push(segment);
        }
    }
    pilot_segments.sort_by_key(|segment| (segment.start_ms, segment.end_ms));

    let segment_count = pilot_segments.len();
    Ok(NativeBatchSegmentation {
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
    diarize: bool,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<Vec<TranscriptionSegment>> {
    let mut segments = Vec::with_capacity(pilot_segments.len());
    for (index, seg) in pilot_segments.iter().enumerate() {
        if let Some(tok) = token {
            tok.checkpoint()?;
        }
        segments.push(TranscriptionSegment {
            start_sec: Some(seg.start_ms as f64 / 1_000.0),
            end_sec: Some(seg.end_ms as f64 / 1_000.0),
            text: seg.text.trim().to_owned(),
            speaker: diarize.then(|| format!("SPEAKER_{:02}", index % 2)),
            confidence: Some(seg.confidence.clamp(0.0, 1.0)),
        });
    }
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use tempfile::tempdir;

    use crate::backend::{Engine, InsanelyFastPilot};
    use crate::model::{BackendKind, BackendParams, InputSource, TranscribeRequest};
    use crate::orchestrator::CancellationToken;

    use super::*;

    fn request() -> TranscribeRequest {
        TranscribeRequest {
            input: InputSource::File {
                path: PathBuf::from("input.wav"),
            },
            backend: BackendKind::InsanelyFast,
            model: Some("openai/whisper-large-v3".to_owned()),
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
        let engine = super::super::InsanelyFastNativeEngine;
        assert_eq!(engine.name(), "insanely-fast-native");
    }

    #[test]
    fn native_engine_kind_matches_bridge_adapter() {
        let native = super::super::InsanelyFastNativeEngine;
        let bridge = super::super::InsanelyFastEngine;
        assert_eq!(native.kind(), bridge.kind());
        assert_eq!(native.kind(), BackendKind::InsanelyFast);
    }

    #[test]
    fn native_engine_capabilities_superset_of_bridge() {
        let native = super::super::InsanelyFastNativeEngine;
        let bridge = super::super::InsanelyFastEngine;
        let native_caps = native.capabilities();
        let bridge_caps = bridge.capabilities();

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
        let native = super::super::InsanelyFastNativeEngine;
        let bridge = super::super::InsanelyFastEngine;
        assert_ne!(native.name(), bridge.name());
        assert!(native.name().contains("native"));
    }

    #[test]
    fn native_engine_availability_is_in_process_true() {
        assert!(is_available());
    }

    #[test]
    fn native_run_produces_segments_and_artifact() {
        let tmp = tempdir().expect("tempdir");
        let result = run(
            &request(),
            Path::new("does-not-need-to-exist.wav"),
            tmp.path(),
            Duration::from_secs(1),
            None,
        )
        .expect("native pilot run should succeed");

        assert_eq!(result.backend, BackendKind::InsanelyFast);
        assert!(!result.segments.is_empty());
        assert_eq!(result.artifact_paths.len(), 1);
        assert!(
            Path::new(&result.artifact_paths[0]).exists(),
            "native pilot should write transcript artifact"
        );
        assert_eq!(
            result.raw_output["engine"].as_str(),
            Some("insanely-fast-native")
        );
        assert_eq!(
            result.raw_output["telemetry"]["analysis"]["mode"].as_str(),
            Some("duration_fallback")
        );
    }

    #[test]
    fn native_run_diarize_without_token_fails() {
        let tmp = tempdir().expect("tempdir");
        let mut req = request();
        req.diarize = true;
        let result = run(
            &req,
            Path::new("does-not-need-to-exist.wav"),
            tmp.path(),
            Duration::from_secs(1),
            None,
        );
        assert!(result.is_err(), "diarize without HF token should fail");
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
    fn native_batch_segmentation_is_content_sensitive() {
        let dir = tempdir().expect("tempdir");
        let silence = dir.path().join("silence.wav");
        let mixed = dir.path().join("mixed.wav");
        write_pcm16_mono_wav(&silence, 16_000, &vec![0i16; 16_000]);
        let mut mixed_samples = vec![0i16; 4_000];
        mixed_samples.extend((0..16_000).map(|i| if i % 2 == 0 { 7_500 } else { -7_500 }));
        mixed_samples.extend(vec![0i16; 4_000]);
        write_pcm16_mono_wav(&mixed, 16_000, &mixed_samples);

        let req = request();
        let silence_result =
            run(&req, &silence, dir.path(), Duration::from_secs(1), None).expect("silence run");
        let mixed_result =
            run(&req, &mixed, dir.path(), Duration::from_secs(1), None).expect("mixed run");

        assert!(
            silence_result.segments.len() < mixed_result.segments.len(),
            "mixed signal should yield richer segmentation than silence"
        );
        assert_eq!(
            mixed_result.raw_output["telemetry"]["analysis"]["mode"].as_str(),
            Some("waveform")
        );
    }

    #[test]
    fn native_batch_run_is_deterministic_and_keeps_invariants() {
        let dir = tempdir().expect("tempdir");
        let wav = dir.path().join("deterministic.wav");
        let mut samples = vec![0i16; 4_000];
        samples.extend((0..16_000).map(|i| if i % 2 == 0 { 7_000 } else { -7_000 }));
        samples.extend(vec![0i16; 4_000]);
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let req = request();
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
        assert_eq!(
            first.raw_output["telemetry"]["analysis"],
            second.raw_output["telemetry"]["analysis"]
        );
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
    fn native_run_observes_cancellation_checkpoints() {
        let tmp = tempdir().expect("tempdir");
        let wav = tmp.path().join("tone.wav");
        let mut samples = vec![0i16; 4_000];
        samples.extend((0..16_000).map(|i| if i % 2 == 0 { 7_000 } else { -7_000 }));
        samples.extend(vec![0i16; 4_000]);
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let req = request();
        let pilot = InsanelyFastPilot::new(
            "openai/whisper-large-v3".to_owned(),
            8,
            "cpu".to_owned(),
            "float32".to_owned(),
        );
        let token = CancellationToken::with_deadline_from_now(Duration::from_millis(0));
        let result = build_native_segmentation(&req, &wav, &pilot, Some(&token));
        assert!(
            result.is_err(),
            "expired token should cancel waveform-aware segmentation"
        );
    }
}
