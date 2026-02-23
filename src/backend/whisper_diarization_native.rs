//! Native diarization engine pilot (bd-1rj.11).
//!
//! Deterministic in-process diarization pipeline with explicit stage
//! decomposition metadata and stable segment/speaker invariants.

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
const DEFAULT_DURATION_MS: u64 = 18_000;
const MIN_REGION_DURATION_MS: u64 = 250;

#[derive(Debug, Clone)]
struct NativeDiarizationProjection {
    duration_ms: u64,
    diarized: super::DiarizedTranscript,
    analysis_provenance: Value,
}

/// In-process pilot is always available.
pub fn is_available() -> bool {
    true
}

pub fn run(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    work_dir: &Path,
    _timeout: Duration,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<TranscriptionResult> {
    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    let num_speakers = requested_num_speakers(request);
    let pilot = super::DiarizationPilot::new(
        request
            .model
            .clone()
            .or_else(|| {
                request
                    .backend_params
                    .diarization_config
                    .as_ref()
                    .and_then(|cfg| cfg.whisper_model.clone())
            })
            .unwrap_or_else(|| "whisper-large-v3".to_owned()),
        request
            .backend_params
            .alignment
            .as_ref()
            .and_then(|cfg| cfg.alignment_model.clone())
            .unwrap_or_else(|| "WAV2VEC2_ASR_LARGE_LV60K_960H".to_owned()),
        num_speakers,
        request.language.clone().unwrap_or_else(|| "en".to_owned()),
    );

    let projection = build_native_projection(request, normalized_wav, &pilot, token)?;
    let punctuation_enabled = request
        .backend_params
        .punctuation
        .as_ref()
        .is_some_and(|cfg| cfg.enabled);
    let suppress_numerals = request
        .backend_params
        .diarization_config
        .as_ref()
        .is_some_and(|cfg| cfg.suppress_numerals);
    let segments = to_transcription_segments(
        &projection.diarized,
        punctuation_enabled,
        suppress_numerals,
        token,
    )?;
    let transcript = super::transcript_from_segments(&segments);

    let txt_path = work_dir.join("diarization_native_output.txt");
    let srt_path = work_dir.join("diarization_native_output.srt");
    fs::write(&txt_path, format!("{transcript}\n"))?;
    fs::write(&srt_path, render_srt(&segments))?;

    Ok(TranscriptionResult {
        backend: BackendKind::WhisperDiarization,
        transcript,
        language: request.language.clone(),
        segments: segments.clone(),
        acceleration: None,
        raw_output: json!({
            "engine": "whisper-diarization-native",
            "schema_version": "native-pilot-v1",
            "in_process": true,
            "duration_ms": projection.duration_ms,
            "stages": [
                {
                    "name": "asr",
                    "status": "ok",
                    "backend": request.model.clone().unwrap_or_else(|| "whisper-large-v3".to_owned()),
                    "segment_count": segments.len(),
                },
                {
                    "name": "alignment",
                    "status": "ok",
                    "model": request.backend_params.alignment.as_ref().and_then(|cfg| cfg.alignment_model.clone()),
                },
                {
                    "name": "punctuation",
                    "status": "ok",
                    "enabled": punctuation_enabled,
                },
                {
                    "name": "speaker_assignment",
                    "status": "ok",
                    "speaker_count": projection.diarized.speakers.len(),
                }
            ],
            "speakers": projection.diarized.speakers,
            "analysis": projection.analysis_provenance,
            "segments": segments,
            "fallback": {
                "deterministic_bridge_fallback_supported": true,
                "trigger_contract": "native-failure-or-contract-violation",
            }
        }),
        artifact_paths: vec![
            txt_path.display().to_string(),
            srt_path.display().to_string(),
        ],
    })
}

fn build_native_projection(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    pilot: &super::DiarizationPilot,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<NativeDiarizationProjection> {
    let duration_hint = request.backend_params.duration_ms;
    let analysis = match analyze_wav(normalized_wav, duration_hint) {
        Ok(analysis) => analysis,
        Err(error) => {
            let duration_ms = estimate_duration_ms(request, normalized_wav);
            return Ok(NativeDiarizationProjection {
                duration_ms,
                diarized: pilot.process(duration_ms),
                analysis_provenance: json!({
                    "mode": "duration_fallback",
                    "reason": error,
                    "duration_ms": duration_ms,
                }),
            });
        }
    };

    let lane_count = requested_num_speakers(request)
        .or(pilot.num_speakers)
        .unwrap_or(2)
        .max(1);
    let mut merged_segments = Vec::new();
    for (region_index, region) in analysis.active_regions.iter().enumerate() {
        if let Some(tok) = token {
            tok.checkpoint()?;
        }
        let region_duration_ms = region
            .end_ms
            .saturating_sub(region.start_ms)
            .max(MIN_REGION_DURATION_MS);
        let mut region_diarized = pilot.process(region_duration_ms);
        for (segment_index, mut segment) in region_diarized.segments.drain(..).enumerate() {
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

            let lane = (region_index + segment_index) % lane_count;
            let energy_bonus = (region.avg_rms as f64 * 2.5).clamp(0.0, 0.08);
            let continuity_penalty = ((region_index + segment_index) as f64) * 0.002;
            segment.start_ms = start_ms;
            segment.end_ms = end_ms;
            segment.speaker_id = format!("SPEAKER_{lane:02}");
            segment.confidence =
                (segment.confidence + energy_bonus - continuity_penalty).clamp(0.45, 0.99);
            merged_segments.push(segment);
        }
    }
    merged_segments.sort_by_key(|segment| (segment.start_ms, segment.end_ms));
    let speakers = speaker_inventory(&merged_segments, lane_count);

    Ok(NativeDiarizationProjection {
        duration_ms: analysis.duration_ms,
        diarized: super::DiarizedTranscript {
            segments: merged_segments.clone(),
            speakers,
        },
        analysis_provenance: json!({
            "mode": "waveform",
            "duration_hint_ms": duration_hint,
            "analysis": analysis.as_json(),
            "speaker_lane_count": lane_count,
            "segments_from_active_regions": merged_segments.len(),
        }),
    })
}

fn speaker_inventory(
    segments: &[super::DiarizedSegment],
    lane_count: usize,
) -> Vec<super::SpeakerInfo> {
    let mut durations = std::collections::HashMap::new();
    for segment in segments {
        let duration = segment.end_ms.saturating_sub(segment.start_ms);
        *durations.entry(segment.speaker_id.clone()).or_insert(0u64) += duration;
    }

    let mut speakers = (0..lane_count)
        .map(|lane| {
            let id = format!("SPEAKER_{lane:02}");
            super::SpeakerInfo {
                id: id.clone(),
                label: format!("Speaker {}", (b'A' + (lane as u8)) as char),
                total_duration_ms: durations.get(&id).copied().unwrap_or(0),
            }
        })
        .collect::<Vec<_>>();
    speakers.sort_by(|left, right| left.id.cmp(&right.id));
    speakers
}

fn requested_num_speakers(request: &TranscribeRequest) -> Option<usize> {
    let sc = request.backend_params.speaker_constraints.as_ref()?;
    if let Some(n) = sc.num_speakers {
        return usize::try_from(n).ok().filter(|n| *n > 0);
    }
    match (sc.min_speakers, sc.max_speakers) {
        (Some(min), Some(max)) if min > 0 && max >= min => {
            usize::try_from(((min + max) / 2).max(1)).ok()
        }
        (Some(min), None) if min > 0 => usize::try_from(min).ok(),
        (None, Some(max)) if max > 0 => usize::try_from(max).ok(),
        _ => None,
    }
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
    diarized: &super::DiarizedTranscript,
    punctuation_enabled: bool,
    suppress_numerals: bool,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<Vec<TranscriptionSegment>> {
    let mut segments = Vec::with_capacity(diarized.segments.len());
    for seg in &diarized.segments {
        if let Some(tok) = token {
            tok.checkpoint()?;
        }

        let mut text = seg.text.trim().to_owned();
        if suppress_numerals {
            text = text
                .chars()
                .filter(|ch| !ch.is_ascii_digit())
                .collect::<String>()
                .trim()
                .to_owned();
        }
        if punctuation_enabled
            && !text.is_empty()
            && !text.ends_with('.')
            && !text.ends_with('!')
            && !text.ends_with('?')
        {
            text.push('.');
        }

        segments.push(TranscriptionSegment {
            start_sec: Some(seg.start_ms as f64 / 1_000.0),
            end_sec: Some(seg.end_ms as f64 / 1_000.0),
            text,
            speaker: Some(seg.speaker_id.clone()),
            confidence: Some(seg.confidence.clamp(0.0, 1.0)),
        });
    }
    Ok(segments)
}

fn render_srt(segments: &[TranscriptionSegment]) -> String {
    let mut out = String::new();
    for (index, seg) in segments.iter().enumerate() {
        let start = format_srt_time(seg.start_sec.unwrap_or(0.0));
        let end = format_srt_time(seg.end_sec.unwrap_or(seg.start_sec.unwrap_or(0.0)));
        let speaker = seg
            .speaker
            .as_deref()
            .map_or(String::new(), |s| format!("[{s}] "));
        let line = format!(
            "{}\n{} --> {}\n{}{}\n\n",
            index + 1,
            start,
            end,
            speaker,
            seg.text.trim()
        );
        out.push_str(&line);
    }
    out
}

fn format_srt_time(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1_000.0).round() as u64;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60_000;
    let secs = (total_ms % 60_000) / 1_000;
    let millis = total_ms % 1_000;
    format!("{hours:02}:{minutes:02}:{secs:02},{millis:03}")
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use tempfile::tempdir;

    use crate::model::{
        BackendKind, BackendParams, InputSource, SpeakerConstraints, TranscribeRequest,
    };

    use crate::backend::{DiarizationPilot, DiarizedSegment, DiarizedTranscript};

    use super::{
        build_native_projection, estimate_duration_ms, format_srt_time, is_available, render_srt,
        requested_num_speakers, run, speaker_inventory, to_transcription_segments,
    };
    use crate::model::TranscriptionSegment;

    fn request() -> TranscribeRequest {
        TranscribeRequest {
            input: InputSource::File {
                path: PathBuf::from("input.wav"),
            },
            backend: BackendKind::WhisperDiarization,
            model: Some("whisper-large-v3".to_owned()),
            language: Some("en".to_owned()),
            translate: false,
            diarize: true,
            persist: false,
            db_path: PathBuf::from("state.sqlite3"),
            timeout_ms: None,
            backend_params: BackendParams::default(),
        }
    }

    #[test]
    fn native_diarization_is_available_in_process() {
        assert!(is_available());
    }

    #[test]
    fn run_writes_txt_and_srt_artifacts() {
        let tmp = tempdir().expect("tempdir");
        let mut request = request();
        request.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: Some(2),
            min_speakers: None,
            max_speakers: None,
        });

        let result = run(
            &request,
            Path::new("missing.wav"),
            tmp.path(),
            Duration::from_secs(1),
            None,
        )
        .expect("native diarization should succeed");

        assert_eq!(result.backend, BackendKind::WhisperDiarization);
        assert_eq!(result.artifact_paths.len(), 2);
        for path in &result.artifact_paths {
            assert!(Path::new(path).exists(), "artifact should exist: {path}");
        }
        assert!(
            result
                .segments
                .windows(2)
                .all(|w| w[0].end_sec <= w[1].start_sec),
            "native diarization should preserve monotonic timestamp ordering"
        );
        assert_eq!(
            result.raw_output["analysis"]["mode"].as_str(),
            Some("duration_fallback")
        );
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
    fn native_diarization_is_content_sensitive_for_active_regions() {
        let tmp = tempdir().expect("tempdir");
        let silence_path = tmp.path().join("silence.wav");
        let speech_path = tmp.path().join("speech.wav");
        write_pcm16_mono_wav(&silence_path, 16_000, &vec![0i16; 16_000]);
        let mut speech = vec![0i16; 4_000];
        speech.extend((0..16_000).map(|i| if i % 2 == 0 { 8_000 } else { -8_000 }));
        speech.extend(vec![0i16; 4_000]);
        write_pcm16_mono_wav(&speech_path, 16_000, &speech);

        let mut req = request();
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: Some(2),
            min_speakers: None,
            max_speakers: None,
        });

        let silence = run(
            &req,
            &silence_path,
            tmp.path(),
            Duration::from_secs(1),
            None,
        )
        .expect("silence run");
        let speech =
            run(&req, &speech_path, tmp.path(), Duration::from_secs(1), None).expect("speech run");

        assert!(
            silence.segments.len() < speech.segments.len(),
            "speech audio should produce richer diarization segmentation than silence"
        );
        assert_eq!(
            speech.raw_output["analysis"]["mode"].as_str(),
            Some("waveform")
        );
    }

    #[test]
    fn native_diarization_is_deterministic_and_speaker_labels_are_valid() {
        let tmp = tempdir().expect("tempdir");
        let wav = tmp.path().join("deterministic.wav");
        let mut speech = vec![0i16; 4_000];
        speech.extend((0..16_000).map(|i| if i % 2 == 0 { 8_000 } else { -8_000 }));
        speech.extend(vec![0i16; 4_000]);
        write_pcm16_mono_wav(&wav, 16_000, &speech);

        let mut req = request();
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: Some(2),
            min_speakers: None,
            max_speakers: None,
        });

        let first = run(&req, &wav, tmp.path(), Duration::from_secs(1), None)
            .expect("first deterministic run");
        let second = run(&req, &wav, tmp.path(), Duration::from_secs(1), None)
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
        assert!(first.segments.iter().all(|segment| {
            segment
                .speaker
                .as_deref()
                .is_none_or(|speaker| !speaker.trim().is_empty())
        }));
    }

    #[test]
    fn native_diarization_observes_cancellation_token() {
        let tmp = tempdir().expect("tempdir");
        let wav = tmp.path().join("speech.wav");
        let mut samples = vec![0i16; 4_000];
        samples.extend((0..16_000).map(|i| if i % 2 == 0 { 8_000 } else { -8_000 }));
        samples.extend(vec![0i16; 4_000]);
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let req = request();
        let pilot = DiarizationPilot::new(
            "whisper-large-v3".to_owned(),
            "WAV2VEC2_ASR_LARGE_LV60K_960H".to_owned(),
            Some(2),
            "en".to_owned(),
        );
        let token = crate::orchestrator::CancellationToken::with_deadline_from_now(
            Duration::from_millis(0),
        );
        let result = build_native_projection(&req, &wav, &pilot, Some(&token));
        assert!(
            result.is_err(),
            "expired token should cancel waveform-aware diarization projection"
        );
    }

    #[test]
    fn requested_num_speakers_all_branches() {
        // num_speakers = Some(0) should be filtered out
        let mut req = request();
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: Some(0),
            min_speakers: None,
            max_speakers: None,
        });
        assert_eq!(requested_num_speakers(&req), None);

        // num_speakers = Some(3)
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: Some(3),
            min_speakers: None,
            max_speakers: None,
        });
        assert_eq!(requested_num_speakers(&req), Some(3));

        // (min, max) average
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: None,
            min_speakers: Some(2),
            max_speakers: Some(6),
        });
        assert_eq!(requested_num_speakers(&req), Some(4));

        // min only
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: None,
            min_speakers: Some(5),
            max_speakers: None,
        });
        assert_eq!(requested_num_speakers(&req), Some(5));

        // max only
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: None,
            min_speakers: None,
            max_speakers: Some(4),
        });
        assert_eq!(requested_num_speakers(&req), Some(4));

        // no constraints
        req.backend_params.speaker_constraints = None;
        assert_eq!(requested_num_speakers(&req), None);
    }

    #[test]
    fn estimate_duration_ms_clamps_hint_and_falls_back() {
        let mut req = request();

        // When duration_ms hint is provided, clamp to [1_000, 1_800_000]
        req.backend_params.duration_ms = Some(500);
        assert_eq!(estimate_duration_ms(&req, Path::new("no_such.wav")), 1_000);

        req.backend_params.duration_ms = Some(5_000_000);
        assert_eq!(
            estimate_duration_ms(&req, Path::new("no_such.wav")),
            1_800_000
        );

        // When no hint and no file → DEFAULT_DURATION_MS (18_000)
        req.backend_params.duration_ms = None;
        assert_eq!(estimate_duration_ms(&req, Path::new("no_such.wav")), 18_000);
    }

    #[test]
    fn format_srt_time_edge_cases() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(1.5), "00:00:01,500");
        assert_eq!(format_srt_time(3661.123), "01:01:01,123");
        // Negative clamped to 0
        assert_eq!(format_srt_time(-5.0), "00:00:00,000");
    }

    #[test]
    fn to_transcription_segments_suppress_numerals_and_punctuation() {
        let diarized = DiarizedTranscript {
            segments: vec![
                DiarizedSegment {
                    text: "Hello 123 world".to_owned(),
                    start_ms: 0,
                    end_ms: 1000,
                    speaker_id: "SPEAKER_00".to_owned(),
                    confidence: 0.8,
                },
                DiarizedSegment {
                    text: "Already ends!".to_owned(),
                    start_ms: 1000,
                    end_ms: 2000,
                    speaker_id: "SPEAKER_01".to_owned(),
                    confidence: 0.9,
                },
            ],
            speakers: vec![],
        };

        // suppress_numerals=true, punctuation=true
        let result = to_transcription_segments(&diarized, true, true, None).expect("segments");
        assert_eq!(result[0].text, "Hello  world.");
        // Already ends with '!' — no extra period
        assert_eq!(result[1].text, "Already ends!");
        assert_eq!(result[0].speaker, Some("SPEAKER_00".to_owned()));
        assert!((result[0].start_sec.unwrap() - 0.0).abs() < 0.001);
        assert!((result[0].end_sec.unwrap() - 1.0).abs() < 0.001);
    }

    #[test]
    fn speaker_inventory_with_empty_segments() {
        let segments: Vec<DiarizedSegment> = vec![];
        let speakers = speaker_inventory(&segments, 3);
        assert_eq!(speakers.len(), 3);
        assert_eq!(speakers[0].id, "SPEAKER_00");
        assert_eq!(speakers[0].label, "Speaker A");
        assert_eq!(speakers[0].total_duration_ms, 0);
        assert_eq!(speakers[2].id, "SPEAKER_02");
        assert_eq!(speakers[2].label, "Speaker C");
    }

    // ── Task #208 — whisper_diarization_native edge-case tests ──────

    #[test]
    fn requested_num_speakers_inverted_range_returns_none() {
        let mut req = request();
        // min > max → falls through to wildcard → None.
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: None,
            min_speakers: Some(5),
            max_speakers: Some(2),
        });
        assert_eq!(
            requested_num_speakers(&req),
            None,
            "inverted min/max range should return None"
        );

        // min == max → valid (both guards satisfied).
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: None,
            min_speakers: Some(3),
            max_speakers: Some(3),
        });
        assert_eq!(
            requested_num_speakers(&req),
            Some(3),
            "min == max should produce that value"
        );
    }

    #[test]
    fn requested_num_speakers_zero_min_and_zero_max_return_none() {
        let mut req = request();
        // min = 0 only → fails the `min > 0` guard → None.
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: None,
            min_speakers: Some(0),
            max_speakers: None,
        });
        assert_eq!(requested_num_speakers(&req), None);

        // max = 0 only → fails the `max > 0` guard → None.
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: None,
            min_speakers: None,
            max_speakers: Some(0),
        });
        assert_eq!(requested_num_speakers(&req), None);
    }

    #[test]
    fn estimate_duration_ms_tiny_file_clamps_to_min() {
        let tmp = tempdir().expect("tempdir");
        let tiny = tmp.path().join("tiny.raw");
        // 4 bytes < WAV_HEADER_BYTES (44) → saturating_sub → 0 audio bytes
        // → estimated 0 → clamped to MIN_DURATION_MS (1_000).
        std::fs::write(&tiny, b"RIFF").unwrap();

        let mut req = request();
        req.backend_params.duration_ms = None;
        assert_eq!(
            estimate_duration_ms(&req, &tiny),
            1_000,
            "sub-header file should clamp to MIN_DURATION_MS"
        );
    }

    #[test]
    fn to_transcription_segments_question_mark_not_double_punctuated() {
        let diarized = DiarizedTranscript {
            segments: vec![DiarizedSegment {
                text: "Is this correct?".to_owned(),
                start_ms: 0,
                end_ms: 1000,
                speaker_id: "SPEAKER_00".to_owned(),
                confidence: 0.8,
            }],
            speakers: vec![],
        };
        // punctuation enabled, but text already ends with '?'.
        let result = to_transcription_segments(&diarized, true, false, None).unwrap();
        assert_eq!(
            result[0].text, "Is this correct?",
            "trailing '?' should not receive extra period"
        );
    }

    #[test]
    fn render_srt_none_timestamps_fall_back() {
        // end_sec = None → falls back to start_sec.
        let segs = vec![
            TranscriptionSegment {
                start_sec: Some(1.0),
                end_sec: None,
                text: "Hello".to_owned(),
                speaker: Some("SPEAKER_00".to_owned()),
                confidence: Some(0.9),
            },
            // Both None → both fall back to 0.0.
            TranscriptionSegment {
                start_sec: None,
                end_sec: None,
                text: "World".to_owned(),
                speaker: None,
                confidence: None,
            },
        ];
        let srt = render_srt(&segs);

        // Entry 1: start=1.0, end falls back to start=1.0.
        assert!(
            srt.contains("00:00:01,000 --> 00:00:01,000"),
            "None end_sec should fall back to start_sec: {srt}"
        );
        assert!(srt.contains("[SPEAKER_00] Hello"));

        // Entry 2: both None → both 0.0.
        assert!(
            srt.contains("00:00:00,000 --> 00:00:00,000"),
            "both None should fall back to 0.0: {srt}"
        );
        // No speaker prefix.
        assert!(srt.contains("World"));
        assert!(!srt.contains("[SPEAKER_") || srt.matches("[SPEAKER_").count() == 1);
    }

    // ── Task #216 — whisper_diarization_native pass 2 edge-case tests ──

    #[test]
    fn to_transcription_segments_empty_text_after_numeral_suppression_no_period() {
        let dt = DiarizedTranscript {
            segments: vec![DiarizedSegment {
                text: "42".to_owned(),
                start_ms: 0,
                end_ms: 1000,
                speaker_id: "SPEAKER_00".to_owned(),
                confidence: 0.9,
            }],
            speakers: vec![],
        };
        let result = to_transcription_segments(&dt, true, true, None).unwrap();
        // After numeral suppression "42" → "", punctuation should NOT add ".".
        assert_eq!(
            result[0].text, "",
            "empty after suppression should stay empty, not become '.'"
        );
    }

    #[test]
    fn speaker_inventory_zero_lanes_returns_empty() {
        let inventory = speaker_inventory(&[], 0);
        assert!(
            inventory.is_empty(),
            "zero lanes should produce empty inventory"
        );
    }

    #[test]
    fn speaker_inventory_one_lane_accumulates_duration() {
        let segments = vec![
            DiarizedSegment {
                text: "hello".to_owned(),
                start_ms: 0,
                end_ms: 1000,
                speaker_id: "SPEAKER_00".to_owned(),
                confidence: 0.9,
            },
            DiarizedSegment {
                text: "world".to_owned(),
                start_ms: 1000,
                end_ms: 3000,
                speaker_id: "SPEAKER_00".to_owned(),
                confidence: 0.8,
            },
        ];
        let inventory = speaker_inventory(&segments, 1);
        assert_eq!(inventory.len(), 1);
        assert_eq!(inventory[0].id, "SPEAKER_00");
        assert_eq!(inventory[0].total_duration_ms, 3000); // 1000 + 2000
    }

    #[test]
    fn estimate_duration_ms_real_file_computes_from_pcm_size() {
        let dir = tempdir().unwrap();
        let wav = dir.path().join("test.wav");
        // Write exactly WAV_HEADER_BYTES + 2 seconds of PCM16 mono 16kHz data.
        let file_size = 44 + 32_000 * 2; // 64_044 bytes
        std::fs::write(&wav, vec![0u8; file_size as usize]).unwrap();

        let mut req = request();
        req.backend_params.duration_ms = None;
        let dur = estimate_duration_ms(&req, &wav);
        assert_eq!(dur, 2000, "64044 bytes should yield 2000ms");
    }

    #[test]
    fn render_srt_sequence_numbers_and_text_trimming() {
        let segs = vec![
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.0),
                text: "  hello  ".to_owned(),
                speaker: Some("SPEAKER_00".to_owned()),
                confidence: Some(0.9),
            },
            TranscriptionSegment {
                start_sec: Some(1.0),
                end_sec: Some(2.5),
                text: "\tworld\t".to_owned(),
                speaker: None,
                confidence: None,
            },
        ];
        let srt = render_srt(&segs);
        // First block starts with sequence number 1.
        assert!(
            srt.starts_with("1\n"),
            "first entry should start with '1\\n': {srt}"
        );
        // Second block contains sequence number 2.
        assert!(
            srt.contains("\n2\n"),
            "second entry should have sequence number 2: {srt}"
        );
        // Text should be trimmed.
        assert!(srt.contains("hello"), "trimmed 'hello' should appear");
        assert!(
            !srt.contains("  hello"),
            "leading whitespace should be trimmed"
        );
        assert!(srt.contains("world"), "trimmed 'world' should appear");
        assert!(!srt.contains("\tworld"), "tab should be trimmed from text");
        // Blocks separated by double newline.
        assert!(
            srt.contains("\n\n2\n"),
            "blocks should be separated by blank line"
        );
    }

    // ── Task #221 — whisper_diarization_native pass 3 edge-case tests ──

    #[test]
    fn requested_num_speakers_zero_min_with_some_max_returns_none() {
        let mut req = request();
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: None,
            min_speakers: Some(0),
            max_speakers: Some(3),
        });
        assert_eq!(
            requested_num_speakers(&req),
            None,
            "min=0 with Some(max) must return None (guard min>0 fails)"
        );
    }

    #[test]
    fn estimate_duration_ms_hint_at_boundary_values_passes_through() {
        let mut req = request();

        req.backend_params.duration_ms = Some(1_000);
        assert_eq!(
            estimate_duration_ms(&req, Path::new("no.wav")),
            1_000,
            "hint at MIN_DURATION_MS should pass through unchanged"
        );

        req.backend_params.duration_ms = Some(1_800_000);
        assert_eq!(
            estimate_duration_ms(&req, Path::new("no.wav")),
            1_800_000,
            "hint at MAX_DURATION_MS should pass through unchanged"
        );
    }

    #[test]
    fn to_transcription_segments_clamps_out_of_range_confidence() {
        let diarized = DiarizedTranscript {
            segments: vec![
                DiarizedSegment {
                    text: "Too confident".to_owned(),
                    start_ms: 0,
                    end_ms: 1000,
                    speaker_id: "SPEAKER_00".to_owned(),
                    confidence: 1.5,
                },
                DiarizedSegment {
                    text: "Negative".to_owned(),
                    start_ms: 1000,
                    end_ms: 2000,
                    speaker_id: "SPEAKER_01".to_owned(),
                    confidence: -0.3,
                },
            ],
            speakers: vec![],
        };
        let result = to_transcription_segments(&diarized, false, false, None).unwrap();
        assert_eq!(
            result[0].confidence,
            Some(1.0),
            "confidence above 1.0 must be clamped to 1.0"
        );
        assert_eq!(
            result[1].confidence,
            Some(0.0),
            "negative confidence must be clamped to 0.0"
        );
    }

    #[test]
    fn render_srt_empty_segments_returns_empty_string() {
        let srt = render_srt(&[]);
        assert!(
            srt.is_empty(),
            "render_srt with no segments should produce empty string, got: {srt:?}"
        );
    }

    #[test]
    fn speaker_inventory_mismatched_speaker_ids_not_counted() {
        let segments = vec![DiarizedSegment {
            text: "orphan".to_owned(),
            start_ms: 0,
            end_ms: 5000,
            speaker_id: "SPEAKER_99".to_owned(),
            confidence: 0.7,
        }];
        let inventory = speaker_inventory(&segments, 2);
        assert_eq!(inventory.len(), 2);
        assert_eq!(inventory[0].id, "SPEAKER_00");
        assert_eq!(
            inventory[0].total_duration_ms, 0,
            "SPEAKER_00 should have 0 duration"
        );
        assert_eq!(inventory[1].id, "SPEAKER_01");
        assert_eq!(
            inventory[1].total_duration_ms, 0,
            "SPEAKER_01 should have 0 duration"
        );
    }

    // ── Task #226 — whisper_diarization_native pass 4 edge-case tests ──

    #[test]
    fn render_srt_none_start_with_some_end_uses_zero_start() {
        let segs = vec![TranscriptionSegment {
            start_sec: None,
            end_sec: Some(2.5),
            text: "late start".to_owned(),
            speaker: None,
            confidence: None,
        }];
        let srt = render_srt(&segs);
        // start_sec = None → unwrap_or(0.0) → "00:00:00,000"
        assert!(
            srt.contains("00:00:00,000 --> 00:00:02,500"),
            "None start_sec should fall back to 0.0 while end_sec stays 2.5: {srt}"
        );
    }

    #[test]
    fn requested_num_speakers_odd_sum_floors_on_average() {
        let mut req = request();
        // (1 + 2) / 2 = 1 (integer floor division)
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: None,
            min_speakers: Some(1),
            max_speakers: Some(2),
        });
        assert_eq!(
            requested_num_speakers(&req),
            Some(1),
            "(1+2)/2 should floor to 1 via integer division"
        );

        // (3 + 6) / 2 = 4 (also integer floor)
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: None,
            min_speakers: Some(3),
            max_speakers: Some(6),
        });
        assert_eq!(
            requested_num_speakers(&req),
            Some(4),
            "(3+6)/2 should floor to 4"
        );
    }

    #[test]
    fn to_transcription_segments_no_suppression_no_punctuation_text_passes_through_verbatim() {
        let diarized = DiarizedTranscript {
            segments: vec![
                DiarizedSegment {
                    text: "  Hello 42 world  ".to_owned(),
                    start_ms: 0,
                    end_ms: 1000,
                    speaker_id: "SPEAKER_00".to_owned(),
                    confidence: 0.5,
                },
                DiarizedSegment {
                    text: "No trailing dot".to_owned(),
                    start_ms: 1000,
                    end_ms: 2000,
                    speaker_id: "SPEAKER_01".to_owned(),
                    confidence: 0.6,
                },
            ],
            speakers: vec![],
        };
        // Both suppress_numerals=false and punctuation_enabled=false
        let result = to_transcription_segments(&diarized, false, false, None).unwrap();
        assert_eq!(
            result[0].text, "Hello 42 world",
            "text should be trimmed but numerals kept and no period added"
        );
        assert_eq!(
            result[1].text, "No trailing dot",
            "text without trailing punctuation should stay as-is"
        );
    }

    #[test]
    fn speaker_inventory_labels_for_many_lanes_wraps_past_z() {
        // lane >= 26 → (b'A' + 26) = b'[' which is ASCII '['.
        // This tests the actual behavior: labels use raw arithmetic wrap.
        let inventory = speaker_inventory(&[], 27);
        assert_eq!(inventory.len(), 27);
        assert_eq!(
            inventory[0].label, "Speaker A",
            "lane 0 should be Speaker A"
        );
        assert_eq!(
            inventory[25].label, "Speaker Z",
            "lane 25 should be Speaker Z"
        );
        // lane 26 → b'A' + 26 = 91 = '[' in ASCII
        assert_eq!(
            inventory[26].label, "Speaker [",
            "lane 26 wraps past 'Z' to '['"
        );
        assert_eq!(inventory[26].id, "SPEAKER_26");
    }

    #[test]
    fn build_native_projection_fallback_provenance_includes_reason_and_duration_ms() {
        // Point to a non-existent WAV to trigger the analysis fallback path.
        let mut req = request();
        req.backend_params.duration_ms = Some(5_000);
        let pilot = DiarizationPilot::new(
            "whisper-large-v3".to_owned(),
            "wav2vec2".to_owned(),
            None,
            "en".to_owned(),
        );
        let projection =
            build_native_projection(&req, Path::new("/no/such/file.wav"), &pilot, None).unwrap();

        assert_eq!(
            projection.duration_ms, 5_000,
            "duration_ms should come from the hint"
        );
        // Provenance must contain mode, reason, and duration_ms keys.
        assert_eq!(
            projection.analysis_provenance["mode"], "duration_fallback",
            "provenance mode should be duration_fallback"
        );
        assert!(
            projection.analysis_provenance.get("reason").is_some(),
            "provenance should include a reason for the fallback"
        );
        assert_eq!(
            projection.analysis_provenance["duration_ms"], 5_000,
            "provenance should echo the resolved duration_ms"
        );
    }

    // ── Task #231 — whisper_diarization_native pass 5 edge-case tests ──

    #[test]
    fn estimate_duration_ms_file_exactly_header_size_clamps_to_min() {
        let dir = tempdir().unwrap();
        let wav = dir.path().join("exact_header.wav");
        // Exactly 44 bytes (WAV_HEADER_BYTES) → 0 audio bytes → 0 ms → clamped to MIN.
        std::fs::write(&wav, vec![0u8; 44]).unwrap();
        let mut req = request();
        req.backend_params.duration_ms = None;
        assert_eq!(
            estimate_duration_ms(&req, &wav),
            1_000,
            "file of exactly WAV_HEADER_BYTES should produce 0 audio bytes, clamped to MIN_DURATION_MS"
        );
    }

    #[test]
    fn requested_num_speakers_one_speaker_accepted_as_minimum_valid() {
        let mut req = request();
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: Some(1),
            min_speakers: None,
            max_speakers: None,
        });
        assert_eq!(
            requested_num_speakers(&req),
            Some(1),
            "num_speakers=1 should be accepted (> 0 guard passes)"
        );
    }

    #[test]
    fn render_srt_both_timestamps_none_with_speaker_emits_prefix() {
        let segs = vec![TranscriptionSegment {
            start_sec: None,
            end_sec: None,
            text: "hello".to_owned(),
            speaker: Some("SPEAKER_02".to_owned()),
            confidence: Some(0.7),
        }];
        let srt = render_srt(&segs);
        // Both timestamps None → both 0.0.
        assert!(
            srt.contains("00:00:00,000 --> 00:00:00,000"),
            "both None timestamps should produce 00:00:00,000: {srt}"
        );
        // Speaker prefix should still appear.
        assert!(
            srt.contains("[SPEAKER_02]"),
            "speaker prefix must appear even when timestamps are None: {srt}"
        );
        assert!(srt.contains("hello"), "text should appear");
    }

    #[test]
    fn to_transcription_segments_whitespace_only_text_stays_empty_with_punctuation() {
        let diarized = DiarizedTranscript {
            segments: vec![DiarizedSegment {
                text: "   \t  ".to_owned(),
                start_ms: 0,
                end_ms: 1000,
                speaker_id: "SPEAKER_00".to_owned(),
                confidence: 0.8,
            }],
            speakers: vec![],
        };
        // punctuation_enabled=true, suppress_numerals=false
        let result = to_transcription_segments(&diarized, true, false, None).unwrap();
        assert_eq!(
            result[0].text, "",
            "whitespace-only text after trim should be empty, not '.'"
        );
    }

    #[test]
    fn to_transcription_segments_suppress_numerals_on_punctuation_off() {
        let diarized = DiarizedTranscript {
            segments: vec![
                DiarizedSegment {
                    text: "Room 42 was empty".to_owned(),
                    start_ms: 0,
                    end_ms: 1000,
                    speaker_id: "SPEAKER_00".to_owned(),
                    confidence: 0.9,
                },
                DiarizedSegment {
                    text: "Check items 1, 2, 3".to_owned(),
                    start_ms: 1000,
                    end_ms: 2000,
                    speaker_id: "SPEAKER_01".to_owned(),
                    confidence: 0.8,
                },
            ],
            speakers: vec![],
        };
        // suppress_numerals=true, punctuation_enabled=false
        let result = to_transcription_segments(&diarized, false, true, None).unwrap();
        // "Room 42 was empty" → strip digits → "Room  was empty" → trim → "Room  was empty"
        assert!(
            !result[0].text.contains('4'),
            "numerals should be stripped: got {:?}",
            result[0].text
        );
        assert!(
            !result[0].text.ends_with('.'),
            "no period should be added when punctuation is off"
        );
        // "Check items 1, 2, 3" → "Check items , , " → trim → "Check items , ,"
        assert!(
            !result[1].text.contains('1'),
            "numerals should be stripped from second segment"
        );
    }

    // -- bd-241: whisper_diarization_native.rs edge-case tests pass 6 --

    #[test]
    fn format_srt_time_exactly_one_hour_boundary() {
        assert_eq!(format_srt_time(3600.0), "01:00:00,000");
        assert_eq!(format_srt_time(3661.5), "01:01:01,500");
        assert_eq!(format_srt_time(7200.0), "02:00:00,000");
    }

    #[test]
    fn format_srt_time_negative_input_saturates_to_zero() {
        assert_eq!(format_srt_time(-5.0), "00:00:00,000");
        assert_eq!(format_srt_time(-0.001), "00:00:00,000");
        assert_eq!(format_srt_time(f64::NEG_INFINITY), "00:00:00,000");
    }

    #[test]
    fn requested_num_speakers_min_max_averaging() {
        let mut req = request();
        // min=2, max=6 → average (2+6)/2 = 4
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: None,
            min_speakers: Some(2),
            max_speakers: Some(6),
        });
        assert_eq!(requested_num_speakers(&req), Some(4));

        // min only → returns min
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: None,
            min_speakers: Some(3),
            max_speakers: None,
        });
        assert_eq!(requested_num_speakers(&req), Some(3));

        // max only → returns max
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: None,
            min_speakers: None,
            max_speakers: Some(5),
        });
        assert_eq!(requested_num_speakers(&req), Some(5));

        // no constraints at all → None
        req.backend_params.speaker_constraints = None;
        assert_eq!(requested_num_speakers(&req), None);
    }

    #[test]
    fn requested_num_speakers_zero_num_speakers_returns_none() {
        let mut req = request();
        // num_speakers=0 should be filtered out (filter(|n| *n > 0))
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: Some(0),
            min_speakers: None,
            max_speakers: None,
        });
        assert_eq!(requested_num_speakers(&req), None);

        // min=0, max=0 should also return None (guard: min > 0 && max >= min)
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: None,
            min_speakers: Some(0),
            max_speakers: Some(0),
        });
        assert_eq!(requested_num_speakers(&req), None);
    }

    #[test]
    fn to_transcription_segments_cancellation_token_fires() {
        let diarized = DiarizedTranscript {
            segments: vec![
                DiarizedSegment {
                    text: "First segment".to_owned(),
                    start_ms: 0,
                    end_ms: 1000,
                    speaker_id: "SPEAKER_00".to_owned(),
                    confidence: 0.9,
                },
                DiarizedSegment {
                    text: "Second segment".to_owned(),
                    start_ms: 1000,
                    end_ms: 2000,
                    speaker_id: "SPEAKER_01".to_owned(),
                    confidence: 0.8,
                },
            ],
            speakers: vec![],
        };
        let token = crate::orchestrator::CancellationToken::with_deadline_from_now(
            Duration::from_millis(0),
        );
        std::thread::sleep(Duration::from_millis(5));
        let result = to_transcription_segments(&diarized, false, false, Some(&token));
        assert!(result.is_err(), "should fail with expired token");
        let err = result.unwrap_err();
        assert!(
            matches!(err, crate::error::FwError::Cancelled(_)),
            "expected Cancelled error, got: {err:?}"
        );
    }

    // ── Task #259 — whisper_diarization_native pass 7 edge-case tests ──

    #[test]
    fn run_with_custom_alignment_model_appears_in_raw_output() {
        // When backend_params.alignment is set with a custom alignment_model,
        // it should appear in raw_output.stages[1].model (line 115).
        // No existing test ever sets alignment config.
        let tmp = tempdir().expect("tempdir");
        let mut req = request();
        req.backend_params.alignment = Some(crate::model::AlignmentConfig {
            alignment_model: Some("WAV2VEC2_CUSTOM_MODEL".to_owned()),
            interpolate_method: None,
            return_char_alignments: false,
        });
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: Some(2),
            min_speakers: None,
            max_speakers: None,
        });

        let result = run(&req, Path::new("missing.wav"), tmp.path(), Duration::from_secs(1), None)
            .expect("should succeed with custom alignment model");
        assert_eq!(
            result.raw_output["stages"][1]["model"].as_str(),
            Some("WAV2VEC2_CUSTOM_MODEL"),
            "alignment model should appear in raw_output stages"
        );
    }

    #[test]
    fn run_model_none_alignment_none_uses_defaults_in_raw_output() {
        // When request.model is None, raw_output.stages[0].backend falls back
        // to "whisper-large-v3" (line 109). When alignment is also None,
        // stages[1].model is null (line 115).
        let tmp = tempdir().expect("tempdir");
        let mut req = request();
        req.model = None;
        req.backend_params.alignment = None;
        req.backend_params.diarization_config = None;
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: Some(2),
            min_speakers: None,
            max_speakers: None,
        });

        let result = run(&req, Path::new("missing.wav"), tmp.path(), Duration::from_secs(1), None)
            .expect("should succeed with all defaults");
        assert_eq!(
            result.raw_output["stages"][0]["backend"].as_str(),
            Some("whisper-large-v3"),
            "model should fall back to 'whisper-large-v3' when request.model is None"
        );
        assert!(
            result.raw_output["stages"][1]["model"].is_null(),
            "alignment model should be null when alignment config is None"
        );
    }

    #[test]
    fn build_native_projection_waveform_uses_pilot_num_speakers_for_lane_count() {
        // When requested_num_speakers returns None (no speaker_constraints)
        // but pilot.num_speakers is Some(3), lane_count should be 3 via .or()
        // at line 167. The provenance should reflect speaker_lane_count = 3.
        let tmp = tempdir().expect("tempdir");
        let wav = tmp.path().join("speech.wav");
        let mut speech = vec![0i16; 4_000];
        speech.extend((0..16_000).map(|i| if i % 2 == 0 { 8_000 } else { -8_000 }));
        speech.extend(vec![0i16; 4_000]);
        write_pcm16_mono_wav(&wav, 16_000, &speech);

        let mut req = request();
        req.backend_params.speaker_constraints = None; // → requested_num_speakers returns None

        let pilot = DiarizationPilot::new(
            "whisper-large-v3".to_owned(),
            "WAV2VEC2_ASR_LARGE_LV60K_960H".to_owned(),
            Some(3), // pilot provides 3 speakers via .or()
            "en".to_owned(),
        );
        let projection = build_native_projection(&req, &wav, &pilot, None)
            .expect("waveform projection should succeed");
        assert_eq!(
            projection.analysis_provenance["mode"].as_str(),
            Some("waveform"),
            "should use waveform mode with real WAV"
        );
        assert_eq!(
            projection.analysis_provenance["speaker_lane_count"].as_u64(),
            Some(3),
            "lane_count should come from pilot.num_speakers via .or()"
        );
    }

    #[test]
    fn speaker_inventory_two_lanes_accumulates_durations_per_speaker() {
        // Verify that with 2 lanes and segments from both speakers,
        // durations are accumulated correctly per speaker_id.
        let segments = vec![
            DiarizedSegment {
                text: "hello".to_owned(),
                start_ms: 0,
                end_ms: 2000,
                speaker_id: "SPEAKER_00".to_owned(),
                confidence: 0.8,
            },
            DiarizedSegment {
                text: "world".to_owned(),
                start_ms: 2000,
                end_ms: 5000,
                speaker_id: "SPEAKER_01".to_owned(),
                confidence: 0.9,
            },
            DiarizedSegment {
                text: "again".to_owned(),
                start_ms: 5000,
                end_ms: 7000,
                speaker_id: "SPEAKER_00".to_owned(),
                confidence: 0.7,
            },
        ];
        let inventory = speaker_inventory(&segments, 2);
        assert_eq!(inventory.len(), 2);
        // SPEAKER_00: 2000 + 2000 = 4000 ms
        assert_eq!(inventory[0].id, "SPEAKER_00");
        assert_eq!(
            inventory[0].total_duration_ms, 4000,
            "SPEAKER_00 should have 2000 + 2000 = 4000 ms"
        );
        // SPEAKER_01: 3000 ms
        assert_eq!(inventory[1].id, "SPEAKER_01");
        assert_eq!(
            inventory[1].total_duration_ms, 3000,
            "SPEAKER_01 should have 3000 ms"
        );
        // Labels: A and B
        assert_eq!(inventory[0].label, "Speaker A");
        assert_eq!(inventory[1].label, "Speaker B");
    }

    #[test]
    fn estimate_duration_ms_large_file_clamps_to_max() {
        // File large enough that computed duration exceeds MAX_DURATION_MS (30 min = 1_800_000 ms).
        // 30 min at 32_000 bytes/sec = 57_600_000 audio bytes + 44 header = 57_600_044 bytes.
        // Use set_len to create a sparse file to avoid allocating that memory.
        let dir = tempdir().unwrap();
        let wav = dir.path().join("huge.wav");
        let huge_size: u64 = 57_600_044 + 32_000; // ~30 min + 1 sec → should clamp to max
        let file = std::fs::File::create(&wav).unwrap();
        file.set_len(huge_size).unwrap();
        drop(file);

        let mut req = request();
        req.backend_params.duration_ms = None;
        let dur = estimate_duration_ms(&req, &wav);
        assert_eq!(
            dur, 1_800_000,
            "file exceeding 30 min should clamp to MAX_DURATION_MS (1_800_000)"
        );
    }

    // ── Task #268 — whisper_diarization_native pass 8 edge-case tests ──

    #[test]
    fn estimate_duration_ms_nonexistent_file_no_hint_returns_default() {
        let mut req = request();
        req.backend_params.duration_ms = None;
        let dur = estimate_duration_ms(&req, Path::new("/no/such/file.wav"));
        assert_eq!(
            dur, 18_000,
            "nonexistent file with no hint should return DEFAULT_DURATION_MS (18_000)"
        );
    }

    #[test]
    fn run_raw_output_fallback_block_contains_contract_fields() {
        let tmp = tempdir().expect("tempdir");
        let mut req = request();
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: Some(2),
            min_speakers: None,
            max_speakers: None,
        });

        let result = run(&req, Path::new("missing.wav"), tmp.path(), Duration::from_secs(1), None)
            .expect("native diarization should succeed");

        let fallback = &result.raw_output["fallback"];
        assert_eq!(
            fallback["deterministic_bridge_fallback_supported"].as_bool(),
            Some(true),
            "fallback block must have deterministic_bridge_fallback_supported: true"
        );
        assert_eq!(
            fallback["trigger_contract"].as_str(),
            Some("native-failure-or-contract-violation"),
            "fallback block must have trigger_contract field"
        );
    }

    #[test]
    fn to_transcription_segments_exclamation_mark_not_double_punctuated() {
        let diarized = DiarizedTranscript {
            segments: vec![DiarizedSegment {
                text: "Stop right there!".to_owned(),
                start_ms: 0,
                end_ms: 1000,
                speaker_id: "SPEAKER_00".to_owned(),
                confidence: 0.8,
            }],
            speakers: vec![],
        };
        let result = to_transcription_segments(&diarized, true, false, None).unwrap();
        assert_eq!(
            result[0].text, "Stop right there!",
            "trailing '!' should not receive extra period"
        );
    }

    #[test]
    fn build_native_projection_defaults_lane_count_to_two_when_no_speakers_specified() {
        let tmp = tempdir().expect("tempdir");
        let wav = tmp.path().join("speech.wav");
        let mut speech = vec![0i16; 4_000];
        speech.extend((0..16_000).map(|i| if i % 2 == 0 { 8_000 } else { -8_000 }));
        speech.extend(vec![0i16; 4_000]);
        write_pcm16_mono_wav(&wav, 16_000, &speech);

        let mut req = request();
        req.backend_params.speaker_constraints = None; // → requested_num_speakers returns None

        let pilot = DiarizationPilot::new(
            "whisper-large-v3".to_owned(),
            "WAV2VEC2_ASR_LARGE_LV60K_960H".to_owned(),
            None, // pilot also has no num_speakers → .or() returns None → unwrap_or(2)
            "en".to_owned(),
        );
        let projection = build_native_projection(&req, &wav, &pilot, None)
            .expect("waveform projection should succeed");
        assert_eq!(
            projection.analysis_provenance["speaker_lane_count"].as_u64(),
            Some(2),
            "lane_count should default to 2 when both request and pilot have no speaker count"
        );
    }

    #[test]
    fn run_language_none_produces_none_language_in_result() {
        let tmp = tempdir().expect("tempdir");
        let mut req = request();
        req.language = None;
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: Some(2),
            min_speakers: None,
            max_speakers: None,
        });

        let result = run(&req, Path::new("missing.wav"), tmp.path(), Duration::from_secs(1), None)
            .expect("should succeed with language=None");
        assert_eq!(
            result.language, None,
            "result.language should be None when request.language is None"
        );
    }
}
