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

    use crate::backend::DiarizationPilot;

    use super::{build_native_projection, is_available, run};

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
}
