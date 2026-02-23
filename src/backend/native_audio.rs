//! Shared waveform-aware audio analysis for native backend adapters.
//!
//! This module provides deterministic, allocation-safe WAV parsing and
//! energy-based speech-region extraction used by native backend runtimes.

use std::fs;
use std::path::Path;

use serde_json::{Value, json};

const FRAME_MS: u32 = 20;
const MIN_THRESHOLD: f32 = 0.003;
const GAP_BRIDGE_MAX_FRAMES: usize = 2;
const MIN_REGION_FRAMES: usize = 2;

#[derive(Debug, Clone)]
pub(crate) struct AudioRegion {
    pub start_ms: u64,
    pub end_ms: u64,
    pub avg_rms: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct NativeAudioAnalysis {
    pub duration_ms: u64,
    pub sample_rate_hz: u32,
    pub frame_ms: u32,
    pub frame_count: usize,
    pub avg_rms: f32,
    pub max_rms: f32,
    pub activity_threshold: f32,
    pub active_regions: Vec<AudioRegion>,
}

impl NativeAudioAnalysis {
    pub(crate) fn as_json(&self) -> Value {
        json!({
            "sample_rate_hz": self.sample_rate_hz,
            "duration_ms": self.duration_ms,
            "frame_ms": self.frame_ms,
            "frame_count": self.frame_count,
            "avg_rms": self.avg_rms,
            "max_rms": self.max_rms,
            "activity_threshold": self.activity_threshold,
            "active_region_count": self.active_regions.len(),
            "active_regions": self.active_regions.iter().map(|r| {
                json!({
                    "start_ms": r.start_ms,
                    "end_ms": r.end_ms,
                    "avg_rms": r.avg_rms,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

#[derive(Debug, Clone)]
struct WavPcm16Mono {
    sample_rate_hz: u32,
    samples: Vec<i16>,
}

pub(crate) fn analyze_wav(
    path: &Path,
    duration_override_ms: Option<u64>,
) -> Result<NativeAudioAnalysis, String> {
    let wav = parse_pcm16_mono_wav(path)?;

    if wav.samples.is_empty() {
        return Ok(NativeAudioAnalysis {
            duration_ms: duration_override_ms.unwrap_or(0),
            sample_rate_hz: wav.sample_rate_hz,
            frame_ms: FRAME_MS,
            frame_count: 0,
            avg_rms: 0.0,
            max_rms: 0.0,
            activity_threshold: MIN_THRESHOLD,
            active_regions: Vec::new(),
        });
    }

    let frame_samples = (((wav.sample_rate_hz as u64) * (FRAME_MS as u64)) / 1_000)
        .max(1)
        .try_into()
        .unwrap_or(1usize);

    let frame_rms = compute_frame_rms(&wav.samples, frame_samples);
    let frame_count = frame_rms.len();

    let avg_rms = if frame_rms.is_empty() {
        0.0
    } else {
        frame_rms.iter().copied().sum::<f32>() / frame_rms.len() as f32
    };
    let max_rms = frame_rms
        .iter()
        .copied()
        .fold(0.0_f32, |acc, v| if v > acc { v } else { acc });

    // Deterministic adaptive threshold: scales with corpus energy while
    // preserving a conservative floor for quiet speech.
    let activity_threshold =
        ((avg_rms * 1.5).max(MIN_THRESHOLD)).min((max_rms * 0.8).max(MIN_THRESHOLD));

    let mut active = frame_rms
        .iter()
        .map(|rms| *rms >= activity_threshold)
        .collect::<Vec<_>>();

    bridge_short_gaps(&mut active, GAP_BRIDGE_MAX_FRAMES);
    let active_regions =
        active_regions_from_frames(&frame_rms, &active, FRAME_MS, MIN_REGION_FRAMES);

    let computed_duration_ms = (((wav.samples.len() as f64) / (wav.sample_rate_hz as f64))
        * 1_000.0)
        .round()
        .max(0.0) as u64;

    Ok(NativeAudioAnalysis {
        duration_ms: duration_override_ms
            .filter(|value| *value > 0)
            .unwrap_or(computed_duration_ms),
        sample_rate_hz: wav.sample_rate_hz,
        frame_ms: FRAME_MS,
        frame_count,
        avg_rms,
        max_rms,
        activity_threshold,
        active_regions,
    })
}

fn parse_pcm16_mono_wav(path: &Path) -> Result<WavPcm16Mono, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("failed to read wav `{}`: {error}", path.display()))?;
    if bytes.len() < 44 {
        return Err(format!("wav too small: {} bytes", bytes.len()));
    }
    if bytes.get(0..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WAVE") {
        return Err("unsupported wav container; expected RIFF/WAVE".to_owned());
    }

    let mut cursor = 12usize;
    let mut sample_rate_hz: Option<u32> = None;
    let mut channels: Option<u16> = None;
    let mut bits_per_sample: Option<u16> = None;
    let mut audio_format: Option<u16> = None;
    let mut data: Option<Vec<u8>> = None;

    while cursor + 8 <= bytes.len() {
        let chunk_id = &bytes[cursor..cursor + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[cursor + 4],
            bytes[cursor + 5],
            bytes[cursor + 6],
            bytes[cursor + 7],
        ]) as usize;
        let data_start = cursor + 8;
        let data_end_unclamped = data_start.saturating_add(chunk_size);
        let data_end = data_end_unclamped.min(bytes.len());

        if chunk_id == b"fmt " {
            if data_end.saturating_sub(data_start) < 16 {
                return Err("invalid wav fmt chunk".to_owned());
            }
            let fmt = &bytes[data_start..data_end];
            audio_format = Some(u16::from_le_bytes([fmt[0], fmt[1]]));
            channels = Some(u16::from_le_bytes([fmt[2], fmt[3]]));
            sample_rate_hz = Some(u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]));
            bits_per_sample = Some(u16::from_le_bytes([fmt[14], fmt[15]]));
        } else if chunk_id == b"data" {
            data = Some(bytes[data_start..data_end].to_vec());
        }

        // Chunks are word-aligned.
        cursor = data_start
            .saturating_add(chunk_size)
            .saturating_add(chunk_size % 2);
    }

    let sample_rate_hz = sample_rate_hz.ok_or_else(|| "missing wav fmt sample_rate".to_owned())?;
    let channels = channels.ok_or_else(|| "missing wav fmt channels".to_owned())?;
    let bits_per_sample =
        bits_per_sample.ok_or_else(|| "missing wav fmt bits_per_sample".to_owned())?;
    let audio_format = audio_format.ok_or_else(|| "missing wav fmt audio_format".to_owned())?;
    let data = data.ok_or_else(|| "missing wav data chunk".to_owned())?;

    if audio_format != 1 {
        return Err(format!(
            "unsupported wav audio_format={audio_format}; expected PCM (1)"
        ));
    }
    if channels != 1 {
        return Err(format!(
            "unsupported wav channels={channels}; expected mono (1)"
        ));
    }
    if bits_per_sample != 16 {
        return Err(format!(
            "unsupported wav bits_per_sample={bits_per_sample}; expected 16"
        ));
    }
    if sample_rate_hz == 0 {
        return Err("invalid wav sample_rate 0".to_owned());
    }

    let mut samples = Vec::with_capacity(data.len() / 2);
    for pair in data.chunks_exact(2) {
        samples.push(i16::from_le_bytes([pair[0], pair[1]]));
    }

    Ok(WavPcm16Mono {
        sample_rate_hz,
        samples,
    })
}

fn compute_frame_rms(samples: &[i16], frame_samples: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for chunk in samples.chunks(frame_samples.max(1)) {
        if chunk.is_empty() {
            continue;
        }
        let sum_sq = chunk.iter().fold(0.0_f64, |acc, value| {
            let normalized = (*value as f64) / 32768.0;
            acc + (normalized * normalized)
        });
        let rms = (sum_sq / (chunk.len() as f64)).sqrt() as f32;
        out.push(rms);
    }
    out
}

fn bridge_short_gaps(active: &mut [bool], max_gap_frames: usize) {
    if active.is_empty() {
        return;
    }

    let mut idx = 0usize;
    while idx < active.len() {
        if active[idx] {
            idx += 1;
            continue;
        }

        let gap_start = idx;
        while idx < active.len() && !active[idx] {
            idx += 1;
        }
        let gap_end = idx;
        let gap_len = gap_end.saturating_sub(gap_start);

        if gap_start > 0 && gap_end < active.len() && gap_len <= max_gap_frames {
            for frame in &mut active[gap_start..gap_end] {
                *frame = true;
            }
        }
    }
}

fn active_regions_from_frames(
    frame_rms: &[f32],
    active: &[bool],
    frame_ms: u32,
    min_region_frames: usize,
) -> Vec<AudioRegion> {
    let mut regions = Vec::new();
    let mut idx = 0usize;

    while idx < active.len() {
        if !active[idx] {
            idx += 1;
            continue;
        }

        let start = idx;
        while idx < active.len() && active[idx] {
            idx += 1;
        }
        let end = idx;

        if end.saturating_sub(start) < min_region_frames {
            continue;
        }

        let avg_rms = if end > start {
            frame_rms[start..end].iter().copied().sum::<f32>() / (end - start) as f32
        } else {
            0.0
        };

        regions.push(AudioRegion {
            start_ms: (start as u64) * (frame_ms as u64),
            end_ms: (end as u64) * (frame_ms as u64),
            avg_rms,
        });
    }

    regions
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tempfile::tempdir;

    use super::analyze_wav;

    fn write_pcm16_mono_wav(path: &Path, sample_rate: u32, samples: &[i16]) {
        let data_len = (samples.len() * 2) as u32;
        let mut bytes = Vec::with_capacity(44 + data_len as usize);

        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");

        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        let byte_rate = sample_rate * 2;
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
        bytes.extend_from_slice(&16u16.to_le_bytes());

        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        std::fs::write(path, bytes).expect("write wav");
    }

    #[test]
    fn silence_has_no_active_regions() {
        let dir = tempdir().expect("tempdir");
        let wav = dir.path().join("silence.wav");
        let samples = vec![0i16; 16_000];
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let analysis = analyze_wav(&wav, None).expect("analysis should succeed");
        assert!(analysis.active_regions.is_empty());
        assert_eq!(analysis.duration_ms, 1_000);
    }

    #[test]
    fn tone_has_detectable_active_region() {
        let dir = tempdir().expect("tempdir");
        let wav = dir.path().join("tone.wav");

        let mut samples = vec![0i16; 16_000];
        // 1s of moderate-amplitude pulse train.
        for i in 0..16_000usize {
            if (i / 40) % 2 == 0 {
                samples.push(8_000);
            } else {
                samples.push(-8_000);
            }
        }
        samples.extend(vec![0i16; 16_000]);

        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let analysis = analyze_wav(&wav, None).expect("analysis should succeed");
        assert!(!analysis.active_regions.is_empty());
        assert!(analysis.active_regions[0].start_ms <= 1_100);
        assert!(analysis.active_regions[0].end_ms >= 1_900);
    }

    #[test]
    fn analysis_is_deterministic_for_same_input() {
        let dir = tempdir().expect("tempdir");
        let wav = dir.path().join("deterministic.wav");

        let mut samples = Vec::new();
        samples.extend(vec![0i16; 4_000]);
        samples.extend(vec![7_000i16; 8_000]);
        samples.extend(vec![0i16; 4_000]);
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let first = analyze_wav(&wav, None).expect("first analysis");
        let second = analyze_wav(&wav, None).expect("second analysis");

        assert_eq!(first.duration_ms, second.duration_ms);
        assert_eq!(first.frame_count, second.frame_count);
        assert_eq!(first.active_regions.len(), second.active_regions.len());
        for (a, b) in first
            .active_regions
            .iter()
            .zip(second.active_regions.iter())
        {
            assert_eq!(a.start_ms, b.start_ms);
            assert_eq!(a.end_ms, b.end_ms);
            assert!((a.avg_rms - b.avg_rms).abs() < 1e-9);
        }
    }

    #[test]
    fn invalid_header_is_rejected() {
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("invalid.wav");
        std::fs::write(&file, b"not-a-wav").expect("write invalid data");

        let error = analyze_wav(&file, None).expect_err("invalid wav should fail");
        assert!(
            error.contains("unsupported wav") || error.contains("too small"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn minimal_wav_with_no_samples_is_valid_and_empty() {
        let dir = tempdir().expect("tempdir");
        let wav = dir.path().join("minimal.wav");
        write_pcm16_mono_wav(&wav, 16_000, &[]);

        let analysis = analyze_wav(&wav, None).expect("minimal wav should parse");
        assert_eq!(analysis.duration_ms, 0);
        assert_eq!(analysis.frame_count, 0);
        assert!(analysis.active_regions.is_empty());
    }

    #[test]
    fn corrupt_fmt_chunk_is_rejected() {
        let dir = tempdir().expect("tempdir");
        let wav = dir.path().join("corrupt_fmt.wav");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&(8u32).to_le_bytes()); // invalid (<16 bytes)
        bytes.extend_from_slice(&[0u8; 8]);
        std::fs::write(&wav, bytes).expect("write corrupt wav");

        let error = analyze_wav(&wav, None).expect_err("corrupt fmt should fail");
        assert!(
            error.contains("invalid wav fmt chunk")
                || error.contains("missing wav")
                || error.contains("unsupported")
                || error.contains("too small"),
            "unexpected corrupt fmt error: {error}"
        );
    }

    #[test]
    fn as_json_contains_expected_fields() {
        use super::{AudioRegion, NativeAudioAnalysis};
        let analysis = NativeAudioAnalysis {
            duration_ms: 5000,
            sample_rate_hz: 16_000,
            frame_ms: 20,
            frame_count: 250,
            avg_rms: 0.05,
            max_rms: 0.3,
            activity_threshold: 0.075,
            active_regions: vec![AudioRegion {
                start_ms: 100,
                end_ms: 500,
                avg_rms: 0.1,
            }],
        };
        let json = analysis.as_json();
        assert_eq!(json["sample_rate_hz"], 16_000);
        assert_eq!(json["duration_ms"], 5000);
        assert_eq!(json["frame_count"], 250);
        assert_eq!(json["active_region_count"], 1);
        let regions = json["active_regions"].as_array().expect("array");
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0]["start_ms"], 100);
        assert_eq!(regions[0]["end_ms"], 500);
    }

    #[test]
    fn bridge_short_gaps_fills_short_preserves_long() {
        use super::bridge_short_gaps;
        // Gap of 2 frames (= GAP_BRIDGE_MAX_FRAMES) between two active runs → bridged.
        let mut active = vec![true, true, false, false, true, true];
        bridge_short_gaps(&mut active, 2);
        assert!(active.iter().all(|&v| v), "gap of 2 should be bridged");

        // Gap of 3 frames (> max 2) → NOT bridged.
        let mut active2 = vec![true, false, false, false, true];
        bridge_short_gaps(&mut active2, 2);
        assert!(!active2[1]);
        assert!(!active2[2]);
        assert!(!active2[3]);

        // Empty input → no panic.
        let mut empty: Vec<bool> = vec![];
        bridge_short_gaps(&mut empty, 2);
        assert!(empty.is_empty());
    }

    #[test]
    fn active_regions_filters_short_regions() {
        use super::active_regions_from_frames;
        // Active run of 1 frame with min_region_frames=2 → filtered out.
        let rms = vec![0.1, 0.0, 0.1, 0.1, 0.0];
        let active = vec![true, false, true, true, false];
        let regions = active_regions_from_frames(&rms, &active, 20, 2);
        // Only the 2-frame run at index 2-3 should survive.
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].start_ms, 40);
        assert_eq!(regions[0].end_ms, 80);
    }

    #[test]
    fn compute_frame_rms_max_amplitude() {
        use super::compute_frame_rms;
        // Max amplitude 16-bit: 32767/32768 ≈ 1.0 per sample → RMS ≈ 1.0.
        let samples = vec![i16::MAX; 160]; // one 10ms frame at 16kHz
        let rms = compute_frame_rms(&samples, 160);
        assert_eq!(rms.len(), 1);
        assert!(
            (rms[0] - 1.0).abs() < 0.001,
            "max amplitude RMS ≈ 1.0, got {}",
            rms[0]
        );

        // Silence → RMS 0.
        let silence = vec![0i16; 160];
        let rms_silent = compute_frame_rms(&silence, 160);
        assert!((rms_silent[0] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn duration_override_zero_falls_back_to_computed() {
        let dir = tempdir().expect("tempdir");
        let wav = dir.path().join("override_zero.wav");
        let samples = vec![0i16; 16_000]; // 1 second at 16kHz
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        // Some(0) is filtered out by `.filter(|value| *value > 0)`.
        let analysis = analyze_wav(&wav, Some(0)).expect("analysis should succeed");
        assert_eq!(
            analysis.duration_ms, 1_000,
            "Some(0) should fall back to computed duration"
        );
    }

    #[test]
    fn duration_override_is_honored_when_positive() {
        let dir = tempdir().expect("tempdir");
        let wav = dir.path().join("override.wav");
        let samples = vec![0i16; 16_000];
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let analysis = analyze_wav(&wav, Some(2_500)).expect("analysis should succeed");
        assert_eq!(analysis.duration_ms, 2_500);
    }
}
