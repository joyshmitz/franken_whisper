//! Honesty regression guard (bd-jryr).
//!
//! The native whisper.cpp engine used to fabricate transcripts from a small set
//! of canned phrases (the "pilot"). bd-jryr rewired it to real inference and
//! deleted those phrases from production. This test makes that deletion
//! *stick*: it scans every `.rs` file under `src/` and asserts the
//! whisper-cpp canned phrases appear **only inside `#[cfg(test)]` regions** (or
//! not at all). A reintroduction in production code fails the build.
//!
//! ## Scope
//!
//! All three former native-engine pilots are now guarded: the whisper-cpp pilot
//! (bd-jryr), the insanely-fast pilot (bd-s8w8), and the diarization pilot
//! (bd-cidv, "Good morning, everyone.", …). Every one of these mocks has been
//! deleted from production; this guard makes that deletion stick across all of
//! them.

use std::path::{Path, PathBuf};

/// The native pilots' canned phrases. None of these may appear in production
/// (non-`#[cfg(test)]`) code. Covers the whisper-cpp pilot (bd-jryr), the
/// insanely-fast pilot (bd-s8w8), and the diarization pilot (bd-cidv).
const CANNED_PHRASES: &[&str] = &[
    // whisper-cpp pilot (bd-jryr)
    "The quick brown fox jumps over the lazy dog",
    "Hello world, this is a test transcription",
    "Speech recognition is a fascinating field",
    "Artificial intelligence continues to advance",
    // insanely-fast pilot (bd-s8w8)
    "Batch processing enables higher throughput",
    "GPU acceleration reduces latency significantly",
    "Parallel inference is the future of ASR",
    // diarization pilot (bd-cidv)
    "Good morning, everyone.",
    "Thank you for joining us today.",
    "Let's begin with the first topic.",
    "I have a question about that.",
    "That's an excellent point.",
    "Could you elaborate further?",
];

#[test]
fn no_canned_whisper_cpp_phrases_in_production() {
    let src_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut rs_files = Vec::new();
    collect_rs_files(&src_root, &mut rs_files);
    assert!(
        !rs_files.is_empty(),
        "found no .rs files under {}",
        src_root.display()
    );

    let mut violations = Vec::new();
    for file in &rs_files {
        let Ok(contents) = std::fs::read_to_string(file) else {
            continue;
        };
        for (line_no, line) in contents.lines().enumerate() {
            for phrase in CANNED_PHRASES {
                if line.contains(phrase) && !line_is_in_cfg_test(&contents, line_no) {
                    violations.push(format!(
                        "{}:{}: canned phrase {:?} in production code",
                        file.display(),
                        line_no + 1,
                        phrase
                    ));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "canned whisper-cpp phrases must only appear inside #[cfg(test)] regions, found:\n{}",
        violations.join("\n")
    );
}

/// Recursively collect every `.rs` file under `dir`.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Heuristic: is the line at `target_line` (0-based) inside a `#[cfg(test)]`
/// region?
///
/// We scan the preceding lines for the **nearest** structural marker. If that
/// marker is a `#[cfg(test)]` attribute (with `mod`/`fn`/block following), the
/// line is test-only. If it is a `pub fn` / `fn` / `impl` / `pub struct` /
/// `struct` / `const` at column 0 *without* a `#[cfg(test)]` above it, the line
/// is production. This deliberately simple line-based heuristic matches the
/// project's conventions (test code lives in `#[cfg(test)] mod tests { ... }`
/// at the bottom of each file, or in clearly-attributed test fns) and is good
/// enough for an honesty guard; it errs toward flagging (treating ambiguous
/// hits as production) so a real reintroduction is never silently allowed.
fn line_is_in_cfg_test(contents: &str, target_line: usize) -> bool {
    let lines: Vec<&str> = contents.lines().collect();
    // Walk backward looking for the nearest `#[cfg(test)]` attribute or a
    // top-level (column-0) item boundary.
    for i in (0..target_line).rev() {
        let trimmed = lines[i].trim_start();
        if trimmed.starts_with("#[cfg(test)]") {
            return true;
        }
        // A column-0 (un-indented) item that is NOT preceded by #[cfg(test)]
        // marks the start of a production item: stop here.
        let at_col0 = !lines[i].starts_with(char::is_whitespace) && !lines[i].is_empty();
        let is_item = at_col0
            && (trimmed.starts_with("pub fn ")
                || trimmed.starts_with("fn ")
                || trimmed.starts_with("pub struct ")
                || trimmed.starts_with("struct ")
                || trimmed.starts_with("impl ")
                || trimmed.starts_with("pub const ")
                || trimmed.starts_with("const ")
                || trimmed.starts_with("pub mod ")
                || trimmed.starts_with("mod "));
        if is_item {
            // Only treat as a production boundary if the line just above it is
            // not a #[cfg(test)] attribute.
            if i > 0 && lines[i - 1].trim_start().starts_with("#[cfg(test)]") {
                return true;
            }
            return false;
        }
    }
    false
}
