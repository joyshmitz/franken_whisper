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
/// line is test-only. If it is any column-0 item keyword (`fn`, `struct`,
/// `enum`, `trait`, `impl`, `const`, `static`, `type`, `mod`, `use`, `union`,
/// `macro_rules!`, with or without a `pub` / `pub(crate)` prefix — see
/// [`line_starts_production_item`]) *without* a `#[cfg(test)]` above it, the
/// line is production. This deliberately simple line-based heuristic matches the
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
        if at_col0 && line_starts_production_item(trimmed) {
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

/// Does `trimmed` (a column-0 line, already left-trimmed) begin a top-level
/// Rust item that marks a production boundary?
///
/// The list must cover **every** item kind that can legally sit at column 0,
/// because a production item appearing *below* a trailing `#[cfg(test)] mod`
/// would otherwise be misclassified as test-only: the backward scan must hit
/// the production item's own keyword before it ever reaches the test module's
/// `#[cfg(test)]`. We therefore include not just `fn`/`struct`/`impl`/`const`/
/// `mod` but also `static`, `type`, `enum`, `trait`, `use`, and
/// `macro_rules!` — with their `pub` / `pub(crate)` visibility prefixes.
fn line_starts_production_item(trimmed: &str) -> bool {
    // Strip an optional leading visibility modifier so we match the item keyword
    // regardless of `pub`, `pub(crate)`, `pub(super)`, `pub(in ...)`, etc.
    let rest = strip_visibility(trimmed);
    // `macro_rules!` has no trailing space before `!`; handle it explicitly.
    if rest.starts_with("macro_rules!") {
        return true;
    }
    const ITEM_KEYWORDS: &[&str] = &[
        "fn ", "struct ", "enum ", "trait ", "impl ", "const ", "static ", "type ", "mod ", "use ",
        "union ",
    ];
    ITEM_KEYWORDS.iter().any(|kw| rest.starts_with(kw))
}

/// Strip a leading `pub`, `pub(crate)`, `pub(super)`, `pub(self)`, or
/// `pub(in path)` visibility modifier (and the following whitespace) from a
/// left-trimmed line, returning the remainder. Lines without a visibility
/// prefix are returned unchanged.
fn strip_visibility(trimmed: &str) -> &str {
    let Some(after_pub) = trimmed.strip_prefix("pub") else {
        return trimmed;
    };
    // Bare `pub ` (e.g. `pub fn`).
    if let Some(rest) = after_pub.strip_prefix(' ') {
        return rest.trim_start();
    }
    // Restricted `pub(...)` — skip the balanced parenthesis group.
    if after_pub.starts_with('(')
        && let Some(close) = after_pub.find(')')
    {
        return after_pub[close + 1..].trim_start();
    }
    // `pub` immediately followed by something unexpected: leave as-is.
    trimmed
}

#[cfg(test)]
mod classifier_tests {
    use super::{line_is_in_cfg_test, line_starts_production_item, strip_visibility};

    #[test]
    fn strip_visibility_handles_all_forms() {
        assert_eq!(strip_visibility("pub fn foo()"), "fn foo()");
        assert_eq!(strip_visibility("pub(crate) struct S;"), "struct S;");
        assert_eq!(
            strip_visibility("pub(super) static X: u8 = 0;"),
            "static X: u8 = 0;"
        );
        assert_eq!(strip_visibility("pub(in crate::a) enum E {}"), "enum E {}");
        assert_eq!(strip_visibility("fn bare()"), "fn bare()");
        // `pubfoo` is not a visibility modifier.
        assert_eq!(strip_visibility("pubfoo"), "pubfoo");
    }

    #[test]
    fn production_item_keywords_are_recognized() {
        for line in [
            "fn f() {}",
            "pub fn f() {}",
            "struct S;",
            "pub struct S;",
            "enum E {}",
            "pub enum E {}",
            "trait T {}",
            "pub trait T {}",
            "impl S {}",
            "const C: u8 = 0;",
            "pub const C: u8 = 0;",
            "static S: u8 = 0;",
            "pub static S: u8 = 0;",
            "type Alias = u8;",
            "pub type Alias = u8;",
            "mod m {}",
            "pub mod m {}",
            "use crate::Foo;",
            "pub use crate::Foo;",
            "union U { a: u8 }",
            "macro_rules! mac { () => {} }",
            "pub(crate) fn f() {}",
        ] {
            assert!(
                line_starts_production_item(line),
                "should be recognized as a production item: {line:?}"
            );
        }
    }

    #[test]
    fn non_items_are_not_production_boundaries() {
        // `line_starts_production_item` is only ever called on a left-trimmed
        // line that the caller has already confirmed sits at column 0, so each
        // case here is the trimmed form of a non-item line.
        for line in [
            "let x = 5;",
            "// a comment",
            "#[derive(Debug)]",
            "return Ok(());",
            "}",
            "usize_value += 1;",    // starts with "us" but not the `use ` keyword
            "structure.field = 1;", // starts with "struct" but not `struct `
            "fnord();",             // starts with "fn" but not `fn `
        ] {
            assert!(
                !line_starts_production_item(line),
                "should NOT be a production item: {line:?}"
            );
        }
    }

    #[test]
    fn production_item_below_trailing_test_mod_is_not_test_only() {
        // The motivating case: a production `static`/`enum`/`use` placed AFTER a
        // trailing `#[cfg(test)] mod tests`. Without the extended keyword list,
        // the backward scan would skip past the production item and reach the
        // test module's `#[cfg(test)]`, misclassifying the production line as
        // test-only. With the extension it correctly stops at the item.
        let src = "\
#[cfg(test)]
mod tests {
    fn helper() {}
}

static PRODUCTION_TABLE: &[&str] = &[
    \"The quick brown fox jumps over the lazy dog\",
];
";
        let lines: Vec<&str> = src.lines().collect();
        let target = lines
            .iter()
            .position(|l| l.contains("The quick brown fox"))
            .expect("fixture line present");
        assert!(
            !line_is_in_cfg_test(src, target),
            "a production static below a trailing test mod must NOT be classified test-only"
        );
    }

    #[test]
    fn line_inside_real_test_mod_is_test_only() {
        let src = "\
fn production() {}

#[cfg(test)]
mod tests {
    fn t() {
        let _ = \"Hello world, this is a test transcription\";
    }
}
";
        let lines: Vec<&str> = src.lines().collect();
        let target = lines
            .iter()
            .position(|l| l.contains("Hello world"))
            .expect("fixture line present");
        assert!(
            line_is_in_cfg_test(src, target),
            "a line inside #[cfg(test)] mod tests must be classified test-only"
        );
    }
}
