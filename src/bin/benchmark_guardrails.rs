use std::fs;
use std::path::{Path, PathBuf};

use chrono::Utc;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser)]
#[command(
    name = "benchmark_guardrails",
    about = "Checks criterion benchmark estimates against regression guardrails"
)]
struct Args {
    /// Path to criterion benchmark artifacts root.
    #[arg(long, default_value = "target/criterion")]
    criterion_root: PathBuf,

    /// Path to benchmark guardrail policy JSON.
    #[arg(long, default_value = "docs/benchmark_guardrails.json")]
    policy: PathBuf,

    /// Treat provisional (non-enforcing) entries as hard failures.
    #[arg(long, default_value_t = false)]
    strict_provisional: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct GuardrailPolicy {
    schema_version: String,
    default_max_regression_pct: f64,
    benchmarks: Vec<GuardrailEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct GuardrailEntry {
    id: String,
    baseline_ns: f64,
    #[serde(default)]
    max_regression_pct: Option<f64>,
    #[serde(default = "default_true")]
    enforce: bool,
    #[serde(default)]
    notes: Option<String>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
struct CriterionEstimates {
    mean: Option<EstimatePoint>,
    median: Option<EstimatePoint>,
    slope: Option<EstimatePoint>,
}

#[derive(Debug, Clone, Deserialize)]
struct EstimatePoint {
    point_estimate: f64,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkCheck {
    id: String,
    status: String,
    enforce: bool,
    measured_ns: Option<f64>,
    baseline_ns: f64,
    max_allowed_ns: f64,
    regression_pct: Option<f64>,
    threshold_pct: f64,
    estimate_source: Option<String>,
    estimates_path: String,
    message: String,
    notes: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct GuardrailReport {
    schema_version: String,
    generated_at_rfc3339: String,
    criterion_root: String,
    policy_path: String,
    total_checks: usize,
    failures: usize,
    warnings: usize,
    pass: bool,
    checks: Vec<BenchmarkCheck>,
}

fn main() {
    let args = Args::parse();
    let report = match run(&args) {
        Ok(report) => report,
        Err(error) => {
            let failure_report = GuardrailReport {
                schema_version: "1.0.0".to_owned(),
                generated_at_rfc3339: Utc::now().to_rfc3339(),
                criterion_root: args.criterion_root.display().to_string(),
                policy_path: args.policy.display().to_string(),
                total_checks: 0,
                failures: 1,
                warnings: 0,
                pass: false,
                checks: vec![BenchmarkCheck {
                    id: "<policy>".to_owned(),
                    status: "error".to_owned(),
                    enforce: true,
                    measured_ns: None,
                    baseline_ns: 0.0,
                    max_allowed_ns: 0.0,
                    regression_pct: None,
                    threshold_pct: 0.0,
                    estimate_source: None,
                    estimates_path: args.policy.display().to_string(),
                    message: error,
                    notes: None,
                }],
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&failure_report)
                    .expect("guardrail failure report should serialize")
            );
            std::process::exit(1);
        }
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("guardrail report should serialize")
    );
    if !report.pass {
        std::process::exit(1);
    }
}

fn run(args: &Args) -> Result<GuardrailReport, String> {
    let policy = read_policy(&args.policy)?;
    let mut checks = Vec::new();
    let mut failures = 0usize;
    let mut warnings = 0usize;

    for entry in &policy.benchmarks {
        let threshold_pct = entry
            .max_regression_pct
            .unwrap_or(policy.default_max_regression_pct);
        let max_allowed_ns = entry.baseline_ns * (1.0 + (threshold_pct / 100.0));

        let effective_enforce = entry.enforce || args.strict_provisional;
        let estimates_path = args
            .criterion_root
            .join(&entry.id)
            .join("new")
            .join("estimates.json");

        let check = match read_measurement_ns(&estimates_path) {
            Ok((measured_ns, source)) => {
                let regression_pct =
                    ((measured_ns - entry.baseline_ns) / entry.baseline_ns) * 100.0;
                if measured_ns > max_allowed_ns {
                    if effective_enforce {
                        failures += 1;
                        BenchmarkCheck {
                            id: entry.id.clone(),
                            status: "fail".to_owned(),
                            enforce: effective_enforce,
                            measured_ns: Some(measured_ns),
                            baseline_ns: entry.baseline_ns,
                            max_allowed_ns,
                            regression_pct: Some(regression_pct),
                            threshold_pct,
                            estimate_source: Some(source),
                            estimates_path: estimates_path.display().to_string(),
                            message: format!(
                                "regression {:.2}% exceeds threshold {:.2}%",
                                regression_pct, threshold_pct
                            ),
                            notes: entry.notes.clone(),
                        }
                    } else {
                        warnings += 1;
                        BenchmarkCheck {
                            id: entry.id.clone(),
                            status: "warn".to_owned(),
                            enforce: effective_enforce,
                            measured_ns: Some(measured_ns),
                            baseline_ns: entry.baseline_ns,
                            max_allowed_ns,
                            regression_pct: Some(regression_pct),
                            threshold_pct,
                            estimate_source: Some(source),
                            estimates_path: estimates_path.display().to_string(),
                            message: format!(
                                "provisional regression {:.2}% exceeds threshold {:.2}%",
                                regression_pct, threshold_pct
                            ),
                            notes: entry.notes.clone(),
                        }
                    }
                } else {
                    BenchmarkCheck {
                        id: entry.id.clone(),
                        status: "pass".to_owned(),
                        enforce: effective_enforce,
                        measured_ns: Some(measured_ns),
                        baseline_ns: entry.baseline_ns,
                        max_allowed_ns,
                        regression_pct: Some(regression_pct),
                        threshold_pct,
                        estimate_source: Some(source),
                        estimates_path: estimates_path.display().to_string(),
                        message: format!(
                            "regression {:.2}% within threshold {:.2}%",
                            regression_pct, threshold_pct
                        ),
                        notes: entry.notes.clone(),
                    }
                }
            }
            Err(error) => {
                if effective_enforce {
                    failures += 1;
                    BenchmarkCheck {
                        id: entry.id.clone(),
                        status: "fail".to_owned(),
                        enforce: effective_enforce,
                        measured_ns: None,
                        baseline_ns: entry.baseline_ns,
                        max_allowed_ns,
                        regression_pct: None,
                        threshold_pct,
                        estimate_source: None,
                        estimates_path: estimates_path.display().to_string(),
                        message: error,
                        notes: entry.notes.clone(),
                    }
                } else {
                    warnings += 1;
                    BenchmarkCheck {
                        id: entry.id.clone(),
                        status: "warn".to_owned(),
                        enforce: effective_enforce,
                        measured_ns: None,
                        baseline_ns: entry.baseline_ns,
                        max_allowed_ns,
                        regression_pct: None,
                        threshold_pct,
                        estimate_source: None,
                        estimates_path: estimates_path.display().to_string(),
                        message: format!("provisional benchmark missing: {error}"),
                        notes: entry.notes.clone(),
                    }
                }
            }
        };

        checks.push(check);
    }

    Ok(GuardrailReport {
        schema_version: policy.schema_version,
        generated_at_rfc3339: Utc::now().to_rfc3339(),
        criterion_root: args.criterion_root.display().to_string(),
        policy_path: args.policy.display().to_string(),
        total_checks: checks.len(),
        failures,
        warnings,
        pass: failures == 0,
        checks,
    })
}

fn read_policy(path: &Path) -> Result<GuardrailPolicy, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read policy {}: {error}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|error| format!("failed to parse policy {}: {error}", path.display()))
}

fn read_measurement_ns(path: &Path) -> Result<(f64, String), String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read estimates {}: {error}", path.display()))?;
    let estimates: CriterionEstimates = serde_json::from_str(&raw)
        .map_err(|error| format!("failed to parse estimates {}: {error}", path.display()))?;

    if let Some(slope) = estimates.slope {
        return Ok((slope.point_estimate, "slope".to_owned()));
    }
    if let Some(mean) = estimates.mean {
        return Ok((mean.point_estimate, "mean".to_owned()));
    }
    if let Some(median) = estimates.median {
        return Ok((median.point_estimate, "median".to_owned()));
    }

    Err(format!(
        "no usable estimate fields (slope/mean/median) in {}",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_file(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent");
        }
        fs::write(path, content).expect("write file");
    }

    #[test]
    fn read_measurement_prefers_slope() {
        let dir = tempfile::tempdir().expect("tempdir");
        let estimates_path = dir.path().join("estimates.json");
        write_file(
            &estimates_path,
            r#"{
                "mean": {"point_estimate": 100.0},
                "median": {"point_estimate": 99.0},
                "slope": {"point_estimate": 80.0}
            }"#,
        );

        let (value, source) = read_measurement_ns(&estimates_path).expect("measurement");
        assert_eq!(value, 80.0);
        assert_eq!(source, "slope");
    }

    #[test]
    fn read_measurement_falls_back_to_mean() {
        let dir = tempfile::tempdir().expect("tempdir");
        let estimates_path = dir.path().join("estimates.json");
        write_file(
            &estimates_path,
            r#"{
                "mean": {"point_estimate": 120.0},
                "median": {"point_estimate": 119.0},
                "slope": null
            }"#,
        );

        let (value, source) = read_measurement_ns(&estimates_path).expect("measurement");
        assert_eq!(value, 120.0);
        assert_eq!(source, "mean");
    }

    #[test]
    fn run_flags_hard_regressions() {
        let dir = tempfile::tempdir().expect("tempdir");
        let policy_path = dir.path().join("policy.json");
        let criterion_root = dir.path().join("criterion");

        write_file(
            &policy_path,
            r#"{
                "schema_version": "1.0.0",
                "default_max_regression_pct": 20.0,
                "benchmarks": [
                    {"id": "tty_encode/chunk_ms/20", "baseline_ns": 100.0, "enforce": true}
                ]
            }"#,
        );
        write_file(
            &criterion_root.join("tty_encode/chunk_ms/20/new/estimates.json"),
            r#"{"mean": {"point_estimate": 130.0}, "median": null, "slope": null}"#,
        );

        let args = Args {
            criterion_root,
            policy: policy_path,
            strict_provisional: false,
        };
        let report = run(&args).expect("report");
        assert_eq!(report.failures, 1);
        assert!(!report.pass);
        assert_eq!(report.checks[0].status, "fail");
    }

    #[test]
    fn run_keeps_provisional_regressions_as_warnings() {
        let dir = tempfile::tempdir().expect("tempdir");
        let policy_path = dir.path().join("policy.json");
        let criterion_root = dir.path().join("criterion");

        write_file(
            &policy_path,
            r#"{
                "schema_version": "1.0.0",
                "default_max_regression_pct": 20.0,
                "benchmarks": [
                    {"id": "sync_export/runs/10", "baseline_ns": 100.0, "enforce": false}
                ]
            }"#,
        );
        write_file(
            &criterion_root.join("sync_export/runs/10/new/estimates.json"),
            r#"{"mean": {"point_estimate": 150.0}, "median": null, "slope": null}"#,
        );

        let args = Args {
            criterion_root,
            policy: policy_path,
            strict_provisional: false,
        };
        let report = run(&args).expect("report");
        assert_eq!(report.failures, 0);
        assert_eq!(report.warnings, 1);
        assert!(report.pass);
        assert_eq!(report.checks[0].status, "warn");
    }
}
