use std::sync::mpsc;
use std::time::Duration;

use clap::Parser;
use franken_whisper::cli::{
    Cli, Command, ControlFrameKind, RobotCommand, RunsOutputFormat, ShutdownController,
    SyncCommand, TtyAudioCommand, TtyAudioControlCommand, TtyAudioRecoveryPolicy,
};
use franken_whisper::robot::{
    emit_robot_complete, emit_robot_error, emit_robot_stage, emit_robot_start, robot_schema_value,
};
use franken_whisper::storage::RunStore;
use franken_whisper::tty_audio;
use franken_whisper::{FrankenWhisperEngine, FwError, FwResult};

fn main() {
    franken_whisper::logging::init();

    // bd-38c.6: Install graceful Ctrl+C shutdown handler.
    if let Err(e) = ShutdownController::install(None) {
        tracing::warn!("failed to install Ctrl+C handler: {e}");
    }

    if let Err(error) = run() {
        // If shutdown was triggered via Ctrl+C, exit with signal code.
        if ShutdownController::is_shutting_down() {
            eprintln!("interrupted");
            std::process::exit(ShutdownController::signal_exit_code());
        }
        eprintln!("error: {error}");
        std::process::exit(1);
    }

    // If we completed but Ctrl+C was pressed (e.g. during finalization),
    // use the signal exit code.
    if ShutdownController::is_shutting_down() {
        std::process::exit(ShutdownController::signal_exit_code());
    }
}

fn run() -> FwResult<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Transcribe(args) => {
            let request = args.to_request()?;
            let engine = FrankenWhisperEngine::new()?;
            let report = engine.transcribe(request)?;

            if args.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!("{}", report.result.transcript);
            }
            Ok(())
        }
        Command::Robot { command } => match command {
            RobotCommand::Run(args) => {
                emit_robot_start(args.robot_summary())?;
                let request = args.to_request()?;

                let (event_tx, event_rx) = mpsc::channel();
                let worker = std::thread::spawn(move || -> FwResult<_> {
                    let engine = FrankenWhisperEngine::new()?;
                    engine.transcribe_with_stream(request, event_tx)
                });

                loop {
                    match event_rx.recv_timeout(Duration::from_millis(40)) {
                        Ok(streamed) => emit_robot_stage(&streamed.run_id, &streamed.event)?,
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            if worker.is_finished() {
                                break;
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }

                while let Ok(streamed) = event_rx.try_recv() {
                    emit_robot_stage(&streamed.run_id, &streamed.event)?;
                }

                let join_outcome = worker
                    .join()
                    .map_err(|_| FwError::Unsupported("robot worker thread panicked".to_owned()))?;

                match join_outcome {
                    Ok(report) => emit_robot_complete(&report),
                    Err(error) => {
                        emit_robot_error(&error.to_string(), error.robot_error_code())?;
                        Err(error)
                    }
                }
            }
            RobotCommand::Schema => {
                println!("{}", serde_json::to_string_pretty(&robot_schema_value())?);
                Ok(())
            }
            RobotCommand::RoutingHistory(args) => {
                let store = RunStore::open(&args.db)?;
                let details_list = if let Some(run_id) = &args.run_id {
                    store
                        .load_run_details(run_id)?
                        .into_iter()
                        .collect::<Vec<_>>()
                } else {
                    let summaries = store.list_recent_runs(args.limit)?;
                    summaries
                        .iter()
                        .filter_map(|s| store.load_run_details(&s.run_id).ok().flatten())
                        .collect()
                };

                for details in details_list {
                    for event in &details.events {
                        if event.code == "backend.routing.decision_contract"
                            || event.code == "backend.routing.safe_mode"
                            || event.code == "backend.routing.calibration_guardrail"
                        {
                            let entry = serde_json::json!({
                                "event": "routing_decision",
                                "run_id": details.run_id,
                                "ts": event.ts_rfc3339,
                                "code": event.code,
                                "decision_id": event.payload.get("decision_id"),
                                "chosen_action": event.payload.get("chosen_action"),
                                "calibration_score": event.payload.get("calibration_score"),
                                "e_process": event.payload.get("e_process"),
                                "fallback_active": event.payload.get("fallback_active"),
                                "recommended_order": event.payload.get("recommended_order"),
                                "mode": event.payload.get("mode"),
                            });
                            println!("{}", serde_json::to_string(&entry)?);
                        }
                    }
                }
                Ok(())
            }
            RobotCommand::Backends => {
                let payload = serde_json::json!({
                    "event": "backends",
                    "backends": franken_whisper::backend::diagnostics(),
                    "acceleration": {
                        "frankentorch_feature": cfg!(feature = "gpu-frankentorch"),
                        "frankenjax_feature": cfg!(feature = "gpu-frankenjax"),
                    },
                });
                println!("{}", serde_json::to_string(&payload)?);
                Ok(())
            }
        },
        Command::Runs(args) => {
            let store = RunStore::open(&args.db)?;

            if let Some(run_id) = &args.id {
                match store.load_run_details(run_id)? {
                    Some(details) => match args.format {
                        RunsOutputFormat::Plain => {
                            println!("{}", details.transcript);
                        }
                        RunsOutputFormat::Json => {
                            println!("{}", serde_json::to_string_pretty(&details)?);
                        }
                        RunsOutputFormat::Ndjson => {
                            println!("{}", serde_json::to_string(&details)?);
                        }
                    },
                    None => {
                        return Err(FwError::InvalidRequest(format!(
                            "no run found with id `{run_id}`"
                        )));
                    }
                }
            } else {
                let runs = store.list_recent_runs(args.limit)?;
                match args.format {
                    RunsOutputFormat::Plain => {
                        for run in runs {
                            println!(
                                "{} | {} | {} | {}",
                                run.started_at_rfc3339,
                                run.backend.as_str(),
                                run.run_id,
                                run.transcript_preview
                            );
                        }
                    }
                    RunsOutputFormat::Json => {
                        println!("{}", serde_json::to_string_pretty(&runs)?);
                    }
                    RunsOutputFormat::Ndjson => {
                        for run in runs {
                            println!("{}", serde_json::to_string(&run)?);
                        }
                    }
                }
            }
            Ok(())
        }
        Command::Sync { command } => match command {
            SyncCommand::Export(args) => {
                let manifest =
                    franken_whisper::sync::export(&args.db, &args.output, &args.state_root)?;
                println!("{}", serde_json::to_string_pretty(&manifest)?);
                Ok(())
            }
            SyncCommand::Import(args) => {
                let result = franken_whisper::sync::import(
                    &args.db,
                    &args.input,
                    &args.state_root,
                    args.conflict_policy,
                )?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "runs_imported": result.runs_imported,
                        "segments_imported": result.segments_imported,
                        "events_imported": result.events_imported,
                        "conflicts": result.conflicts,
                    }))?
                );
                Ok(())
            }
        },
        Command::TtyAudio { command } => match command {
            TtyAudioCommand::Encode { input, chunk_ms } => {
                tty_audio::encode_to_stdout(&input, chunk_ms)
            }
            TtyAudioCommand::Decode { output, recovery } => {
                let policy = match recovery {
                    TtyAudioRecoveryPolicy::FailClosed => {
                        tty_audio::DecodeRecoveryPolicy::FailClosed
                    }
                    TtyAudioRecoveryPolicy::SkipMissing => {
                        tty_audio::DecodeRecoveryPolicy::SkipMissing
                    }
                };
                tty_audio::decode_from_stdin_to_wav_with_policy(&output, policy)
            }
            TtyAudioCommand::RetransmitPlan { recovery } => {
                let policy = match recovery {
                    TtyAudioRecoveryPolicy::FailClosed => {
                        tty_audio::DecodeRecoveryPolicy::FailClosed
                    }
                    TtyAudioRecoveryPolicy::SkipMissing => {
                        tty_audio::DecodeRecoveryPolicy::SkipMissing
                    }
                };
                let plan = tty_audio::retransmit_plan_from_stdin(policy)?;
                println!("{}", serde_json::to_string(&plan)?);
                Ok(())
            }
            TtyAudioCommand::Control { command } => match command {
                TtyAudioControlCommand::Handshake {
                    min_version,
                    max_version,
                    supported_codecs,
                } => tty_audio::emit_control_frame_to_stdout(
                    &tty_audio::TtyControlFrame::Handshake {
                        min_version,
                        max_version,
                        supported_codecs,
                    },
                ),
                TtyAudioControlCommand::HandshakeAck {
                    negotiated_version,
                    negotiated_codec,
                } => tty_audio::emit_control_frame_to_stdout(
                    &tty_audio::TtyControlFrame::HandshakeAck {
                        negotiated_version,
                        negotiated_codec,
                    },
                ),
                TtyAudioControlCommand::Ack { up_to_seq } => {
                    tty_audio::emit_control_frame_to_stdout(&tty_audio::TtyControlFrame::Ack {
                        up_to_seq,
                    })
                }
                TtyAudioControlCommand::Backpressure { remaining_capacity } => {
                    tty_audio::emit_control_frame_to_stdout(
                        &tty_audio::TtyControlFrame::Backpressure { remaining_capacity },
                    )
                }
                TtyAudioControlCommand::RetransmitRequest { sequences } => {
                    tty_audio::emit_control_frame_to_stdout(
                        &tty_audio::TtyControlFrame::RetransmitRequest { sequences },
                    )
                }
                TtyAudioControlCommand::RetransmitResponse { sequences } => {
                    tty_audio::emit_control_frame_to_stdout(
                        &tty_audio::TtyControlFrame::RetransmitResponse { sequences },
                    )
                }
                TtyAudioControlCommand::RetransmitLoop { recovery, rounds } => {
                    let policy = match recovery {
                        TtyAudioRecoveryPolicy::FailClosed => {
                            tty_audio::DecodeRecoveryPolicy::FailClosed
                        }
                        TtyAudioRecoveryPolicy::SkipMissing => {
                            tty_audio::DecodeRecoveryPolicy::SkipMissing
                        }
                    };
                    tty_audio::emit_retransmit_loop_from_stdin(policy, rounds)
                }
            },

            // bd-2xe.4: convenience send-control command
            TtyAudioCommand::SendControl { frame_type } => send_control_frame(frame_type),

            // bd-2xe.4: convenience retransmit command
            TtyAudioCommand::Retransmit { recovery, rounds } => {
                let policy = match recovery {
                    TtyAudioRecoveryPolicy::FailClosed => {
                        tty_audio::DecodeRecoveryPolicy::FailClosed
                    }
                    TtyAudioRecoveryPolicy::SkipMissing => {
                        tty_audio::DecodeRecoveryPolicy::SkipMissing
                    }
                };
                tty_audio::emit_retransmit_loop_from_stdin(policy, rounds)
            }
        },
        Command::Tui => franken_whisper::tui::run_tui(),
    }
}

// ---------------------------------------------------------------------------
// bd-2xe.4: send-control helper
// ---------------------------------------------------------------------------

/// Emit a control frame to stdout based on the simplified `ControlFrameKind`.
///
/// - `Handshake` emits a default handshake with protocol v1 and the standard
///   codec.
/// - `Eof` emits an end-of-stream control frame.
/// - `Reset` emits a stream-reset control frame.
fn send_control_frame(kind: ControlFrameKind) -> FwResult<()> {
    match kind {
        ControlFrameKind::Handshake => {
            tty_audio::emit_control_frame_to_stdout(&tty_audio::TtyControlFrame::Handshake {
                min_version: 1,
                max_version: 1,
                supported_codecs: vec!["mulaw+zlib+b64".to_owned()],
            })
        }
        ControlFrameKind::Eof => tty_audio::emit_session_close(
            &mut std::io::stdout().lock(),
            tty_audio::SessionCloseReason::Normal,
            None,
        ),
        ControlFrameKind::Reset => tty_audio::emit_session_close(
            &mut std::io::stdout().lock(),
            tty_audio::SessionCloseReason::Error,
            None,
        ),
    }
}
