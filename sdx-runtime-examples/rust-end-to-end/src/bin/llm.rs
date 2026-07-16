// ******************************************************************************
//
// This program and the accompanying materials are made available under the
// terms of the Apache License, Version 2.0 which is available at
// https://www.apache.org/licenses/LICENSE-2.0.
//
// SPDX-License-Identifier: Apache-2.0
// ******************************************************************************

//! LLM/VLM/STT end-to-end walkthrough using the AOT `libsdx_llm` C ABI.
//!
//! Demonstrates the full lifecycle of the SDX LLM surface — no JVM required:
//!
//! 1. Runtime / model creation (`LlmRuntime` / `LlmModel` with `Drop`).
//! 2. Model info JSON query.
//! 3. Tokenize → detokenize round-trip.
//! 4. Text generation with greedy sampling.
//! 5. `GenerateStats` telemetry from `sdxLlmLastResultJson`.
//! 6. Canonical output assertion: generated text must contain `" Paris."`.
//! 7. Optional: VLM document extraction (enabled via `--vlm`).
//! 8. Optional: Whisper STT transcription (enabled via `--transcribe`).
//! 9. Error-path demonstration.
//!
//! ## Running
//!
//! ```bash
//! export SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8   # unpacked AOT SDK package root
//!
//! # Model paths default to the local development cache locations; override:
//! # SDX_LLM_MODEL_PATH  — .gguf model file
//! # SDX_LLM_TOKENIZER   — tokenizer.json file
//! # SDX_VLM_MODEL_PATH  — SmolDocling model directory (for --vlm)
//! # SDX_VLM_IMAGE_PATH  — image/PDF for VLM (for --vlm)
//! # SDX_STT_MODEL_PATH  — Whisper model directory (for --transcribe)
//! # SDX_STT_AUDIO_PATH  — .wav file (for --transcribe)
//!
//! SDX_LLM_LIB_DIR=$SDX_LLM_AOT_HOME/lib \
//!   LD_LIBRARY_PATH=$SDX_LLM_AOT_HOME/lib \
//!   cargo run --release --bin llm [-- --vlm] [-- --transcribe]
//! ```

// The LLM wrapper is now provided by the canonical `sdx-runtime` crate
// under `sdx_runtime::llm` (feature "llm").  The vendored `src/sdx_llm.rs`
// has been removed; all callers should use the canonical module instead.
use sdx_runtime::llm::{ensure_native_lib_dir, GenerateStats, LlmError, LlmRuntime};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

// ── Default asset paths ───────────────────────────────────────────────────────

fn home() -> PathBuf {
    dirs_home().unwrap_or_else(|| PathBuf::from("/root"))
}

fn dirs_home() -> Option<PathBuf> {
    // Avoid pulling in the `dirs` crate — resolve from HOME env var.
    std::env::var("HOME").ok().map(PathBuf::from)
}

fn env_or(var: &str, default: PathBuf) -> PathBuf {
    std::env::var(var).map(PathBuf::from).unwrap_or(default)
}

fn default_model() -> PathBuf {
    env_or(
        "SDX_LLM_MODEL_PATH",
        home().join(".cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf"),
    )
}

fn default_tokenizer() -> PathBuf {
    env_or(
        "SDX_LLM_TOKENIZER",
        home().join(".cache/dl4j-llm-models/qwen35-0.8B-tokenizer.json"),
    )
}

fn default_vlm_model() -> PathBuf {
    env_or(
        "SDX_VLM_MODEL_PATH",
        home().join(".kompile/models/vlm/smoldocling-256m"),
    )
}

fn default_vlm_image() -> PathBuf {
    env_or("SDX_VLM_IMAGE_PATH", PathBuf::from("/tmp/vlm-agent-test-doc.png"))
}

fn default_stt_model() -> PathBuf {
    env_or(
        "SDX_STT_MODEL_PATH",
        home().join(".cache/dl4j-whisper-models/whisper-tiny-onnx"),
    )
}

fn default_stt_audio() -> PathBuf {
    env_or("SDX_STT_AUDIO_PATH", PathBuf::from("/tmp/test-audio-agent.wav"))
}

// The canonical generate prompt and the substring expected in the output.
const CANON_PROMPT:   &str = "The capital of France is";
const CANON_EXPECTED: &str = " Paris.";
const GENERATE_OPTS:  &str = r#"{"maxNewTokens":8,"sampling":{"preset":"greedy"}}"#;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn check_asset(path: &Path, label: &str) -> bool {
    if path.exists() {
        true
    } else {
        println!("  [SKIP] {label} not found at: {}", path.display());
        false
    }
}

fn pass_fail(ok: bool, label: &str) {
    println!("  {}: {label}", if ok { "PASS" } else { "FAIL" });
}

// ── Step implementations ──────────────────────────────────────────────────────

fn step_info(model: &sdx_runtime::llm::LlmModel) -> Result<(), LlmError> {
    println!("\n== Step 2: model info ==");
    match model.info_json() {
        Ok(json) if !json.is_empty() => println!("  info JSON: {}", &json[..json.len().min(300)]),
        Ok(_)    => println!("  (no info returned)"),
        Err(e)   => println!("  info query failed (non-fatal): {e}"),
    }
    Ok(())
}

fn step_tokenize(model: &sdx_runtime::llm::LlmModel) -> Result<bool, LlmError> {
    println!("\n== Step 3: tokenize / detokenize round-trip ==");
    let text = "Hello, world!";
    let ids = model.tokenize(text, false)?;
    println!("  '{text}' → token IDs: {ids:?}");
    let recovered = model.detokenize(&ids, true)?;
    println!("  IDs → '{recovered}'");
    let ok = text.to_lowercase().contains(&recovered.trim().to_lowercase())
        || recovered.trim().to_lowercase().contains(&text.trim().to_lowercase());
    pass_fail(ok, "round-trip matches source text");
    Ok(ok)
}

fn step_generate(model: &sdx_runtime::llm::LlmModel) -> Result<bool, LlmError> {
    println!(
        "\n== Step 4: generate (prompt={CANON_PROMPT:?}, options={GENERATE_OPTS}) =="
    );
    println!("  (first call includes GGUF import + pipeline warmup; may take 1-3 min)");
    let text = model.generate(CANON_PROMPT, Some(GENERATE_OPTS))?;
    println!("  generated text: {text:?}");

    let stats: GenerateStats = model.last_result()?;
    println!("\n== Step 5: GenerateStats (from sdxLlmLastResultJson) ==");
    println!("  prompt_tokens    = {}", stats.prompt_tokens);
    println!("  generated_tokens = {}", stats.generated_tokens);
    println!("  generation_time_ms      = {:.1}", stats.generation_time_ms);
    println!("  first_token_latency_ms  = {:.1}", stats.first_token_latency_ms);
    println!("  tokens_per_sec          = {:.2}", stats.tokens_per_sec);
    println!("  finish_reason    = {:?}", stats.finish_reason);

    let ok = text.contains(CANON_EXPECTED);
    pass_fail(ok, &format!("generated text contains {CANON_EXPECTED:?}"));
    Ok(ok)
}

fn step_vlm(runtime: &LlmRuntime, require: bool) -> Result<bool, LlmError> {
    println!("\n== Step 6 (optional): VLM document extraction ==");
    let model_path = default_vlm_model();
    let image_path = default_vlm_image();
    let assets_ok = check_asset(&model_path, "VLM model")
        && check_asset(&image_path, "VLM test image");

    if !assets_ok {
        if require {
            println!("  FAIL: --vlm requested but assets are missing.");
            return Ok(false);
        }
        println!("  [SKIP] VLM assets not available — skipping.");
        return Ok(true);
    }

    println!("  model : {}", model_path.display());
    println!("  image : {}", image_path.display());
    println!("  (stateless — loads model per call; may take 1-3 min)");

    let text = runtime.vlm_extract(
        &model_path,
        None,
        &image_path,
        Some(r#"{"maxNewTokens":256,"format":"doctags"}"#),
    )?;
    let preview = &text[..text.len().min(200)];
    println!("  extraction result (first 200 chars): {preview:?}");
    let ok = !text.trim().is_empty();
    pass_fail(ok, "VLM returned non-empty text");
    Ok(ok)
}

fn step_transcribe(runtime: &LlmRuntime, require: bool) -> Result<bool, LlmError> {
    println!("\n== Step 7 (optional): Whisper transcription ==");
    let model_path = default_stt_model();
    let audio_path = default_stt_audio();
    let assets_ok = check_asset(&model_path, "Whisper model")
        && check_asset(&audio_path, "test audio");

    if !assets_ok {
        if require {
            println!("  FAIL: --transcribe requested but assets are missing.");
            return Ok(false);
        }
        println!("  [SKIP] STT assets not available — skipping.");
        return Ok(true);
    }

    println!("  model : {}", model_path.display());
    println!("  audio : {}", audio_path.display());
    println!("  (stateless — loads model per call)");

    let text = runtime.audio_transcribe(
        &model_path,
        &audio_path,
        Some(r#"{"language":"en"}"#),
    )?;
    println!("  transcript: {text:?}");
    let ok = !text.trim().is_empty();
    pass_fail(ok, "Whisper returned non-empty transcript");
    Ok(ok)
}

// ── Entry points ──────────────────────────────────────────────────────────────

fn main() -> ExitCode {
    match run() {
        Ok(true) => {
            println!("\nSUCCESS: SDX LLM C ABI outputs verified from pure Rust (no JVM).");
            ExitCode::SUCCESS
        }
        Ok(false) => {
            eprintln!("\nFAILURE: one or more checks did not pass.");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("\nFATAL: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<bool, LlmError> {
    // Parse --vlm and --transcribe flags.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let want_vlm         = args.contains(&"--vlm".to_string());
    let want_transcribe  = args.contains(&"--transcribe".to_string());

    // ── Step 0: pre-flight ───────────────────────────────────────────────────
    println!("== Step 0: pre-flight checks ==");
    let model_path     = default_model();
    let tokenizer_path = default_tokenizer();

    if !check_asset(&model_path, "LLM model (.gguf)") {
        eprintln!(
            "\nSet SDX_LLM_MODEL_PATH to an existing .gguf file.\nDefault: {}",
            model_path.display()
        );
        return Ok(false);
    }
    if !check_asset(&tokenizer_path, "tokenizer.json") {
        eprintln!(
            "\nSet SDX_LLM_TOKENIZER to an existing tokenizer.json file.\nDefault: {}",
            tokenizer_path.display()
        );
        return Ok(false);
    }

    // Set SDX_NATIVE_LIB_DIR so libsdx_llm.so can find its bundled .so files.
    ensure_native_lib_dir();

    // ── Step 1: create runtime and load model ────────────────────────────────
    println!("\n== Step 1: create runtime and load model ==");
    println!("  model     : {}", model_path.display());
    println!("  tokenizer : {}", tokenizer_path.display());

    let runtime = LlmRuntime::new()?;
    println!("  ABI version: {}", runtime.abi_version());

    let load_opts = r#"{"graphOptimizer":true,"sampling":{"preset":"greedy"}}"#;
    let model = runtime.load_model(
        &model_path,
        tokenizer_path.to_str(),
        Some(load_opts),
    )?;

    let mut all_ok = true;

    step_info(&model)?;
    all_ok &= step_tokenize(&model)?;
    all_ok &= step_generate(&model)?;

    // ── Optional steps (stateless — no loaded model needed). Strictly
    // flag-gated: each loads its own model, adding many minutes on CPU, so the
    // default run stays a quick LLM smoke. ────────────────────────────────────
    if want_vlm {
        all_ok &= step_vlm(&runtime, true)?;
    } else {
        println!("\n== Step 6 (optional): VLM document extraction — skipped (pass --vlm) ==");
    }
    if want_transcribe {
        all_ok &= step_transcribe(&runtime, true)?;
    } else {
        println!("\n== Step 7 (optional): Whisper transcription — skipped (pass --transcribe) ==");
    }

    // ── Step 9: error-path demonstration ─────────────────────────────────────
    println!("\n== Step 9: error handling ==");
    match runtime.load_model("/definitely/not/a/model.gguf", None, None) {
        Ok(_) => {
            println!("  unexpected: bogus load succeeded");
            all_ok = false;
        }
        Err(LlmError::ModelLoad { path, status, .. }) => {
            println!(
                "  Loading '{path}' raised LlmError::ModelLoad (status={name}) — as expected.",
                name = status.name()
            );
        }
        Err(e) => println!("  Error: {e}"),
    }

    // Drop order: model is dropped before runtime (LIFO — correct).
    drop(model);

    Ok(all_ok)
}
