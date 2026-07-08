// ******************************************************************************
//
// This program and the accompanying materials are made available under the
// terms of the Apache License, Version 2.0 which is available at
// https://www.apache.org/licenses/LICENSE-2.0.
//
// SPDX-License-Identifier: Apache-2.0
// ******************************************************************************

//! End-to-end SDX runtime walkthrough — idiomatic Rust, no JVM.
//!
//! Loads `../models/mlp.sdz` through the `sdx-runtime` wrapper crate and
//! demonstrates the complete SDK lifecycle:
//!
//! 1. Runtime creation and model loading
//! 2. Input-contract discovery (`input_names()`)
//! 3. Warmup runs via `run_named_shaped()` with ndarray inputs
//! 4. `freeze_shapes()` → DSP replay fast path
//! 5. Execution-report telemetry
//! 6. The error path
//!
//! ## Running
//!
//! ```bash
//! LIBDIR=/path/to/dir/with/libnd4jcpu.so
//! SDX_RUNTIME_LIB_DIR=$LIBDIR LD_LIBRARY_PATH=$LIBDIR \
//!   cargo run --release [-- /path/to/model.sdz]
//! ```

use ndarray::{Array, ArrayD, IxDyn};
use sdx_runtime::{runtime_abi_version, Error, Runtime};
use std::path::PathBuf;
use std::process::ExitCode;

// ── Model metadata ────────────────────────────────────────────────────────────

/// Canonical input: two samples of 4 features each.
const CANONICAL_X_DATA: [f32; 8] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
const INPUT_SHAPE: &[usize] = &[2, 4];

/// Expected softmax output for the canonical input (tolerance ≤ 1e-4).
const EXPECTED_PROBS: [f32; 6] = [
    0.44481823, 0.3220363, 0.23314552,
    0.4567148,  0.31961477, 0.22367041,
];
const OUTPUT_SHAPE: &[usize] = &[2, 3];

/// DSP plan phase names (index = phase code from the ABI).
const PLAN_PHASE_NAMES: [&str; 4] = [
    "SLOT_BY_SLOT (warmup)",
    "SHAPES_FROZEN",
    "REPLAYING",
    "REPLAY_BLOCKED",
];

/// Backend names (index = backend code from the ABI).
const BACKEND_NAMES: [&str; 9] = [
    "AUTO", "SLOT_BY_SLOT", "CUDA_GRAPHS", "NVRTC",
    "PTX", "TRITON", "MLX", "ARM_HYBRID", "NNAPI",
];

fn phase_name(code: i32) -> &'static str {
    PLAN_PHASE_NAMES.get(code as usize).copied().unwrap_or("? (unknown)")
}

fn backend_name(code: i32) -> &'static str {
    BACKEND_NAMES.get(code as usize).copied().unwrap_or("? (unknown)")
}

// ── Weight helpers ────────────────────────────────────────────────────────────

/// Return a linearly-spaced `Vec<f32>` of length `n` from `start` to `end`.
fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| start + (end - start) * i as f32 / (n - 1) as f32)
        .collect()
}

/// Canonical weight tensors for `models/mlp.sdz` (deterministic linspace init).
///
/// In a production deployment these come from the model bundle; this example
/// supplies them externally to keep the fixture self-contained.
fn weight_tensor(name: &str) -> Option<ArrayD<f32>> {
    let (data, shape): (Vec<f32>, &[usize]) = match name {
        "w1" => (linspace(-1.0,  1.0, 32), &[4, 8]),
        "b1" => (linspace( 0.0,  0.7,  8), &[8]),
        "w2" => (linspace( 1.0, -1.0, 24), &[8, 3]),
        "b2" => (linspace(-0.1,  0.1,  3), &[3]),
        _    => return None,
    };
    Some(Array::from_shape_vec(IxDyn(shape), data).expect("linspace shape is correct"))
}

// ── Inference helpers ─────────────────────────────────────────────────────────

/// Run one inference step and verify the output.
///
/// `scale` multiplies the canonical input so values change between warmup runs
/// while the shape stays fixed. `scale == 1.0` gives the canonical expected
/// values used for numerical verification.
fn run_step(
    ctx: &sdx_runtime::Context,
    plan_names: &[String],
    scale: f32,
    step: usize,
) -> Result<(), String> {
    // Build the input map: one entry per plan external, resolved by name.
    let x_data: Vec<f32> = CANONICAL_X_DATA.iter().map(|v| v * scale).collect();
    let x: ArrayD<f32> = Array::from_shape_vec(IxDyn(INPUT_SHAPE), x_data)
        .expect("canonical input shape is correct");

    // Collect named inputs for all plan externals.
    let weights: Vec<(&str, ArrayD<f32>)> = plan_names
        .iter()
        .filter(|n| n.as_str() != "x")
        .filter_map(|n| weight_tensor(n).map(|t| (n.as_str(), t)))
        .collect();

    // Build the slice of (&str, ArrayViewD) that run_named_shaped expects.
    let mut named: Vec<(&str, ndarray::ArrayViewD<f32>)> =
        weights.iter().map(|(n, t)| (*n, t.view())).collect();
    named.push(("x", x.view()));

    // Run — output shape is statically known from the model spec.
    let outputs = ctx
        .run_named_shaped(&named, &[OUTPUT_SHAPE])
        .map_err(|e| format!("step {step}: {e}"))?;

    let probs = &outputs[0];

    // Verify: every row should sum to 1.0 (softmax invariant).
    let rows_ok = probs
        .rows()
        .into_iter()
        .all(|row| (row.sum() - 1.0).abs() <= 1e-5);

    // On the canonical step, also verify numerical match.
    let mut checks = format!("rows sum to 1: {rows_ok}");
    let mut ok = rows_ok;

    if (scale - 1.0).abs() < f32::EPSILON {
        let flat: Vec<f32> = probs.iter().copied().collect();
        let max_diff = flat
            .iter()
            .zip(EXPECTED_PROBS.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        let canon_ok = max_diff <= 1e-4;
        ok = ok && canon_ok;
        checks.push_str(&format!(
            "  canonical match: {canon_ok} (max_diff={max_diff:.2e})"
        ));
    }

    println!(
        "  step {:>2}: phase={:<22} exec_count={:>3}  {}",
        step,
        phase_name(ctx.plan_phase()),
        ctx.execution_count(),
        checks,
    );

    if ok { Ok(()) } else { Err(format!("step {step}: output verification failed")) }
}

// ── Entry points ─────────────────────────────────────────────────────────────

fn main() -> ExitCode {
    match run() {
        Ok(()) => {
            println!("\nSUCCESS: SDX outputs verified from pure Rust (no JVM).");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("\nFAILURE: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let model_path: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/mlp.sdz"));

    if !model_path.exists() {
        return Err(format!(
            "model not found: {}\n  generate it with the java-end-to-end GenerateExampleModel tool",
            model_path.display()
        ));
    }

    // ── Step 1: create runtime and load model ─────────────────────────────────
    println!("== Step 1: create runtime, load model ==");
    println!("SDX ABI version: {}", runtime_abi_version());

    let runtime = Runtime::new().map_err(|e| e.to_string())?;
    let model   = runtime.load(&model_path).map_err(|e| e.to_string())?;
    let ctx     = model.context(&["probs"]).map_err(|e| e.to_string())?;

    println!("Loaded: {}", model_path.display());

    // ── Step 2: discover the plan's input contract ────────────────────────────
    println!("\n== Step 2: input-contract discovery ==");
    let plan_names = ctx.input_names();
    println!(
        "Plan: {} external inputs, {} output(s)",
        ctx.num_inputs(),
        ctx.num_outputs()
    );
    for (i, name) in plan_names.iter().enumerate() {
        println!("  input[{i}] = \"{name}\"");
    }

    // Mark `x` as a placeholder (value and potentially shape may vary per run).
    for (i, name) in plan_names.iter().enumerate() {
        if name == "x" {
            ctx.mark_input_placeholder(i as i32)
                .map_err(|e| e.to_string())?;
        }
    }

    // ── Step 3: warmup runs (SLOT_BY_SLOT phase) ──────────────────────────────
    println!("\n== Step 3: warmup runs ==");
    for step in 1..=3usize {
        run_step(&ctx, &plan_names, step as f32, step)?;
    }

    // ── Step 4: freeze → DSP replay fast path ────────────────────────────────
    println!("\n== Step 4: freeze_shapes() → DSP replay fast path ==");
    ctx.freeze_shapes().map_err(|e| e.to_string())?;
    println!("Phase after freeze: {}", phase_name(ctx.plan_phase()));

    for step in 4..=6usize {
        run_step(&ctx, &plan_names, step as f32, step)?;
    }

    // ── Step 5: execution-report telemetry ───────────────────────────────────
    println!("\n== Step 5: execution report ==");
    let r = ctx.execution_report().map_err(|e| e.to_string())?;
    let fallback = match r.used_fallback {
        f if f < 0 => "unknown",
        0 => "no",
        _ => "yes",
    };
    println!("  status_code       = {}", r.status_code);
    println!("  requested_backend = {}", backend_name(r.requested_backend));
    println!("  applied_backend   = {}", backend_name(r.applied_backend));
    println!("  used_fallback     = {fallback}");
    println!("  plan_phase        = {}", phase_name(r.plan_phase));
    println!("  execution_count   = {}", r.execution_count);
    println!("  execution_time    = {:.3} ms", r.execution_time_ns as f64 / 1.0e6);

    // ── Step 6: typed error path ──────────────────────────────────────────────
    println!("\n== Step 6: error handling ==");
    match runtime.load("/definitely/not/a/model.sdz") {
        Ok(_)  => return Err("unexpected: bogus load succeeded".into()),
        Err(Error::ModelLoad { path, status, .. }) =>
            println!("  Loading '{path}' raised ModelLoad (status={status}) — as expected."),
        Err(e) => println!("  Error: {e}"),
    }

    Ok(())
}
