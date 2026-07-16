/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/**
 * End-to-end SDX runtime walkthrough from TypeScript / Node.js — no JVM.
 *
 * Loads `../../models/mlp.sdz` through the SDX C ABI and demonstrates the
 * full SDK lifecycle using the idiomatic {@link SdxRuntime}/{@link SdxModel}/
 * {@link SdxContext} wrappers:
 *
 *  1. Runtime and model creation (like onnxruntime-node's `InferenceSession.create`).
 *  2. Input-contract discovery via `inputNames()`.
 *  3. Named-tensor `run()` — pass `Record<string, Tensor>`, receive named outputs.
 *  4. Warmup, `freezeShapes()`, and the DSP graph-replay fast path.
 *  5. Typed `ExecutionReport` telemetry.
 *  6. Canonical output verification (≤1e-4 max-abs error).
 *  7. Error path demonstration.
 *
 * Run:
 * ```bash
 * # Explicit library path (most common in CI and dev):
 * SDX_RUNTIME_LIBRARY=/path/to/libsdx_cpu.so npm start
 * # Unpacked SDK:
 * SDX_RUNTIME_HOME=/path/to/unpacked-sdk npm start
 * # Optional: supply a different .sdz:
 * npm start -- /path/to/other.sdz
 * ```
 */

import * as fs   from 'fs';
import * as path from 'path';
import {
  SdxRuntime,
  SdxModel,
  SdxContext,
  Tensor,
  TensorMap,
  ExecutionReport,
  phaseName,
  backendName,
} from './sdx';

// ── Fixtures ──────────────────────────────────────────────────────────────────

/** Canonical input for mlp.sdz: x[2,4] = [0.1, 0.2, ..., 0.8] */
const CANONICAL_X: Tensor = {
  data: Float32Array.from([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
  dims: [2, 4],
};

/** Expected softmax output for CANONICAL_X (tolerance ≤1e-4). */
const EXPECTED_PROBS = Float32Array.from([
  0.44481823, 0.3220363, 0.23314552,
  0.4567148,  0.31961477, 0.22367041,
]);

function linspace(start: number, end: number, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = start + ((end - start) * i) / (n - 1);
  return out;
}

/**
 * Model weights — in a real deployment these come from the model provider.
 * Here they are deterministic linspace initializers matching GenerateExampleModel.
 */
const WEIGHTS: Readonly<Record<string, Tensor>> = {
  w1: { data: linspace(-1.0, 1.0, 32), dims: [4, 8] },
  b1: { data: linspace(0.0, 0.7,  8),  dims: [8]    },
  w2: { data: linspace(1.0, -1.0, 24), dims: [8, 3] },
  b2: { data: linspace(-0.1, 0.1, 3),  dims: [3]    },
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function printReport(report: ExecutionReport): void {
  const fallback = report.usedFallback ? 'yes' : 'no';
  console.log(`  status_code       = ${report.statusCode}`);
  console.log(`  requested_backend = ${backendName(report.requestedBackend)}`);
  console.log(`  applied_backend   = ${backendName(report.appliedBackend)}`);
  console.log(`  used_fallback     = ${fallback}`);
  console.log(`  plan_phase        = ${phaseName(report.planPhase)}`);
  console.log(`  execution_count   = ${report.executionCount}`);
  console.log(`  execution_time    = ${(Number(report.executionTimeNs) / 1e6).toFixed(3)} ms`);
}

/** Builds the full input feed for one run, scaling x by `step`. */
function buildFeeds(step: number): TensorMap {
  const xData = CANONICAL_X.data.map((v) => v * step);
  return {
    ...WEIGHTS,
    x: { data: xData, dims: CANONICAL_X.dims },
  };
}

/** Output buffer large enough for probs[2,3]. */
function makeOutputBuffers(): TensorMap {
  return { probs: { data: new Float32Array(6), dims: [2, 3] } };
}

// ── Walkthrough ───────────────────────────────────────────────────────────────

function runStep(
  session: SdxContext,
  step: number,
  checkCanonical: boolean,
): boolean {
  const feeds   = buildFeeds(step);
  const buffers = makeOutputBuffers();
  const result  = session.run(feeds, buffers);
  const probs   = result['probs'].data;

  // Row sums must be 1 (softmax invariant).
  const row0Sum = probs[0] + probs[1] + probs[2];
  const row1Sum = probs[3] + probs[4] + probs[5];
  const rowSumsOk = Math.abs(row0Sum - 1) <= 1e-5 && Math.abs(row1Sum - 1) <= 1e-5;

  let ok     = rowSumsOk;
  let detail = `rows sum to 1: ${rowSumsOk}`;

  if (checkCanonical) {
    let maxDiff = 0;
    for (let i = 0; i < EXPECTED_PROBS.length; i++)
      maxDiff = Math.max(maxDiff, Math.abs(probs[i] - EXPECTED_PROBS[i]));
    const matches = maxDiff <= 1e-4;
    ok      = ok && matches;
    detail += `; matches canonical: ${matches} (maxDiff=${maxDiff.toExponential(2)})`;
  }

  const phase = phaseName(session.planPhase()).padEnd(22);
  console.log(`  run ${step}: phase=${phase} execCount=${session.executionCount()}  ${detail}`);
  return ok;
}

function main(): number {
  const modelPath = process.argv[2]
    ?? path.resolve(__dirname, '../../../models/mlp.sdz');
  if (!fs.existsSync(modelPath)) {
    console.error(
      `Model not found: ${modelPath}\n` +
      'Generate it with the Java GenerateExampleModel tool.');
    return 2;
  }

  // ── Step 1: create the runtime and load the model ──────────────────────────
  console.log(`== Step 1: load runtime and bundle (${path.basename(modelPath)}) ==`);
  const runtime = SdxRuntime.create();
  console.log(`Runtime library : ${runtime.libraryPath}`);
  console.log(`ABI version     : ${runtime.abiVersion}`);

  let exitCode = 1;
  let model: SdxModel | null    = null;
  let session: SdxContext | null = null;

  try {
    model   = runtime.loadBundle(modelPath);
    session = model.createContext(['probs']);

    // ── Step 2: discover the input contract ───────────────────────────────────
    console.log("\n== Step 2: discover the plan's input contract ==");
    const names = session.inputNames();
    console.log(`Plan expects ${names.length} external inputs, ${session.numOutputs()} output(s):`);
    names.forEach((name, i) => console.log(`  input[${i}] = "${name}"`));

    // Mark 'x' as the placeholder (changes every run); weights are variables.
    const xIndex = names.indexOf('x');
    if (xIndex < 0) throw new Error('"x" not found in input names');
    session.markInputPlaceholder(xIndex);

    // ── Step 3: warmup runs ───────────────────────────────────────────────────
    console.log('\n== Step 3: warmup runs ==');
    for (let step = 1; step <= 3; step++) {
      // step=1 uses the canonical vector (scale factor = 1) — verify it.
      if (!runStep(session, step, step === 1)) return 1;
    }

    // ── Step 4: freeze shapes → DSP replay fast path ─────────────────────────
    console.log('\n== Step 4: sdxFreezeShapes -> DSP replay fast path ==');
    session.freezeShapes();
    console.log(`Plan phase after freeze: ${phaseName(session.planPhase())}`);
    for (let step = 4; step <= 6; step++) {
      if (!runStep(session, step, false)) return 1;
    }

    // ── Step 5: execution report ──────────────────────────────────────────────
    console.log('\n== Step 5: execution report ==');
    printReport(session.executionReport());
    exitCode = 0;

  } finally {
    session?.dispose();
    model?.dispose();
  }

  // ── Step 6: error path ────────────────────────────────────────────────────
  // Model and context are already closed; use runtime directly to verify the
  // error-reporting path.
  console.log('\n== Step 6: error path ==');
  try {
    runtime.loadBundle('/definitely/not/a/model.sdz');
  } catch (err) {
    console.log(`Loading bogus path throws: ${(err as Error).message}`);
  }

  runtime.dispose();

  if (exitCode === 0) {
    console.log('\nSUCCESS: SDX C ABI outputs verified from TypeScript/Node.js (no JVM).');
  } else {
    console.error('\nFAILURE: output verification failed.');
  }
  return exitCode;
}

process.exit(main());
