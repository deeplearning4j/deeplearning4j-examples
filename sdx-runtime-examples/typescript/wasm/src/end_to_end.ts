/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/**
 * End-to-end walkthrough of the SDX runtime on WebAssembly.
 *
 * Same lifecycle as every other language example — input-contract discovery,
 * placeholder marking, warmup, freeze → replay, execution-report telemetry,
 * canonical output verification, and the error path — through the
 * onnxruntime-web-style wrapper in ./sdx-wasm.
 *
 * Module resolution:
 *  - default: the Emscripten build output `../sdx_runtime_wasm.js`
 *    (produced by ./build-wasm.sh once libnd4j has an Emscripten port);
 *  - `--mock`: the pure-JS reference implementation of the C ABI
 *    (./mock-module), which computes the real MLP math from the marshaled
 *    heap bytes — this verifies every struct offset and pointer the wrapper
 *    writes, and runs anywhere Node runs.
 */

import * as fs from 'fs';
import * as path from 'path';
import {
  backendName,
  phaseName,
  SdxContext,
  SdxRuntime,
  SdxWasmModule,
  Tensor,
  TensorMap,
} from './sdx-wasm';

/** Canonical verification vector for models/mlp.sdz (printed by the Java
 *  GenerateExampleModel tool that produced the fixture). */
const CANONICAL_X = Float32Array.from([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
const EXPECTED_PROBS = Float32Array.from([
  0.44481823, 0.3220363, 0.23314552, 0.4567148, 0.31961477, 0.22367041]);

function linspace(start: number, end: number, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = start + ((end - start) * i) / (n - 1);
  return out;
}

/** The model's weights travel INSIDE the .sdz — a generic client obtains them
 *  from the model provider. This example ships the deterministic linspace
 *  initializers next to the fixture for simplicity. */
const WEIGHTS: TensorMap = {
  w1: { data: linspace(-1.0, 1.0, 32), dims: [4, 8] },
  b1: { data: linspace(0.0, 0.7, 8), dims: [8] },
  w2: { data: linspace(1.0, -1.0, 24), dims: [8, 3] },
  b2: { data: linspace(-0.1, 0.1, 3), dims: [3] },
};

async function instantiateModule(useMock: boolean): Promise<SdxWasmModule> {
  if (useMock) {
    console.log('[module] Using the pure-JS REFERENCE ABI (marshaling verification mode).');
    console.log('[module] Build the real runtime with ./build-wasm.sh and rerun without --mock.\n');
    const { createMockSdxModule } = await import('./mock-module');
    return createMockSdxModule();
  }
  const gluePath = path.resolve(__dirname, '../sdx_runtime_wasm.js');
  if (!fs.existsSync(gluePath)) {
    throw new Error(
      `Emscripten build output not found: ${gluePath}\n` +
      'Build it with ./build-wasm.sh, or run the marshaling verification ' +
      'against the JS reference ABI with: npm run start:mock');
  }
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const createSdxModule = require(gluePath) as () => Promise<SdxWasmModule>;
  return createSdxModule();
}

function runOnceAndVerify(context: SdxContext, inputNames: string[], step: number): boolean {
  // Scale the canonical input per step so values change while the shape stays
  // fixed; step 1 uses the canonical vector so the baked expectation applies.
  const x: Tensor = { data: CANONICAL_X.map((v) => v * step), dims: [2, 4] };
  const feeds: TensorMap = { ...WEIGHTS, x };

  const started = process.hrtime.bigint();
  const [probs] = context.run(feeds, [[2, 3]]);
  const tookUs = Number((process.hrtime.bigint() - started) / 1000n);

  const p = probs.data;
  const rowSumsOk =
    Math.abs(p[0] + p[1] + p[2] - 1) <= 1e-5 &&
    Math.abs(p[3] + p[4] + p[5] - 1) <= 1e-5;
  let ok = rowSumsOk;
  let checks = `rows sum to 1: ${rowSumsOk}`;
  if (step === 1) {
    let maxDiff = 0;
    for (let i = 0; i < EXPECTED_PROBS.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(p[i] - EXPECTED_PROBS[i]));
    }
    const matches = maxDiff <= 1e-4;
    ok = ok && matches;
    checks += `; matches canonical expectation: ${matches} (maxDiff=${maxDiff.toExponential(2)})`;
  }
  const phase = phaseName(context.planPhase()).padEnd(22);
  console.log(`  run ${step}: phase=${phase} execCount=${context.executionCount()}  ${tookUs} us  ${checks}`);
  return ok;
}

async function main(): Promise<number> {
  const args = process.argv.slice(2);
  const useMock = args.includes('--mock');
  const modelPath = args.find((a) => !a.startsWith('--'))
    ?? path.resolve(__dirname, '../../../models/mlp.sdz');
  if (!fs.existsSync(modelPath)) {
    console.error(`Model not found: ${modelPath} — generate it with the ` +
      'java-end-to-end GenerateExampleModel tool.');
    return 2;
  }

  console.log(`== Step 1: instantiate the wasm module and load ${path.basename(modelPath)} ==`);
  const module = await instantiateModule(useMock);
  const runtime = SdxRuntime.create(module);
  console.log(`SDX runtime ABI version: ${runtime.abiVersion()}`);

  let exitCode = 1;
  try {
    const modelBytes = fs.readFileSync(modelPath);
    const model = runtime.loadBundle('mlp.sdz', new Uint8Array(modelBytes));
    try {
      const context = model.createContext(['probs']);
      try {
        console.log("\n== Step 2: discover the plan's input contract ==");
        const inputNames = context.inputNames();
        console.log(`Plan expects ${inputNames.length} external inputs, ` +
          `${context.numOutputs()} outputs:`);
        inputNames.forEach((name, i) => console.log(`  input[${i}] = "${name}"`));
        context.markInputPlaceholder('x');

        console.log('\n== Step 3: warmup runs ==');
        for (let step = 1; step <= 3; step++) {
          if (!runOnceAndVerify(context, inputNames, step)) return 1;
        }

        console.log('\n== Step 4: freezeShapes -> DSP replay fast path ==');
        context.freezeShapes();
        console.log(`Plan phase after freeze: ${phaseName(context.planPhase())}`);
        for (let step = 4; step <= 6; step++) {
          if (!runOnceAndVerify(context, inputNames, step)) return 1;
        }

        console.log('\n== Step 5: execution report ==');
        const report = context.executionReport();
        const fallback = report.usedFallback < 0
          ? 'unknown' : report.usedFallback === 1 ? 'yes' : 'no';
        console.log(`  status_code       = ${report.statusCode}`);
        console.log(`  requested_backend = ${backendName(report.requestedBackend)}`);
        console.log(`  applied_backend   = ${backendName(report.appliedBackend)}`);
        console.log(`  used_fallback     = ${fallback}`);
        console.log(`  plan_phase        = ${phaseName(report.planPhase)}`);
        console.log(`  execution_count   = ${report.executionCount}`);
        console.log(`  execution_time    = ${(report.executionTimeNs / 1e6).toFixed(3)} ms`);
        exitCode = 0;
      } finally {
        context.dispose();
      }
    } finally {
      model.dispose();
    }

    console.log('\n== Step 6: error handling ==');
    try {
      // Attempt to load a bundle name that was never written into MEMFS.
      const heapPath = '/definitely-not-a-model.sdz';
      const module2 = runtime.module;
      const pathBytes = module2.lengthBytesUTF8(heapPath) + 1;
      const pathPtr = module2._malloc(pathBytes);
      module2.stringToUTF8(heapPath, pathPtr, pathBytes);
      const outPtr = module2._malloc(4);
      const status = module2._sdxLoadBundle(runtime.rawHandle, pathPtr, 0, outPtr);
      module2._free(outPtr);
      module2._free(pathPtr);
      console.log(`Loading a bogus path -> status=${status}, ` +
        `sdxGetLastError="${runtime.lastError()}"`);
    } catch (e) {
      console.log(`Loading a bogus path threw: ${(e as Error).message}`);
    }
  } finally {
    runtime.dispose();
  }

  console.log(exitCode === 0
    ? `\nSUCCESS: SDX C ABI outputs verified on ${useMock ? 'the JS reference ABI' : 'WebAssembly'}.`
    : '\nFAILURE: output verification failed.');
  return exitCode;
}

main().then((code) => process.exit(code));
