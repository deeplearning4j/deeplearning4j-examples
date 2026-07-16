/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/**
 * A pure-JavaScript REFERENCE IMPLEMENTATION of the SDX C ABI over a
 * simulated Emscripten linear memory.
 *
 * Purpose: verify the wrapper's heap marshaling end-to-end without the real
 * `sdx_runtime.wasm`. It decodes the `sdx_tensor_view_t` structs the wrapper
 * writes into the (simulated) heap at the exact wasm32 offsets, computes the
 * example MLP forward pass (`probs = softmax(relu(x·W1+b1)·W2+b2)`) from the
 * marshaled bytes, and writes the results back through the output views. If
 * any struct offset, pointer write, or byte length in the wrapper were wrong,
 * the example's canonical-output verification would fail.
 *
 * It intentionally implements the same observable semantics as the native
 * runtime: positional inputs over the discovered name order, caller-provided
 * output buffers, struct_size-capped report copies, `sdxGetLastError`
 * strings, and the plan-phase lifecycle (warmup → frozen → replaying).
 */

import type { SdxWasmModule } from './sdx-wasm';

const HEAP_SIZE = 32 * 1024 * 1024;

/** Fixed plan input order used by the reference plan (order is arbitrary by
 *  contract — clients must discover it via sdxGetInputName). */
const PLAN_INPUT_NAMES = ['b2', 'w2', 'w1', 'b1', 'x'] as const;

const EXPECTED_SHAPES: Record<string, readonly number[]> = {
  w1: [4, 8],
  b1: [8],
  w2: [8, 3],
  b2: [3],
  x: [2, 4],
};

export function createMockSdxModule(): SdxWasmModule {
  const buffer = new ArrayBuffer(HEAP_SIZE);
  const HEAPU8 = new Uint8Array(buffer);
  const u32 = new Uint32Array(buffer);
  const i32 = new Int32Array(buffer);
  const f32 = new Float32Array(buffer);
  const i64 = new BigInt64Array(buffer);
  const u64 = new BigUint64Array(buffer);

  // Bump allocator; 0 stays NULL.
  let brk = 1024;
  const malloc = (size: number): number => {
    const ptr = (brk + 7) & ~7;
    brk = ptr + Math.max(1, size);
    if (brk > HEAP_SIZE) {
      throw new Error('mock heap exhausted');
    }
    return ptr;
  };

  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  const writeUtf8 = (text: string, ptr: number, maxBytes: number): void => {
    const bytes = encoder.encode(text);
    const n = Math.min(bytes.length, maxBytes - 1);
    HEAPU8.set(bytes.subarray(0, n), ptr);
    HEAPU8[ptr + n] = 0;
  };

  const readUtf8 = (ptr: number): string => {
    let end = ptr;
    while (HEAPU8[end] !== 0) end++;
    return decoder.decode(HEAPU8.subarray(ptr, end));
  };

  const allocUtf8 = (text: string): number => {
    const bytes = encoder.encode(text);
    const ptr = malloc(bytes.length + 1);
    HEAPU8.set(bytes, ptr);
    HEAPU8[ptr + bytes.length] = 0;
    return ptr;
  };

  // ── Runtime state ──────────────────────────────────────────────────────────
  const files = new Map<string, Uint8Array>();
  let lastErrorPtr = allocUtf8('');
  const setError = (message: string): void => {
    lastErrorPtr = allocUtf8(message);
  };

  const inputNamePtrs = PLAN_INPUT_NAMES.map((n) => allocUtf8(n));

  const RUNTIME_HANDLE = 0x101;
  const MODEL_HANDLE = 0x202;
  const CONTEXT_HANDLE = 0x303;

  let modelLoaded = false;
  let planPhase = 0; // SLOT_BY_SLOT (warmup)
  let executionCount = 0;
  let postFreezeRuns = 0;
  let frozen = false;
  let lastRunNs = 0;

  // ── Tensor view decoding (must match the wrapper's wasm32 layout) ─────────
  interface DecodedView {
    dataPtr: number;
    dims: number[];
    byteLength: number;
    deviceType: number;
  }

  const decodeView = (structPtr: number): DecodedView => {
    const b = structPtr >> 2;
    const dataPtr = u32[b];
    const shapePtr = u32[b + 1];
    const rank = i32[b + 2];
    const dtype = i32[b + 3];
    const byteLength = u32[b + 4];
    const deviceType = i32[b + 5];
    if (dtype !== 5) {
      throw new Error(`mock: only FLOAT32 tensors supported (dtype=${dtype})`);
    }
    const dims: number[] = [];
    for (let i = 0; i < rank; i++) {
      dims.push(Number(i64[(shapePtr >> 3) + i]));
    }
    const elements = dims.reduce((a, d) => a * d, 1);
    if (byteLength !== elements * 4) {
      throw new Error(
        `mock: bytes field (${byteLength}) does not match shape [${dims}] (${elements * 4})`);
    }
    return { dataPtr, dims, byteLength, deviceType };
  };

  const readFloats = (view: DecodedView): Float32Array =>
    f32.slice(view.dataPtr >> 2, (view.dataPtr >> 2) + view.byteLength / 4);

  // ── The reference forward pass (independent JS implementation) ────────────
  const forward = (t: Record<string, Float32Array>): Float32Array => {
    const [batch] = [2];
    const hidden = new Float32Array(batch * 8);
    for (let r = 0; r < batch; r++) {
      for (let c = 0; c < 8; c++) {
        let sum = t.b1[c];
        for (let k = 0; k < 4; k++) {
          sum += t.x[r * 4 + k] * t.w1[k * 8 + c];
        }
        hidden[r * 8 + c] = Math.max(0, sum);
      }
    }
    const probs = new Float32Array(batch * 3);
    for (let r = 0; r < batch; r++) {
      const logits = new Float32Array(3);
      let maxLogit = -Infinity;
      for (let c = 0; c < 3; c++) {
        let sum = t.b2[c];
        for (let k = 0; k < 8; k++) {
          sum += hidden[r * 8 + k] * t.w2[k * 3 + c];
        }
        logits[c] = sum;
        maxLogit = Math.max(maxLogit, sum);
      }
      let denom = 0;
      for (let c = 0; c < 3; c++) {
        logits[c] = Math.exp(logits[c] - maxLogit);
        denom += logits[c];
      }
      for (let c = 0; c < 3; c++) {
        probs[r * 3 + c] = logits[c] / denom;
      }
    }
    return probs;
  };

  // ── The module object ──────────────────────────────────────────────────────
  return {
    HEAPU8,

    _malloc: malloc,
    _free: (_ptr: number): void => {
      // bump allocator: no-op
    },

    UTF8ToString: readUtf8,
    stringToUTF8: writeUtf8,
    lengthBytesUTF8: (s: string): number => encoder.encode(s).length,

    FS: {
      writeFile: (path: string, data: Uint8Array): void => {
        files.set(path, data);
      },
    },

    _sdxGetRuntimeAbiVersion: () => 1,

    _sdxCreateRuntime: (optionsPtr: number, outPtr: number): number => {
      if (outPtr === 0) return 1; // SDX_STATUS_INVALID_ARGUMENT
      if (optionsPtr !== 0 && u32[optionsPtr >> 2] !== 0 && u32[optionsPtr >> 2] < 4) {
        return 2; // SDX_STATUS_INCOMPATIBLE_ABI
      }
      u32[outPtr >> 2] = RUNTIME_HANDLE;
      return 0;
    },
    _sdxDestroyRuntime: (): void => {},

    _sdxLoadBundle: (runtime: number, pathPtr: number, _optionsPtr: number,
                     outPtr: number): number => {
      if (runtime !== RUNTIME_HANDLE || pathPtr === 0 || outPtr === 0) return 1;
      const path = readUtf8(pathPtr);
      const bytes = files.get(path);
      if (bytes === undefined) {
        setError(`Bundle path does not exist: ${path}`);
        return 6; // SDX_STATUS_IO_ERROR
      }
      // .sdz bundles are ZIP archives: verify the local-file-header magic
      // actually made it through MEMFS intact.
      if (bytes.length < 4 || bytes[0] !== 0x50 || bytes[1] !== 0x4b) {
        setError(`Not a ZIP (.sdz) archive: ${path}`);
        return 3; // SDX_STATUS_MODEL_LOAD_FAILED
      }
      modelLoaded = true;
      u32[outPtr >> 2] = MODEL_HANDLE;
      return 0;
    },
    _sdxUnloadModel: (): void => {},

    _sdxCreateContext: (model: number, namesPtr: number, numNames: number,
                        outPtr: number): number => {
      if (model !== MODEL_HANDLE || !modelLoaded || outPtr === 0) return 1;
      // Validate the requested-output marshaling (array of char*).
      for (let i = 0; i < numNames; i++) {
        const p = u32[(namesPtr >> 2) + i];
        if (readUtf8(p).length === 0) {
          setError('empty requested output name');
          return 1;
        }
      }
      u32[outPtr >> 2] = CONTEXT_HANDLE;
      return 0;
    },
    _sdxDestroyContext: (): void => {},

    _sdxGetNumInputs: (ctx: number) => (ctx === CONTEXT_HANDLE ? PLAN_INPUT_NAMES.length : -1),
    _sdxGetNumOutputs: (ctx: number) => (ctx === CONTEXT_HANDLE ? 1 : -1),
    _sdxGetInputName: (ctx: number, index: number): number =>
      ctx === CONTEXT_HANDLE && index >= 0 && index < inputNamePtrs.length
        ? inputNamePtrs[index] : 0,

    _sdxMarkInputVariable: (ctx: number, index: number): number =>
      ctx === CONTEXT_HANDLE && index >= 0 && index < PLAN_INPUT_NAMES.length ? 0 : 1,
    _sdxMarkInputPlaceholder: (ctx: number, index: number): number =>
      ctx === CONTEXT_HANDLE && index >= 0 && index < PLAN_INPUT_NAMES.length ? 0 : 1,

    _sdxFreezeShapes: (ctx: number): number => {
      if (ctx !== CONTEXT_HANDLE) return 1;
      frozen = true;
      planPhase = 1; // SHAPES_FROZEN
      return 0;
    },
    _sdxGetPlanPhase: (ctx: number) => (ctx === CONTEXT_HANDLE ? planPhase : -1),
    _sdxGetExecutionCount: (ctx: number) => (ctx === CONTEXT_HANDLE ? executionCount : -1),

    _sdxRun: (ctx: number, inputsPtr: number, numInputs: number,
              outputsPtr: number, numOutputs: number, _optionsPtr: number): number => {
      if (ctx !== CONTEXT_HANDLE) return 1;
      if (numInputs !== PLAN_INPUT_NAMES.length || numOutputs !== 1) {
        setError(`Input tensor count mismatch (got ${numInputs}, expected ${PLAN_INPUT_NAMES.length})`);
        return 4; // SDX_STATUS_EXECUTION_FAILED
      }
      const started = Date.now();

      const tensors: Record<string, Float32Array> = {};
      for (let index = 0; index < numInputs; index++) {
        const view = decodeView(inputsPtr + index * 28);
        const name = PLAN_INPUT_NAMES[index];
        const expected = EXPECTED_SHAPES[name];
        if (view.dims.length !== expected.length ||
            view.dims.some((d, i) => d !== expected[i])) {
          setError(`shape mismatch for '${name}': [${view.dims}] vs [${expected}]`);
          return 4;
        }
        tensors[name] = readFloats(view);
      }

      const probs = forward(tensors);
      const outView = decodeView(outputsPtr);
      if (outView.byteLength !== probs.byteLength) {
        setError(`output buffer size mismatch: ${outView.byteLength} vs ${probs.byteLength}`);
        return 4;
      }
      f32.set(probs, outView.dataPtr >> 2);

      executionCount++;
      if (frozen) {
        postFreezeRuns++;
        if (postFreezeRuns >= 2) {
          planPhase = 2; // REPLAYING
        }
      }
      lastRunNs = Math.max(1, Date.now() - started) * 1_000_000;
      return 0;
    },

    _sdxGetLastError: () => lastErrorPtr,

    _sdxGetExecutionReport: (ctx: number, reportPtr: number): number => {
      if (ctx !== CONTEXT_HANDLE || reportPtr === 0) return 1;
      // Mirror the C semantics: honor the caller's struct_size (copy cap).
      const callerSize = u32[reportPtr >> 2];
      const size = callerSize === 0 || callerSize > 48 ? 48 : callerSize;
      const scratch = malloc(48);
      const b = scratch >> 2;
      u32[b] = 48;
      i32[b + 1] = 0;                    // requested_backend AUTO
      i32[b + 2] = 0;                    // applied_backend AUTO
      i32[b + 3] = 0;                    // status_code OK
      i32[b + 4] = 0;                    // used_fallback: no
      u64[(scratch + 24) >> 3] = BigInt(lastRunNs);
      i32[b + 8] = 0;                    // requested_gpu_target
      i32[b + 9] = 0;                    // applied_gpu_target
      i32[b + 10] = planPhase;
      i32[b + 11] = executionCount;
      HEAPU8.copyWithin(reportPtr, scratch, scratch + size);
      return 0;
    },
  };
}
