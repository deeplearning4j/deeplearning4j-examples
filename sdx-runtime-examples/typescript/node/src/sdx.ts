/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/**
 * SDX runtime wrapper for Node.js — idiomatic TypeScript classes following the
 * onnxruntime-node naming and lifecycle conventions.
 *
 * API overview:
 *
 * ```ts
 * // 1. Load the runtime (binds the C ABI via koffi):
 * const runtime = SdxRuntime.create('/path/to/libsdx_cpu.so');
 *
 * // 2. Load a .sdz bundle and create an inference context:
 * const model   = runtime.loadBundle('/path/to/model.sdz');
 * const session = model.createContext(['probs']);
 *
 * // 3. Discover the positional input contract and run with NAMED inputs:
 * const names = session.inputNames();          // ['w1', 'b1', 'w2', 'b2', 'x']
 * const result = session.run({
 *   w1: { data: w1Data, dims: [4, 8] },
 *   x:  { data: xData,  dims: [2, 4] },
 *   // ... remaining names ...
 * });
 * const probs = result['probs'].data;          // Float32Array
 *
 * // 4. Lifecycle — explicit dispose() or `using` (Node 22+):
 * session.dispose();
 * model.dispose();
 * runtime.dispose();
 * ```
 *
 * All three objects implement `Disposable` (Symbol.dispose) so they work with
 * the TC39 explicit resource management `using` keyword on Node 22+.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import koffi from 'koffi';

// ── Constants ─────────────────────────────────────────────────────────────────

/** sdx_status_t OK */
const SDX_STATUS_OK = 0;
/** sd::DataType::FLOAT32 (value 5 in the C ABI) */
const SDX_DTYPE_FLOAT = 5;
/** Device type: host / CPU memory */
const SDX_DEVICE_HOST = 0;

export const PLAN_PHASE_NAMES: ReadonlyArray<string> = [
  'SLOT_BY_SLOT',
  'SHAPES_FROZEN',
  'REPLAYING',
  'REPLAY_BLOCKED',
];

export const BACKEND_NAMES: ReadonlyArray<string> = [
  'AUTO', 'SLOT_BY_SLOT', 'CUDA_GRAPHS', 'NVRTC', 'PTX', 'TRITON',
  'MLX', 'ARM_HYBRID', 'NNAPI', 'HIP_GRAPHS', 'LEVEL_ZERO', 'VULKAN',
  'METAL', 'TPU', 'HEXAGON',
];

export const phaseName   = (p: number): string => PLAN_PHASE_NAMES[p]  ?? `UNKNOWN(${p})`;
export const backendName = (b: number): string => BACKEND_NAMES[b]     ?? `UNKNOWN(${b})`;

// ── Public types ──────────────────────────────────────────────────────────────

/**
 * A host-side tensor carrying a typed data buffer and its shape.
 *
 * Follows the onnxruntime-node convention of keeping data and dims together —
 * all bulk data MUST be Float32Array, never number[].
 */
export interface Tensor {
  /** Raw float values. Float32Array maps directly to the C `float*` ABI. */
  data: Float32Array;
  /**
   * Shape in elements, e.g. [2, 4] for a (2×4) matrix.
   * number[] (not bigint[]) because dims are a metadata concern — the C ABI
   * receives them as BigInt64Array internally.
   */
  dims: number[];
}

/**
 * Named-tensor map: the caller passes inputs and receives outputs by name,
 * mirroring the onnxruntime-node `session.run(feeds)` convention.
 */
export type TensorMap = Record<string, Tensor>;

/**
 * Typed execution telemetry returned by {@link SdxContext.executionReport}.
 */
export interface ExecutionReport {
  /** sdx_status_t from the last run */
  statusCode: number;
  /** Backend the caller requested (see BACKEND_NAMES) */
  requestedBackend: number;
  /** Backend the runtime actually used */
  appliedBackend: number;
  /** true if the runtime fell back from the requested backend */
  usedFallback: boolean;
  /** DSP plan phase (see PLAN_PHASE_NAMES) */
  planPhase: number;
  /** Number of times this context has been executed */
  executionCount: number;
  /** Wall time of the last run in nanoseconds */
  executionTimeNs: bigint;
}

/**
 * Options forwarded to `sdxLoadBundle`.
 */
export interface ModelOptions {
  /** Backend preference (default: AUTO = 0) */
  backend?: number;
  /** If true, fail instead of falling back when the backend is unavailable */
  strictBackend?: boolean;
  /** Whether to allow runtime JIT compilation (Triton/NVRTC) */
  allowRuntimeJit?: boolean;
  /** GPU device index to target (-1 = auto) */
  gpuTarget?: number;
}

/**
 * Options forwarded to `sdxRun`.
 */
export interface RunOptions {
  /** Backend for this specific run (default: AUTO = 0) */
  backend?: number;
  /**
   * When true, sdxRun will fail if the input signature does not exactly match
   * the plan contract. Useful during development.
   */
  strictSignature?: boolean;
  /** GPU device index for this run (-1 = auto) */
  gpuTarget?: number;
}

// ── C ABI struct declarations ─────────────────────────────────────────────────

// Structs are registered once at module load (koffi caches by name).
koffi.struct('sdx_runtime_options_t', { struct_size: 'uint32_t' });
koffi.struct('sdx_model_options_t', {
  struct_size:       'uint32_t',
  backend:           'int32_t',
  strict_backend:    'int32_t',
  allow_runtime_jit: 'int32_t',
  gpu_target:        'int32_t',
});
koffi.struct('sdx_run_options_t', {
  struct_size:      'uint32_t',
  backend:          'int32_t',
  strict_signature: 'int32_t',
  gpu_target:       'int32_t',
});
koffi.struct('sdx_tensor_view_t', {
  data:        'void *',
  shape:       'const int64_t *',
  rank:        'int32_t',
  dtype:       'int32_t',
  bytes:       'size_t',
  device_type: 'int32_t',
  device_id:   'int32_t',
});
koffi.struct('sdx_execution_report_t', {
  struct_size:         'uint32_t',
  requested_backend:   'int32_t',
  applied_backend:     'int32_t',
  status_code:         'int32_t',
  used_fallback:       'int32_t',
  execution_time_ns:   'uint64_t',
  requested_gpu_target:'int32_t',
  applied_gpu_target:  'int32_t',
  plan_phase:          'int32_t',
  execution_count:     'int32_t',
});

// ── Internal ABI binding ──────────────────────────────────────────────────────

/** Raw koffi function table — internal, not exported. */
interface SdxAbi {
  sdxGetRuntimeAbiVersion: koffi.KoffiFunction;
  sdxCreateRuntime:        koffi.KoffiFunction;
  sdxDestroyRuntime:       koffi.KoffiFunction;
  sdxLoadBundle:           koffi.KoffiFunction;
  sdxUnloadModel:          koffi.KoffiFunction;
  sdxCreateContext:        koffi.KoffiFunction;
  sdxDestroyContext:       koffi.KoffiFunction;
  sdxRun:                  koffi.KoffiFunction;
  sdxGetLastError:         koffi.KoffiFunction;
  sdxGetExecutionReport:   koffi.KoffiFunction;
  sdxMarkInputVariable:    koffi.KoffiFunction;
  sdxMarkInputPlaceholder: koffi.KoffiFunction;
  sdxFreezeShapes:         koffi.KoffiFunction;
  sdxGetPlanPhase:         koffi.KoffiFunction;
  sdxGetExecutionCount:    koffi.KoffiFunction;
  sdxGetNumInputs:         koffi.KoffiFunction;
  sdxGetNumOutputs:        koffi.KoffiFunction;
  sdxGetInputName:         koffi.KoffiFunction;
}

function loadAbi(libraryPath: string): SdxAbi {
  const lib = koffi.load(libraryPath);
  return {
    sdxGetRuntimeAbiVersion: lib.func('int sdxGetRuntimeAbiVersion()'),
    sdxCreateRuntime:        lib.func(
      'int sdxCreateRuntime(const sdx_runtime_options_t *options, _Out_ void **out_runtime)'),
    sdxDestroyRuntime:       lib.func('void sdxDestroyRuntime(void *runtime)'),
    sdxLoadBundle:           lib.func(
      'int sdxLoadBundle(void *runtime, const char *bundle_path, const sdx_model_options_t *options, _Out_ void **out_model)'),
    sdxUnloadModel:          lib.func('void sdxUnloadModel(void *model)'),
    sdxCreateContext:        lib.func(
      'int sdxCreateContext(void *model, const char **requested_output_names, int32_t num_requested_outputs, _Out_ void **out_context)'),
    sdxDestroyContext:       lib.func('void sdxDestroyContext(void *context)'),
    sdxRun:                  lib.func(
      'int sdxRun(void *context, const sdx_tensor_view_t *inputs, int32_t num_inputs, const sdx_tensor_view_t *outputs, int32_t num_outputs, const sdx_run_options_t *options)'),
    sdxGetLastError:         lib.func('const char *sdxGetLastError(const void *runtime)'),
    sdxGetExecutionReport:   lib.func(
      'int sdxGetExecutionReport(const void *context, _Inout_ sdx_execution_report_t *out_report)'),
    sdxMarkInputVariable:    lib.func('int sdxMarkInputVariable(void *context, int32_t input_index)'),
    sdxMarkInputPlaceholder: lib.func('int sdxMarkInputPlaceholder(void *context, int32_t input_index)'),
    sdxFreezeShapes:         lib.func('int sdxFreezeShapes(void *context)'),
    sdxGetPlanPhase:         lib.func('int32_t sdxGetPlanPhase(const void *context)'),
    sdxGetExecutionCount:    lib.func('int32_t sdxGetExecutionCount(const void *context)'),
    sdxGetNumInputs:         lib.func('int32_t sdxGetNumInputs(const void *context)'),
    sdxGetNumOutputs:        lib.func('int32_t sdxGetNumOutputs(const void *context)'),
    sdxGetInputName:         lib.func('const char *sdxGetInputName(const void *context, int32_t input_index)'),
  };
}

// ── Internal tensor bridge ────────────────────────────────────────────────────

/** A tensor view that keeps its backing Buffers alive across sdxRun. */
interface TensorView {
  view:       Record<string, unknown>;
  dataBuffer: Buffer;          // pins Float32Array memory for the native call
  shapeBuffer: BigInt64Array;  // pins shape memory for the native call
}

function toTensorView(t: Tensor): TensorView {
  // Copy into a node Buffer so koffi can pass the pointer safely.
  const dataBuffer = Buffer.alloc(t.data.byteLength);
  dataBuffer.set(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength));
  const shapeBuffer = BigInt64Array.from(t.dims.map(BigInt));
  return {
    dataBuffer,
    shapeBuffer,
    view: {
      data:        dataBuffer,
      shape:       shapeBuffer,
      rank:        t.dims.length,
      dtype:       SDX_DTYPE_FLOAT,
      bytes:       t.data.byteLength,
      device_type: SDX_DEVICE_HOST,
      device_id:   -1,
    },
  };
}

function viewToTensor(v: TensorView, numElements: number): Tensor {
  return {
    data: new Float32Array(
      v.dataBuffer.buffer, v.dataBuffer.byteOffset, numElements),
    dims: Array.from(v.shapeBuffer).map(Number),
  };
}

// ── Library resolution ────────────────────────────────────────────────────────

function findFileRecursive(root: string, names: string[], maxDepth = 12): string | null {
  const stack: Array<{ dir: string; depth: number }> = [{ dir: root, depth: 0 }];
  while (stack.length > 0) {
    const { dir, depth } = stack.pop()!;
    let entries: fs.Dirent[];
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); }
    catch { continue; }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isFile() && names.includes(entry.name)) return full;
      if (entry.isDirectory() && depth < maxDepth)
        stack.push({ dir: full, depth: depth + 1 });
    }
  }
  return null;
}

/**
 * Resolves the SDX runtime shared library path.
 *
 * Search order:
 *  1. `SDX_RUNTIME_LIBRARY` — explicit path to the library file.
 *  2. `SDX_RUNTIME_HOME/lib/` — unpacked SDK distribution directory.
 *  3. `~/.javacpp/cache/` — JavaCPP-extracted library left by an ND4J process.
 */
export function resolveRuntimeLibrary(): string {
  const explicit = process.env.SDX_RUNTIME_LIBRARY;
  if (explicit && fs.existsSync(explicit)) return explicit;

  const ext = process.platform === 'darwin' ? '.dylib' : '.so';
  const names = [`libsdx_cpu${ext}`, `libnd4jcpu${ext}`];

  const sdkHome = process.env.SDX_RUNTIME_HOME;
  if (sdkHome) {
    for (const name of names) {
      const candidate = path.join(sdkHome, 'lib', name);
      if (fs.existsSync(candidate)) return candidate;
    }
  }

  const javacppCache = path.join(os.homedir(), '.javacpp', 'cache');
  if (fs.existsSync(javacppCache)) {
    const hit = findFileRecursive(javacppCache, names);
    if (hit) return hit;
  }

  throw new Error(
    `SDX runtime library not found. ` +
    `Set SDX_RUNTIME_LIBRARY=/path/to/libsdx_cpu${ext} ` +
    `or SDX_RUNTIME_HOME=/path/to/unpacked-sdk`);
}

// ── SdxRuntime ────────────────────────────────────────────────────────────────

/**
 * Entry point: loads the native SDX library and owns a `sdxRuntime` handle.
 *
 * @example
 * ```ts
 * const runtime = SdxRuntime.create();          // auto-resolves the library
 * // ... use runtime.loadBundle() ...
 * runtime.dispose();                             // or `using runtime = ...` on Node 22+
 * ```
 */
export class SdxRuntime implements Disposable {
  private readonly _abi:     SdxAbi;
  private readonly _handle:  unknown;
  private          _disposed = false;

  /** ABI version reported by the native library. */
  readonly abiVersion: number;
  /** Absolute path to the loaded library. */
  readonly libraryPath: string;

  private constructor(abi: SdxAbi, handle: unknown, abiVersion: number, libraryPath: string) {
    this._abi       = abi;
    this._handle    = handle;
    this.abiVersion = abiVersion;
    this.libraryPath = libraryPath;
  }

  /**
   * Loads the SDX runtime from the given library path (or auto-resolves via
   * {@link resolveRuntimeLibrary}).
   */
  static create(libraryPath?: string): SdxRuntime {
    const lib = libraryPath ?? resolveRuntimeLibrary();
    const abi = loadAbi(lib);
    const abiVersion = abi.sdxGetRuntimeAbiVersion() as number;
    const outHandle: unknown[] = [null];
    const status = abi.sdxCreateRuntime(
      { struct_size: koffi.sizeof('sdx_runtime_options_t') }, outHandle) as number;
    if (status !== SDX_STATUS_OK)
      throw new Error(`sdxCreateRuntime failed: status=${status}`);
    return new SdxRuntime(abi, outHandle[0], abiVersion, lib);
  }

  /** Loads a `.sdz` model bundle and returns an {@link SdxModel}. */
  loadBundle(bundlePath: string, options?: ModelOptions): SdxModel {
    this._assertAlive();
    const opts = options ?? {};
    const outModel: unknown[] = [null];
    const status = this._abi.sdxLoadBundle(this._handle, bundlePath, {
      struct_size:       koffi.sizeof('sdx_model_options_t'),
      backend:           opts.backend           ?? 0,
      strict_backend:    opts.strictBackend      ? 1 : 0,
      allow_runtime_jit: opts.allowRuntimeJit    ? 1 : 0,
      gpu_target:        opts.gpuTarget          ?? 0,
    }, outModel) as number;
    if (status !== SDX_STATUS_OK) {
      const err = (this._abi.sdxGetLastError(this._handle) as string | null) ?? '';
      throw new Error(`sdxLoadBundle("${bundlePath}") failed: status=${status}; ${err}`);
    }
    return new SdxModel(this._abi, this._handle, outModel[0]);
  }

  /**
   * Returns the last error string from the runtime, or empty string.
   * Useful for diagnosing failed calls from outside the wrapper.
   */
  lastError(): string {
    return (this._abi.sdxGetLastError(this._handle) as string | null) ?? '';
  }

  /** Releases the runtime handle. Safe to call multiple times. */
  dispose(): void {
    if (!this._disposed) {
      this._disposed = true;
      this._abi.sdxDestroyRuntime(this._handle);
    }
  }

  /** TC39 explicit resource management — called automatically by `using`. */
  [Symbol.dispose](): void { this.dispose(); }

  private _assertAlive(): void {
    if (this._disposed) throw new Error('SdxRuntime has been disposed');
  }
}

// ── SdxModel ──────────────────────────────────────────────────────────────────

/**
 * A loaded model bundle (`sdxModel` handle).
 *
 * Obtain via {@link SdxRuntime.loadBundle}. Create inference contexts with
 * {@link SdxModel.createContext}.
 */
export class SdxModel implements Disposable {
  private readonly _abi:     SdxAbi;
  private readonly _runtime: unknown;
  private readonly _handle:  unknown;
  private          _disposed = false;

  /** @internal — use SdxRuntime.loadBundle() */
  constructor(abi: SdxAbi, runtime: unknown, handle: unknown) {
    this._abi     = abi;
    this._runtime = runtime;
    this._handle  = handle;
  }

  /**
   * Creates an inference context that will produce the named output variables.
   *
   * @param requestedOutputs - names of the output variables to compute.
   *   Pass `[]` to compute all outputs defined in the bundle.
   */
  createContext(requestedOutputs: string[]): SdxContext {
    this._assertAlive();
    const outCtx: unknown[] = [null];
    const status = this._abi.sdxCreateContext(
      this._handle, requestedOutputs, requestedOutputs.length, outCtx) as number;
    if (status !== SDX_STATUS_OK) {
      const err = (this._abi.sdxGetLastError(this._runtime) as string | null) ?? '';
      throw new Error(`sdxCreateContext failed: status=${status}; ${err}`);
    }
    return new SdxContext(this._abi, this._runtime, outCtx[0]);
  }

  /** Unloads the model. Safe to call multiple times. */
  dispose(): void {
    if (!this._disposed) {
      this._disposed = true;
      this._abi.sdxUnloadModel(this._handle);
    }
  }

  [Symbol.dispose](): void { this.dispose(); }

  private _assertAlive(): void {
    if (this._disposed) throw new Error('SdxModel has been disposed');
  }
}

// ── SdxContext ────────────────────────────────────────────────────────────────

/**
 * An inference context (`sdxContext` handle) — the main inference interface.
 *
 * Follows the onnxruntime-node `InferenceSession` naming pattern:
 * - Named inputs via `run(feeds)` — a `Record<string, Tensor>`.
 * - Named output discovery via `outputNames()`.
 * - Input contract discovery via `inputNames()`.
 *
 * Lifecycle:
 * 1. Call `inputNames()` to discover what the plan expects.
 * 2. Mark variable and placeholder slots: `markInputVariable` / `markInputPlaceholder`.
 * 3. Warmup: call `run()` several times.
 * 4. Call `freezeShapes()` to enable the DSP replay fast path.
 * 5. Continue calling `run()` — the runtime is now replaying a compiled graph.
 *
 * @example
 * ```ts
 * const session = model.createContext(['probs']);
 *
 * // Discover the positional contract, mark the placeholder:
 * const names = session.inputNames();   // ['w1','b1','w2','b2','x']
 * session.markInputPlaceholder(names.indexOf('x'));
 *
 * // Warmup:
 * for (let i = 0; i < 3; i++) session.run({ w1: ..., b1: ..., w2: ..., b2: ..., x: ... });
 *
 * // Freeze and replay:
 * session.freezeShapes();
 * const result = session.run({ w1: ..., x: ... });
 * const probs = result['probs'].data;   // Float32Array
 *
 * session.dispose();
 * ```
 */
export class SdxContext implements Disposable {
  private readonly _abi:         SdxAbi;
  private readonly _runtime:     unknown;
  private readonly _handle:      unknown;
  private          _inputNames:  string[] | null = null;
  private          _disposed   = false;

  /** @internal — use SdxModel.createContext() */
  constructor(abi: SdxAbi, runtime: unknown, handle: unknown) {
    this._abi     = abi;
    this._runtime = runtime;
    this._handle  = handle;
  }

  // ── Contract discovery ──────────────────────────────────────────────────────

  /**
   * Returns the ordered list of external input names the plan expects.
   *
   * The order is positional — it must match the order you pass tensors to
   * {@link run}. The result is cached after the first call.
   *
   * Mirrors onnxruntime-node's `InferenceSession.inputNames`.
   */
  inputNames(): string[] {
    if (this._inputNames) return this._inputNames;
    const n = this._abi.sdxGetNumInputs(this._handle) as number;
    const names: string[] = [];
    for (let i = 0; i < n; i++)
      names.push(this._abi.sdxGetInputName(this._handle, i) as string);
    this._inputNames = names;
    return names;
  }

  /** Number of output tensors this context produces. */
  numOutputs(): number {
    return this._abi.sdxGetNumOutputs(this._handle) as number;
  }

  // ── Slot annotation ─────────────────────────────────────────────────────────

  /**
   * Marks an input slot as a variable (weight / constant that seldom changes).
   * DSP uses this to avoid unnecessary re-syncs.
   */
  markInputVariable(inputIndex: number): void {
    const status = this._abi.sdxMarkInputVariable(this._handle, inputIndex) as number;
    if (status !== SDX_STATUS_OK) this._throw(`sdxMarkInputVariable(${inputIndex})`, status);
  }

  /**
   * Marks an input slot as a placeholder (data that changes every run, e.g.
   * the batch input `x`). DSP will always treat these as dynamic.
   */
  markInputPlaceholder(inputIndex: number): void {
    const status = this._abi.sdxMarkInputPlaceholder(this._handle, inputIndex) as number;
    if (status !== SDX_STATUS_OK) this._throw(`sdxMarkInputPlaceholder(${inputIndex})`, status);
  }

  // ── Execution ───────────────────────────────────────────────────────────────

  /**
   * Runs the model with named inputs and returns named output tensors.
   *
   * Inputs are a `Record<string, Tensor>` keyed on the names returned by
   * {@link inputNames}. The outputs are a `Record<string, Tensor>` keyed on
   * the names passed to {@link SdxModel.createContext}.
   *
   * Output buffer sizes are inferred from the provided `outputSpecs` map.
   * If omitted, the caller must supply `outputSpecs` via a second argument.
   *
   * @example
   * ```ts
   * const result = session.run(
   *   { w1, b1, w2, b2, x },
   *   { probs: { data: new Float32Array(6), dims: [2, 3] } }
   * );
   * const probs = result['probs'].data; // Float32Array
   * ```
   */
  run(feeds: TensorMap, outputBuffers: TensorMap, opts?: RunOptions): TensorMap {
    this._assertAlive();
    const names = this.inputNames();

    // Build the positional input array from the named feeds.
    const inputViews: TensorView[] = names.map((name, i) => {
      const t = feeds[name];
      if (!t) throw new Error(
        `Missing input "${name}" (index ${i}). ` +
        `Expected inputs: [${names.join(', ')}].`);
      return toTensorView(t);
    });

    // Build the output array from the caller-supplied output buffers.
    const outputEntries = Object.entries(outputBuffers);
    const outputViews: TensorView[] = outputEntries.map(([, t]) => toTensorView(t));

    const runOpts = opts ?? {};
    const runOptsStruct = {
      struct_size:      koffi.sizeof('sdx_run_options_t'),
      backend:          runOpts.backend ?? 0,
      strict_signature: runOpts.strictSignature ? 1 : 0,
      gpu_target:       runOpts.gpuTarget ?? 0,
    };

    const status = this._abi.sdxRun(
      this._handle,
      inputViews.map((v) => v.view),
      inputViews.length,
      outputViews.map((v) => v.view),
      outputViews.length,
      runOptsStruct,
    ) as number;

    if (status !== SDX_STATUS_OK) this._throw('sdxRun', status);

    // Build the named output map from the results.
    const result: TensorMap = {};
    for (let i = 0; i < outputEntries.length; i++) {
      const [name, spec] = outputEntries[i];
      const numElements = spec.dims.reduce((a, b) => a * b, 1);
      result[name] = viewToTensor(outputViews[i], numElements);
    }
    return result;
  }

  // ── DSP lifecycle ───────────────────────────────────────────────────────────

  /**
   * Signals to the DSP that input shapes are stable. After this call the
   * runtime enters the fast-path graph-replay mode (REPLAYING phase).
   *
   * Call after the warmup runs, before production inference.
   */
  freezeShapes(): void {
    const status = this._abi.sdxFreezeShapes(this._handle) as number;
    if (status !== SDX_STATUS_OK) this._throw('sdxFreezeShapes', status);
  }

  /** Current DSP plan phase (0–3, see {@link PLAN_PHASE_NAMES}). */
  planPhase(): number {
    return this._abi.sdxGetPlanPhase(this._handle) as number;
  }

  /** Number of times this context has been executed. */
  executionCount(): number {
    return this._abi.sdxGetExecutionCount(this._handle) as number;
  }

  /**
   * Returns a typed {@link ExecutionReport} for the last run.
   */
  executionReport(): ExecutionReport {
    this._assertAlive();
    const raw: Record<string, unknown> = {
      struct_size:          koffi.sizeof('sdx_execution_report_t'),
      requested_backend:    0,
      applied_backend:      0,
      status_code:          0,
      used_fallback:        -1,
      execution_time_ns:    0,
      requested_gpu_target: 0,
      applied_gpu_target:   0,
      plan_phase:           -1,
      execution_count:      0,
    };
    const status = this._abi.sdxGetExecutionReport(this._handle, raw) as number;
    if (status !== SDX_STATUS_OK) this._throw('sdxGetExecutionReport', status);
    return {
      statusCode:       raw.status_code       as number,
      requestedBackend: raw.requested_backend  as number,
      appliedBackend:   raw.applied_backend    as number,
      usedFallback:     (raw.used_fallback as number) === 1,
      planPhase:        raw.plan_phase         as number,
      executionCount:   raw.execution_count    as number,
      executionTimeNs:  BigInt(raw.execution_time_ns as number),
    };
  }

  // ── Lifecycle ───────────────────────────────────────────────────────────────

  /** Destroys the native context handle. Safe to call multiple times. */
  dispose(): void {
    if (!this._disposed) {
      this._disposed = true;
      this._abi.sdxDestroyContext(this._handle);
    }
  }

  [Symbol.dispose](): void { this.dispose(); }

  // ── Internal ────────────────────────────────────────────────────────────────

  private _throw(op: string, status: number): never {
    const err = (this._abi.sdxGetLastError(this._runtime) as string | null) ?? '';
    throw new Error(`${op} failed: status=${status}${err ? `; ${err}` : ''}`);
  }

  private _assertAlive(): void {
    if (this._disposed) throw new Error('SdxContext has been disposed');
  }
}
