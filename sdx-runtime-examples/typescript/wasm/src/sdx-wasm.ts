/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/**
 * TypeScript wrapper for the SDX runtime compiled to WebAssembly with
 * Emscripten, following the conventions established by onnxruntime-web
 * (session-style API, `Tensor { data: Float32Array, dims }`, named input
 * feeds) and the TC39 explicit-resource-management proposal
 * (`Symbol.dispose` on every native handle).
 *
 * The wrapper talks to the standard Emscripten `MODULARIZE` surface:
 * `_sdx*` exports, `_malloc`/`_free`, `HEAPU8`, `UTF8ToString`/`stringToUTF8`,
 * and `FS` (MEMFS) for model files. All C structs are marshaled manually at
 * the wasm32 (ILP32) layouts of `dsp_runtime_c.h` — pointers and `size_t`
 * are 4 bytes wide:
 *
 * ```
 * sdx_runtime_options_t   :  4 bytes { struct_size:u32@0 }
 * sdx_model_options_t     : 20 bytes { struct_size@0, backend@4,
 *                                      strict_backend@8, allow_runtime_jit@12,
 *                                      gpu_target@16 }
 * sdx_run_options_t       : 16 bytes { struct_size@0, backend@4,
 *                                      strict_signature@8, gpu_target@12 }
 * sdx_tensor_view_t       : 28 bytes { data:ptr@0, shape:ptr@4, rank:i32@8,
 *                                      dtype:i32@12, bytes:size_t@16,
 *                                      device_type:i32@20, device_id:i32@24 }
 * sdx_execution_report_t  : 48 bytes { struct_size@0, requested_backend@4,
 *                                      applied_backend@8, status_code@12,
 *                                      used_fallback@16, (4-byte pad @20),
 *                                      execution_time_ns:u64@24,
 *                                      requested_gpu_target@32,
 *                                      applied_gpu_target@36, plan_phase@40,
 *                                      execution_count@44 }
 * ```
 *
 * Heap views are re-derived from `module.HEAPU8.buffer` on every access —
 * `ALLOW_MEMORY_GROWTH=1` detaches cached TypedArray views when the linear
 * memory grows, so caching them across calls is a correctness bug.
 */

export const SDX_STATUS_OK = 0;
export const SDX_DEVICE_HOST = 0;
/** sd::DataType::FLOAT32 */
export const SDX_DTYPE_FLOAT = 5;

export const PLAN_PHASE_NAMES = [
  'SLOT_BY_SLOT (warmup)', 'SHAPES_FROZEN', 'REPLAYING', 'REPLAY_BLOCKED'] as const;
export const BACKEND_NAMES = [
  'AUTO', 'SLOT_BY_SLOT', 'CUDA_GRAPHS', 'NVRTC', 'PTX', 'TRITON',
  'MLX', 'ARM_HYBRID', 'NNAPI', 'HIP_GRAPHS', 'LEVEL_ZERO', 'VULKAN',
  'METAL', 'TPU', 'HEXAGON'] as const;

export const phaseName = (p: number): string => PLAN_PHASE_NAMES[p] ?? `? (${p})`;
export const backendName = (b: number): string => BACKEND_NAMES[b] ?? `? (${b})`;

/** An float32 host tensor, onnxruntime-web style. */
export interface Tensor {
  readonly data: Float32Array;
  readonly dims: readonly number[];
}

export type TensorMap = Record<string, Tensor>;
export type ShapeMap = Record<string, readonly number[]>;

export interface ExecutionReport {
  readonly statusCode: number;
  readonly requestedBackend: number;
  readonly appliedBackend: number;
  /** -1 = unknown, 0 = no, 1 = yes */
  readonly usedFallback: number;
  readonly executionTimeNs: number;
  /** 0=SLOT_BY_SLOT (warmup), 1=SHAPES_FROZEN, 2=REPLAYING, 3=REPLAY_BLOCKED */
  readonly planPhase: number;
  readonly executionCount: number;
}

/**
 * The Emscripten module surface this wrapper needs. Produced by a build with
 * `-sMODULARIZE=1 -sEXPORT_NAME=createSdxModule -sFORCE_FILESYSTEM=1` and the
 * `_sdx*` functions exported (see build-wasm.sh).
 */
export interface SdxWasmModule {
  HEAPU8: Uint8Array;

  _malloc(size: number): number;
  _free(ptr: number): void;

  UTF8ToString(ptr: number): string;
  stringToUTF8(str: string, outPtr: number, maxBytes: number): void;
  lengthBytesUTF8(str: string): number;

  FS: {
    writeFile(path: string, data: Uint8Array): void;
  };

  _sdxGetRuntimeAbiVersion(): number;
  _sdxCreateRuntime(optionsPtr: number, outRuntimePtr: number): number;
  _sdxDestroyRuntime(runtime: number): void;
  _sdxLoadBundle(runtime: number, pathPtr: number, optionsPtr: number, outModelPtr: number): number;
  _sdxUnloadModel(model: number): void;
  _sdxCreateContext(model: number, namesPtr: number, numNames: number, outContextPtr: number): number;
  _sdxDestroyContext(context: number): void;
  _sdxRun(context: number, inputsPtr: number, numInputs: number,
          outputsPtr: number, numOutputs: number, optionsPtr: number): number;
  _sdxGetLastError(runtime: number): number;
  _sdxGetExecutionReport(context: number, reportPtr: number): number;
  _sdxMarkInputVariable(context: number, index: number): number;
  _sdxMarkInputPlaceholder(context: number, index: number): number;
  _sdxFreezeShapes(context: number): number;
  _sdxGetPlanPhase(context: number): number;
  _sdxGetExecutionCount(context: number): number;
  _sdxGetNumInputs(context: number): number;
  _sdxGetNumOutputs(context: number): number;
  _sdxGetInputName(context: number, index: number): number;
}

// ── wasm32 struct layout constants ───────────────────────────────────────────

const RUNTIME_OPTIONS_SIZE = 4;
const MODEL_OPTIONS_SIZE = 20;
const RUN_OPTIONS_SIZE = 16;
const TENSOR_VIEW_SIZE = 28;
const EXECUTION_REPORT_SIZE = 48;

// ── Heap access helpers (growth-safe: views derived per access) ──────────────

class Heap {
  constructor(private readonly m: SdxWasmModule) {}

  get u8(): Uint8Array {
    return this.m.HEAPU8;
  }
  get u32(): Uint32Array {
    return new Uint32Array(this.m.HEAPU8.buffer);
  }
  get i32(): Int32Array {
    return new Int32Array(this.m.HEAPU8.buffer);
  }
  get f32(): Float32Array {
    return new Float32Array(this.m.HEAPU8.buffer);
  }
  get i64(): BigInt64Array {
    return new BigInt64Array(this.m.HEAPU8.buffer);
  }
  get u64(): BigUint64Array {
    return new BigUint64Array(this.m.HEAPU8.buffer);
  }

  allocUtf8(text: string): number {
    const bytes = this.m.lengthBytesUTF8(text) + 1;
    const ptr = this.m._malloc(bytes);
    this.m.stringToUTF8(text, ptr, bytes);
    return ptr;
  }
}

class SdxWasmError extends Error {
  constructor(op: string, readonly status: number, detail: string) {
    super(`${op} failed: status=${status}${detail ? `, error=${detail}` : ''}`);
    this.name = 'SdxWasmError';
  }
}

// ── Public wrapper classes ───────────────────────────────────────────────────

export class SdxRuntime implements Disposable {
  private handle: number;
  private readonly heap: Heap;

  private constructor(readonly module: SdxWasmModule, handle: number) {
    this.module = module;
    this.handle = handle;
    this.heap = new Heap(module);
  }

  /** Create a runtime over an instantiated Emscripten module. */
  static create(module: SdxWasmModule): SdxRuntime {
    const heap = new Heap(module);
    const optionsPtr = module._malloc(RUNTIME_OPTIONS_SIZE);
    const outPtr = module._malloc(4);
    try {
      heap.u32[optionsPtr >> 2] = RUNTIME_OPTIONS_SIZE;
      heap.u32[outPtr >> 2] = 0;
      const status = module._sdxCreateRuntime(optionsPtr, outPtr);
      if (status !== SDX_STATUS_OK) {
        throw new SdxWasmError('sdxCreateRuntime', status, '');
      }
      return new SdxRuntime(module, heap.u32[outPtr >> 2]);
    } finally {
      module._free(outPtr);
      module._free(optionsPtr);
    }
  }

  abiVersion(): number {
    return this.module._sdxGetRuntimeAbiVersion();
  }

  lastError(): string {
    const ptr = this.module._sdxGetLastError(this.handle);
    return ptr === 0 ? '' : this.module.UTF8ToString(ptr);
  }

  /**
   * Write the model bytes into the module's MEMFS and load them through
   * sdxLoadBundle — the wasm counterpart of passing a filesystem path.
   */
  loadBundle(name: string, bytes: Uint8Array): SdxModel {
    const path = `/${name}`;
    this.module.FS.writeFile(path, bytes);

    const pathPtr = this.heap.allocUtf8(path);
    const optionsPtr = this.module._malloc(MODEL_OPTIONS_SIZE);
    const outPtr = this.module._malloc(4);
    try {
      const base = optionsPtr >> 2;
      this.heap.u32[base] = MODEL_OPTIONS_SIZE;
      this.heap.i32[base + 1] = 0; // backend AUTO
      this.heap.i32[base + 2] = 0; // strict_backend
      this.heap.i32[base + 3] = 0; // allow_runtime_jit
      this.heap.i32[base + 4] = 0; // gpu_target AUTO
      this.heap.u32[outPtr >> 2] = 0;

      const status = this.module._sdxLoadBundle(this.handle, pathPtr, optionsPtr, outPtr);
      if (status !== SDX_STATUS_OK) {
        throw new SdxWasmError('sdxLoadBundle', status, this.lastError());
      }
      return new SdxModel(this, this.heap.u32[outPtr >> 2]);
    } finally {
      this.module._free(outPtr);
      this.module._free(optionsPtr);
      this.module._free(pathPtr);
    }
  }

  dispose(): void {
    if (this.handle !== 0) {
      this.module._sdxDestroyRuntime(this.handle);
      this.handle = 0;
    }
  }

  [Symbol.dispose](): void {
    this.dispose();
  }

  /** @internal */
  get rawHandle(): number {
    return this.handle;
  }
}

export class SdxModel implements Disposable {
  private handle: number;

  /** @internal */
  constructor(readonly runtime: SdxRuntime, handle: number) {
    this.handle = handle;
  }

  createContext(outputNames: readonly string[]): SdxContext {
    const m = this.runtime.module;
    const heap = new Heap(m);

    const namePtrs = outputNames.map((n) => heap.allocUtf8(n));
    const arrayPtr = m._malloc(Math.max(4, namePtrs.length * 4));
    const outPtr = m._malloc(4);
    try {
      namePtrs.forEach((p, i) => {
        heap.u32[(arrayPtr >> 2) + i] = p;
      });
      heap.u32[outPtr >> 2] = 0;
      const status = m._sdxCreateContext(this.handle, arrayPtr, outputNames.length, outPtr);
      if (status !== SDX_STATUS_OK) {
        throw new SdxWasmError('sdxCreateContext', status, this.runtime.lastError());
      }
      return new SdxContext(this.runtime, heap.u32[outPtr >> 2]);
    } finally {
      m._free(outPtr);
      m._free(arrayPtr);
      namePtrs.forEach((p) => m._free(p));
    }
  }

  dispose(): void {
    if (this.handle !== 0) {
      this.runtime.module._sdxUnloadModel(this.handle);
      this.handle = 0;
    }
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}

export class SdxContext implements Disposable {
  private handle: number;
  private cachedInputNames: string[] | null = null;

  /** @internal */
  constructor(readonly runtime: SdxRuntime, handle: number) {
    this.handle = handle;
  }

  /** The plan's positional input contract, by name (cached). */
  inputNames(): string[] {
    if (this.cachedInputNames === null) {
      const m = this.runtime.module;
      const count = m._sdxGetNumInputs(this.handle);
      const names: string[] = [];
      for (let i = 0; i < count; i++) {
        const ptr = m._sdxGetInputName(this.handle, i);
        names.push(ptr === 0 ? '' : m.UTF8ToString(ptr));
      }
      this.cachedInputNames = names;
    }
    return this.cachedInputNames;
  }

  numOutputs(): number {
    return this.runtime.module._sdxGetNumOutputs(this.handle);
  }

  markInputPlaceholder(name: string): void {
    const index = this.inputNames().indexOf(name);
    if (index < 0) {
      throw new Error(`Unknown plan input: '${name}'`);
    }
    const status = this.runtime.module._sdxMarkInputPlaceholder(this.handle, index);
    if (status !== SDX_STATUS_OK) {
      throw new SdxWasmError('sdxMarkInputPlaceholder', status, this.runtime.lastError());
    }
  }

  freezeShapes(): void {
    const status = this.runtime.module._sdxFreezeShapes(this.handle);
    if (status !== SDX_STATUS_OK) {
      throw new SdxWasmError('sdxFreezeShapes', status, this.runtime.lastError());
    }
  }

  planPhase(): number {
    return this.runtime.module._sdxGetPlanPhase(this.handle);
  }

  executionCount(): number {
    return this.runtime.module._sdxGetExecutionCount(this.handle);
  }

  /**
   * Execute the plan with NAMED input feeds (mapped onto the positional C
   * ABI via {@link inputNames}) and caller-declared output shapes. Returns
   * one tensor per requested output, in request order.
   */
  run(feeds: TensorMap, outputShapes: readonly (readonly number[])[]): Tensor[] {
    const m = this.runtime.module;
    const heap = new Heap(m);
    const names = this.inputNames();

    for (const name of names) {
      if (!(name in feeds)) {
        throw new Error(`Missing feed for plan input '${name}' (required: ${names.join(', ')})`);
      }
    }

    const allocations: number[] = [];
    const alloc = (size: number): number => {
      const ptr = m._malloc(size);
      allocations.push(ptr);
      return ptr;
    };

    const writeTensorView = (structBase: number, dataPtr: number, dims: readonly number[],
                             byteLength: number): void => {
      const shapePtr = alloc(Math.max(8, dims.length * 8));
      const i64 = heap.i64;
      dims.forEach((d, i) => {
        i64[(shapePtr >> 3) + i] = BigInt(d);
      });
      const base32 = structBase >> 2;
      const u32 = heap.u32;
      const i32 = heap.i32;
      u32[base32] = dataPtr;          // data @0
      u32[base32 + 1] = shapePtr;     // shape @4
      i32[base32 + 2] = dims.length;  // rank @8
      i32[base32 + 3] = SDX_DTYPE_FLOAT; // dtype @12
      u32[base32 + 4] = byteLength;   // bytes (size_t) @16
      i32[base32 + 5] = SDX_DEVICE_HOST; // device_type @20
      i32[base32 + 6] = -1;           // device_id @24
    };

    try {
      // Inputs: copy each feed into the wasm heap, positionally per the plan.
      const inputViews = alloc(names.length * TENSOR_VIEW_SIZE);
      names.forEach((name, i) => {
        const tensor = feeds[name];
        const dataPtr = alloc(tensor.data.byteLength);
        heap.f32.set(tensor.data, dataPtr >> 2);
        writeTensorView(inputViews + i * TENSOR_VIEW_SIZE, dataPtr, tensor.dims,
                        tensor.data.byteLength);
      });

      // Outputs: caller-provided buffers per the declared specs.
      const outputViews = alloc(outputShapes.length * TENSOR_VIEW_SIZE);
      const outputDataPtrs: number[] = [];
      const outputLengths: number[] = [];
      outputShapes.forEach((dims, i) => {
        const elements = dims.reduce((a, b) => a * b, 1);
        const dataPtr = alloc(elements * 4);
        heap.u8.fill(0, dataPtr, dataPtr + elements * 4);
        outputDataPtrs.push(dataPtr);
        outputLengths.push(elements);
        writeTensorView(outputViews + i * TENSOR_VIEW_SIZE, dataPtr, dims, elements * 4);
      });

      // Run options.
      const optionsPtr = alloc(RUN_OPTIONS_SIZE);
      const ob = optionsPtr >> 2;
      heap.u32[ob] = RUN_OPTIONS_SIZE;
      heap.i32[ob + 1] = 0; // backend AUTO
      heap.i32[ob + 2] = 1; // strict_signature
      heap.i32[ob + 3] = 0; // gpu_target AUTO

      const status = m._sdxRun(this.handle, inputViews, names.length,
                               outputViews, outputShapes.length, optionsPtr);
      if (status !== SDX_STATUS_OK) {
        throw new SdxWasmError('sdxRun', status, this.runtime.lastError());
      }

      // Copy results out of the heap before freeing.
      return outputShapes.map((dims, i) => {
        const f32 = heap.f32;
        const start = outputDataPtrs[i] >> 2;
        return {
          data: f32.slice(start, start + outputLengths[i]),
          dims,
        };
      });
    } finally {
      for (let i = allocations.length - 1; i >= 0; i--) {
        m._free(allocations[i]);
      }
    }
  }

  executionReport(): ExecutionReport {
    const m = this.runtime.module;
    const heap = new Heap(m);
    const ptr = m._malloc(EXECUTION_REPORT_SIZE);
    try {
      heap.u8.fill(0, ptr, ptr + EXECUTION_REPORT_SIZE);
      heap.u32[ptr >> 2] = EXECUTION_REPORT_SIZE;
      const status = m._sdxGetExecutionReport(this.handle, ptr);
      if (status !== SDX_STATUS_OK) {
        throw new SdxWasmError('sdxGetExecutionReport', status, this.runtime.lastError());
      }
      const i32 = heap.i32;
      const b = ptr >> 2;
      return {
        requestedBackend: i32[b + 1],
        appliedBackend: i32[b + 2],
        statusCode: i32[b + 3],
        usedFallback: i32[b + 4],
        executionTimeNs: Number(heap.u64[(ptr + 24) >> 3]),
        planPhase: i32[b + 10],       // offset 40
        executionCount: i32[b + 11],  // offset 44
      };
    } finally {
      m._free(ptr);
    }
  }

  dispose(): void {
    if (this.handle !== 0) {
      this.runtime.module._sdxDestroyContext(this.handle);
      this.handle = 0;
    }
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}
