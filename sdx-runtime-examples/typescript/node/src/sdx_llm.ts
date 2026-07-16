/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/**
 * SDX LLM AOT runtime wrapper for Node.js — idiomatic TypeScript classes
 * following the onnxruntime-node naming and lifecycle conventions.
 *
 * Binds `sdx_llm_c.h` (libsdx_llm — GraalVM AOT native-image, no JVM) via
 * [koffi](https://koffi.dev) running on a dedicated Worker thread.
 *
 * ## Why a Worker thread?
 *
 * The GraalVM AOT isolate requires a **large OS thread stack** (≥64 MB) during
 * `sdxLlmCreateRuntime` — it bootstraps JVM runtime frames that exceed Node's
 * 8 MB main-thread stack, causing a fatal `StackOverflowError`.  A
 * `worker_threads.Worker` with `resourceLimits.stackSizeMb = 128` gives the
 * isolate the stack space it needs.
 *
 * All koffi FFI calls run on that single worker thread, preserving the
 * **single-thread affinity** required by the GraalVM isolate.  From the
 * caller's perspective every method is `async` and returns a `Promise`.
 *
 * ## Quick start
 *
 * ```ts
 * import { SdxLlmRuntime } from './sdx_llm';
 *
 * const rt    = await SdxLlmRuntime.create();
 * console.log('ABI version:', rt.abiVersion);    // 1
 *
 * const model = await rt.loadModel(
 *   '~/.cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf',
 *   '~/.cache/dl4j-llm-models/qwen35-0.8B-tokenizer.json',
 *   JSON.stringify({ maxNewTokens: 8, sampling: { preset: 'greedy' } }),
 * );
 *
 * const text = await model.generate('The capital of France is');
 * console.log(text);   // " Paris."
 *
 * const stats = await model.lastResultStats();
 * console.log(stats?.tokensPerSecond, 'tok/s');
 *
 * await model.dispose();
 * await rt.dispose();
 * ```
 *
 * ## Threading
 *
 * All library calls are serialised on one dedicated worker thread with a 128 MB
 * stack.  The main thread is never blocked.  Do not call any ABI function from
 * a different thread.
 *
 * ## Deployment
 *
 * Unpack the AOT SDK and set `SDX_LLM_AOT_HOME`.  Library resolution order
 * (see {@link resolveLibsdxLlm}):
 *  1. `SDX_LLM_LIBRARY` — explicit path to `libsdx_llm.so` / `.dylib`.
 *  2. `SDX_LLM_AOT_HOME/lib/libsdx_llm.*`.
 *
 * `libsdx_llm` side-loads its native companions relative to the host process.
 * Set `SDX_NATIVE_LIB_DIR` to `$SDX_LLM_AOT_HOME/lib`; {@link SdxLlmRuntime.create}
 * does this automatically when `SDX_LLM_AOT_HOME` is set and `SDX_NATIVE_LIB_DIR`
 * is not.
 */

import * as fs             from 'fs';
import * as path           from 'path';
import { Worker }          from 'worker_threads';

// ── Status codes ──────────────────────────────────────────────────────────────

export const SDX_LLM_STATUS_OK               = 0;
export const SDX_LLM_STATUS_INVALID_ARGUMENT = 1;
export const SDX_LLM_STATUS_MODEL_LOAD_FAILED = 3;
export const SDX_LLM_STATUS_EXECUTION_FAILED = 4;
export const SDX_LLM_STATUS_IO_ERROR         = 6;

export const STATUS_NAMES: ReadonlyMap<number, string> = new Map([
  [0, 'OK'],
  [1, 'INVALID_ARGUMENT'],
  [3, 'MODEL_LOAD_FAILED'],
  [4, 'EXECUTION_FAILED'],
  [6, 'IO_ERROR'],
]);
export const statusName = (s: number): string =>
  STATUS_NAMES.get(s) ?? `UNKNOWN(${s})`;

// ── Public types ──────────────────────────────────────────────────────────────

/**
 * Parsed subset of the JSON returned by `sdxLlmLastResultJson`.
 *
 * All fields are optional — the JSON schema may evolve across SDK versions.
 * Field names match the SDK's serialised form (e.g. `generatedTokens`, not
 * `newTokens`).
 */
export interface GenerationStats {
  /** Tokens generated (prompt not counted). SDK key: `generatedTokens`. */
  generatedTokens?: number;
  /** Prompt tokens consumed. SDK key: `promptTokens`. */
  promptTokens?: number;
  /** Total wall-clock generation time in milliseconds. SDK key: `generationTimeMs`. */
  generationTimeMs?: number;
  /** Throughput: `generatedTokens / generationTimeMs * 1000`. SDK key: `tokensPerSecond`. */
  tokensPerSecond?: number;
  /** Why generation stopped: `"MAX_TOKENS"`, `"EOS"`, … SDK key: `finishReason`. */
  finishReason?: string;
}

/**
 * Options forwarded to `sdxLlmLoadModel`.
 *
 * Pass as a JSON string via the `optionsJson` parameter of
 * {@link SdxLlmRuntime.loadModel}.  This interface documents the known fields.
 */
export interface LoadModelOptions {
  /** Maximum tokens to generate per call (default: model-dependent). */
  maxNewTokens?: number;
  /** Whether to run graph optimizer before inference (default: true). */
  graphOptimizer?: boolean;
  /** Sampling configuration. */
  sampling?: {
    /** Preset: `"greedy"` | `"top_p"` | … */
    preset?: string;
    /** Softmax temperature (ignored for greedy). */
    temperature?: number;
    /** Top-k sampling parameter. */
    topK?: number;
    /** Nucleus sampling probability. */
    topP?: number;
    /** Repetition penalty (1.0 = none). */
    repetitionPenalty?: number;
    /** RNG seed for reproducible sampling. */
    seed?: number;
  };
}

// ── Library resolution ────────────────────────────────────────────────────────

/**
 * Resolves the path to `libsdx_llm.so` / `libsdx_llm.dylib`.
 *
 * Search order:
 *  1. `SDX_LLM_LIBRARY`     — explicit absolute path to the library file.
 *  2. `SDX_LLM_AOT_HOME/lib/` — unpacked AOT SDK distribution directory.
 *
 * @throws {Error} when the library cannot be found.
 */
export function resolveLibsdxLlm(): string {
  const explicit = process.env.SDX_LLM_LIBRARY;
  if (explicit && fs.existsSync(explicit)) return explicit;

  const ext  = process.platform === 'darwin' ? '.dylib' : '.so';

  const aotHome = process.env.SDX_LLM_AOT_HOME;
  if (aotHome) {
    for (const name of [`libsdx_llm${ext}`, `sdx_llm${ext}`]) {
      const candidate = path.join(aotHome, 'lib', name);
      if (fs.existsSync(candidate)) return candidate;
    }
  }

  throw new Error(
    `libsdx_llm not found. ` +
    `Set SDX_LLM_LIBRARY=/path/to/libsdx_llm${ext} ` +
    `or SDX_LLM_AOT_HOME=/path/to/unpacked-sdk.`
  );
}

// ── Worker-thread bridge ──────────────────────────────────────────────────────

/** A pending RPC call: resolves / rejects when the worker replies. */
interface Pending {
  resolve: (v: unknown) => void;
  reject:  (e: Error)   => void;
}

/**
 * Creates a Worker thread with a 128 MB OS stack and establishes a simple
 * request/reply RPC over `postMessage`.
 *
 * The GraalVM AOT isolate needs ≥64 MB of OS stack during `sdxLlmCreateRuntime`.
 * A worker with `resourceLimits.stackSizeMb = 128` provides that; Node's
 * main thread with its 8 MB default stack would crash with a StackOverflowError.
 *
 * Returns a `call(op, ...args)` function that serialises every ABI call onto
 * the worker and returns a Promise.
 */
function createLlmWorker(): { call: (op: string, args: unknown[]) => Promise<unknown>; terminate: () => Promise<number> } {
  // The worker runs the compiled JS from dist/llm_worker.js.
  const workerScript = path.join(__dirname, 'llm_worker.js');
  const worker = new Worker(workerScript, {
    resourceLimits: {
      stackSizeMb: 128,    // GraalVM isolate init needs >> 8 MB OS stack
    },
    // Forward environment so the worker inherits SDX_LLM_AOT_HOME etc.
    env: process.env as NodeJS.ProcessEnv,
  });

  const pending = new Map<number, Pending>();
  let   nextId = 1;

  worker.on('message', (msg: { id: number; ok: boolean; result: unknown; error?: string }) => {
    const p = pending.get(msg.id);
    if (!p) return;
    pending.delete(msg.id);
    if (msg.ok) p.resolve(msg.result);
    else        p.reject(new Error(msg.error ?? 'Unknown worker error'));
  });

  worker.on('error', (err) => {
    // Reject all outstanding calls on a fatal worker error.
    for (const p of pending.values()) p.reject(err);
    pending.clear();
  });

  return {
    call(op: string, args: unknown[]): Promise<unknown> {
      return new Promise((resolve, reject) => {
        const id = nextId++;
        pending.set(id, { resolve, reject });
        worker.postMessage({ id, op, args });
      });
    },
    terminate: () => worker.terminate(),
  };
}

// ── SdxLlmModel ───────────────────────────────────────────────────────────────

/**
 * A loaded LLM pipeline.  Obtain via {@link SdxLlmRuntime.loadModel}.
 *
 * All methods are `async` — the actual FFI calls run on the worker thread.
 */
export class SdxLlmModel {
  private readonly _call: (op: string, args: unknown[]) => Promise<unknown>;
  private          _disposed = false;

  /** @internal — use SdxLlmRuntime.loadModel() */
  constructor(call: (op: string, args: unknown[]) => Promise<unknown>) {
    this._call = call;
  }

  // ── Generation ─────────────────────────────────────────────────────────────

  /**
   * Blocking text generation (awaited on worker thread).
   *
   * @param prompt      Input text (UTF-8).
   * @param optionsJson Optional per-call JSON overrides.
   *                    Pass `null` to use the load-time defaults.
   * @returns The generated text (prompt not included).
   */
  async generate(prompt: string, optionsJson?: string | null): Promise<string> {
    this._assertAlive();
    return this._call('generate', [prompt, optionsJson ?? null]) as Promise<string>;
  }

  // ── Tokenization ───────────────────────────────────────────────────────────

  /**
   * Encode text to token IDs.
   *
   * @returns Int32Array of token IDs.
   */
  async tokenize(text: string, addSpecialTokens = true): Promise<Int32Array> {
    this._assertAlive();
    const ids = await this._call('tokenize', [text, addSpecialTokens]) as number[];
    return Int32Array.from(ids);
  }

  /**
   * Decode token IDs back to text.
   */
  async detokenize(ids: Int32Array | number[], skipSpecialTokens = true): Promise<string> {
    this._assertAlive();
    return this._call('detokenize', [Array.from(ids), skipSpecialTokens]) as Promise<string>;
  }

  // ── Metadata ───────────────────────────────────────────────────────────────

  /**
   * Statistics from the most recent {@link generate} call.
   *
   * @returns Parsed {@link GenerationStats}, or `null` if no generation has run.
   */
  async lastResultStats(): Promise<GenerationStats | null> {
    this._assertAlive();
    return this._call('lastResultJson', []) as Promise<GenerationStats | null>;
  }

  /**
   * Model/tokenizer summary JSON (inputs, outputs, vocab, chat-template flag).
   */
  async infoJson(): Promise<string> {
    this._assertAlive();
    return this._call('infoJson', []) as Promise<string>;
  }

  // ── Lifecycle ───────────────────────────────────────────────────────────────

  /** Unload the model.  Safe to call multiple times. */
  async dispose(): Promise<void> {
    if (!this._disposed) {
      this._disposed = true;
      await this._call('unloadModel', []);
    }
  }

  private _assertAlive(): void {
    if (this._disposed) throw new Error('SdxLlmModel has been disposed');
  }
}

// ── SdxLlmRuntime ─────────────────────────────────────────────────────────────

/**
 * Entry point for the SDX LLM AOT runtime.
 *
 * Runs the GraalVM isolate on a dedicated Worker thread with a 128 MB stack.
 *
 * @example
 * ```ts
 * const rt    = await SdxLlmRuntime.create();
 * const model = await rt.loadModel(modelPath, tokenizerPath,
 *   JSON.stringify({ maxNewTokens: 8, sampling: { preset: 'greedy' } }));
 * const text  = await model.generate('The capital of France is');
 * console.log(text);   // " Paris."
 * await model.dispose();
 * await rt.dispose();
 * ```
 */
export class SdxLlmRuntime {
  private readonly _worker: ReturnType<typeof createLlmWorker>;
  private          _disposed = false;

  /** ABI version of the loaded `libsdx_llm` (must equal `SDX_LLM_ABI_VERSION = 1`). */
  readonly abiVersion: number;
  /** Absolute path to the loaded `libsdx_llm` shared library. */
  readonly libraryPath: string;

  private constructor(
    worker: ReturnType<typeof createLlmWorker>,
    abiVersion: number,
    libraryPath: string,
  ) {
    this._worker   = worker;
    this.abiVersion = abiVersion;
    this.libraryPath = libraryPath;
  }

  /**
   * Spawn the worker thread and initialise the GraalVM isolate.
   *
   * Sets `process.env.SDX_NATIVE_LIB_DIR` from `SDX_LLM_AOT_HOME/lib` when
   * `SDX_NATIVE_LIB_DIR` is not already set, so `libsdx_llm` finds its
   * side-loaded native companions (ND4J CPU kernels, tokenizers).
   *
   * @param libraryPath Optional explicit path to `libsdx_llm.so`.
   */
  static async create(libraryPath?: string): Promise<SdxLlmRuntime> {
    // Propagate to child worker via process.env before spawning.
    if (!process.env.SDX_NATIVE_LIB_DIR) {
      const aotHome = process.env.SDX_LLM_AOT_HOME;
      if (aotHome) {
        process.env.SDX_NATIVE_LIB_DIR = path.join(aotHome, 'lib');
      }
    }

    const lib    = libraryPath ?? resolveLibsdxLlm();
    const worker = createLlmWorker();

    const nativeLibDir = process.env.SDX_NATIVE_LIB_DIR ?? null;
    const init = await worker.call('init', [lib, nativeLibDir]) as
      { libraryPath: string; abiVersion: number };

    return new SdxLlmRuntime(worker, init.abiVersion, init.libraryPath);
  }

  // ── Model lifecycle ─────────────────────────────────────────────────────────

  /**
   * Load a model and build the generation pipeline.
   *
   * @param modelPath      Path to `.gguf`/`.ggml` or `.sdz`/`.sdnb`/`.fb`.
   * @param tokenizerPath  Path to `tokenizer.json` or directory (`null` = auto).
   * @param optionsJson    Optional JSON options string.
   * @returns A loaded {@link SdxLlmModel} ready for generation.
   */
  async loadModel(
    modelPath: string,
    tokenizerPath?: string | null,
    optionsJson?: string | null,
  ): Promise<SdxLlmModel> {
    this._assertAlive();
    await this._worker.call('loadModel', [modelPath, tokenizerPath ?? null, optionsJson ?? null]);
    // All calls share the same runtime + model handle on the worker.
    return new SdxLlmModel(this._worker.call.bind(this._worker));
  }

  // ── VLM / audio (stateless) ─────────────────────────────────────────────────

  /**
   * Extract content from an image or PDF using the SmolDocling VLM.
   */
  async vlmExtract(
    modelPath: string,
    inputPath: string,
    tokenizerPath?: string | null,
    optionsJson?: string | null,
  ): Promise<string> {
    this._assertAlive();
    return this._worker.call('vlmExtract',
      [modelPath, inputPath, tokenizerPath ?? null, optionsJson ?? null]) as Promise<string>;
  }

  /**
   * Transcribe audio using a Whisper ONNX model.
   */
  async audioTranscribe(
    modelPath: string,
    audioPath: string,
    optionsJson?: string | null,
  ): Promise<string> {
    this._assertAlive();
    return this._worker.call('audioTranscribe',
      [modelPath, audioPath, optionsJson ?? null]) as Promise<string>;
  }

  // ── Lifecycle ───────────────────────────────────────────────────────────────

  /**
   * Tear down the GraalVM isolate and terminate the worker thread.
   * All model handles become invalid.  Always dispose models first.
   */
  async dispose(): Promise<void> {
    if (!this._disposed) {
      this._disposed = true;
      await this._worker.call('destroyRuntime', []);
      await this._worker.terminate();
    }
  }

  private _assertAlive(): void {
    if (this._disposed) throw new Error('SdxLlmRuntime has been disposed');
  }
}
