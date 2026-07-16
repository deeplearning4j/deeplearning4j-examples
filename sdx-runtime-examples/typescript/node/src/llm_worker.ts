/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/**
 * Worker-thread body for the SDX LLM AOT runtime.
 *
 * The GraalVM AOT isolate (libsdx_llm) requires a large native thread stack
 * (≥64 MB) for its JVM frame bootstrapping during `sdxLlmCreateRuntime`.
 * Node's main thread only has ~8 MB of OS stack, which is too small.
 *
 * This file runs inside a `worker_threads.Worker` created with
 * `resourceLimits: { stackSizeMb: 128 }`, which raises the OS thread stack to
 * 128 MB — enough for GraalVM initialisation.
 *
 * The host script (`llm_example.ts`) drives this worker via
 * `postMessage(request)` / `on('message', reply)`.
 *
 * Request shape: `{ id, op, args }` — all arguments are plain JS values.
 * Reply shape:   `{ id, ok, result, error }`.
 *
 * This file is imported at worker startup via the `workerData.script` path.
 */

import { parentPort, workerData } from 'worker_threads';
import koffi from 'koffi';
import * as path from 'path';
import * as fs   from 'fs';

// ── Library resolution (mirrors sdx_llm.ts) ─────────────────────────────���─────

function resolveLibsdxLlm(): string {
  const explicit = process.env.SDX_LLM_LIBRARY;
  if (explicit && fs.existsSync(explicit)) return explicit;
  const ext  = process.platform === 'darwin' ? '.dylib' : '.so';
  const aotHome = process.env.SDX_LLM_AOT_HOME;
  if (aotHome) {
    for (const name of [`libsdx_llm${ext}`, `sdx_llm${ext}`]) {
      const p = path.join(aotHome, 'lib', name);
      if (fs.existsSync(p)) return p;
    }
  }
  throw new Error(
    `libsdx_llm not found. Set SDX_LLM_LIBRARY or SDX_LLM_AOT_HOME.`);
}

// ── C ABI binding ─────────────────────────────────────────────────────────────

let abi: ReturnType<typeof loadAbi>;
let rtHandle: unknown = null;
let modelHandle: unknown = null;

// koffi char** out-param strategy:
//
// When koffi processes `_Out_ char **`, it automatically dereferences the native
// char* and copies the string content into a JS string in outArr[0].  The
// original native char* (allocated by the SDK) is no longer accessible — koffi
// does not expose it.  Calling sdxLlmFree on a JS string value causes glibc to
// report "free(): invalid pointer".
//
// For char** outputs we simply do NOT call sdxLlmFree.  This leaks one small
// string allocation per call — negligible for a CLI tool.  The tokenize int32**
// output is declared as `_Out_ int32_t **` so koffi returns a Buffer there;
// we decode the int array and then free via sdxLlmFree.

function loadAbi(libraryPath: string) {
  const lib = koffi.load(libraryPath);
  return {
    sdxLlmCreateRuntime:  lib.func('void * sdxLlmCreateRuntime()'),
    sdxLlmDestroyRuntime: lib.func('int sdxLlmDestroyRuntime(void *runtime)'),
    sdxLlmAbiVersion:     lib.func('int sdxLlmAbiVersion(void *runtime)'),
    sdxLlmLoadModel:      lib.func(
      'void * sdxLlmLoadModel(void *runtime, const char *model_path, ' +
      'const char *tokenizer_path, const char *options_json)'),
    sdxLlmUnloadModel:    lib.func('int sdxLlmUnloadModel(void *runtime, void *model)'),
    sdxLlmGenerate:       lib.func(
      'int sdxLlmGenerate(void *runtime, void *model, const char *prompt, ' +
      'const char *options_json, _Out_ char **out_text)'),
    sdxLlmLastResultJson: lib.func(
      'int sdxLlmLastResultJson(void *runtime, void *model, _Out_ char **out_json)'),
    sdxLlmInfoJson:       lib.func(
      'int sdxLlmInfoJson(void *runtime, void *model, _Out_ char **out_json)'),
    // int32_t** — use _Out_ int32_t ** so koffi exposes the raw Buffer for freeing
    sdxLlmTokenize:       lib.func(
      'int sdxLlmTokenize(void *runtime, void *model, const char *text, ' +
      'int32_t add_special, _Out_ int32_t **out_ids, _Out_ int32_t *out_count)'),
    sdxLlmDetokenize:     lib.func(
      'int sdxLlmDetokenize(void *runtime, void *model, const int32_t *ids, ' +
      'int32_t count, int32_t skip_special, _Out_ char **out_text)'),
    sdxVlmExtract:        lib.func(
      'int sdxVlmExtract(void *runtime, const char *model_path, ' +
      'const char *tokenizer_path, const char *input_path, ' +
      'const char *options_json, _Out_ char **out_text)'),
    sdxAudioTranscribe:   lib.func(
      'int sdxAudioTranscribe(void *runtime, const char *model_path, ' +
      'const char *audio_path, const char *options_json, _Out_ char **out_text)'),
    sdxLlmFree:           lib.func('void sdxLlmFree(void *runtime, void *pointer)'),
    sdxLlmGetLastError:   lib.func(
      'int sdxLlmGetLastError(void *runtime, char *buffer, int32_t capacity)'),
  };
}

function lastError(): string {
  if (!rtHandle) return '';
  const buf = Buffer.alloc(4096);
  abi.sdxLlmGetLastError(rtHandle, buf, buf.length);
  return buf.toString('utf8').replace(/\0.*/, '');
}

// ── Request handler ───────────────────────────────────────────────────────────

function handle(op: string, args: unknown[]): unknown {
  switch (op) {

    case 'init': {
      // Set SDX_NATIVE_LIB_DIR before loading the library.
      const [libPath, nativeLibDir] = args as [string | null, string | null];
      if (nativeLibDir && !process.env.SDX_NATIVE_LIB_DIR) {
        process.env.SDX_NATIVE_LIB_DIR = nativeLibDir;
      }
      const resolved = libPath ?? resolveLibsdxLlm();
      abi = loadAbi(resolved);
      rtHandle = abi.sdxLlmCreateRuntime();
      if (rtHandle == null)
        throw new Error('sdxLlmCreateRuntime returned NULL');
      return { libraryPath: resolved, abiVersion: abi.sdxLlmAbiVersion(rtHandle) as number };
    }

    case 'loadModel': {
      const [modelPath, tokenizerPath, optionsJson] = args as [string, string | null, string | null];
      modelHandle = abi.sdxLlmLoadModel(rtHandle, modelPath, tokenizerPath ?? null, optionsJson ?? null);
      if (modelHandle == null)
        throw new Error(`sdxLlmLoadModel("${modelPath}") failed; ${lastError()}`);
      return null;
    }

    case 'infoJson': {
      const outJson: unknown[] = [null];
      const s = abi.sdxLlmInfoJson(rtHandle, modelHandle, outJson) as number;
      if (s !== 0) throw new Error(`sdxLlmInfoJson failed status=${s}; ${lastError()}`);
      // koffi decoded the native char* into a JS string; do NOT call sdxLlmFree here
      // (sdxLlmFree on a JS string causes "free(): invalid pointer")
      return (outJson[0] as string) || '{}';
    }

    case 'tokenize': {
      const [text, addSpecial] = args as [string, boolean];
      const outIds: unknown[] = [null]; const outCount: number[] = [0];
      const s = abi.sdxLlmTokenize(rtHandle, modelHandle, text, addSpecial ? 1 : 0, outIds, outCount) as number;
      if (s !== 0) throw new Error(`sdxLlmTokenize failed status=${s}; ${lastError()}`);
      const count = outCount[0]; const ptr = outIds[0];
      if (ptr == null || count <= 0) return [];
      // koffi _Out_ int32_t** returns a koffi pointer object — use koffi.decode to read the array
      const ids = koffi.decode(ptr as object, 'int32_t', count) as number[];
      abi.sdxLlmFree(rtHandle, ptr);
      return ids;
    }

    case 'detokenize': {
      const [ids, skipSpecial] = args as [number[], boolean];
      const arr = Int32Array.from(ids);
      const outText: unknown[] = [null];
      const s = abi.sdxLlmDetokenize(rtHandle, modelHandle, arr, arr.length, skipSpecial ? 1 : 0, outText) as number;
      if (s !== 0) throw new Error(`sdxLlmDetokenize failed status=${s}; ${lastError()}`);
      return (outText[0] as string) ?? '';
    }

    case 'generate': {
      const [prompt, optionsJson] = args as [string, string | null];
      const outText: unknown[] = [null];
      const s = abi.sdxLlmGenerate(rtHandle, modelHandle, prompt, optionsJson ?? null, outText) as number;
      if (s !== 0) throw new Error(`sdxLlmGenerate failed status=${s}; ${lastError()}`);
      if (outText[0] == null) throw new Error('sdxLlmGenerate returned null text');
      return outText[0] as string;
    }

    case 'lastResultJson': {
      const outJson: unknown[] = [null];
      const s = abi.sdxLlmLastResultJson(rtHandle, modelHandle, outJson) as number;
      if (s !== 0) return null;
      const json = (outJson[0] as string) ?? '';
      if (!json) return null;
      try { return JSON.parse(json); } catch { return null; }
    }

    case 'vlmExtract': {
      const [modelPath, inputPath, tokenizerPath, optionsJson] = args as [string, string, string | null, string | null];
      const outText: unknown[] = [null];
      const s = abi.sdxVlmExtract(rtHandle, modelPath, tokenizerPath ?? null, inputPath, optionsJson ?? null, outText) as number;
      if (s !== 0) throw new Error(`sdxVlmExtract failed status=${s}; ${lastError()}`);
      return (outText[0] as string) ?? '';
    }

    case 'audioTranscribe': {
      const [modelPath, audioPath, optionsJson] = args as [string, string, string | null];
      const outText: unknown[] = [null];
      const s = abi.sdxAudioTranscribe(rtHandle, modelPath, audioPath, optionsJson ?? null, outText) as number;
      if (s !== 0) throw new Error(`sdxAudioTranscribe failed status=${s}; ${lastError()}`);
      return (outText[0] as string) ?? '';
    }

    case 'unloadModel': {
      if (modelHandle) { abi.sdxLlmUnloadModel(rtHandle, modelHandle); modelHandle = null; }
      return null;
    }

    case 'destroyRuntime': {
      if (modelHandle) { abi.sdxLlmUnloadModel(rtHandle, modelHandle); modelHandle = null; }
      if (rtHandle)    { abi.sdxLlmDestroyRuntime(rtHandle); rtHandle = null; }
      return null;
    }

    default:
      throw new Error(`Unknown op: ${op}`);
  }
}

// ── Message loop ──────────────────────────────────────────────────────────────

parentPort!.on('message', (msg: { id: number; op: string; args: unknown[] }) => {
  try {
    const result = handle(msg.op, msg.args);
    parentPort!.postMessage({ id: msg.id, ok: true, result });
  } catch (err) {
    parentPort!.postMessage({ id: msg.id, ok: false, error: (err as Error).message });
  }
});
