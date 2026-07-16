/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.examples.sdx.client;

import com.sun.jna.Library;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.PointerByReference;

/**
 * Low-level JNA binding for the SDX LLM C ABI ({@code sdx_llm_c.h}).
 *
 * <p><b>NOTE:</b> This is a vendored copy kept for reference. The canonical
 * implementation is {@code org.nd4j.dsp.runtime.SdxLlm.NativeApi} in the
 * {@code nd4j-sdx} Maven module
 * ({@code nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx/src/main/java/org/nd4j/dsp/runtime/SdxLlm.java}).
 * New application code should depend on {@code nd4j-sdx} and use
 * {@code SdxLlm.Runtime} / {@code SdxLlm.Model} directly.
 *
 * <p>This interface is <em>package-private</em>. Application code uses the
 * higher-level {@link SdxLlmEnvironment} / {@link SdxLlmModel} API instead.
 *
 * <p>The ABI is implemented by GraalVM {@code @CEntryPoint} methods inside
 * {@code libsdx_llm.so}. It is JVM-free — {@code SdxLlmRuntime} methods bind
 * a GraalVM isolate thread, not a JVM thread.
 *
 * <p><b>Threading contract (v1):</b> each runtime handle is bound to the OS
 * thread that created it. Create, use, and destroy a runtime from one thread.
 * For concurrent generation use one runtime per thread — models cannot be
 * shared across runtimes.
 */
interface SdxLlmAbi extends Library {

    // ── Status codes ─────────────────────────────────────────────────────────

    /** Successful completion. */
    int SDX_LLM_STATUS_OK                = 0;
    /** Caller supplied a null or invalid argument. */
    int SDX_LLM_STATUS_INVALID_ARGUMENT  = 1;
    /** Model file not found or not parseable. */
    int SDX_LLM_STATUS_MODEL_LOAD_FAILED = 3;
    /** Generation / tokenization failed at runtime. */
    int SDX_LLM_STATUS_EXECUTION_FAILED  = 4;
    /** I/O error (file read, audio decode, …). */
    int SDX_LLM_STATUS_IO_ERROR          = 6;

    // ── Runtime lifecycle ─────────────────────────────────────────────────────

    /**
     * Creates a GraalVM isolate (runtime). Returns {@code null} on failure.
     * <p>Each call returns an independent runtime bound to the calling thread.
     */
    Pointer sdxLlmCreateRuntime();

    /**
     * Tears down the runtime. All model handles created in it become invalid.
     * Returns 0 on success.
     */
    int sdxLlmDestroyRuntime(Pointer runtime);

    /**
     * Returns the ABI version compiled into the library. Compare with
     * {@link SdxLlmEnvironment#SDX_LLM_ABI_VERSION}.
     */
    int sdxLlmAbiVersion(Pointer runtime);

    // ── Model lifecycle ───────────────────────────────────────────────────────

    /**
     * Loads a model and builds the generation pipeline.
     *
     * @param runtime        Runtime handle.
     * @param model_path     Path to a {@code .gguf}/{@code .ggml}/{@code .sdz}/{@code .sdnb}.
     * @param tokenizer_path Optional path to a {@code tokenizer.json} file or directory; pass
     *                       {@code null} to let the library probe the model's directory.
     * @param options_json   Optional JSON string, e.g.
     *                       {@code {"maxNewTokens":128,"sampling":{"preset":"greedy"}}}, or
     *                       {@code null} to use defaults.
     * @return Opaque model handle, or {@code null} on failure — call
     *         {@link #sdxLlmGetLastError} for details.
     */
    Pointer sdxLlmLoadModel(Pointer runtime,
                             String model_path,
                             String tokenizer_path,
                             String options_json);

    /**
     * Unloads a model and releases all associated resources.
     */
    int sdxLlmUnloadModel(Pointer runtime, Pointer model);

    // ── Generation ────────────────────────────────────────────────────────────

    /**
     * Blocking text generation.
     *
     * <p>{@code *out_text} receives a malloc'd UTF-8 string owned by the caller.
     * Release it with {@link #sdxLlmFree}. {@code options_json} may override
     * generation settings per call; pass {@code null} to use load-time defaults.
     * The pipeline's compiled plan and KV state are reused across calls.
     *
     * @param out_text  Receives a pointer to a NUL-terminated UTF-8 string.
     */
    int sdxLlmGenerate(Pointer runtime,
                        Pointer model,
                        String prompt,
                        String options_json,
                        PointerByReference out_text);

    /**
     * Returns a JSON string containing stats for the most recent
     * {@link #sdxLlmGenerate} on this model: token counts, timings, tok/s,
     * and finish reason. Caller frees {@code *out_json} with
     * {@link #sdxLlmFree}.
     */
    int sdxLlmLastResultJson(Pointer runtime,
                              Pointer model,
                              PointerByReference out_json);

    /**
     * Returns a model/tokenizer summary JSON (inputs, outputs, vocab size,
     * chat template flag). Caller frees {@code *out_json} with
     * {@link #sdxLlmFree}.
     */
    int sdxLlmInfoJson(Pointer runtime,
                        Pointer model,
                        PointerByReference out_json);

    // ── Tokenization ──────────────────────────────────────────────────────────

    /**
     * Encodes {@code text} to token IDs. {@code *out_ids} receives a malloc'd
     * {@code int32} array; {@code *out_count} receives its length. Caller frees
     * {@code *out_ids} with {@link #sdxLlmFree}.
     *
     * @param add_special_tokens Non-zero to prepend/append BOS/EOS tokens.
     */
    int sdxLlmTokenize(Pointer runtime,
                        Pointer model,
                        String text,
                        int add_special_tokens,
                        PointerByReference out_ids,
                        IntByReference out_count);

    /**
     * Decodes {@code count} token IDs to text. {@code *out_text} receives a
     * malloc'd UTF-8 string. Caller frees with {@link #sdxLlmFree}.
     *
     * @param skip_special_tokens Non-zero to omit special tokens from the output.
     */
    int sdxLlmDetokenize(Pointer runtime,
                          Pointer model,
                          int[] ids,
                          int count,
                          int skip_special_tokens,
                          PointerByReference out_text);

    // ── VLM + audio ───────────────────────────────────────────────────────────

    /**
     * VLM document extraction (SmolDocling) — stateless, loads/releases per call.
     * Caller frees {@code *out_text} with {@link #sdxLlmFree}.
     *
     * @param options_json  e.g. {@code {"maxNewTokens":512,"format":"doctags"}}.
     */
    int sdxVlmExtract(Pointer runtime,
                       String model_path,
                       String tokenizer_path,
                       String input_path,
                       String options_json,
                       PointerByReference out_text);

    /**
     * Whisper speech-to-text — stateless, loads/releases per call.
     * Caller frees {@code *out_text} with {@link #sdxLlmFree}.
     *
     * @param options_json  e.g. {@code {"language":"en","maxNewTokens":448}}.
     */
    int sdxAudioTranscribe(Pointer runtime,
                            String model_path,
                            String audio_path,
                            String options_json,
                            PointerByReference out_text);

    // ── Memory + errors ───────────────────────────────────────────────────────

    /**
     * Releases any pointer returned through an out-parameter of this ABI.
     * Always prefer this over {@code free()} — the library manages its own heap.
     */
    void sdxLlmFree(Pointer runtime, Pointer pointer);

    /**
     * Copies the last error message (UTF-8, NUL-terminated) into {@code buffer}.
     *
     * @param buffer    Caller-allocated buffer (may be null if capacity is 0).
     * @param capacity  Size of {@code buffer} in bytes.
     * @return Full message length in bytes (excluding the NUL), or 0 if no error.
     */
    int sdxLlmGetLastError(Pointer runtime, byte[] buffer, int capacity);
}
