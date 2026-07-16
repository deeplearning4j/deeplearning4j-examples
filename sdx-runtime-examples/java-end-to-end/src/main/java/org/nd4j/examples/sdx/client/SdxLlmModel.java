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

import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.PointerByReference;

import java.nio.charset.StandardCharsets;

/**
 * A loaded LLM/VLM model in the SDX runtime.
 *
 * <p><b>NOTE:</b> This is a vendored copy kept for reference. The canonical
 * implementation is {@code org.nd4j.dsp.runtime.SdxLlm.Model} in the
 * {@code nd4j-sdx} Maven module. New code should depend on {@code nd4j-sdx}.</p>
 *
 * <p>Obtain via {@link SdxLlmEnvironment#loadModel}. The model is thread-bound
 * to the same OS thread as its parent {@link SdxLlmEnvironment}.
 *
 * <p>Usage:
 * <pre>{@code
 * try (SdxLlmModel model = env.loadModel(modelPath, tokenizerPath, null)) {
 *
 *     // Greedy generation — KV state and plan are reused between calls
 *     String text = model.generate(
 *         "The capital of France is",
 *         "{\"maxNewTokens\":8,\"sampling\":{\"preset\":\"greedy\"}}");
 *     System.out.println(text);  // " Paris."
 *
 *     // Per-call stats
 *     String stats = model.lastResultJson();  // JSON with tok/s, token counts, …
 *
 *     // Model info
 *     String info = model.infoJson();  // vocab size, chat template flag, …
 *
 *     // Tokenization round-trip
 *     int[] ids = model.tokenize("Hello world", true);
 *     String back = model.detokenize(ids, true);
 * }
 * }</pre>
 *
 * <p><b>Threading:</b> same thread constraint as {@link SdxLlmEnvironment}.
 * Do not share a model across threads; instead give each thread its own
 * {@link SdxLlmEnvironment} and load the model in each.
 */
public final class SdxLlmModel implements AutoCloseable {

    private final SdxLlmEnvironment env;
    private Pointer modelHandle;
    private volatile boolean closed = false;

    SdxLlmModel(SdxLlmEnvironment env, Pointer modelHandle) {
        this.env         = env;
        this.modelHandle = modelHandle;
    }

    // ── Generation ────────────────────────────────────────────────────────────

    /**
     * Generates text for {@code prompt} and returns the generated continuation.
     *
     * <p>This is a blocking call. On first call the model's SameDiff graph is
     * compiled (DSP warmup); subsequent calls reuse the captured plan and KV
     * state for the same input length, making them much faster.
     *
     * @param prompt      The input prompt (UTF-8).
     * @param optionsJson Optional per-call overrides, e.g.
     *                    {@code {"maxNewTokens":64,"sampling":{"temperature":0.8}}}.
     *                    Pass {@code null} to use load-time defaults.
     * @return The generated text (the continuation, not the prompt).
     * @throws IllegalStateException if generation fails.
     */
    public String generate(String prompt, String optionsJson) {
        checkNotClosed();
        PointerByReference outText = new PointerByReference();
        int status = env.api.sdxLlmGenerate(env.runtime, modelHandle, prompt, optionsJson, outText);
        if (status != SdxLlmAbi.SDX_LLM_STATUS_OK) {
            throw new IllegalStateException(
                "sdxLlmGenerate failed (status=" + status + "): " + env.lastError());
        }
        return readAndFree(outText.getValue());
    }

    /**
     * Convenience overload — uses load-time generation defaults.
     */
    public String generate(String prompt) {
        return generate(prompt, null);
    }

    // ── Stats + info ──────────────────────────────────────────────────────────

    /**
     * Returns a JSON string with stats for the most recent {@link #generate}
     * call: token counts, timings, tok/s, finish reason.
     *
     * @throws IllegalStateException if the ABI call fails.
     */
    public String lastResultJson() {
        checkNotClosed();
        PointerByReference outJson = new PointerByReference();
        int status = env.api.sdxLlmLastResultJson(env.runtime, modelHandle, outJson);
        if (status != SdxLlmAbi.SDX_LLM_STATUS_OK) {
            throw new IllegalStateException(
                "sdxLlmLastResultJson failed (status=" + status + "): " + env.lastError());
        }
        return readAndFree(outJson.getValue());
    }

    /**
     * Returns a JSON summary of this model and its tokenizer: inputs, outputs,
     * vocab size, and whether a chat template is present.
     *
     * @throws IllegalStateException if the ABI call fails.
     */
    public String infoJson() {
        checkNotClosed();
        PointerByReference outJson = new PointerByReference();
        int status = env.api.sdxLlmInfoJson(env.runtime, modelHandle, outJson);
        if (status != SdxLlmAbi.SDX_LLM_STATUS_OK) {
            throw new IllegalStateException(
                "sdxLlmInfoJson failed (status=" + status + "): " + env.lastError());
        }
        return readAndFree(outJson.getValue());
    }

    // ── Tokenization ──────────────────────────────────────────────────────────

    /**
     * Encodes {@code text} to an array of token IDs.
     *
     * @param text             Input text (UTF-8).
     * @param addSpecialTokens {@code true} to prepend/append BOS/EOS tokens.
     * @return Array of token IDs.
     * @throws IllegalStateException if tokenization fails.
     */
    public int[] tokenize(String text, boolean addSpecialTokens) {
        checkNotClosed();
        PointerByReference outIds = new PointerByReference();
        IntByReference outCount = new IntByReference();
        int status = env.api.sdxLlmTokenize(
            env.runtime, modelHandle, text,
            addSpecialTokens ? 1 : 0,
            outIds, outCount);
        if (status != SdxLlmAbi.SDX_LLM_STATUS_OK) {
            throw new IllegalStateException(
                "sdxLlmTokenize failed (status=" + status + "): " + env.lastError());
        }
        int count = outCount.getValue();
        Pointer idsPtr = outIds.getValue();
        int[] ids = idsPtr.getIntArray(0, count);
        env.api.sdxLlmFree(env.runtime, idsPtr);
        return ids;
    }

    /**
     * Decodes an array of token IDs back to text.
     *
     * @param ids                Token ID array.
     * @param skipSpecialTokens  {@code true} to omit BOS/EOS/pad tokens from output.
     * @return Decoded text (UTF-8).
     * @throws IllegalStateException if detokenization fails.
     */
    public String detokenize(int[] ids, boolean skipSpecialTokens) {
        checkNotClosed();
        PointerByReference outText = new PointerByReference();
        int status = env.api.sdxLlmDetokenize(
            env.runtime, modelHandle,
            ids, ids.length,
            skipSpecialTokens ? 1 : 0,
            outText);
        if (status != SdxLlmAbi.SDX_LLM_STATUS_OK) {
            throw new IllegalStateException(
                "sdxLlmDetokenize failed (status=" + status + "): " + env.lastError());
        }
        return readAndFree(outText.getValue());
    }

    // ── Resource management ───────────────────────────────────────────────────

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            if (modelHandle != null) {
                env.api.sdxLlmUnloadModel(env.runtime, modelHandle);
                modelHandle = null;
            }
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("SdxLlmModel has been closed");
        }
    }

    /**
     * Reads the NUL-terminated string at {@code ptr}, frees it via the ABI's
     * {@code sdxLlmFree}, and returns the Java string.
     */
    private String readAndFree(Pointer ptr) {
        if (ptr == null) return "";
        try {
            String value = ptr.getString(0, StandardCharsets.UTF_8.name());
            return value == null ? "" : value;
        } finally {
            env.api.sdxLlmFree(env.runtime, ptr);
        }
    }
}
