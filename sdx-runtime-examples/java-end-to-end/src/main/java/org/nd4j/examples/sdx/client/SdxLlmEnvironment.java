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
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.examples.sdx.client;

import com.sun.jna.Native;
import com.sun.jna.Pointer;

import java.io.File;
import java.nio.charset.StandardCharsets;

/**
 * Global environment for the SDX LLM runtime — analogous to
 * {@code OrtEnvironment} in the ONNX Runtime Java API.
 *
 * <p><b>NOTE:</b> This is a vendored copy kept for reference. The canonical
 * implementation is {@code org.nd4j.dsp.runtime.SdxLlm.Runtime} in the
 * {@code nd4j-sdx} Maven module. New code should depend on {@code nd4j-sdx}
 * and use {@code SdxLlm.Runtime} / {@code SdxLlm.Model} directly.</p>
 *
 * <p>There is typically one {@code SdxLlmEnvironment} per process (or per
 * dedicated generation thread). It owns the native {@code sdx_llm_runtime_t*}
 * pointer and is the factory for {@link SdxLlmModel} objects.
 *
 * <p><b>Threading contract (v1):</b> the runtime handle is bound to the OS
 * thread that created it. Create, use, and destroy the environment on a single
 * thread. For concurrent generation, create one {@code SdxLlmEnvironment} per
 * thread — models are not shared across runtimes.
 *
 * <p>Usage (try-with-resources mirrors the ONNX Runtime pattern):
 * <pre>{@code
 * try (SdxLlmEnvironment env = SdxLlmEnvironment.create()) {
 *     try (SdxLlmModel model = env.loadModel(modelPath, tokenizerPath, null)) {
 *         String text = model.generate("The capital of France is",
 *                 "{\"maxNewTokens\":8,\"sampling\":{\"preset\":\"greedy\"}}");
 *         System.out.println(text);  // " Paris."
 *     }
 * }
 * }</pre>
 *
 * <h3>Library resolution</h3>
 * <ol>
 *   <li>System property {@code sdx.llm.library} or env-var
 *       {@code SDX_LLM_LIBRARY} — explicit path.</li>
 *   <li>{@code SDX_LLM_AOT_HOME/lib/libsdx_llm.so} — preferred; mirrors the
 *       unpacked AOT SDK ZIP layout.</li>
 *   <li>Bare names {@code sdx_llm} / {@code sdx_llm_cpu} tried via JNA's
 *       default library search.</li>
 * </ol>
 *
 * <h3>Side-loaded natives (CRITICAL)</h3>
 * <p>{@code libsdx_llm.so} resolves its side-loaded ND4J/libnd4j natives
 * relative to the <em>host executable</em>, not the JVM. The environment
 * variable {@code SDX_NATIVE_LIB_DIR} must point at the SDK {@code lib/}
 * directory <em>before the first {@link #loadModel} call</em>. A JVM process
 * <b>cannot set process environment after start</b> — the runner must export
 * this before launching the JVM:
 * <pre>{@code
 *   export SDX_LLM_AOT_HOME=/path/to/sdx-sdk
 *   export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib
 *   mvn exec:java -Dexec.mainClass=org.nd4j.examples.sdx.LlmEndToEnd
 * }</pre>
 * A warning is printed at construction time when {@code SDX_LLM_AOT_HOME} is
 * set but {@code SDX_NATIVE_LIB_DIR} is not — catch this early.
 */
public final class SdxLlmEnvironment implements AutoCloseable {

    /** ABI version this wrapper was written against. */
    public static final int SDX_LLM_ABI_VERSION = 1;

    // Visible inside the package for SdxLlmModel
    final SdxLlmAbi api;
    final Pointer runtime;

    private volatile boolean closed = false;

    private SdxLlmEnvironment(SdxLlmAbi api, Pointer runtime) {
        this.api     = api;
        this.runtime = runtime;
    }

    // ── Factory ───────────────────────────────────────────────────────────────

    /**
     * Creates the environment by auto-detecting {@code libsdx_llm.so}.
     *
     * @throws IllegalStateException if the library cannot be found or the
     *                               runtime cannot be initialized.
     */
    public static SdxLlmEnvironment create() {
        warnIfNativeLibDirMissing();
        return create(resolveLibrary());
    }

    /**
     * Creates the environment using an explicit library path.
     *
     * @param libraryPath Absolute path to {@code libsdx_llm.so} (or the bare
     *                    library name for JNA lookup, e.g. {@code "sdx_llm"}).
     */
    public static SdxLlmEnvironment create(String libraryPath) {
        SdxLlmAbi api = Native.load(libraryPath, SdxLlmAbi.class);
        Pointer runtime = api.sdxLlmCreateRuntime();
        if (runtime == null) {
            throw new IllegalStateException(
                "sdxLlmCreateRuntime() returned null — check that " +
                "SDX_NATIVE_LIB_DIR is set before starting the JVM.");
        }
        return new SdxLlmEnvironment(api, runtime);
    }

    // ── API ───────────────────────────────────────────────────────────────────

    /**
     * Returns the ABI version reported by the loaded library.
     */
    public int abiVersion() {
        checkNotClosed();
        return api.sdxLlmAbiVersion(runtime);
    }

    /**
     * Loads a GGUF/SDZ/SameDiff model and builds the generation pipeline.
     * Analogous to {@code new OrtSession(env, modelPath, options)} in ONNX Runtime.
     *
     * <p><b>Note:</b> import warmup takes 30–60 s the first time a GGUF model
     * is loaded. Subsequent calls reuse cached plans.
     *
     * @param modelPath     Path to the model file (e.g. {@code .gguf}).
     * @param tokenizerPath Path to {@code tokenizer.json} or its directory; pass
     *                      {@code null} to probe the model's directory.
     * @param optionsJson   Optional generation defaults JSON, e.g.
     *                      {@code {"maxNewTokens":128,"sampling":{"preset":"greedy"}}}.
     *                      Pass {@code null} for library defaults.
     * @return A new {@link SdxLlmModel}; close it when done.
     * @throws IllegalStateException if loading fails.
     */
    public SdxLlmModel loadModel(String modelPath, String tokenizerPath, String optionsJson) {
        checkNotClosed();
        Pointer model = api.sdxLlmLoadModel(runtime, modelPath, tokenizerPath, optionsJson);
        if (model == null) {
            throw new IllegalStateException(
                "sdxLlmLoadModel failed: " + lastError() +
                "\n  model_path=" + modelPath);
        }
        return new SdxLlmModel(this, model);
    }

    /** Returns the last error string from the native runtime, or empty. */
    public String lastError() {
        if (closed || runtime == null) return "";
        byte[] buf = new byte[2048];
        int len = api.sdxLlmGetLastError(runtime, buf, buf.length);
        if (len <= 0) return "";
        return new String(buf, 0, Math.min(len, buf.length - 1), StandardCharsets.UTF_8);
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            api.sdxLlmDestroyRuntime(runtime);
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("SdxLlmEnvironment has been closed");
        }
    }

    private static void warnIfNativeLibDirMissing() {
        String aotHome = System.getenv("SDX_LLM_AOT_HOME");
        String nativeLibDir = System.getenv("SDX_NATIVE_LIB_DIR");
        if (aotHome != null && !aotHome.isEmpty() && (nativeLibDir == null || nativeLibDir.isEmpty())) {
            System.err.println(
                "[SdxLlmEnvironment] WARNING: SDX_LLM_AOT_HOME is set but SDX_NATIVE_LIB_DIR is not.\n" +
                "  libsdx_llm.so resolves its side-loaded natives relative to the host executable.\n" +
                "  Export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib BEFORE starting the JVM, e.g.:\n" +
                "    export SDX_NATIVE_LIB_DIR=" + aotHome + "/lib");
        }
    }

    private static String resolveLibrary() {
        // 1) Explicit override.
        String explicit = System.getProperty("sdx.llm.library",
                System.getenv("SDX_LLM_LIBRARY"));
        if (explicit != null && !explicit.isEmpty()) {
            return explicit;
        }

        // 2) AOT SDK unpacked at SDX_LLM_AOT_HOME.
        String aotHome = System.getenv("SDX_LLM_AOT_HOME");
        if (aotHome != null && !aotHome.isEmpty()) {
            for (String name : new String[]{"libsdx_llm.so", "sdx_llm.so", "libsdx_llm.dylib"}) {
                File candidate = new File(aotHome, "lib/" + name);
                if (candidate.exists()) {
                    // Set jna.library.path so JNA can find side-loaded deps too.
                    String jnaPath = System.getProperty("jna.library.path", "");
                    String libDir = new File(aotHome, "lib").getAbsolutePath();
                    if (!jnaPath.contains(libDir)) {
                        System.setProperty("jna.library.path",
                            jnaPath.isEmpty() ? libDir : libDir + File.pathSeparator + jnaPath);
                    }
                    return candidate.getAbsolutePath();
                }
            }
        }

        // 3) JNA bare-name fallback.
        return "sdx_llm";
    }
}
