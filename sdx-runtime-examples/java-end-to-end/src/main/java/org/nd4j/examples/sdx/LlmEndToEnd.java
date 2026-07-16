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
package org.nd4j.examples.sdx;

// Canonical wrapper: org.nd4j.dsp.runtime.SdxLlm (nd4j-sdx Maven module).
// The vendored client/ classes are kept for reference but are no longer used here.
import org.nd4j.dsp.runtime.SdxLlm;

/**
 * End-to-end walkthrough of the SDX LLM C ABI ({@code sdx_llm_c.h}) from Java.
 *
 * <p>This example embeds the <em>JVM-free</em> AOT-compiled LLM library
 * ({@code libsdx_llm.so}) from a standard Java process using JNA — no
 * {@code samediff-llm.jar}, no ND4J classloader, no JVM spinup in the
 * library. The library was compiled with GraalVM native-image and exports a
 * plain C ABI that any language can bind. The POINT of this example is showing
 * that a JVM host app can embed the AOT library without any Java LLM
 * dependencies on its classpath.
 *
 * <p>What this example does:</p>
 * <ol>
 *   <li>Creates an {@link SdxLlmEnvironment} (GraalVM isolate + runtime).</li>
 *   <li>Checks the ABI version for compatibility.</li>
 *   <li>Loads a GGUF model and tokenizer through the C ABI via JNA.</li>
 *   <li>Queries model info JSON (vocab size, chat template, etc.).</li>
 *   <li>Tokenizes a prompt and round-trips it through detokenize.</li>
 *   <li>Generates 8 tokens with greedy sampling — asserts the output
 *       contains {@code "Paris"}.</li>
 *   <li>Reads the stats JSON (tok/s, token counts, finish reason).</li>
 *   <li>Demonstrates the error path ({@code sdxLlmGetLastError}).</li>
 * </ol>
 *
 * <h3>Prerequisites</h3>
 * <pre>{@code
 * export SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8
 * export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib
 * }</pre>
 * {@code SDX_NATIVE_LIB_DIR} MUST be set <em>before</em> the JVM starts —
 * a JVM process cannot set process environment after launch, and
 * {@code libsdx_llm.so} resolves side-loaded natives (libnd4jcpu, libjemalloc,
 * …) relative to the host executable using this variable.
 *
 * <h3>Run</h3>
 * <pre>{@code
 * export SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8
 * export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib
 * /home/agibsonccc/dev-apps/mvn/bin/mvn -q compile exec:java \
 *   -Dexec.mainClass=org.nd4j.examples.sdx.LlmEndToEnd \
 *   -Dexec.args="$HOME/.cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf \
 *                $HOME/.cache/dl4j-llm-models/qwen35-0.8B-tokenizer.json"
 * }</pre>
 *
 * <p>First run takes 1–3 minutes (GGUF import + DSP warmup). Subsequent runs
 * are much faster because the compiled plan is cached.
 */
public class LlmEndToEnd {

    /**
     * Default model path — override via command-line arg[0].
     */
    private static final String DEFAULT_MODEL_PATH =
        System.getProperty("user.home") +
        "/.cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf";

    /**
     * Default tokenizer path — override via command-line arg[1].
     */
    private static final String DEFAULT_TOKENIZER_PATH =
        System.getProperty("user.home") +
        "/.cache/dl4j-llm-models/qwen35-0.8B-tokenizer.json";

    /** Prompt used for the greedy-generation assertion. */
    private static final String PROBE_PROMPT = "The capital of France is";

    /** Expected substring in the generated continuation. */
    private static final String EXPECTED_SUBSTRING = "Paris";

    /** Generation options: 8 tokens, greedy. */
    private static final String GREEDY_8 =
        "{\"maxNewTokens\":8,\"sampling\":{\"preset\":\"greedy\"}}";

    public static void main(String[] args) throws Exception {
        String modelPath     = args.length > 0 ? args[0] : DEFAULT_MODEL_PATH;
        String tokenizerPath = args.length > 1 ? args[1] : DEFAULT_TOKENIZER_PATH;

        System.out.println("=== SDX LLM Java end-to-end ===");
        System.out.println("Model    : " + modelPath);
        System.out.println("Tokenizer: " + tokenizerPath);
        System.out.println();

        // ── Step 1: create the environment ────────────────────────────────────
        System.out.println("== Step 1: create SdxLlm.Runtime (analogous to OrtEnvironment) ==");
        try (SdxLlm.Runtime env = SdxLlm.Runtime.create()) {
            int abiVer = env.abiVersion();
            System.out.printf("ABI version: %d (expected %d)%n%n",
                abiVer, SdxLlm.SDX_LLM_ABI_VERSION);
            if (abiVer != SdxLlm.SDX_LLM_ABI_VERSION) {
                System.err.println("WARNING: ABI version mismatch — wrapper may be out of date.");
            }

            // ── Step 2: load the model ─────────────────────────────────────────
            System.out.println("== Step 2: loadModel (analogous to OrtSession constructor) ==");
            System.out.println("Loading model — first load compiles DSP plan (1-3 min on CPU)…");
            try (SdxLlm.Model model = env.loadModel(modelPath, tokenizerPath, null)) {
                System.out.println("Model loaded successfully.");
                System.out.println();

                // ── Step 3: model info ─────────────────────────────────────────
                System.out.println("== Step 3: model info JSON ==");
                String info = model.infoJson();
                System.out.println(abbreviate(info, 400));
                System.out.println();

                // ── Step 4: tokenization round-trip ───────────────────────────
                System.out.println("== Step 4: tokenize / detokenize ==");
                int[] ids = model.tokenize(PROBE_PROMPT, false);
                System.out.printf("tokenize(\"%s\") → %d tokens: %s%n",
                    PROBE_PROMPT, ids.length, tokStr(ids));
                String back = model.detokenize(ids, true);
                System.out.printf("detokenize → \"%s\"%n%n", back);

                // ── Step 5: greedy generation ──────────────────────────────────
                System.out.println("== Step 5: generate (greedy, 8 tokens) ==");
                System.out.println("Prompt  : \"" + PROBE_PROMPT + "\"");
                long t0 = System.currentTimeMillis();
                String generated = model.generate(PROBE_PROMPT, GREEDY_8);
                long ms = System.currentTimeMillis() - t0;
                System.out.printf("Output  : \"%s\"%n", generated);
                System.out.printf("Elapsed : %d ms%n%n", ms);

                // Assert the key fact.
                boolean containsParis = generated.contains(EXPECTED_SUBSTRING);
                System.out.printf("Contains \"%s\": %s%n%n",
                    EXPECTED_SUBSTRING, containsParis ? "YES ✓" : "NO ✗");

                // ── Step 6: stats JSON ─────────────────────────────────────────
                System.out.println("== Step 6: lastResultJson (tok/s, finish reason, …) ==");
                String stats = model.lastResultJson();
                System.out.println(abbreviate(stats, 400));
                System.out.println();

                // ── Step 7: error handling ─────────────────────────────────────
                System.out.println("== Step 7: error handling ==");
                try (SdxLlm.Model bogus = env.loadModel("/definitely/not/a/model.gguf", null, null)) {
                    System.err.println("UNEXPECTED: bogus load succeeded");
                    System.exit(1);
                } catch (IllegalStateException e) {
                    System.out.println("Loading a bogus path threw: " + e.getMessage());
                }
                System.out.println();

                if (!containsParis) {
                    System.err.println("FAILURE: generated text does not contain \"" +
                        EXPECTED_SUBSTRING + "\".");
                    System.exit(1);
                }
                System.out.println(
                    "SUCCESS: SDX LLM C ABI verified from Java (no samediff-llm on classpath).");
            }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static String abbreviate(String s, int maxLen) {
        if (s == null) return "(null)";
        if (s.length() <= maxLen) return s;
        return s.substring(0, maxLen) + " … [" + s.length() + " chars total]";
    }

    private static String tokStr(int[] ids) {
        if (ids.length == 0) return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < Math.min(ids.length, 8); i++) {
            if (i > 0) sb.append(", ");
            sb.append(ids[i]);
        }
        if (ids.length > 8) sb.append(", …");
        sb.append("]");
        return sb.toString();
    }
}
