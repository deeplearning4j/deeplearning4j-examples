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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.examples.sdx.client.SdxEnvironment;
import org.nd4j.examples.sdx.client.SdxExecutionReport;
import org.nd4j.examples.sdx.client.SdxSession;
import org.nd4j.examples.sdx.client.SdxSessionOptions;
import org.nd4j.examples.sdx.client.SdxTensor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * End-to-end walkthrough of the SDX runtime C ABI ({@code dsp_runtime_c.h}) —
 * the exportable, language-agnostic serving SDK with DSP execution built in.
 *
 * <p>The client-side API ({@link SdxEnvironment}, {@link SdxSession},
 * {@link SdxTensor}) is modelled after the ONNX Runtime Java API
 * ({@code OrtEnvironment} / {@code OrtSession} / {@code OnnxTensor}) so that
 * Java developers familiar with ONNX Runtime find the patterns immediately
 * recognisable.  Try-with-resources, named-input maps, and builder-pattern
 * options are used throughout.
 *
 * <p>What this example does:</p>
 * <ol>
 *   <li>Builds a small MLP in SameDiff and saves it as a {@code .sdz} bundle.</li>
 *   <li>Loads the bundle through {@link SdxEnvironment#openSession} — internally
 *       this binds the exported {@code sdx*} C symbols via JNA, the exact route
 *       every non-JVM language binding (Python, Rust, C#, Kotlin, Swift) takes.</li>
 *   <li>Discovers the plan's input contract with {@code sdxGetNumInputs} /
 *       {@code sdxGetInputName}: external inputs cover the model's constants,
 *       variables (weights!), and placeholders, bound positionally.</li>
 *   <li>Runs warmup steps, freezes shapes ({@code sdxFreezeShapes}) to enter the
 *       DSP replay fast path, then runs again.</li>
 *   <li>Reads the execution report: plan phase, execution count, applied
 *       backend, fallback flag, and per-run wall time.</li>
 *   <li>Verifies every output against the SameDiff reference and demonstrates
 *       the error path ({@code sdxGetLastError}).</li>
 * </ol>
 *
 * <p>Run with: {@code mvn -q compile exec:java} (CPU backend by default; switch
 * the {@code nd4j.backend} property in the pom for CUDA).</p>
 */
public class SdxRuntimeEndToEndExample {

    public static void main(String[] args) throws Exception {

        // ── Step 1: build a small MLP and save it as .sdz ────────────────────
        // probs = softmax(relu(x·W1 + b1)·W2 + b2), x: [batch, 4] float
        System.out.println("== Step 1: build a SameDiff MLP and save it as .sdz ==");
        SameDiff sd = SameDiff.create();
        SDVariable x  = sd.placeHolder("x",  DataType.FLOAT, -1, 4);
        SDVariable w1 = sd.var("w1", Nd4j.linspace(-1.0, 1.0, 32,  DataType.FLOAT).reshape('c', 4, 8));
        SDVariable b1 = sd.var("b1", Nd4j.linspace( 0.0, 0.7,  8,  DataType.FLOAT));
        SDVariable w2 = sd.var("w2", Nd4j.linspace( 1.0,-1.0, 24,  DataType.FLOAT).reshape('c', 8, 3));
        SDVariable b2 = sd.var("b2", Nd4j.linspace(-0.1, 0.1,  3,  DataType.FLOAT));
        SDVariable hidden = sd.nn.relu(x.mmul(w1).add(b1), 0.0);
        sd.nn.softmax("probs", hidden.mmul(w2).add(b2), 1);

        File sdzFile = File.createTempFile("sdx-example-mlp-", ".sdz");
        sdzFile.deleteOnExit();
        SDZSerializer.save(sd, sdzFile, false, Collections.emptyMap());
        System.out.printf("Saved model: %s (%d bytes)%n%n", sdzFile.getAbsolutePath(), sdzFile.length());

        // ── Step 2: open the environment (analogous to OrtEnvironment) ────────
        System.out.println("== Step 2: create SdxEnvironment (analogous to OrtEnvironment) ==");

        try (SdxEnvironment env = SdxEnvironment.create()) {
            System.out.println("SDX runtime ABI version: " + env.abiVersion());

            // ── Step 3: open a session (analogous to OrtSession) ──────────────
            System.out.println("\n== Step 3: openSession (analogous to OrtSession) ==");
            SdxSessionOptions options = new SdxSessionOptions();
            // options.withBackend(SdxSessionOptions.SdxBackend.CUDA_GRAPHS)
            //        .withGpuTarget(0);  // uncomment for explicit GPU config

            try (SdxSession session = env.openSession(
                    sdzFile.getAbsolutePath(), new String[]{"probs"}, options)) {

                int exitCode = runInference(session, sd, env);

                // ── Step 8: the error path is part of the ABI too ─────────────
                System.out.println("== Step 8: error handling ==");
                try {
                    env.openSession("/definitely/not/a/model.sdz", new String[]{"probs"});
                    System.err.println("Unexpected: bogus load succeeded");
                    System.exit(1);
                } catch (IllegalStateException e) {
                    System.out.println("Loading a bogus path threw: " + e.getMessage() + "\n");
                }

                System.out.println(exitCode == 0
                        ? "SUCCESS: C ABI outputs matched the SameDiff reference on every run."
                        : "FAILURE: C ABI outputs diverged from the SameDiff reference.");
                if (exitCode != 0) {
                    System.exit(exitCode);
                }
            }
        }
    }

    // ── Inference walkthrough ─────────────────────────────────────────────────

    private static int runInference(SdxSession session, SameDiff sd, SdxEnvironment env) {

        // ── Step 4: discover the input contract ─────────────────────────────
        // External inputs are the model's constants + variables + placeholders,
        // bound positionally in plan order.  inputNames() is how a generic
        // client learns what to feed and in which slot.
        System.out.println("\n== Step 4: discover the plan's input contract ==");
        List<String> inputNames = session.inputNames();
        System.out.printf("Plan expects %d external inputs, %d outputs:%n",
                inputNames.size(), session.numOutputs());
        for (int i = 0; i < inputNames.size(); i++) {
            String name = inputNames.get(i);
            INDArray value = valueFor(sd, name, 1);
            System.out.printf("  input[%d] = \"%s\" shape=%s%n",
                    i, name, java.util.Arrays.toString(value.shape()));
        }

        // Mark lifecycle hints: the batch placeholder changes shape between
        // runs; the weights keep value AND shape.  Marking lets the DSP engine
        // pick the right staging strategy per input.
        for (String name : inputNames) {
            if ("x".equals(name)) {
                session.markInputPlaceholder(name);
            }
        }

        // ── Step 5: warmup runs (each verified against the reference) ───────
        System.out.println("\n== Step 5: warmup runs ==");
        for (int step = 1; step <= 3; step++) {
            if (!runOnceAndVerify(session, sd, step)) {
                return 1;
            }
        }

        // ── Step 6: freeze shapes → replay fast path ────────────────────────
        System.out.println("\n== Step 6: session.freezeShapes() -> DSP replay fast path ==");
        session.freezeShapes();
        System.out.println("Plan phase after freeze: " + session.planPhaseName());
        for (int step = 4; step <= 6; step++) {
            if (!runOnceAndVerify(session, sd, step)) {
                return 1;
            }
        }

        // ── Step 7: execution-report telemetry ──────────────────────────────
        System.out.println("\n== Step 7: session.getExecutionReport() ==");
        SdxExecutionReport report = session.getExecutionReport();
        System.out.printf("  status_code       = %d%n",   report.statusCode());
        System.out.printf("  requestedBackend  = %s%n",   report.requestedBackendName());
        System.out.printf("  appliedBackend    = %s%n",   report.appliedBackendName());
        System.out.printf("  usedFallback      = %s%n",   report.fallbackDescription());
        System.out.printf("  planPhase         = %s%n",   report.planPhaseName());
        System.out.printf("  executionCount    = %d%n",   report.executionCount());
        System.out.printf("  executionTime     = %.3f ms%n%n", report.executionTimeMs());

        return 0;
    }

    /**
     * Builds a named-input map, calls {@link SdxSession#runWithShapes}, and
     * compares the output against the SameDiff reference — mirroring the
     * canonical ONNX Runtime usage pattern:
     * <pre>{@code
     * Map<String, OnnxTensor> inputs = Map.of("x", tensor);
     * try (OrtSession.Result result = session.run(inputs)) { ... }
     * }</pre>
     */
    private static boolean runOnceAndVerify(SdxSession session, SameDiff sd, int step) {
        int batch = 2;
        INDArray xValue = Nd4j.linspace(0.1 * step, 0.1 * step + 0.7, batch * 4, DataType.FLOAT)
                .reshape('c', batch, 4);

        // Reference result through the normal SameDiff engine.
        Map<String, INDArray> refOut = sd.output(
                Collections.singletonMap("x", xValue),
                Collections.singletonList("probs"));
        float[] expected = refOut.get("probs").dup('c').data().asFloat();

        // Build the named-input map: weights come from the SameDiff graph
        // (in production they come from the model provider); the placeholder
        // is the per-request batch.
        Map<String, SdxTensor> inputs = new LinkedHashMap<>();
        for (String name : session.inputNames()) {
            // Use the same xValue that went into the SameDiff reference call,
            // so the C ABI path and the reference are fed identical data.
            INDArray arr = "x".equals(name) ? xValue : valueFor(sd, name, batch);
            inputs.put(name, SdxTensor.fromArray(arr.dup('c').data().asFloat(), arr.shape()));
        }

        // Output shape is known from the model contract: probs[batch, 3].
        Map<String, long[]> outputShapes = Collections.singletonMap("probs", new long[]{batch, 3});

        long start = System.nanoTime();
        Map<String, SdxTensor> outputs = session.runWithShapes(inputs, outputShapes);
        long tookUs = (System.nanoTime() - start) / 1_000;

        float[] actual = outputs.get("probs").toFloatArray();
        float maxDiff = 0f;
        for (int i = 0; i < expected.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(expected[i] - actual[i]));
        }
        boolean match = maxDiff <= 1e-4f;
        System.out.printf("  run %d: phase=%-22s execCount=%d  %d us  maxDiff=%.2e  %s%n",
                step, session.planPhaseName(), session.executionCount(),
                tookUs, maxDiff, match ? "MATCH" : "MISMATCH");
        return match;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static INDArray valueFor(SameDiff sd, String name, int batch) {
        if ("x".equals(name)) {
            return Nd4j.zeros(DataType.FLOAT, batch, 4);
        }
        INDArray arr = sd.getVariable(name).getArr();
        if (arr == null) {
            throw new IllegalStateException("No value for plan input '" + name + "'");
        }
        return arr;
    }
}
