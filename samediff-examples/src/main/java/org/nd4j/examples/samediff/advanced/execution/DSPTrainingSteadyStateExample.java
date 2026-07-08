/* *****************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.examples.samediff.advanced.execution;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * DSP Training with Steady-State Replay.
 *
 * <p>DSP training is <b>implicit</b>: when the compile classifier is enabled,
 * the entire training path (forward + backward + updater + weight update) executes
 * via DynamicShapePlan automatically. No separate "training mode" flag is needed.</p>
 *
 * <h3>How DSP training works:</h3>
 * <ol>
 *   <li>SameDiff's {@code fit()} method builds a <b>training graph</b> that includes:
 *       forward ops, loss computation, gradient ops (backward pass), and updater ops
 *       (Adam/SGD/etc state updates + weight updates).</li>
 *   <li>The DSP compiler compiles this entire training graph into a single plan with
 *       flat integer-indexed slots — no HashMap lookups during execution.</li>
 *   <li>The plan progresses through phases just like inference:
 *       SLOT_BY_SLOT → SHAPES_FROZEN → REPLAYING.</li>
 *   <li>Once in REPLAYING, each training step runs as a single graph replay
 *       (CUDA graph replay on GPU, compiled backend replay on CPU).</li>
 * </ol>
 *
 * <h3>Key requirement: fixed batch size for replay</h3>
 * <p>Graph replay requires <b>frozen shapes</b>. For training, this means you must
 * use a <b>fixed batch size</b>. If the last batch in an epoch is smaller (partial batch),
 * the plan recompiles for that new shape and loses replay state. Strategy: drop the
 * partial last batch, or pad it to the full batch size.</p>
 *
 * <h3>Compile classifier must be enabled:</h3>
 * <pre>
 *   sd.setDspAutoCompileEnabled(true);         // Java plan compilation
 *   sd.setDspNativeAutoCompileEnabled(true);    // Native C++ plan compilation
 * </pre>
 * <p>Both default to true. The compile classifier determines whether a plan is
 * eligible for graph capture/replay based on op traits, shape stability, and
 * pointer stability.</p>
 */
public class DSPTrainingSteadyStateExample {
    private static final Logger log = LoggerFactory.getLogger(DSPTrainingSteadyStateExample.class);

    public static void main(String[] args) {

        // =====================================================================
        // 1. Build a training graph
        // =====================================================================
        log.info("=== Building training graph ===");

        SameDiff sd = SameDiff.create();

        int inputSize = 128;
        int hiddenSize = 64;
        int outputSize = 10;
        int batchSize = 32;   // FIXED batch size — required for steady-state replay

        // Placeholders with explicit batch dimension
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, inputSize);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, outputSize);

        // Two-layer network
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, inputSize, hiddenSize).muli(0.01));
        SDVariable b1 = sd.var("b1", Nd4j.zeros(DataType.FLOAT, hiddenSize));
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, hiddenSize, outputSize).muli(0.01));
        SDVariable b2 = sd.var("b2", Nd4j.zeros(DataType.FLOAT, outputSize));

        SDVariable h1 = sd.nn.relu(input.mmul(w1).add(b1), 0);
        SDVariable logits = h1.mmul(w2).add(b2);
        SDVariable predictions = sd.nn.softmax("predictions", logits, -1);

        // Loss function
        SDVariable loss = sd.loss.softmaxCrossEntropy("loss", label, logits, null);

        log.info("Graph: {} variables, {} ops (includes only forward — "
                + "backward/updater ops are added by setTrainingConfig)", sd.variables().size(), sd.ops().length);

        // =====================================================================
        // 2. Enable DSP diagnostics to observe training plan lifecycle
        // =====================================================================
        log.info("=== Enabling DSP diagnostics ===");

        // Enable COMPILE + EXECUTE + TIMING to observe the training plan's
        // progression to steady-state replay.
        // Equivalent command line: -Dnd4j.dsp.diagnostics=COMPILE,EXECUTE,TIMING
        DspDiagnostics.initialize();
        DspDiagnostics.setCategories(
                DspDiagnostics.COMPILE | DspDiagnostics.EXECUTE | DspDiagnostics.TIMING);
        DspDiagnostics.setLevel(DspDiagnostics.LEVEL_DETAILED);
        log.info("  DSP_DIAG enabled: COMPILE, EXECUTE, TIMING (level=DETAILED)");

        // =====================================================================
        // 3. Enable the compile classifier
        // =====================================================================
        log.info("=== Enabling compile classifier ===");

        // BOTH flags must be true for the training plan to compile and replay.
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        log.info("  dspAutoCompileEnabled:       {}", sd.isDspAutoCompileEnabled());
        log.info("  dspNativeAutoCompileEnabled: {}", sd.isDspNativeAutoCompileEnabled());

        // =====================================================================
        // 3. Configure training
        // =====================================================================
        log.info("=== Configuring training ===");

        TrainingConfig config = TrainingConfig.builder()
                .updater(new Adam(1e-3))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .build();

        sd.setTrainingConfig(config);
        log.info("  Updater: Adam(lr=1e-3)");

        // =====================================================================
        // 4. Train with fixed batch size — observe plan phase progression
        // =====================================================================
        log.info("=== Training loop with DSP phase tracking ===");

        int numSteps = 15;  // enough steps for the plan to reach REPLAYING

        // Track warmup vs steady-state throughput
        long warmupTotalNs = 0;
        long steadyTotalNs = 0;
        int warmupSteps = 0;
        int steadySteps = 0;
        boolean reachedReplay = false;

        for (int step = 0; step < numSteps; step++) {
            // Generate random training data (fixed batch size)
            INDArray inputData = Nd4j.randn(batchSize, inputSize);
            INDArray labelData = Nd4j.zeros(batchSize, outputSize);
            // Random one-hot labels
            for (int i = 0; i < batchSize; i++) {
                labelData.putScalar(new int[]{i, Nd4j.getRandom().nextInt(outputSize)}, 1.0);
            }

            DataSet ds = new DataSet(inputData, labelData);

            long t0 = System.nanoTime();
            sd.fit(ds);
            long elapsed = System.nanoTime() - t0;
            long ms = elapsed / 1_000_000;

            // Track the training plan's phase progression
            DspHandle dsp = sd.dsp();
            if (dsp.isCompiled()) {
                int phase = dsp.planPhase();
                String phaseName = phase == 0 ? "SLOT_BY_SLOT"
                        : phase == 1 ? "SHAPES_FROZEN"
                        : phase == 2 ? "REPLAYING"
                        : phase == 3 ? "REPLAY_BLOCKED"
                        : "UNKNOWN";

                // Per-execution performance metrics: how did this step execute?
                int segsReplayed = dsp.lastExecSegmentsReplayed();
                int segsSlotBySlot = dsp.lastExecSegmentsSlotBySlot();
                int segsTotal = dsp.lastExecSegmentsTotal();

                // Compute samples/sec for this step
                double samplesPerSec = (batchSize * 1_000_000_000.0) / elapsed;

                log.info("  Step {}: {}ms ({} samples/sec)  phase={}  segs[replay={}/sbs={}/total={}]",
                        step, ms, String.format("%.0f", samplesPerSec), phaseName,
                        segsReplayed, segsSlotBySlot, segsTotal);

                // Classify as warmup or steady-state
                if (phase == PlanPhase.REPLAYING.getNativeCode()) {
                    reachedReplay = true;
                    steadyTotalNs += elapsed;
                    steadySteps++;
                } else {
                    warmupTotalNs += elapsed;
                    warmupSteps++;
                }
            } else {
                warmupTotalNs += elapsed;
                warmupSteps++;
                log.info("  Step {}: {}ms  (plan not yet compiled)", step, ms);
            }
        }

        // Report warmup vs steady-state throughput
        log.info("");
        log.info("=== Warmup vs Steady-State Throughput ===");
        if (warmupSteps > 0) {
            double warmupSps = (warmupSteps * batchSize * 1_000_000_000.0) / warmupTotalNs;
            log.info("  Warmup ({} steps): {} samples/sec, {} ms/step",
                    warmupSteps, String.format("%.0f", warmupSps),
                    String.format("%.1f", (double) warmupTotalNs / warmupSteps / 1_000_000));
        }
        if (steadySteps > 0) {
            double steadySps = (steadySteps * batchSize * 1_000_000_000.0) / steadyTotalNs;
            log.info("  Steady-state ({} steps, REPLAYING): {} samples/sec, {} ms/step",
                    steadySteps, String.format("%.0f", steadySps),
                    String.format("%.1f", (double) steadyTotalNs / steadySteps / 1_000_000));
            if (warmupSteps > 0) {
                double warmupSps = (warmupSteps * batchSize * 1_000_000_000.0) / warmupTotalNs;
                log.info("  Speedup: steady-state is {}x faster than warmup",
                        String.format("%.1f", steadySps / warmupSps));
            }
        } else {
            log.info("  Plan did not reach REPLAYING in {} steps — increase numSteps.", numSteps);
        }

        // =====================================================================
        // 5. Verify training plan reached steady state
        // =====================================================================
        log.info("=== Training plan steady-state check ===");

        DspHandle dsp = sd.dsp();
        if (dsp.isCompiled()) {
            int phase = dsp.planPhase();
            boolean inReplay = (phase == PlanPhase.REPLAYING.getNativeCode());
            int numSegs = dsp.numSegments();
            int capturedSegs = dsp.numCapturedGraphSegments();

            log.info("  Plan phase: {} (target: REPLAYING=2)", phase);
            log.info("  Steady-state replay: {}", inReplay);
            log.info("  Total slots: {}", dsp.totalSlots());
            log.info("  Segments: {} total, {} captured for replay", numSegs, capturedSegs);
            log.info("  Total graph replays: {}", dsp.totalGraphReplays());
            log.info("  Pointers stable: {}", dsp.pointersStable());
            log.info("  Compilation sealed: {}", dsp.isCompilationSealed());
            log.info("  Frozen exec count: {}", dsp.frozenExecCount());
            log.info("  Mid-execution recompiles: {}", dsp.midExecutionCompileCount());

            // Capture stats — how many segments captured vs failed?
            DspHandle.CaptureStats cs = dsp.parsedCaptureStats();
            log.info("  Capture stats: captured={}, permFailed={}, nonCapt={}, addrUnstable={}",
                    cs.captured, cs.permFailed, cs.nonCapturable, cs.addrUnstable);

            // Segment breakdown
            for (int s = 0; s < Math.min(numSegs, 10); s++) {
                log.info("    Seg {}: phase={} (3=REPLAYING, 4=SLOT_BY_SLOT), replays={}, backend={}",
                        s, dsp.segmentExecutionPhase(s),
                        dsp.segmentReplayCount(s),
                        dsp.segmentBackendName(s));
            }
            if (numSegs > 10) {
                log.info("    ... ({} more segments)", numSegs - 10);
            }

            // DSP_DIAG: print the diagnostic report from the training run
            String diagReport = DspDiagnostics.getPlanReport();
            if (diagReport != null && !diagReport.isEmpty()) {
                log.info("  DSP_DIAG training plan report (first 500 chars):");
                log.info("    {}", diagReport.substring(0, Math.min(500, diagReport.length())));
            }
            log.info("  DSP_DIAG total events recorded: {}", dsp.diagTotalEventCount());

            if (inReplay) {
                log.info("  TRAINING PLAN IS IN STEADY-STATE REPLAY.");
                log.info("  Each fit() call now executes the full training step");
                log.info("  (forward + backward + updater + weight update) as a single");
                log.info("  graph replay — minimal kernel launch overhead.");
            } else {
                log.info("  Training plan has not reached REPLAYING yet.");
                log.info("  This may happen if the graph has non-capturable ops or");
                log.info("  if shapes are not stable. Additional warmup steps may help.");
                log.info("  Use DSP_DIAG to diagnose: -Dnd4j.dsp.diagnostics=all -Dnd4j.dsp.diagnostics.level=full");
            }
        }

        // Clean up diagnostics for the benchmark section
        DspDiagnostics.setCategories(DspDiagnostics.NONE);

        // =====================================================================
        // 6. Compare: DSP training vs non-DSP training
        // =====================================================================
        log.info("=== Comparing DSP vs non-DSP training ===");

        // Build a fresh graph for comparison
        SameDiff sdNoDsp = SameDiff.create();
        SDVariable in2 = sdNoDsp.placeHolder("input", DataType.FLOAT, -1, inputSize);
        SDVariable lb2 = sdNoDsp.placeHolder("label", DataType.FLOAT, -1, outputSize);
        SDVariable ww1 = sdNoDsp.var("w1", Nd4j.randn(DataType.FLOAT, inputSize, hiddenSize).muli(0.01));
        SDVariable bb1 = sdNoDsp.var("b1", Nd4j.zeros(DataType.FLOAT, hiddenSize));
        SDVariable ww2 = sdNoDsp.var("w2", Nd4j.randn(DataType.FLOAT, hiddenSize, outputSize).muli(0.01));
        SDVariable bb2 = sdNoDsp.var("b2", Nd4j.zeros(DataType.FLOAT, outputSize));
        SDVariable hh1 = sdNoDsp.nn.relu(in2.mmul(ww1).add(bb1), 0);
        SDVariable ll = hh1.mmul(ww2).add(bb2);
        sdNoDsp.loss.softmaxCrossEntropy("loss", lb2, ll, null);

        // DISABLE DSP — forces SLOT_BY_SLOT execution (no graph replay)
        sdNoDsp.setDspAutoCompileEnabled(false);
        sdNoDsp.setDspNativeAutoCompileEnabled(false);
        sdNoDsp.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        sdNoDsp.setTrainingConfig(TrainingConfig.builder()
                .updater(new Adam(1e-3))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .build());

        // Warm up both
        DataSet warmupDs = new DataSet(Nd4j.randn(batchSize, inputSize),
                Nd4j.randn(batchSize, outputSize));
        for (int i = 0; i < 5; i++) {
            sd.fit(warmupDs);
            sdNoDsp.fit(warmupDs);
        }

        // Benchmark
        int benchSteps = 20;
        DataSet benchDs = new DataSet(Nd4j.randn(batchSize, inputSize),
                Nd4j.randn(batchSize, outputSize));

        long dspStart = System.nanoTime();
        for (int i = 0; i < benchSteps; i++) sd.fit(benchDs);
        long dspMs = (System.nanoTime() - dspStart) / 1_000_000;

        long noDspStart = System.nanoTime();
        for (int i = 0; i < benchSteps; i++) sdNoDsp.fit(benchDs);
        long noDspMs = (System.nanoTime() - noDspStart) / 1_000_000;

        double dspSps = (benchSteps * batchSize * 1000.0) / dspMs;
        double noDspSps = (benchSteps * batchSize * 1000.0) / noDspMs;

        log.info("  DSP training:     {} ms for {} steps ({} ms/step, {} samples/sec)",
                dspMs, benchSteps, String.format("%.1f", (double) dspMs / benchSteps),
                String.format("%.0f", dspSps));
        log.info("  Non-DSP training: {} ms for {} steps ({} ms/step, {} samples/sec)",
                noDspMs, benchSteps, String.format("%.1f", (double) noDspMs / benchSteps),
                String.format("%.0f", noDspSps));
        if (noDspMs > 0) {
            log.info("  Speedup: {}x", String.format("%.2f", (double) noDspMs / dspMs));
        }

        // =====================================================================
        // Summary
        // =====================================================================
        log.info("");
        log.info("=== Summary ===");
        log.info("DSP training is implicit:");
        log.info("  - Enable compile classifier: setDspAutoCompileEnabled(true) +");
        log.info("    setDspNativeAutoCompileEnabled(true)");
        log.info("  - Configure training: sd.setTrainingConfig(config)");
        log.info("  - Call sd.fit(dataset) — DSP compiles and replays the full");
        log.info("    training graph (forward + backward + updater + weight update)");
        log.info("");
        log.info("Fixed batch size is key:");
        log.info("  - Graph replay requires frozen shapes");
        log.info("  - Variable batch sizes cause recompilation on shape change");
        log.info("  - Drop or pad partial last batches");
        log.info("");
        log.info("Phase progression:");
        log.info("  SLOT_BY_SLOT → SHAPES_FROZEN → REPLAYING");
        log.info("  Steps 1-2: warmup (shapes tracked)");
        log.info("  Steps 3-4: shapes freeze, compilation");
        log.info("  Steps 5+:  steady-state replay (minimal overhead)");
        log.info("");
        log.info("Performance metrics:");
        log.info("  dsp.lastExecSegmentsReplayed()  — segments in graph replay this step");
        log.info("  dsp.lastExecSegmentsSlotBySlot() — segments in slot-by-slot this step");
        log.info("  dsp.totalGraphReplays()           — cumulative replay count");
        log.info("  dsp.parsedCaptureStats()          — capture success/failure counts");
        log.info("");
        log.info("Throughput measurement:");
        log.info("  samples/sec = (numSteps × batchSize × 1e9) / totalNanos");
        log.info("  Compare warmup (compilation) vs steady-state (REPLAYING)");
        log.info("  Steady-state should be significantly faster");
        log.info("");
        log.info("DSP_DIAG for training:");
        log.info("  DspDiagnostics.setCategories(COMPILE | EXECUTE | TIMING)");
        log.info("  Or: -Dnd4j.dsp.diagnostics=COMPILE,EXECUTE,TIMING");
        log.info("  -Dnd4j.dsp.diagnostics.level=detailed");
        log.info("  -Dnd4j.dsp.diagnostics.file=/tmp/training-dsp.json");

        log.info("**************** DSP Training Steady State Example finished ********************");
    }
}
