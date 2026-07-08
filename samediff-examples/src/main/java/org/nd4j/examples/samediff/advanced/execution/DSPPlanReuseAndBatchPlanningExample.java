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
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.autodiff.samediff.execution.DspCompilationMode;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * DSP Plan Reuse and Batch Planning — Reaching Steady-State Replay for Inference.
 *
 * <p>This example demonstrates how the DSP (Dynamic Shape Plan) system compiles,
 * caches, and replays execution plans for inference workloads. The key insight is
 * that DSP plans are <b>keyed by input shapes</b>: when you call {@code sd.output()}
 * with the same input shapes as a previous call, the cached plan is reused
 * (no recompilation). When shapes change, a new plan is compiled and cached.</p>
 *
 * <h3>The path to steady-state replay:</h3>
 * <pre>
 *   SLOT_BY_SLOT → SHAPES_FROZEN → REPLAYING
 *      (warmup)     (shapes lock)   (graph replay — near-zero overhead)
 * </pre>
 *
 * <ol>
 *   <li><b>SLOT_BY_SLOT</b> — First execution. Ops run individually. Shapes and
 *       pointers are tracked.</li>
 *   <li><b>SHAPES_FROZEN</b> — After repeated calls with the same shapes, DSP detects
 *       that shapes are stable and freezes them. Shape inference is skipped.</li>
 *   <li><b>REPLAYING</b> — Shapes frozen + pointers stable + backend compiled. On CUDA,
 *       this means CUDA graph replay with near-zero kernel launch overhead.
 *       On CPU, this means compiled backend replay (oneDNN/MLX/OpenVINO).</li>
 * </ol>
 *
 * <h3>Batch planning strategy:</h3>
 * <p>For production inference servers that receive variable batch sizes, the recommended
 * pattern is to <b>pre-warm</b> the plan cache with expected batch sizes, then rely on
 * cached plan reuse during serving. Each unique batch size gets its own compiled plan.</p>
 *
 * <h3>Important: Enable the compile classifier</h3>
 * <p>The compile classifier ({@code sd.setDspAutoCompileEnabled(true)} +
 * {@code sd.setDspNativeAutoCompileEnabled(true)}) must be enabled for DSP to
 * automatically compile and replay plans. Both default to true, but if you've
 * explicitly disabled them, plans will execute in SLOT_BY_SLOT mode only.</p>
 *
 * @see DspHandle
 * @see DspCompilationMode
 * @see PlanPhase
 */
public class DSPPlanReuseAndBatchPlanningExample {
    private static final Logger log = LoggerFactory.getLogger(DSPPlanReuseAndBatchPlanningExample.class);

    public static void main(String[] args) {

        // =====================================================================
        // 1. Build a multi-layer graph (simulates a real inference model)
        // =====================================================================
        log.info("=== Building inference graph ===");

        SameDiff sd = SameDiff.create();

        // Placeholders: batch dimension is dynamic (-1)
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 256);

        // Three dense layers with ReLU activations
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, 256, 128).muli(0.01));
        SDVariable b1 = sd.var("b1", Nd4j.zeros(DataType.FLOAT, 128));
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, 128, 64).muli(0.01));
        SDVariable b2 = sd.var("b2", Nd4j.zeros(DataType.FLOAT, 64));
        SDVariable w3 = sd.var("w3", Nd4j.randn(DataType.FLOAT, 64, 10).muli(0.01));
        SDVariable b3 = sd.var("b3", Nd4j.zeros(DataType.FLOAT, 10));

        SDVariable h1 = sd.nn.relu(input.mmul(w1).add(b1), 0);
        SDVariable h2 = sd.nn.relu(h1.mmul(w2).add(b2), 0);
        SDVariable logits = h2.mmul(w3).add(b3);
        SDVariable output = sd.nn.softmax("output", logits, -1);

        log.info("Graph: {} variables, {} ops", sd.variables().size(), sd.ops().length);

        // =====================================================================
        // 2. Enable DSP diagnostics to observe plan lifecycle
        // =====================================================================
        log.info("=== Enabling DSP diagnostics ===");

        // DspDiagnostics provides 18 category-based diagnostic channels.
        // Enable COMPILE + EXECUTE + SEGMENT + GRAPH_REPLAY to observe the
        // plan reaching steady-state replay.
        //
        // Can also be set via system property:
        //   -Dnd4j.dsp.diagnostics=COMPILE,EXECUTE,SEGMENT,GRAPH_REPLAY
        //   -Dnd4j.dsp.diagnostics.level=detailed
        DspDiagnostics.initialize();
        DspDiagnostics.setCategories(
                DspDiagnostics.COMPILE | DspDiagnostics.EXECUTE
                | DspDiagnostics.SEGMENT | DspDiagnostics.GRAPH_REPLAY
                | DspDiagnostics.TIMING);
        DspDiagnostics.setLevel(DspDiagnostics.LEVEL_DETAILED);

        log.info("  DSP_DIAG categories enabled: COMPILE, EXECUTE, SEGMENT, GRAPH_REPLAY, TIMING");
        log.info("  DSP_DIAG level: DETAILED");
        log.info("  (Diagnostic events appear in native C++ log output)");

        // =====================================================================
        // 3. Enable the compile classifier (REQUIRED for graph replay)
        // =====================================================================
        log.info("=== Enabling compile classifier ===");

        // Both flags must be true for DSP to compile and replay.
        // They default to true, but we set them explicitly for clarity.
        sd.setDspAutoCompileEnabled(true);         // Java-side plan compilation
        sd.setDspNativeAutoCompileEnabled(true);    // Native C++ plan compilation

        // Choose a compilation mode. This is analogous to torch.compile() modes:
        //   REDUCE_OVERHEAD — fast startup, uses PTX JIT
        //   SPLIT_STITCH    — balanced Triton compilation
        //   MAX_AUTOTUNE    — best throughput, full Triton optimization
        sd.setDspCompilationMode(DspCompilationMode.REDUCE_OVERHEAD);

        log.info("  dspAutoCompileEnabled:       {}", sd.isDspAutoCompileEnabled());
        log.info("  dspNativeAutoCompileEnabled: {}", sd.isDspNativeAutoCompileEnabled());
        log.info("  graphExecutionMode:          {}", sd.getGraphExecutionMode());

        // =====================================================================
        // 4. First execution — plan compilation (SLOT_BY_SLOT phase)
        // =====================================================================
        log.info("=== First execution: plan compilation ===");

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.randn(32, 256));

        long t0 = System.nanoTime();
        INDArray result = sd.outputSingle(ph, "output");
        long firstMs = (System.nanoTime() - t0) / 1_000_000;

        log.info("  First call (batch=32): {} ms — includes plan compilation", firstMs);
        log.info("  Output shape: {}", Arrays.toString(result.shape()));

        // After first execution, the plan is compiled and cached.
        // Use DspHandle to inspect the plan state.
        DspHandle dsp = sd.dsp();
        if (dsp.isCompiled()) {
            log.info("  Plan compiled: {} total slots, {} external inputs",
                    dsp.totalSlots(), dsp.numExternalInputs());
            log.info("  Plan phase: {} (0=SLOT_BY_SLOT, 1=SHAPES_FROZEN, 2=REPLAYING)",
                    dsp.planPhase());
            log.info("  Execute count: {}", dsp.executeCount());

            // Check the DSP_DIAG report after compilation
            String diagReport = DspDiagnostics.getPlanReport();
            if (diagReport != null && !diagReport.isEmpty()) {
                log.info("  DSP_DIAG plan report (first 300 chars):");
                log.info("    {}", diagReport.substring(0, Math.min(300, diagReport.length())));
            }
        }

        // =====================================================================
        // 5. Plan reuse with StepSnapshot — track per-step metrics
        // =====================================================================
        log.info("=== Plan reuse: tracking per-step performance metrics ===");

        // StepSnapshot captures a complete state snapshot after each execution.
        // It answers: is the plan replaying? are pointers stable? did D2D fire?
        DspHandle.StepSnapshot prevSnapshot = null;
        int warmupReps = 8;
        for (int i = 0; i < warmupReps; i++) {
            ph.put("input", Nd4j.randn(32, 256));
            t0 = System.nanoTime();
            result = sd.outputSingle(ph, "output");
            long ms = (System.nanoTime() - t0) / 1_000_000;

            // Capture a StepSnapshot — structured plan state, no log parsing needed
            DspHandle.StepSnapshot snapshot = dsp.captureStepSnapshot();

            int phase = snapshot.planPhase;
            String phaseName = phase == 0 ? "SLOT_BY_SLOT"
                    : phase == 1 ? "SHAPES_FROZEN"
                    : phase == 2 ? "REPLAYING"
                    : phase == 3 ? "REPLAY_BLOCKED"
                    : "UNKNOWN(" + phase + ")";

            // Per-execution stats: how many segments were replayed vs slot-by-slot?
            int segsReplayed = dsp.lastExecSegmentsReplayed();
            int segsSlotBySlot = dsp.lastExecSegmentsSlotBySlot();
            int segsCaptured = dsp.lastExecSegmentsCaptured();
            int segsTotal = dsp.lastExecSegmentsTotal();

            log.info("  Rep {}: {}ms  phase={}  exec={}  segs[replay={}/capture={}/sbs={}/total={}]",
                    i + 1, ms, phaseName, snapshot.executeCount,
                    segsReplayed, segsCaptured, segsSlotBySlot, segsTotal);

            // Show phase transitions via snapshot diffs
            if (prevSnapshot != null) {
                String changes = snapshot.describeChangesFrom(prevSnapshot);
                if (!changes.isEmpty() && !changes.equals("no previous snapshot")) {
                    log.info("         transition: {}", changes);
                }
            }
            prevSnapshot = snapshot;
        }

        // =====================================================================
        // 6. Verify steady-state replay with detailed metrics
        // =====================================================================
        log.info("=== Verifying steady state ===");

        if (dsp.isCompiled()) {
            int phase = dsp.planPhase();
            boolean inReplay = (phase == PlanPhase.REPLAYING.getNativeCode());
            int numSegs = dsp.numSegments();
            int capturedSegs = dsp.numCapturedGraphSegments();

            log.info("  Plan phase: {} (target: REPLAYING=2)", phase);
            log.info("  Steady-state replay: {}", inReplay);
            log.info("  Segments: {} total, {} captured", numSegs, capturedSegs);
            log.info("  Total graph replays: {}", dsp.totalGraphReplays());
            log.info("  Pointers stable: {}", dsp.pointersStable());
            log.info("  Compilation sealed: {}", dsp.isCompilationSealed());
            log.info("  Frozen exec count: {}", dsp.frozenExecCount());
            log.info("  Mid-execution recompiles: {} (should be 0 in steady state)",
                    dsp.midExecutionCompileCount());
            log.info("  Buffer coloring: applied={}, colors={}, saved={} bytes",
                    dsp.bufferColoringApplied(),
                    dsp.bufferColoringNumColors(),
                    dsp.bufferColoringBytesSaved());

            // Per-segment details
            for (int s = 0; s < numSegs; s++) {
                log.info("    Segment {}: phase={} (3=REPLAYING), replays={}, capturable={}, backend={}",
                        s, dsp.segmentExecutionPhase(s),
                        dsp.segmentReplayCount(s),
                        dsp.isSegmentCapturable(s),
                        dsp.segmentBackendName(s));
            }

            // Capture stats — how many segments captured vs failed?
            DspHandle.CaptureStats cs = dsp.parsedCaptureStats();
            log.info("  Capture stats: captured={}, permFailed={}, oomRetrying={}, "
                    + "nonCapt={}, tooSmall={}, addrUnstable={}",
                    cs.captured, cs.permFailed, cs.oomRetrying,
                    cs.nonCapturable, cs.tooSmall, cs.addrUnstable);

            // DSP_DIAG: get the final diagnostic report
            String finalReport = DspDiagnostics.getPlanReport();
            if (finalReport != null && !finalReport.isEmpty()) {
                log.info("  DSP_DIAG final plan report:");
                // Print first 500 chars — in production, write to file via
                // -Dnd4j.dsp.diagnostics.file=/path/report.json
                for (String line : finalReport.split("\n")) {
                    log.info("    {}", line);
                    if (line.length() > 500) break;
                }
            }

            // DSP_DIAG: category event counts
            log.info("  DSP_DIAG event counts:");
            log.info("    COMPILE events:      {}", DspDiagnostics.isEnabled(DspDiagnostics.COMPILE)
                    ? dsp.diagCategoryEventCount(0) : "disabled");
            log.info("    EXECUTE events:      {}", DspDiagnostics.isEnabled(DspDiagnostics.EXECUTE)
                    ? dsp.diagCategoryEventCount(2) : "disabled");
            log.info("    TIMING events:       {}", DspDiagnostics.isEnabled(DspDiagnostics.TIMING)
                    ? dsp.diagCategoryEventCount(3) : "disabled");
            log.info("    GRAPH_REPLAY events: {}", DspDiagnostics.isEnabled(DspDiagnostics.GRAPH_REPLAY)
                    ? dsp.diagCategoryEventCount(16) : "disabled");
            log.info("    Total events:        {}", dsp.diagTotalEventCount());
        }

        // =====================================================================
        // 6. Batch planning — pre-warm cache for expected batch sizes
        // =====================================================================
        log.info("=== Batch planning: pre-warming plan cache ===");

        // In production, you know the expected batch sizes. Pre-warm the
        // plan cache so that serving requests hit cached plans immediately.
        int[] expectedBatches = {1, 4, 8, 16, 32, 64};

        for (int batch : expectedBatches) {
            ph.put("input", Nd4j.randn(batch, 256));

            t0 = System.nanoTime();
            result = sd.outputSingle(ph, "output");
            long ms = (System.nanoTime() - t0) / 1_000_000;

            log.info("  Pre-warm batch={}: {} ms, output shape={}",
                    batch, ms, Arrays.toString(result.shape()));
        }

        // Now re-run the same batch sizes — all should reuse cached plans
        log.info("  --- Cached plan reuse (should be faster) ---");
        for (int batch : expectedBatches) {
            ph.put("input", Nd4j.randn(batch, 256));

            t0 = System.nanoTime();
            result = sd.outputSingle(ph, "output");
            long ms = (System.nanoTime() - t0) / 1_000_000;

            log.info("  Reuse batch={}: {} ms (cached plan)", batch, ms);
        }

        // =====================================================================
        // 7. Plan swapping — observe cache hits/misses as shapes change
        // =====================================================================
        log.info("=== Plan swapping: cache hits/misses across shape changes ===");

        // When input shapes change, the plan cache dispatches a different plan.
        // replayCacheHits() / replayCacheMisses() track how the native cache
        // dispatches: a "hit" means a previously compiled plan was reused,
        // a "miss" means a new plan was compiled for an unseen shape.
        //
        // This is critical for serving: every cache miss adds compile latency.
        // Plan swapping is fast (pointer swap) but the first call with a new
        // shape compiles. Pre-warming avoids misses during serving.

        // Reset and re-warm with two known batch sizes
        sd.clearDynamicShapePlanCache();

        int[] swapBatches = {8, 16};
        for (int batch : swapBatches) {
            ph.put("input", Nd4j.randn(batch, 256));
            for (int i = 0; i < 5; i++) sd.outputSingle(ph, "output");
        }

        dsp = sd.dsp();
        int hitsBeforeSwap = dsp.replayCacheHits();
        int missesBeforeSwap = dsp.replayCacheMisses();
        log.info("  After warm-up:  cacheHits={}, cacheMisses={}", hitsBeforeSwap, missesBeforeSwap);

        // Now interleave batch=8 and batch=16 — all should be cache hits
        log.info("  --- Interleaving warm shapes (expect cache HITS) ---");
        int interleaveReps = 10;
        long[] swapTimes = new long[interleaveReps];
        for (int i = 0; i < interleaveReps; i++) {
            int batch = swapBatches[i % swapBatches.length];
            ph.put("input", Nd4j.randn(batch, 256));
            t0 = System.nanoTime();
            sd.outputSingle(ph, "output");
            swapTimes[i] = (System.nanoTime() - t0) / 1_000;  // microseconds
        }

        int hitsAfterSwap = dsp.replayCacheHits();
        int missesAfterSwap = dsp.replayCacheMisses();
        int newHits = hitsAfterSwap - hitsBeforeSwap;
        int newMisses = missesAfterSwap - missesBeforeSwap;
        log.info("  After interleave: cacheHits={} (+{}), cacheMisses={} (+{})",
                hitsAfterSwap, newHits, missesAfterSwap, newMisses);

        // Show per-swap timing
        for (int i = 0; i < interleaveReps; i++) {
            int batch = swapBatches[i % swapBatches.length];
            log.info("    swap {}: batch={}, time={}μs (plan swap = pointer swap, fast)",
                    i, batch, swapTimes[i]);
        }

        // Now hit an UNSEEN batch size — expect a cache miss (new compilation)
        log.info("  --- Unseen batch size (expect cache MISS + compile) ---");
        ph.put("input", Nd4j.randn(24, 256));
        t0 = System.nanoTime();
        sd.outputSingle(ph, "output");
        long unseenUs = (System.nanoTime() - t0) / 1_000;

        int hitsAfterUnseen = dsp.replayCacheHits();
        int missesAfterUnseen = dsp.replayCacheMisses();
        log.info("  batch=24 (unseen): time={}μs, cacheHits={} (+{}), cacheMisses={} (+{})",
                unseenUs, hitsAfterUnseen, hitsAfterUnseen - hitsAfterSwap,
                missesAfterUnseen, missesAfterUnseen - missesAfterSwap);
        log.info("  NOTE: First call with unseen shape is slower (plan compilation).");
        log.info("  Subsequent calls with batch=24 will be cache hits.");

        // Verify: call batch=24 again — should be a cache hit now
        t0 = System.nanoTime();
        ph.put("input", Nd4j.randn(24, 256));
        sd.outputSingle(ph, "output");
        long cachedUs = (System.nanoTime() - t0) / 1_000;
        log.info("  batch=24 (cached): time={}μs (fast — plan was cached on first call)",
                cachedUs);

        // =====================================================================
        // 8. Steady-state throughput measurement (samples/sec)
        // =====================================================================
        log.info("=== Steady-state throughput: samples/sec ===");

        // For inference serving, the key metric is samples/sec at steady state.
        // Warmup steps let the plan reach REPLAYING; then we measure throughput
        // over many reps with the same shape (no plan swapping, pure replay).
        int batchForBench = 32;
        ph.put("input", Nd4j.randn(batchForBench, 256));

        // Warmup to reach steady state
        for (int i = 0; i < 10; i++) sd.outputSingle(ph, "output");

        // Measure warmup phase (first 5 reps after a fresh start)
        SameDiff sdFresh = SameDiff.create();
        SDVariable inF = sdFresh.placeHolder("input", DataType.FLOAT, -1, 256);
        SDVariable w1F = sdFresh.var("w1", Nd4j.randn(DataType.FLOAT, 256, 128).muli(0.01));
        SDVariable b1F = sdFresh.var("b1", Nd4j.zeros(DataType.FLOAT, 128));
        SDVariable w2F = sdFresh.var("w2", Nd4j.randn(DataType.FLOAT, 128, 64).muli(0.01));
        SDVariable b2F = sdFresh.var("b2", Nd4j.zeros(DataType.FLOAT, 64));
        SDVariable w3F = sdFresh.var("w3", Nd4j.randn(DataType.FLOAT, 64, 10).muli(0.01));
        SDVariable b3F = sdFresh.var("b3", Nd4j.zeros(DataType.FLOAT, 10));
        SDVariable h1F = sdFresh.nn.relu(inF.mmul(w1F).add(b1F), 0);
        SDVariable h2F = sdFresh.nn.relu(h1F.mmul(w2F).add(b2F), 0);
        SDVariable logitsF = h2F.mmul(w3F).add(b3F);
        sdFresh.nn.softmax("output", logitsF, -1);
        sdFresh.setDspAutoCompileEnabled(true);
        sdFresh.setDspNativeAutoCompileEnabled(true);
        sdFresh.setDspCompilationMode(DspCompilationMode.REDUCE_OVERHEAD);

        Map<String, INDArray> phF = new HashMap<>();
        phF.put("input", Nd4j.randn(batchForBench, 256));

        // Measure warmup (includes compilation)
        int warmupMeasure = 5;
        long warmStart = System.nanoTime();
        for (int i = 0; i < warmupMeasure; i++) sdFresh.outputSingle(phF, "output");
        long warmMs = (System.nanoTime() - warmStart) / 1_000_000;
        double warmSamplesPerSec = (warmupMeasure * batchForBench * 1000.0) / warmMs;

        // Measure steady state (plan in REPLAYING, pure graph replay)
        int steadyReps = 100;
        long steadyStart = System.nanoTime();
        for (int i = 0; i < steadyReps; i++) sdFresh.outputSingle(phF, "output");
        long steadyMs = (System.nanoTime() - steadyStart) / 1_000_000;
        double steadySamplesPerSec = (steadyReps * batchForBench * 1000.0) / steadyMs;
        double steadyMsPerStep = (double) steadyMs / steadyReps;

        log.info("  Warmup phase ({} reps × batch {}):", warmupMeasure, batchForBench);
        log.info("    Total: {}ms, throughput: {} samples/sec",
                warmMs, String.format("%.0f", warmSamplesPerSec));
        log.info("  Steady-state ({} reps × batch {}):", steadyReps, batchForBench);
        log.info("    Total: {}ms, {} ms/step, throughput: {} samples/sec",
                steadyMs, String.format("%.2f", steadyMsPerStep),
                String.format("%.0f", steadySamplesPerSec));

        DspHandle dspF = sdFresh.dsp();
        if (dspF.isCompiled()) {
            log.info("    Plan phase: {} (2=REPLAYING)", dspF.planPhase());
            log.info("    Graph replays: {}", dspF.totalGraphReplays());
            log.info("    Cache hits/misses: {}/{}", dspF.replayCacheHits(), dspF.replayCacheMisses());
        }

        if (steadySamplesPerSec > warmSamplesPerSec) {
            log.info("  Speedup: steady state is {}x faster than warmup",
                    String.format("%.1f", steadySamplesPerSec / warmSamplesPerSec));
        }

        // =====================================================================
        // 9. Explicit pre-compilation (alternative to lazy compilation)
        // =====================================================================
        log.info("=== Explicit pre-compilation ===");

        // Instead of relying on lazy compilation during the first sd.output() call,
        // you can explicitly compile the native plan ahead of time.
        SameDiff sd2 = SameDiff.create();
        SDVariable in2 = sd2.placeHolder("input", DataType.FLOAT, -1, 64);
        SDVariable w = sd2.var("w", Nd4j.randn(DataType.FLOAT, 64, 32).muli(0.01));
        SDVariable out2 = sd2.nn.softmax("output", in2.mmul(w), -1);

        // Enable the compile classifier
        sd2.setDspAutoCompileEnabled(true);
        sd2.setDspNativeAutoCompileEnabled(true);

        // Explicit compilation with a specific mode
        GraphExecutionMode effectiveMode = sd2.compileNativeDynamicShapePlan(
                DspCompilationMode.REDUCE_OVERHEAD, "output");
        log.info("  Pre-compiled with mode: {}", effectiveMode);

        // First output call now skips compilation — plan already exists
        Map<String, INDArray> ph2 = new HashMap<>();
        ph2.put("input", Nd4j.randn(16, 64));
        t0 = System.nanoTime();
        INDArray r2 = sd2.outputSingle(ph2, "output");
        long precompiledMs = (System.nanoTime() - t0) / 1_000_000;
        log.info("  First call after pre-compile: {} ms (no compilation overhead)",
                precompiledMs);

        // =====================================================================
        // 10. DspHandle replay API (bypass InferenceSession)
        // =====================================================================
        log.info("=== DspHandle direct replay ===");

        // Once a plan is compiled, you can replay it directly via DspHandle.
        // This skips InferenceSession bookkeeping and is the fastest path.
        DspHandle dsp2 = sd2.dsp();
        if (dsp2.isCompiled()) {
            ph2.put("input", Nd4j.randn(16, 64));
            Map<String, INDArray> outputs = dsp2.replay(ph2);
            log.info("  Direct replay output shape: {}",
                    Arrays.toString(outputs.get("output").shape()));
            log.info("  Execute count after replay: {}", dsp2.executeCount());
        }

        // =====================================================================
        // Summary
        // =====================================================================
        log.info("");
        log.info("=== Summary ===");
        log.info("Plan reuse lifecycle:");
        log.info("  1. First call with new shapes  → compile plan (slow)");
        log.info("  2. Repeat calls same shapes     → reuse plan, advance toward replay");
        log.info("  3. Shapes frozen + ptrs stable  → REPLAYING (near-zero overhead)");
        log.info("");
        log.info("Plan swapping:");
        log.info("  - Native plan cache is keyed by placeholder-shape signature");
        log.info("  - Shape change → cache lookup: HIT (swap) or MISS (compile)");
        log.info("  - dsp.replayCacheHits() / dsp.replayCacheMisses() track dispatch");
        log.info("  - Pre-warm all expected shapes to avoid compile misses in production");
        log.info("  - Plan swap itself is fast (pointer swap), compilation is slow");
        log.info("");
        log.info("Batch planning strategy:");
        log.info("  - Pre-warm with expected batch sizes at startup");
        log.info("  - Each batch size gets its own cached plan");
        log.info("  - Serving requests hit cached plans (no compile latency)");
        log.info("");
        log.info("Compile classifier (REQUIRED):");
        log.info("  sd.setDspAutoCompileEnabled(true)");
        log.info("  sd.setDspNativeAutoCompileEnabled(true)");
        log.info("  Both default to true — set explicitly if you've disabled them.");
        log.info("");
        log.info("DSP_DIAG (observing plan lifecycle):");
        log.info("  Programmatic:");
        log.info("    DspDiagnostics.setCategories(COMPILE | EXECUTE | TIMING | GRAPH_REPLAY)");
        log.info("    DspDiagnostics.setLevel(LEVEL_DETAILED)");
        log.info("  System property:");
        log.info("    -Dnd4j.dsp.diagnostics=COMPILE,EXECUTE,TIMING,GRAPH_REPLAY");
        log.info("    -Dnd4j.dsp.diagnostics.level=detailed");
        log.info("  File output:");
        log.info("    -Dnd4j.dsp.diagnostics.file=/tmp/dsp-report.json");
        log.info("");
        log.info("Performance metrics API (DspHandle):");
        log.info("  dsp.captureStepSnapshot()          — structured per-step state");
        log.info("  dsp.lastExecSegmentsReplayed()     — segments replayed this step");
        log.info("  dsp.lastExecSegmentsSlotBySlot()   — segments in slot-by-slot this step");
        log.info("  dsp.totalGraphReplays()             — cumulative replay count");
        log.info("  dsp.parsedCaptureStats()            — capture success/failure breakdown");
        log.info("  dsp.replayCacheHits()               — plan cache hits (shape reuse)");
        log.info("  dsp.replayCacheMisses()             — plan cache misses (new compile)");
        log.info("  dsp.diagTotalEventCount()           — total DSP_DIAG events");

        // Clean up diagnostics
        DspDiagnostics.setCategories(DspDiagnostics.NONE);

        log.info("**************** DSP Plan Reuse & Batch Planning Example finished ********************");
    }
}
