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
import org.nd4j.autodiff.samediff.execution.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * DSP Diagnostics, Debugging, and Plan Introspection Example.
 *
 * When DSP compilation or execution produces unexpected results, these tools
 * help diagnose the issue:
 *
 * <h3>1. DspDiagnostics — Category-based logging</h3>
 * Enable fine-grained diagnostic logging for specific DSP subsystems using
 * a bitmask of 20 categories. Categories can be set via system properties
 * or programmatically.
 *
 * <h3>2. DspDebugger — Attach to a SameDiff graph</h3>
 * The debugger analyzes the compiled plan, validates execution steps,
 * detects NaN/Inf issues, checks phase contracts, and generates reports.
 *
 * <h3>3. DspHandle — Introspect the live execution plan</h3>
 * Access via {@code sd.dsp()}, provides real-time plan state: slot outputs,
 * segment replay modes, external input addresses, buffer pool stats, and more.
 *
 * <h3>4. DspPlanAssertions — Test assertions for DSP correctness</h3>
 * Static assertions for verifying plan phases, segment states, capture
 * quality, pointer stability, and KV cache positions. Used in tests but
 * also useful for production health checks.
 *
 * <h3>System Properties:</h3>
 * <pre>
 * -Dnd4j.dsp.diagnostics=COMPILE,EXECUTE,TIMING     # comma-separated categories or "all"
 * -Dnd4j.dsp.diagnostics.level=full                  # summary | detailed | full
 * -Dnd4j.dsp.diagnostics.file=/path/report.json      # write JSON report to file
 * </pre>
 *
 * Run with:
 *   cd samediff-examples
 *   mvn exec:java -Dexec.mainClass="org.nd4j.examples.samediff.advanced.execution.DSPDiagnosticsAndDebuggingExample"
 */
public class DSPDiagnosticsAndDebuggingExample {
    private static final Logger log = LoggerFactory.getLogger(DSPDiagnosticsAndDebuggingExample.class);

    public static void main(String[] args) {

        // =====================================================================
        // 1. Build a SameDiff graph for diagnostics
        // =====================================================================
        log.info("=== 1. Building SameDiff graph ===");

        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 128);
        SDVariable w1 = sd.var("w1", Nd4j.randn(128, 64).muli(0.01));
        SDVariable b1 = sd.var("b1", Nd4j.zeros(64));
        SDVariable w2 = sd.var("w2", Nd4j.randn(64, 32).muli(0.01));
        SDVariable b2 = sd.var("b2", Nd4j.zeros(32));
        SDVariable w3 = sd.var("w3", Nd4j.randn(32, 10).muli(0.01));
        SDVariable b3 = sd.var("b3", Nd4j.zeros(10));

        SDVariable z1 = input.mmul(w1).add(b1);
        SDVariable a1 = sd.nn.relu(z1, 0);
        SDVariable z2 = a1.mmul(w2).add(b2);
        SDVariable a2 = sd.nn.relu(z2, 0);
        SDVariable z3 = a2.mmul(w3).add(b3);
        SDVariable output = sd.nn.softmax("output", z3, -1);

        log.info("Graph: {} variables, {} ops", sd.variables().size(), sd.ops().length);

        // =====================================================================
        // 2. DspDiagnostics — Enable Category-Based Logging
        // =====================================================================
        log.info("\n=== 2. DspDiagnostics — Category-Based Logging ===");

        // Initialize the diagnostics subsystem
        DspDiagnostics.initialize();

        // 20 diagnostic categories (bitmask flags):
        log.info("Available diagnostic categories:");
        log.info("  COMPILE  (1<<0)  — graph compilation events");
        log.info("  JIT      (1<<1)  — JIT kernel compilation (NVRTC, PTX, Triton)");
        log.info("  EXECUTE  (1<<2)  — execution events and dispatch");
        log.info("  TIMING   (1<<3)  — per-op and per-segment timing");
        log.info("  MEMORY   (1<<4)  — memory allocation and workspace events");
        log.info("  BACKEND  (1<<5)  — backend selection decisions");
        log.info("  SHAPE    (1<<6)  — shape inference and caching");
        log.info("  SEGMENT  (1<<7)  — segment formation and replay");
        log.info("  FUSION   (1<<8)  — op fusion decisions and scoring");
        log.info("  VERIFY   (1<<9)  — output verification and validation");
        log.info("  KV_CACHE (1<<10) — KV cache position tracking (LLM)");
        log.info("  FALLBACK (1<<11) — fallback events (JIT→slot-by-slot)");
        log.info("  TRANSFER (1<<12) — data transfer events (D2D, H2D, D2H)");
        log.info("  EMULATED_REPLAY (1<<13) — emulated replay diagnostics");
        log.info("  STREAM_SYNC     (1<<14) — CUDA stream synchronization");
        log.info("  MULTI_DEVICE    (1<<15) — multi-device placement");
        log.info("  GRAPH_REPLAY    (1<<16) — graph replay lifecycle");
        log.info("  SEGMENT_BUCKETS (1<<17) — segment bucketing");
        log.info("  LIFECYCLE       (1<<18) — plan lifecycle events");
        log.info("  COLORING        (1<<19) — buffer coloring optimization");
        log.info("  NONE=0, ALL=0xFFFFF");

        // Enable specific categories
        int categories = DspDiagnostics.COMPILE | DspDiagnostics.EXECUTE | DspDiagnostics.TIMING;
        DspDiagnostics.setCategories(categories);
        log.info("Enabled categories: COMPILE | EXECUTE | TIMING");

        // Set detail level: LEVEL_SUMMARY=0, LEVEL_DETAILED=1, LEVEL_FULL=2
        DspDiagnostics.setLevel(DspDiagnostics.LEVEL_DETAILED);
        log.info("Detail level: DETAILED");

        // Add more categories without clearing existing ones
        DspDiagnostics.enableCategories(DspDiagnostics.BACKEND | DspDiagnostics.FUSION);
        log.info("Added BACKEND and FUSION categories");

        // Check if a specific category is enabled
        boolean compileEnabled = DspDiagnostics.isEnabled(DspDiagnostics.COMPILE);
        log.info("COMPILE diagnostics enabled: {}", compileEnabled);

        // Parse categories from a string (as used in system properties)
        int parsed = DspDiagnostics.parseCategories("COMPILE,EXECUTE,TIMING,BACKEND");
        log.info("Parsed category mask: 0x{}", Integer.toHexString(parsed));

        // Record a diagnostic event
        DspDiagnostics.record(DspDiagnostics.COMPILE, "Starting graph compilation for example");
        DspDiagnostics.recordSlot(DspDiagnostics.EXECUTE, 0, "matmul", "Executing first matmul");
        DspDiagnostics.recordTimed(DspDiagnostics.TIMING, 0, 0, "matmul", 1500, "First matmul took 1.5ms");

        // =====================================================================
        // 3. Execute with diagnostics enabled
        // =====================================================================
        log.info("\n=== 3. Execute with diagnostics ===");

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.randn(16, 128));

        // Execute the graph — diagnostics will capture compilation and execution events
        INDArray result = sd.outputSingle(ph, "output");
        log.info("Output shape: {}", Arrays.toString(result.shape()));

        // Get the diagnostic report
        String planReport = DspDiagnostics.getPlanReport();
        if (planReport != null && !planReport.isEmpty()) {
            log.info("Plan report (first 500 chars):");
            log.info("  {}", planReport.substring(0, Math.min(500, planReport.length())));
        }

        // Get JSON report (for programmatic analysis)
        String jsonReport = DspDiagnostics.getJsonReport();
        if (jsonReport != null && !jsonReport.isEmpty()) {
            log.info("JSON report available ({} chars)", jsonReport.length());
        }

        // Clear diagnostics data
        DspDiagnostics.clear();
        log.info("Diagnostics cleared");

        // =====================================================================
        // 4. DspDebugger — Attach and Analyze
        // =====================================================================
        log.info("\n=== 4. DspDebugger — Plan Analysis ===");

        // Attach the debugger to the SameDiff instance
        // This enables DSP if not already enabled and creates the debugger
        DspDebugger debugger = DspDebugger.attach(sd);

        // Execute again to ensure plan is compiled
        result = sd.outputSingle(ph, "output");

        // Analyze the compiled plan — returns a structured PlanReport
        // PlanReport has public final fields: numSlots, numSegments, planPhase,
        // graphNodePhase, slots (List<SlotInfo>), segments (List<SegmentReport>), errorMessage
        DspDebugger.PlanReport planAnalysis = debugger.analyzePlan();
        log.info("Plan analysis:");
        log.info("  Total slots: {}", planAnalysis.numSlots);
        log.info("  Number of segments: {}", planAnalysis.numSegments);
        log.info("  Plan phase: {}", planAnalysis.planPhase);
        log.info("  Graph node phase: {}", planAnalysis.graphNodePhase);

        // Print segment details
        // SegmentReport has public final fields: index, capturable, captureFailed,
        // executionCount, phase (ExecutionPhase), graphNodePhase
        List<DspDebugger.SegmentReport> segments = planAnalysis.segments;
        for (int i = 0; i < segments.size(); i++) {
            DspDebugger.SegmentReport seg = segments.get(i);
            log.info("  Segment {}: capturable={}, captureFailed={}, execCount={}, phase={}",
                    seg.index, seg.capturable, seg.captureFailed,
                    seg.executionCount, seg.phase);
        }

        // Slot flag reference
        log.info("\nSlot flags (bitmask):");
        log.info("  FLAG_VIEW_CAPABLE        (1<<0) — output can be a view");
        log.info("  FLAG_DATA_DEPENDENT      (1<<1) — output depends on input values");
        log.info("  FLAG_SHAPE_DEPENDS_ON_VALUES (1<<2) — output shape depends on values");
        log.info("  FLAG_IDENTITY            (1<<3) — identity op (passthrough)");
        log.info("  FLAG_IN_PLACE_FUSED      (1<<4) — op was fused in-place");
        log.info("  FLAG_FUSED_CHAIN_HEAD    (1<<5) — head of fusion chain");
        log.info("  FLAG_FUSED_CHAIN_TAIL    (1<<6) — tail of fusion chain");
        log.info("  FLAG_NEEDS_ZEROED        (1<<7) — output must be zero-initialized");
        log.info("  FLAG_NEEDS_INT_LONG_SYNC (1<<8) — needs int/long sync");
        log.info("  FLAG_SHAPE_STATIC        (1<<9) — shape is statically known");
        log.info("  FLAG_FROZEN_CONSTANT     (1<<10) — frozen constant value");

        // =====================================================================
        // 5. DspDebugger — Validate Execution Steps
        // =====================================================================
        log.info("\n=== 5. DspDebugger — Validate Steps ===");

        // Validate a single execution step
        // StepReport provides: hasErrors(), hasWarnings(), getErrors(), getWarnings()
        DspDebugger.StepReport stepReport = debugger.validateStep(ph, "output");
        log.info("Step validation:");
        log.info("  Has errors: {}", stepReport.hasErrors());
        log.info("  Has warnings: {}", stepReport.hasWarnings());
        if (stepReport.hasErrors()) {
            log.info("  Errors: {}", stepReport.getErrors());
        }

        // Validate multiple steps with random inputs
        // MultiStepReport provides: hasErrors(), hasStaleData(), getErrorCount()
        DspDebugger.MultiStepReport multiStep = debugger.validateMultipleSteps(
                3,              // number of steps
                "input",        // placeholder name
                new long[]{8, 128},  // shape
                DataType.FLOAT, // dtype
                "output"        // output names
        );
        log.info("Multi-step validation:");
        log.info("  Has errors: {}", multiStep.hasErrors());
        log.info("  Error count: {}", multiStep.getErrorCount());
        log.info("  Has stale data: {}", multiStep.hasStaleData());

        // Validate phase contracts (no illegal phase transitions)
        // PhaseContractReport provides: hasViolations(), getViolations()
        DspDebugger.PhaseContractReport phaseReport = debugger.validatePhaseContract();
        log.info("Phase contract: hasViolations={}, violations={}",
                phaseReport.hasViolations(), phaseReport.getViolations().size());

        // =====================================================================
        // 6. DspHandle — Live Plan Introspection
        // =====================================================================
        log.info("\n=== 6. DspHandle — Live Plan Introspection ===");

        // Get a DspHandle for the current session
        DspHandle dsp = sd.dsp();

        log.info("Plan state:");
        log.info("  Compiled: {}", dsp.isCompiled());
        log.info("  Total slots: {}", dsp.totalSlots());
        log.info("  External inputs: {}", dsp.numExternalInputs());
        log.info("  Execute count: {}", dsp.executeCount());

        // Get plan summary
        String summary = dsp.planSummary();
        if (summary != null) {
            log.info("  Plan summary (first 300 chars):");
            log.info("    {}", summary.substring(0, Math.min(300, summary.length())));
        }

        // Slot inspection
        if (dsp.isCompiled() && dsp.totalSlots() > 0) {
            // Find slot for a specific output variable
            int outputSlot = dsp.slotIndexForOutput("output");
            log.info("  Slot for 'output': {}", outputSlot);

            // Find slot by op name substring
            int matmulSlot = dsp.slotIndexForOp("mmul");
            log.info("  First slot for 'mmul': {}", matmulSlot);

            // Find all slots matching an op name
            List<Integer> allMmulSlots = dsp.allSlotsForOp("mmul");
            log.info("  All 'mmul' slots: {}", allMmulSlots);

            // NaN debugging — find first slot producing NaN
            int nanSlot = dsp.firstNaNSlot();
            log.info("  First NaN slot: {}", nanSlot == -1 ? "none" : nanSlot);

            // Check for any output issues
            boolean hasIssues = dsp.hasOutputIssues();
            log.info("  Has output issues (NaN/Inf): {}", hasIssues);
        }

        // Segment introspection
        // Note: segmentExecutionPhase() returns raw int ordinal, not the enum.
        // Use ExecutionPhase.fromNativeCode() to convert.
        int numSegments = dsp.numSegments();
        log.info("  Segments: {}", numSegments);
        for (int i = 0; i < numSegments; i++) {
            int phaseCode = dsp.segmentExecutionPhase(i);
            ExecutionPhase phase = ExecutionPhase.fromNativeCode(phaseCode);
            boolean capturable = dsp.isSegmentCapturable(i);
            log.info("    Segment {}: phase={} (code={}), capturable={}",
                    i, phase, phaseCode, capturable);
        }

        // Plan phase tracking
        // planPhase() returns raw int ordinal — convert with PlanPhase.fromNativeCode()
        int planPhaseCode = dsp.planPhase();
        PlanPhase planPhase = PlanPhase.fromNativeCode(planPhaseCode);
        log.info("  Plan phase: {} (code={})", planPhase, planPhaseCode);
        log.info("  Frozen exec count: {}", dsp.frozenExecCount());
        log.info("  Compilation sealed: {}", dsp.isCompilationSealed());
        log.info("  Total graph replays: {}", dsp.totalGraphReplays());

        // =====================================================================
        // 7. DspHandle — Replay and Snapshot
        // =====================================================================
        log.info("\n=== 7. DspHandle — Replay and Snapshot ===");

        // Execute via DspHandle.replay()
        Map<String, INDArray> replayResult = dsp.replay(ph);
        if (replayResult.containsKey("output")) {
            log.info("Replay output shape: {}",
                    Arrays.toString(replayResult.get("output").shape()));
        }

        // Capture statistics
        String stats = dsp.captureStats();
        if (stats != null && !stats.isEmpty()) {
            log.info("Capture stats: {}", stats.substring(0, Math.min(200, stats.length())));
        }

        // Validate all outputs
        // Returns int[] flags — one per output. Bit 0=NULL, bit 1=NaN, bit 2=Inf, bit 3=ALL_ZERO
        int[] outputFlags = dsp.validateOutputs();
        boolean allValid = true;
        for (int flag : outputFlags) {
            if (flag != 0) { allValid = false; break; }
        }
        log.info("All outputs valid: {} ({} outputs checked)", allValid, outputFlags.length);

        // Buffer pool stats (static, per device)
        long pooledBytes = DspHandle.bufferPoolPooledBytes(0);
        long pooledCount = DspHandle.bufferPoolPooledCount(0);
        log.info("Buffer pool (device 0): {} bytes in {} buffers", pooledBytes, pooledCount);

        // =====================================================================
        // 8. DspPlanAssertions — Test and Production Health Checks
        // =====================================================================
        log.info("\n=== 8. DspPlanAssertions ===");

        log.info("DspPlanAssertions provides static assertions for DSP plan correctness.");
        log.info("Useful in tests and production health checks.");
        log.info("");
        log.info("Phase assertions:");
        log.info("  assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN)");
        log.info("  assertPhaseExact(sd, PlanPhase.REPLAYING)");
        log.info("  assertFullyReplaying(sd)");
        log.info("  assertSealed(sd)");
        log.info("");
        log.info("Segment assertions:");
        log.info("  assertNoCaptureFailures(sd)");
        log.info("  assertSegmentReachedPhase(sd, segIdx, ExecutionPhase.COMPILED)");
        log.info("  assertAllCapturableSegmentsReachedPhase(sd, ExecutionPhase.REPLAYING)");
        log.info("  assertSegmentBackend(sd, segIdx, \"Triton\")");
        log.info("  assertAllSegmentsCompiledWith(sd, \"NVRTC\")");
        log.info("");
        log.info("Quality assertions:");
        log.info("  assertNoPhaseContractViolations(sd)");
        log.info("  assertOutputsValid(sd)");
        log.info("  assertNoAddressDrift(sd)");
        log.info("  assertPointersStable(sd)");
        log.info("  assertNoSlotBySlotFallback(sd)");
        log.info("  assertCaptureComplete(sd)");
        log.info("  assertZeroPermCaptureFailures(sd)");
        log.info("  assertNoHostOnlyOps(sd)");
        log.info("");
        log.info("Slot-level assertions:");
        log.info("  assertSlotHasTrait(sd, slotIdx, DspPlanAssertions.FLAG_SHAPE_STATIC)");
        log.info("  assertSlotNotDataDependent(sd, slotIdx)");
        log.info("  assertSlotIsFrozenConstant(sd, slotIdx)");
        log.info("  assertOpCompiled(sd, \"matmul\")");
        log.info("");
        log.info("KV cache assertions (LLM inference):");
        log.info("  assertKvCachePositionEquals(sd, expected)");
        log.info("  assertKvCachePositionAdvanced(sd, previousPos)");
        log.info("");
        log.info("Non-asserting queries:");
        log.info("  snapshotPlanState(sd)      — full plan state as string");
        log.info("  snapshotSegmentState(sd, i) — segment state as string");
        log.info("  snapshotExtInputState(sd)   — external input state");
        log.info("  getSegmentCompiledBackend(sd, i) — backend name");
        log.info("  getTotalGraphReplays(sd)   — total replay count");

        // Demonstrate non-asserting queries
        String planState = DspPlanAssertions.snapshotPlanState(sd);
        if (planState != null) {
            log.info("\nPlan state snapshot (first 300 chars):");
            log.info("  {}", planState.substring(0, Math.min(300, planState.length())));
        }

        // =====================================================================
        // 9. System Properties Quick Reference
        // =====================================================================
        log.info("\n=== 9. DSP System Properties Quick Reference ===");
        log.info("");
        log.info("Diagnostics:");
        log.info("  -Dnd4j.dsp.diagnostics=COMPILE,EXECUTE,TIMING  (or 'all')");
        log.info("  -Dnd4j.dsp.diagnostics.level=full              (summary|detailed|full)");
        log.info("  -Dnd4j.dsp.diagnostics.file=/path/report.json");
        log.info("");
        log.info("Execution control:");
        log.info("  -Dnd4j.dsp.graphExecutionMode=TRITON");
        log.info("  -Dnd4j.dsp.nativeExecutor.enabled=true");
        log.info("  -Dnd4j.dsp.cudaGraphs.enabled=true");
        log.info("  -Dnd4j.dsp.jitMode=graph+jit");
        log.info("  -Dnd4j.dsp.noFreeze=true");
        log.info("");
        log.info("Memory:");
        log.info("  -Dnd4j.dsp.capturePoolEnabled=true");
        log.info("  -Dnd4j.dsp.batchZero=true");
        log.info("  -Dnd4j.dsp.maxKvCacheLength=2048");
        log.info("");
        log.info("Compute:");
        log.info("  -Dnd4j.dsp.fp16Compute=true");
        log.info("  -Dnd4j.dsp.castElimination=true");
        log.info("  -Dnd4j.dsp.symbolicShapes=true");
        log.info("  -Dnd4j.dsp.batchedGemm=true");
        log.info("  -Dnd4j.dsp.castSinkMatmul=true");
        log.info("  -Dnd4j.dsp.matmulSegmentation=true");

        // Disable diagnostics to avoid impacting other examples
        DspDiagnostics.setCategories(DspDiagnostics.NONE);

        log.info("\n**************** DSP Diagnostics and Debugging Example finished ********************");
    }
}
