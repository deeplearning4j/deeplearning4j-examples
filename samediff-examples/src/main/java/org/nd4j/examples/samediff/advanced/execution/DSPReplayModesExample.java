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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * DSP Replay Modes: Hardware-Level vs Emulated Replay.
 *
 * <p>The DSP executor supports multiple replay modes that differ fundamentally
 * in how they execute the compiled plan. Understanding the distinction between
 * <b>hardware-level replay</b> and <b>emulated replay</b> is critical for
 * choosing the right mode and debugging performance issues.</p>
 *
 * <hr/>
 *
 * <h3>Hardware-Level Replay (CUDA Graphs, HIP Graphs, Vulkan, Metal, Level Zero)</h3>
 *
 * <p>Hardware-level replay records an entire sequence of GPU kernel launches into
 * a graph object provided by the GPU driver, then replays that graph as a single
 * unit. On CUDA, this is {@code cudaGraphLaunch()} — the driver submits all
 * kernels in the captured graph to the GPU command queue in one shot, with no
 * per-kernel CPU-side submission overhead.</p>
 *
 * <pre>
 *   Capture:  cudaStreamBeginCapture → run all ops → cudaStreamEndCapture
 *             → cudaGraphInstantiate → cudaGraphExec_t ready
 *
 *   Replay:   cudaGraphLaunch(graphExec, stream)
 *             → ALL kernels submitted in ~20μs regardless of graph size
 * </pre>
 *
 * <p><b>Why it's fast:</b> A 200-op graph normally requires 200 individual
 * kernel launch API calls from the CPU. Each launch has ~5-10μs overhead
 * (parameter setup, driver validation, queue submission). With hardware graph
 * replay, all 200 kernels launch in a single driver call — total overhead
 * drops from 1-2ms to ~20μs. For small/fast kernels (element-wise ops), this
 * kernel-launch overhead can dominate execution time.</p>
 *
 * <p><b>Constraints:</b> Hardware graph replay requires that:</p>
 * <ul>
 *   <li>All kernel arguments (device pointers) are stable between replays.
 *       The graph captures exact pointer values; if a buffer is reallocated,
 *       the graph replays with stale pointers → crash or corruption.</li>
 *   <li>No CPU-side control flow (if/else, loops) inside the captured region.</li>
 *   <li>No host-to-device copies inside the captured region.</li>
 *   <li>All ops are GPU-native (no host fallback ops).</li>
 * </ul>
 *
 * <p><b>Hardware backends by platform:</b></p>
 * <table>
 *   <tr><td>{@link GraphExecutionMode#CUDA_GRAPHS}</td><td>NVIDIA CUDA (cudaGraphLaunch)</td></tr>
 *   <tr><td>{@link GraphExecutionMode#HIP_GRAPHS}</td><td>AMD ROCm (hipGraphLaunch)</td></tr>
 *   <tr><td>{@link GraphExecutionMode#VULKAN}</td><td>Cross-platform (VkCommandBuffer replay)</td></tr>
 *   <tr><td>{@link GraphExecutionMode#METAL}</td><td>Apple Silicon (MTLIndirectCommandBuffer)</td></tr>
 *   <tr><td>{@link GraphExecutionMode#LEVEL_ZERO}</td><td>Intel (mutable command list replay)</td></tr>
 *   <tr><td>{@link GraphExecutionMode#TPU}</td><td>Google TPU (PJRT cached executables)</td></tr>
 * </table>
 *
 * <hr/>
 *
 * <h3>Emulated Replay ({@link GraphExecutionMode#EMULATED_REPLAY})</h3>
 *
 * <p>Emulated replay executes ops <b>slot-by-slot</b> (one at a time) but with
 * the full DSP graph replay lifecycle: shape key tracking, address stability
 * monitoring, capture buffer identification, and segment timing. It does NOT
 * use any GPU graph APIs.</p>
 *
 * <p><b>Why it exists:</b></p>
 * <ul>
 *   <li>Works on <b>any platform</b> — no GPU graph APIs required</li>
 *   <li>Diagnostic stepping stone between SLOT_BY_SLOT and CUDA_GRAPHS</li>
 *   <li>Shows what <em>would</em> happen with hardware graph replay</li>
 *   <li>Detects shape drift and pointer instability before capture</li>
 *   <li>Profiles slot-by-slot overhead that graph replay would eliminate</li>
 * </ul>
 *
 * <p>Use EMULATED_REPLAY to answer: "Is my graph ready for hardware capture?
 * Are shapes stable? Are pointers stable? Which segments would be captured?"</p>
 *
 * <hr/>
 *
 * <h3>JIT-Compiled Replay (Triton, NVRTC, PTX, OpenVINO, oneDNN)</h3>
 *
 * <p>JIT-compiled modes don't replay a whole-graph recording. Instead, they
 * <b>compile fusible segments</b> of the graph into optimized kernels, then
 * launch those fused kernels individually. The "replay" is launching the
 * cached compiled kernel — no recompilation, but each segment is still a
 * separate kernel launch.</p>
 *
 * <p>{@link GraphExecutionMode#TRITON} compiles through the full Triton MLIR
 * pipeline (TTIR → TTGIR → LLVM → PTX) and produces the most optimized fused
 * kernels. {@link GraphExecutionMode#NVRTC_JIT} generates CUDA C++ source and
 * compiles with NVRTC at runtime.</p>
 *
 * <p><b>Hybrid execution:</b> In practice, DSP splits the graph into <b>segments</b>.
 * Fusible element-wise segments are compiled via Triton/NVRTC. Non-fusible
 * segments (matmul, conv2d) use cuBLAS/cuDNN and may be captured into CUDA
 * graphs. The result is a mix of compiled kernels and graph replay.</p>
 *
 * <hr/>
 *
 * <h3>SLOT_BY_SLOT — The Baseline</h3>
 *
 * <p>{@link GraphExecutionMode#SLOT_BY_SLOT} executes each op individually with
 * no fusion, no graph capture, and no compilation. This is the correctness
 * baseline — use it to verify that DSP compilation/replay doesn't change
 * numerical results.</p>
 */
public class DSPReplayModesExample {
    private static final Logger log = LoggerFactory.getLogger(DSPReplayModesExample.class);

    public static void main(String[] args) {

        // =====================================================================
        // 0. Enable DSP_DIAG to observe how each mode behaves
        // =====================================================================
        // DSP_DIAG with EMULATED_REPLAY category shows what hardware replay
        // would do. GRAPH_REPLAY category shows actual capture/replay events.
        // SEGMENT shows segment formation. BACKEND shows backend selection.
        DspDiagnostics.initialize();
        DspDiagnostics.setCategories(
                DspDiagnostics.SEGMENT | DspDiagnostics.GRAPH_REPLAY
                | DspDiagnostics.BACKEND | DspDiagnostics.EMULATED_REPLAY);
        DspDiagnostics.setLevel(DspDiagnostics.LEVEL_SUMMARY);
        log.info("DSP_DIAG enabled: SEGMENT, GRAPH_REPLAY, BACKEND, EMULATED_REPLAY");

        // =====================================================================
        // 1. Build a graph that exercises multiple segment types
        // =====================================================================
        log.info("=== Building graph with mixed op types ===");

        SameDiff sd = SameDiff.create();

        int batch = 16, dim = 128, hidden = 64, out = 10;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, dim);
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, dim, hidden).muli(0.01));
        SDVariable b1 = sd.var("b1", Nd4j.zeros(DataType.FLOAT, hidden));
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, hidden, out).muli(0.01));
        SDVariable b2 = sd.var("b2", Nd4j.zeros(DataType.FLOAT, out));

        // Element-wise ops (fusible into a single kernel by Triton/NVRTC)
        SDVariable z1 = input.mmul(w1).add(b1);       // matmul (cuBLAS) + add (fusible)
        SDVariable a1 = sd.nn.relu(z1, 0);             // relu (fusible)
        SDVariable z2 = a1.mmul(w2).add(b2);           // matmul (cuBLAS) + add (fusible)
        SDVariable scaled = sd.math.mul(z2, 0.5);      // scalar mul (fusible)
        SDVariable output = sd.nn.softmax("output", scaled, -1); // softmax (may be fusible)

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.randn(batch, dim));

        log.info("Graph: {} ops", sd.ops().length);
        log.info("Expect: matmul segments (cuBLAS) + element-wise segments (fusible)");

        // =====================================================================
        // 2. SLOT_BY_SLOT — correctness baseline
        // =====================================================================
        log.info("=== Mode 1: SLOT_BY_SLOT (correctness baseline) ===");
        log.info("  Each op executes individually. No fusion, no graph capture.");
        log.info("  This is the reference for numerical correctness.");

        SameDiff sdSlot = cloneGraph(sd);
        sdSlot.setDspAutoCompileEnabled(false);
        sdSlot.setDspNativeAutoCompileEnabled(false);
        sdSlot.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        INDArray slotResult = sdSlot.outputSingle(ph, "output");
        log.info("  Output shape: {}", Arrays.toString(slotResult.shape()));
        log.info("  Output sum: {}", slotResult.sumNumber());

        // =====================================================================
        // 3. EMULATED_REPLAY — diagnostic mode
        // =====================================================================
        log.info("=== Mode 2: EMULATED_REPLAY (diagnostic, no GPU graph APIs) ===");
        log.info("  Runs slot-by-slot but tracks the full DSP lifecycle:");
        log.info("  shape keys, pointer stability, segment identification.");
        log.info("  Works on any platform (CPU or GPU).");

        SameDiff sdEmulated = cloneGraph(sd);
        sdEmulated.setDspAutoCompileEnabled(true);
        sdEmulated.setDspNativeAutoCompileEnabled(true);
        sdEmulated.setGraphExecutionMode(GraphExecutionMode.EMULATED_REPLAY);

        INDArray emulatedResult = sdEmulated.outputSingle(ph, "output");
        log.info("  Output shape: {}", Arrays.toString(emulatedResult.shape()));

        // Run several more times to let the plan stabilize
        for (int i = 0; i < 5; i++) {
            ph.put("input", Nd4j.randn(batch, dim));
            sdEmulated.outputSingle(ph, "output");
        }

        DspHandle dspEmulated = sdEmulated.dsp();
        if (dspEmulated.isCompiled()) {
            log.info("  Plan phase: {} (emulated — no real capture)", dspEmulated.planPhase());
            log.info("  Total slots: {}", dspEmulated.totalSlots());
            log.info("  Segments: {}", dspEmulated.numSegments());
            log.info("  Pointers stable: {}", dspEmulated.pointersStable());
            log.info("  EMULATED_REPLAY shows what hardware replay WOULD do,");
            log.info("  without actually using GPU graph APIs.");

            // DSP_DIAG: The EMULATED_REPLAY category captures diagnostic events
            // about what would have been captured/replayed by hardware graph APIs.
            log.info("  DSP_DIAG EMULATED_REPLAY events: {}",
                    dspEmulated.diagCategoryEventCount(13));
        }

        // =====================================================================
        // 4. CUDA_GRAPHS — hardware-level replay
        // =====================================================================
        log.info("=== Mode 3: CUDA_GRAPHS (hardware graph replay) ===");
        log.info("  Records GPU kernel launches into a CUDA graph, then replays");
        log.info("  the entire graph in a single cudaGraphLaunch() call.");
        log.info("  Eliminates per-kernel launch overhead (~5-10μs per op).");
        log.info("  NOTE: Requires CUDA GPU. Falls back on CPU.");

        SameDiff sdCuda = cloneGraph(sd);
        sdCuda.setDspAutoCompileEnabled(true);
        sdCuda.setDspNativeAutoCompileEnabled(true);
        sdCuda.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        ph.put("input", Nd4j.randn(batch, dim));
        INDArray cudaResult = sdCuda.outputSingle(ph, "output");
        log.info("  Output shape: {}", Arrays.toString(cudaResult.shape()));

        // Warm up for graph capture
        for (int i = 0; i < 5; i++) {
            ph.put("input", Nd4j.randn(batch, dim));
            sdCuda.outputSingle(ph, "output");
        }

        DspHandle dspCuda = sdCuda.dsp();
        if (dspCuda.isCompiled()) {
            log.info("  Plan phase: {}", dspCuda.planPhase());
            log.info("  Captured graph segments: {}", dspCuda.numCapturedGraphSegments());
            log.info("  Total graph replays: {}", dspCuda.totalGraphReplays());
            log.info("  Capture stats: {}", dspCuda.captureStats());

            // Show per-segment capture status
            int numSegs = dspCuda.numSegments();
            for (int s = 0; s < Math.min(numSegs, 8); s++) {
                log.info("    Seg {}: phase={}, replays={}, capturable={}, backend={}",
                        s, dspCuda.segmentExecutionPhase(s),
                        dspCuda.segmentReplayCount(s),
                        dspCuda.isSegmentCapturable(s),
                        dspCuda.segmentBackendName(s));
            }
        }

        // =====================================================================
        // 5. TRITON — JIT-compiled fused kernels
        // =====================================================================
        log.info("=== Mode 4: TRITON (JIT-compiled fused kernels) ===");
        log.info("  Compiles fusible segments through the Triton MLIR pipeline.");
        log.info("  Produces highly optimized fused kernels (element-wise + reductions).");
        log.info("  Non-fusible ops (matmul) still use cuBLAS.");
        log.info("  NOTE: Requires Triton support. Falls back to AUTO if unavailable.");

        SameDiff sdTriton = cloneGraph(sd);
        sdTriton.setDspAutoCompileEnabled(true);
        sdTriton.setDspNativeAutoCompileEnabled(true);

        // Use REDUCE_OVERHEAD for fast compilation
        sdTriton.setDspCompilationMode(DspCompilationMode.REDUCE_OVERHEAD);

        ph.put("input", Nd4j.randn(batch, dim));
        INDArray tritonResult = sdTriton.outputSingle(ph, "output");
        log.info("  Output shape: {}", Arrays.toString(tritonResult.shape()));

        // Warm up
        for (int i = 0; i < 5; i++) {
            ph.put("input", Nd4j.randn(batch, dim));
            sdTriton.outputSingle(ph, "output");
        }

        DspHandle dspTriton = sdTriton.dsp();
        if (dspTriton.isCompiled()) {
            log.info("  Plan phase: {}", dspTriton.planPhase());
            log.info("  Effective mode: {}", sdTriton.getGraphExecutionMode());
            log.info("  Segments: {}", dspTriton.numSegments());
        }

        // =====================================================================
        // 6. Compare numerical accuracy across modes
        // =====================================================================
        log.info("=== Numerical accuracy comparison ===");

        ph.put("input", Nd4j.randn(batch, dim));

        // Re-run all modes with the same input for comparison
        INDArray refOutput = sdSlot.outputSingle(ph, "output");
        INDArray emOutput = sdEmulated.outputSingle(ph, "output");
        INDArray cuOutput = sdCuda.outputSingle(ph, "output");
        INDArray trOutput = sdTriton.outputSingle(ph, "output");

        log.info("  Reference (SLOT_BY_SLOT) sum:   {}", refOutput.sumNumber());
        log.info("  EMULATED_REPLAY sum:             {}", emOutput.sumNumber());
        log.info("  CUDA_GRAPHS sum:                 {}", cuOutput.sumNumber());
        log.info("  TRITON sum:                      {}", trOutput.sumNumber());

        double emDiff = refOutput.sub(emOutput).norm2Number().doubleValue();
        double cuDiff = refOutput.sub(cuOutput).norm2Number().doubleValue();
        double trDiff = refOutput.sub(trOutput).norm2Number().doubleValue();

        log.info("  L2 diff EMULATED vs reference:   {}", emDiff);
        log.info("  L2 diff CUDA_GRAPHS vs reference:{}", cuDiff);
        log.info("  L2 diff TRITON vs reference:     {}", trDiff);

        // =====================================================================
        // 7. Steady-state tokens/sec simulation (autoregressive decode)
        // =====================================================================
        log.info("=== Steady-state tokens/sec (autoregressive decode simulation) ===");

        // In autoregressive decoding (LLMs), each "step" produces one token.
        // The graph runs with sequence_length=1 (single token) and the key
        // metric is tokens/sec at steady state. We simulate this by running
        // the same fixed-shape graph repeatedly — each call = one decode step.
        //
        // The plan stays in REPLAYING because shapes don't change between steps.
        // This is the production decode path: constant shape → constant plan → replay.

        int decodeSteps = 200;
        int warmupDecodeSteps = 20;

        // Use the CUDA_GRAPHS mode (or best available) for decode simulation
        SameDiff sdDecode = cloneGraph(sd);
        sdDecode.setDspAutoCompileEnabled(true);
        sdDecode.setDspNativeAutoCompileEnabled(true);
        sdDecode.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        // Fixed decode shape: batch=1, dim=128 (simulates single-token decode)
        Map<String, INDArray> decodePh = new HashMap<>();
        decodePh.put("input", Nd4j.randn(1, dim));

        // Warmup: let plan reach REPLAYING
        for (int i = 0; i < warmupDecodeSteps; i++) {
            sdDecode.outputSingle(decodePh, "output");
        }

        DspHandle dspDecode = sdDecode.dsp();
        log.info("  After {} warmup decode steps:", warmupDecodeSteps);
        if (dspDecode.isCompiled()) {
            log.info("    Plan phase: {} (2=REPLAYING)", dspDecode.planPhase());
            log.info("    Captured segments: {}", dspDecode.numCapturedGraphSegments());
        }

        // Measure steady-state: each step = 1 token
        long decodeStart = System.nanoTime();
        for (int i = 0; i < decodeSteps; i++) {
            sdDecode.outputSingle(decodePh, "output");
        }
        long decodeNs = System.nanoTime() - decodeStart;
        long decodeMs = decodeNs / 1_000_000;

        // tokens/sec = decodeSteps / (totalTimeSeconds)
        double tokPerSec = (decodeSteps * 1_000_000_000.0) / decodeNs;
        double usPerTok = (double) (decodeNs / 1_000) / decodeSteps;

        log.info("  Steady-state decode benchmark ({} steps, batch=1):", decodeSteps);
        log.info("    Total time:      {} ms", decodeMs);
        log.info("    Tokens/sec:      {}", String.format("%.1f", tokPerSec));
        log.info("    μs/token:        {}", String.format("%.0f", usPerTok));

        if (dspDecode.isCompiled()) {
            log.info("    Graph replays:   {}", dspDecode.totalGraphReplays());
            log.info("    Cache hits:      {}", dspDecode.replayCacheHits());
            log.info("    Cache misses:    {}", dspDecode.replayCacheMisses());
        }

        // Compare tok/sec across all modes at decode shape (batch=1)
        log.info("");
        log.info("  --- Tokens/sec comparison across modes (batch=1, {} steps) ---",
                decodeSteps);

        Map<String, INDArray> decode1 = new HashMap<>();
        decode1.put("input", Nd4j.randn(1, dim));

        String[] modeNames = {"SLOT_BY_SLOT", "EMULATED_REPLAY", "CUDA_GRAPHS", "TRITON"};
        SameDiff[] modeGraphs = {sdSlot, sdEmulated, sdCuda, sdTriton};

        // Warmup all modes at decode shape
        for (SameDiff m : modeGraphs) {
            for (int i = 0; i < 20; i++) m.outputSingle(decode1, "output");
        }

        for (int mi = 0; mi < modeNames.length; mi++) {
            long mStart = System.nanoTime();
            for (int i = 0; i < decodeSteps; i++) {
                modeGraphs[mi].outputSingle(decode1, "output");
            }
            long mNs = System.nanoTime() - mStart;
            double mTokSec = (decodeSteps * 1_000_000_000.0) / mNs;
            double mUsPerTok = (double) (mNs / 1_000) / decodeSteps;
            log.info("    {}: {} tok/s ({} μs/tok)",
                    String.format("%-17s", modeNames[mi]),
                    String.format("%.1f", mTokSec),
                    String.format("%.0f", mUsPerTok));
        }

        // =====================================================================
        // 8. Batch throughput benchmark (samples/sec)
        // =====================================================================
        log.info("=== Benchmark: samples/sec across modes (batch={}) ===", batch);

        int benchReps = 50;
        ph.put("input", Nd4j.randn(batch, dim));

        // Warmup
        for (int i = 0; i < 10; i++) {
            sdSlot.outputSingle(ph, "output");
            sdEmulated.outputSingle(ph, "output");
            sdCuda.outputSingle(ph, "output");
            sdTriton.outputSingle(ph, "output");
        }

        long slotTime = benchMode(sdSlot, ph, benchReps);
        long emTime = benchMode(sdEmulated, ph, benchReps);
        long cuTime = benchMode(sdCuda, ph, benchReps);
        long trTime = benchMode(sdTriton, ph, benchReps);

        double slotSps = (benchReps * batch * 1000.0) / slotTime;
        double emSps = (benchReps * batch * 1000.0) / emTime;
        double cuSps = (benchReps * batch * 1000.0) / cuTime;
        double trSps = (benchReps * batch * 1000.0) / trTime;

        log.info("  SLOT_BY_SLOT:    {} ms ({} ms/rep, {} samples/sec)", slotTime,
                String.format("%.2f", (double) slotTime / benchReps), String.format("%.0f", slotSps));
        log.info("  EMULATED_REPLAY: {} ms ({} ms/rep, {} samples/sec)", emTime,
                String.format("%.2f", (double) emTime / benchReps), String.format("%.0f", emSps));
        log.info("  CUDA_GRAPHS:     {} ms ({} ms/rep, {} samples/sec)", cuTime,
                String.format("%.2f", (double) cuTime / benchReps), String.format("%.0f", cuSps));
        log.info("  TRITON:          {} ms ({} ms/rep, {} samples/sec)", trTime,
                String.format("%.2f", (double) trTime / benchReps), String.format("%.0f", trSps));

        // =====================================================================
        // Summary
        // =====================================================================
        log.info("");
        log.info("=== Replay Mode Summary ===");
        log.info("");
        log.info("HARDWARE-LEVEL REPLAY (CUDA_GRAPHS, HIP_GRAPHS, VULKAN, METAL, LEVEL_ZERO):");
        log.info("  - Records GPU kernel launches into a driver graph object");
        log.info("  - Replays ALL kernels in a single API call (~20μs total)");
        log.info("  - Eliminates per-kernel launch overhead (~5-10μs × N ops)");
        log.info("  - Requires: stable pointers, no CPU control flow, GPU-native ops");
        log.info("  - Best for: fixed-shape inference with many small ops");
        log.info("");
        log.info("EMULATED REPLAY (EMULATED_REPLAY):");
        log.info("  - Runs slot-by-slot but tracks full DSP lifecycle");
        log.info("  - No GPU graph APIs used — works on any platform");
        log.info("  - Diagnostic tool: shows what hardware replay WOULD do");
        log.info("  - Use to debug: shape drift, pointer instability, segment issues");
        log.info("");
        log.info("JIT-COMPILED REPLAY (TRITON, NVRTC_JIT, PTX_JIT, OPENVINO):");
        log.info("  - Compiles fusible op segments into optimized fused kernels");
        log.info("  - Each segment is a separate (but highly optimized) kernel launch");
        log.info("  - Triton produces the best fused kernels (via MLIR pipeline)");
        log.info("  - Non-fusible ops (matmul) still use vendor libraries (cuBLAS)");
        log.info("");
        log.info("SLOT_BY_SLOT:");
        log.info("  - No compilation, no fusion, no graph capture");
        log.info("  - Correctness baseline for validating other modes");
        log.info("  - Use: sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT)");
        log.info("");
        log.info("COMPILE CLASSIFIER (must be enabled for all replay modes):");
        log.info("  sd.setDspAutoCompileEnabled(true)");
        log.info("  sd.setDspNativeAutoCompileEnabled(true)");
        log.info("  These default to true. If disabled, plans stay in SLOT_BY_SLOT.");
        log.info("");
        log.info("Steady-state performance metrics:");
        log.info("  tokens/sec = numSteps / totalTimeSeconds  (for batch=1 decode)");
        log.info("  samples/sec = (numSteps × batchSize) / totalTimeSeconds  (for batched)");
        log.info("  dsp.replayCacheHits()    — plan cache hits (shape reuse)");
        log.info("  dsp.replayCacheMisses()  — plan cache misses (new compile)");
        log.info("  dsp.totalGraphReplays()  — cumulative GPU graph replay count");
        log.info("");
        log.info("DSP_DIAG (observe mode behavior):");
        log.info("  DspDiagnostics.setCategories(SEGMENT | GRAPH_REPLAY | BACKEND | EMULATED_REPLAY)");
        log.info("  Or: -Dnd4j.dsp.diagnostics=SEGMENT,GRAPH_REPLAY,BACKEND,EMULATED_REPLAY");
        log.info("  EMULATED_REPLAY category: events from EMULATED_REPLAY mode");
        log.info("  GRAPH_REPLAY category: real CUDA graph capture/replay events");
        log.info("  BACKEND category: which backend was selected for each segment");
        log.info("  SEGMENT category: segment formation and boundaries");

        // Clean up diagnostics
        DspDiagnostics.setCategories(DspDiagnostics.NONE);

        log.info("**************** DSP Replay Modes Example finished ********************");
    }

    /**
     * Clone a SameDiff graph by serialization round-trip.
     * Each clone gets its own independent plan cache.
     */
    private static SameDiff cloneGraph(SameDiff original) {
        try {
            java.io.File tmp = java.io.File.createTempFile("sd_clone_", ".sdnb");
            tmp.deleteOnExit();
            original.save(tmp, false);
            return SameDiff.load(tmp, false);
        } catch (Exception e) {
            throw new RuntimeException("Failed to clone SameDiff graph", e);
        }
    }

    /**
     * Benchmark a mode by running N reps and returning total time in ms.
     */
    private static long benchMode(SameDiff sd, Map<String, INDArray> ph, int reps) {
        long start = System.nanoTime();
        for (int i = 0; i < reps; i++) {
            sd.outputSingle(ph, "output");
        }
        return (System.nanoTime() - start) / 1_000_000;
    }
}
