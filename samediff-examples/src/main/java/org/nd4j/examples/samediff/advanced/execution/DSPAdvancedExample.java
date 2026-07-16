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
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * DSP (Dynamic Shape Plan) Advanced API Example.
 *
 * DSP is SameDiff's production execution engine. It compiles the computation graph
 * into an optimized execution plan that maps operations to flat integer-indexed
 * "slots" for efficient replay.
 *
 * <h3>Compilation API (SameDiff methods):</h3>
 * <pre>
 * // Compile the graph for specified outputs
 * GraphExecutionMode mode = sd.compileNativeDynamicShapePlan("output1", "output2");
 * GraphExecutionMode mode = sd.compileNativeDynamicShapePlan(DspCompilationMode.REDUCE_OVERHEAD, "output");
 * GraphExecutionMode mode = sd.compileNativeDynamicShapePlan(outputs, mode, fallbackToAuto);
 *
 * // Compilation modes (analogous to torch.compile):
 * //   REDUCE_OVERHEAD — minimize startup, fast JIT path selection
 * //   SPLIT_STITCH    — balanced, split-and-stitch Triton compilation
 * //   MAX_AUTOTUNE    — maximize throughput, best Triton quality
 *
 * // Shape freezing (required for CUDA graph capture)
 * sd.setDspShapesFrozen(true);   // Lock shapes — enables CUDA graph replay
 * sd.isDspShapesFrozen();        // Check if shapes are frozen
 *
 * // Plan cache
 * sd.clearDynamicShapePlanCache();       // Clear cached plans
 * sd.reassignDynamicShapePlanDevices();  // Reassign device placement
 *
 * // Mode comparison (benchmarking)
 * Map&lt;String, Double&gt; times = sd.compareDspModes(modes, inputs, outputs, warmup, reps);
 * </pre>
 *
 * <h3>DspHandle — Debugging and Introspection (sd.dsp()):</h3>
 * <pre>
 * DspHandle dsp = sd.dsp();
 *
 * // Plan state
 * dsp.isCompiled()              // Is plan compiled?
 * dsp.totalSlots()              // Number of operation slots
 * dsp.numExternalInputs()       // Number of external inputs (placeholders)
 * dsp.planSummary()             // Text summary of the plan
 * dsp.executeCount()            // Total graph executions
 *
 * // Slot inspection (debug individual operations)
 * dsp.getSlotOutput(slotIdx)         // Get output of slot i
 * dsp.getSlotOutput("opName")        // Get output by op name substring
 * dsp.snapshotAllSlots()             // Snapshot all slot outputs
 * dsp.slotIndexForOutput("varName")  // Find slot for a variable
 * dsp.slotIndexForOp("opName")       // Find slot for an operation
 *
 * // NaN debugging
 * dsp.firstNaNSlot()            // First slot producing NaN
 * dsp.hasOutputIssues()         // Any NaN/Inf in outputs?
 * dsp.validateOutputs()         // Validate all outputs
 *
 * // Replay
 * dsp.replay(placeholders)      // Execute the compiled plan
 *
 * // Segment introspection (CUDA graph segments)
 * dsp.numSegments()
 * dsp.segmentReplayCount(i)
 * dsp.segmentReplayMode(i)
 * dsp.segmentBackendName(i)
 * dsp.segmentStatisticsJson(i)
 * dsp.isSegmentCapturable(i)
 *
 * // Buffer coloring (memory optimization)
 * dsp.bufferColoringApplied()
 * dsp.bufferColoringNumColors()
 * dsp.bufferColoringBytesSaved()
 *
 * // Chrome trace export (for profiling visualization)
 * dsp.exportChromeTrace("trace.json")
 * dsp.exportCudaGraphHtml("graph.html")
 *
 * // KV cache position tracking (LLM inference)
 * dsp.kvCachePosition()
 * </pre>
 *
 * <h3>DynamicShapePlan — The Compiled Plan Object:</h3>
 * <pre>
 * DynamicShapePlan plan = sd.getCachedDynamicShapePlan(outputNames);
 *
 * // Introspection
 * plan.getSummary()              // Text summary
 * plan.getDetailedSummary()      // Detailed summary
 * plan.toDot()                   // GraphViz DOT format
 * plan.getSegments()             // List of execution segments
 * plan.getParallelGroups()       // Parallel execution groups
 *
 * // Device placement
 * plan.assignDevices()           // Auto device assignment
 * plan.assignDevices(deviceMemBudgets)  // With memory budgets
 * plan.assignDeviceToRange(start, end, deviceId)
 * plan.getDeviceAssignmentSummary()
 *
 * // Serialization
 * byte[] data = plan.serialize()         // Compact binary for C++ executor
 * long hash = plan.computeStructureHash() // Structure hash for caching
 * </pre>
 *
 * <h3>Distributed Execution:</h3>
 * <pre>
 * // Tensor Parallelism (split layers across GPUs)
 * TensorParallelConfig tp = TensorParallelConfig.create(numGpus, rank);
 * tp = tp.withDeviceIds(0, 1, 2, 3);
 *
 * // Pipeline Parallelism (split stages across devices)
 * PipelineParallelRunner pipeRunner = new PipelineParallelRunner(config);
 *
 * // DDP Training
 * DistributedDataParallelTrainer ddp = new DistributedDataParallelTrainer(sd, ddpConfig);
 * ddp.fit(dataIterator, numEpochs);
 * ddp.syncParameters();
 * ddp.saveCheckpoint(file, step);
 *
 * // DDP Config
 * DistributedTrainingConfig ddpConfig = DistributedTrainingConfig.singleNodeMultiGpu(4);
 * DistributedTrainingConfig ddpConfig = DistributedTrainingConfig.multiNode(numNodes, gpusPerNode);
 * </pre>
 */
public class DSPAdvancedExample {
    private static final Logger log = LoggerFactory.getLogger(DSPAdvancedExample.class);

    public static void main(String[] args) {

        // =====================================================================
        // 1. Build a simple graph for DSP demonstration
        // =====================================================================
        log.info("=== Building SameDiff graph ===");

        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 128);
        SDVariable w1 = sd.var("w1", Nd4j.randn(128, 64).muli(0.01));
        SDVariable b1 = sd.var("b1", Nd4j.zeros(64));
        SDVariable w2 = sd.var("w2", Nd4j.randn(64, 10).muli(0.01));
        SDVariable b2 = sd.var("b2", Nd4j.zeros(10));

        SDVariable z1 = input.mmul(w1).add(b1);
        SDVariable a1 = sd.nn.relu(z1, 0);
        SDVariable z2 = a1.mmul(w2).add(b2);
        SDVariable output = sd.nn.softmax("output", z2, -1);

        log.info("Graph: {} variables, {} ops", sd.variables().size(), sd.ops().length);

        // =====================================================================
        // 2. Standard execution (auto DSP compilation)
        // =====================================================================
        log.info("=== Standard execution ===");

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.randn(8, 128));

        INDArray result = sd.outputSingle(ph, "output");
        log.info("Output shape: {}", java.util.Arrays.toString(result.shape()));

        // =====================================================================
        // 3. DSP compilation and execution APIs (documented)
        // =====================================================================
        log.info("=== DSP Compilation API ===");
        log.info("");
        log.info("--- Compilation ---");
        log.info("  sd.compileNativeDynamicShapePlan(\"output\")");
        log.info("  sd.compileNativeDynamicShapePlan(DspCompilationMode.REDUCE_OVERHEAD, \"output\")");
        log.info("  sd.compileNativeDynamicShapePlan(outputSet, mode, fallbackToAuto)");
        log.info("");
        log.info("--- Compilation modes ---");
        log.info("  REDUCE_OVERHEAD — minimize JIT startup time");
        log.info("  SPLIT_STITCH    — balanced Triton compilation");
        log.info("  MAX_AUTOTUNE    — max throughput, best kernel selection");
        log.info("");
        log.info("--- Shape freezing (CUDA graph capture) ---");
        log.info("  sd.setDspShapesFrozen(true)  — lock shapes for CUDA graph replay");
        log.info("  sd.isDspShapesFrozen()        — check frozen state");
        log.info("");
        log.info("--- DspHandle (sd.dsp()) ---");
        log.info("  .isCompiled()         — plan ready?");
        log.info("  .totalSlots()         — number of op slots");
        log.info("  .planSummary()        — text summary");
        log.info("  .replay(placeholders) — execute the plan");
        log.info("  .firstNaNSlot()       — debug NaN issues");
        log.info("  .snapshotAllSlots()   — capture all intermediate outputs");
        log.info("  .exportChromeTrace(f) — profiling visualization");
        log.info("");
        log.info("--- DynamicShapePlan ---");
        log.info("  plan.toDot()                  — GraphViz visualization");
        log.info("  plan.assignDevices()           — multi-device placement");
        log.info("  plan.serialize()              — binary for native executor");
        log.info("");
        log.info("--- Distributed ---");
        log.info("  TensorParallelConfig.create(numGpus, rank)");
        log.info("  DistributedDataParallelTrainer(sd, ddpConfig)");
        log.info("  DistributedTrainingConfig.singleNodeMultiGpu(4)");

        // =====================================================================
        // 4. Dynamic shape caching demonstration
        // =====================================================================
        log.info("=== Dynamic shape caching ===");

        // DSP caches compiled plans by input shapes.
        // When shapes change, a new plan is compiled and cached.
        for (int batch : new int[]{1, 4, 16, 32, 64}) {
            ph.put("input", Nd4j.randn(batch, 128));
            INDArray out = sd.outputSingle(ph, "output");
            log.info("  Batch {}: output shape {}", batch,
                    java.util.Arrays.toString(out.shape()));
        }

        // Subsequent calls with the same shapes reuse cached plans (no recompilation)
        log.info("  Re-running batch 32 (cached plan reused)...");
        ph.put("input", Nd4j.randn(32, 128));
        result = sd.outputSingle(ph, "output");
        log.info("  Output shape: {}", java.util.Arrays.toString(result.shape()));

        // =====================================================================
        // 5. DspHandle live introspection (real API calls, not just docs)
        // =====================================================================
        log.info("=== DspHandle introspection (sd.dsp()) ===");
        // sd.dsp() is valid only after at least one sd.output() call.
        DspHandle dsp = sd.dsp();
        log.info("  isCompiled()      = {}", dsp.isCompiled());
        log.info("  totalSlots()      = {}", dsp.totalSlots());
        log.info("  numExternalInputs = {}", dsp.numExternalInputs());
        log.info("  executeCount()    = {}", dsp.executeCount());
        log.info("  planPhase()       = {} (0=SLOT_BY_SLOT 1=SHAPES_FROZEN 2=REPLAYING)", dsp.planPhase());
        log.info("  frozenExecCount() = {}", dsp.frozenExecCount());
        log.info("  numSegments()     = {}", dsp.numSegments());
        if (dsp.isCompiled()) {
            log.info("  planSummary() summary (first 200 chars): {}",
                    dsp.planSummary().substring(0, Math.min(200, dsp.planSummary().length())));
        }
        // firstNaNSlot returns -1 when no NaN found (healthy)
        int nanSlot = dsp.firstNaNSlot();
        log.info("  firstNaNSlot()    = {} ({})", nanSlot, nanSlot < 0 ? "no NaN — healthy" : "NaN detected");
        // Buffer coloring: memory reuse between non-overlapping slots
        log.info("  bufferColoringApplied() = {}", dsp.bufferColoringApplied());
        if (dsp.bufferColoringApplied()) {
            log.info("  bufferColoringNumColors()  = {}", dsp.bufferColoringNumColors());
            log.info("  bufferColoringBytesSaved() = {} bytes", dsp.bufferColoringBytesSaved());
        }

        log.info("**************** DSP Advanced Example finished ********************");
    }
}
