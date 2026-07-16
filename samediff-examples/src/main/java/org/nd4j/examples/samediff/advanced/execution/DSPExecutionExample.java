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
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Dynamic Shape Plan (DSP) Execution Example.
 *
 * DSP is SameDiff's graph compilation and execution infrastructure that enables
 * efficient execution of computation graphs, especially for LLM inference where
 * shapes change dynamically (variable sequence lengths, batch sizes).
 *
 * Key concepts:
 *
 * 1. GraphExecutionMode: Controls how the graph is executed.
 *    - AUTO: Automatically selects the best execution mode
 *    - SLOT_BY_SLOT: Execute ops one at a time (debugging)
 *    - CUDA_GRAPHS: Capture and replay CUDA graph (eliminates kernel launch overhead)
 *    - TRITON: Use Triton JIT-compiled kernels for fused ops
 *    - OPENVINO: Use Intel OpenVINO backend
 *
 * 2. DspCompilationMode: Controls compilation optimization level.
 *    - REDUCE_OVERHEAD: Minimize compilation time
 *    - SPLIT_STITCH: Split graph into segments for partial recompilation
 *    - MAX_AUTOTUNE: Maximum auto-tuning (slower compile, faster runtime)
 *
 * 3. Shape-keyed plan caching: Compiled plans are cached by input shapes,
 *    so recompilation only happens when shapes change.
 *
 * This example demonstrates building and executing a SameDiff graph with
 * mixed precision and training configuration.
 */
public class DSPExecutionExample {
    private static final Logger log = LoggerFactory.getLogger(DSPExecutionExample.class);

    public static void main(String[] args) {

        // =====================================================================
        // 1. Build a simple SameDiff computation graph
        // =====================================================================
        log.info("=== Building SameDiff graph ===");

        SameDiff sd = SameDiff.create();

        // Define placeholders (dynamic shapes via -1)
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 784);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 10);

        // Build a simple feedforward network
        SDVariable w1 = sd.var("w1", Nd4j.randn(784, 256).muli(0.01));
        SDVariable b1 = sd.var("b1", Nd4j.zeros(256));
        SDVariable w2 = sd.var("w2", Nd4j.randn(256, 10).muli(0.01));
        SDVariable b2 = sd.var("b2", Nd4j.zeros(10));

        SDVariable z1 = input.mmul(w1).add(b1);
        SDVariable a1 = sd.nn.relu(z1, 0);
        SDVariable z2 = a1.mmul(w2).add(b2);
        SDVariable predictions = sd.nn.softmax("predictions", z2, -1);

        // Cross-entropy loss
        SDVariable loss = sd.loss.softmaxCrossEntropy("loss", label, z2, null);

        log.info("Variables: {}", sd.variables().size());
        log.info("Operations: {}", sd.ops().length);

        // =====================================================================
        // 2. Configure training with mixed precision
        // =====================================================================
        log.info("=== Configuring training ===");

        TrainingConfig config = TrainingConfig.builder()
                .updater(new Adam(1e-3))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .build();

        sd.setTrainingConfig(config);
        log.info("Training configured with Adam optimizer");

        // =====================================================================
        // 3. Execute inference with explicit placeholders
        // =====================================================================
        log.info("=== Running inference ===");

        INDArray inputData = Nd4j.randn(32, 784);  // Batch of 32
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("input", inputData);

        // outputSingle executes the graph and returns the named output
        INDArray output = sd.outputSingle(placeholders, "predictions");
        log.info("Output shape: {}", java.util.Arrays.toString(output.shape()));
        log.info("Output sum (should be ~32 for 32 softmax rows): {}", output.sumNumber());

        // =====================================================================
        // 4. Execute with different batch sizes (dynamic shapes)
        // =====================================================================
        log.info("=== Dynamic shape execution ===");

        // DSP recompiles the plan when input shapes change
        for (int batchSize : new int[]{1, 16, 64, 128}) {
            INDArray batchInput = Nd4j.randn(batchSize, 784);
            placeholders.put("input", batchInput);
            INDArray batchOutput = sd.outputSingle(placeholders, "predictions");
            log.info("  Batch {}: output shape {}", batchSize,
                    java.util.Arrays.toString(batchOutput.shape()));
        }

        // =====================================================================
        // 5. DspHandle introspection (available after any sd.output() call)
        // =====================================================================
        log.info("=== DspHandle plan inspection ===");
        // sd.dsp() returns the live plan handle for the CURRENT shape key.
        // Always call AFTER at least one sd.outputSingle() / sd.output().
        DspHandle dsp = sd.dsp();
        log.info("  isCompiled()      = {}", dsp.isCompiled());
        log.info("  totalSlots()      = {} operation slots in plan", dsp.totalSlots());
        log.info("  numExternalInputs = {} (placeholders wired into plan)", dsp.numExternalInputs());
        log.info("  executeCount()    = {} total graph executions", dsp.executeCount());
        log.info("  planPhase()       = {} (0=SLOT_BY_SLOT 1=SHAPES_FROZEN 2=REPLAYING)", dsp.planPhase());
        // On CPU, plan may stay at SHAPES_FROZEN (phase=1) because CUDA graph
        // capture requires GPU. On CUDA, after enough stable executions the plan
        // advances to REPLAYING (phase=2) for lowest-latency replay.
        log.info("  numSegments()     = {}", dsp.numSegments());
        for (int i = 0; i < dsp.numSegments(); i++) {
            log.info("    segment[{}]: phase={} capturable={} backend={}",
                    i, dsp.segmentExecutionPhase(i),
                    dsp.isSegmentCapturable(i),
                    dsp.segmentBackendName(i));
        }

        // =====================================================================
        // 6. Save and load the model
        // =====================================================================
        log.info("=== Model serialization ===");

        // SameDiff models can be saved in SDZ (SameDiff ZIP) format
        // or exported to ONNX for interoperability
        log.info("Model can be saved with sd.save(file, withUpdaterState)");
        log.info("Model can be loaded with SameDiff.load(file, withUpdaterState)");

        log.info("**************** DSP Execution Example finished ********************");
    }
}
