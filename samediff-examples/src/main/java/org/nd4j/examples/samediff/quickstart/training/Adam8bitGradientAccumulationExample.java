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

package org.nd4j.examples.samediff.quickstart.training;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.training.GradientAccumulator;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Adam8bit;

import java.util.Map;

/**
 * 8-bit Adam Optimizer and Gradient Accumulation in SameDiff
 *
 * This example demonstrates two memory-saving techniques for training large models:
 *
 * 1. Adam8bit - quantizes optimizer state (first and second moments) to INT8,
 *    reducing optimizer memory by approximately 4x compared to standard FP32 Adam.
 *    The quantization uses a block-wise scheme: each block of `blockSize` values
 *    is independently scaled to INT8, preserving per-block dynamic range.
 *
 * 2. Gradient Accumulation - accumulates gradients over N micro-batches before
 *    applying a weight update, simulating a larger effective batch size without
 *    requiring proportionally more GPU memory.
 *
 * Optimizer Memory Comparison (per parameter):
 *
 *   Adam (FP32):   2 x FP32 states = 8 bytes/param
 *   Adam (FP16):   2 x FP16 states = 4 bytes/param  (may lose precision)
 *   Adam8bit:      2 x INT8 states = 2 bytes/param  (~4x vs FP32, minimal quality loss)
 *
 * For a 7B parameter model, optimizer states alone cost:
 *   Adam FP32:  56 GB
 *   Adam8bit:   14 GB  (saves 42 GB)
 *
 * Key classes:
 *   - Adam8bit:           8-bit quantized Adam optimizer
 *   - GradientAccumulator: multi-step gradient accumulation
 *   - TrainingConfig:     wires the optimizer into a SameDiff training session
 */
public class Adam8bitGradientAccumulationExample {

    public static void main(String[] args) {

        // ============================================================
        // 1. ADAM8BIT OPTIMIZER - Basics
        // ============================================================
        System.out.println("=== Adam8bit Optimizer ===");
        {
            // Adam8bit.builder() mirrors Adam but stores optimizer moments in INT8.
            //
            // blockSize: number of values grouped together for quantization scaling.
            //   - Smaller blockSize → more scale factors stored → better precision
            //   - Larger blockSize → fewer scale factors → more compression
            //   - Default 2048 is a good balance; 256 gives higher precision.
            //
            // Standard Adam hyperparameters (beta1, beta2, epsilon) are preserved;
            // only the in-memory representation of m1/m2 is quantized.
            Adam8bit adam8bit = Adam8bit.builder()
                    .learningRate(1e-4)
                    .beta1(0.9)
                    .beta2(0.999)
                    .epsilon(1e-8)
                    .blockSize(2048)   // INT8 quantization block granularity
                    .build();

            System.out.println("  Learning rate:  " + adam8bit.getLearningRate(0));
            System.out.println("  Beta1:          " + adam8bit.getBeta1());
            System.out.println("  Beta2:          " + adam8bit.getBeta2());
            System.out.println("  Block size:     " + adam8bit.getBlockSize());
            System.out.println("  Memory per param (approx): 2 bytes (INT8 x2)");
            System.out.println("  vs standard Adam: 8 bytes (FP32 x2)  =>  ~4x savings");
        }

        // ============================================================
        // 2. ADAM8BIT WITH DIFFERENT BLOCK SIZES
        // ============================================================
        System.out.println("\n=== Block Size Comparison ===");
        {
            // blockSize=256: higher precision (more scale factors per parameter)
            Adam8bit highPrecision = Adam8bit.builder()
                    .learningRate(1e-4)
                    .blockSize(256)
                    .build();

            // blockSize=4096: higher compression (fewer scale factors)
            Adam8bit highCompression = Adam8bit.builder()
                    .learningRate(1e-4)
                    .blockSize(4096)
                    .build();

            // blockSize=2048: default trade-off (recommended starting point)
            Adam8bit defaultConfig = new Adam8bit(1e-4);

            System.out.println("  blockSize=256:  higher precision, slightly more scale overhead");
            System.out.println("  blockSize=2048: default, good balance (recommended)");
            System.out.println("  blockSize=4096: maximum compression, may lose some precision");
            System.out.println("  Scale factor overhead = params / blockSize x 4 bytes (FP32)");
        }

        // ============================================================
        // 3. ADAM8BIT IN TRAININGCONFIG
        // ============================================================
        System.out.println("\n=== Adam8bit in TrainingConfig ===");
        {
            SameDiff sd = SameDiff.create();

            // Simple two-layer MLP for demonstration
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 512);
            SDVariable label  = sd.placeHolder("label",  DataType.FLOAT, -1, 10);

            SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, 512, 256).mul(0.02));
            SDVariable b1 = sd.var("b1", Nd4j.zeros(DataType.FLOAT, 256));
            SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, 256, 10).mul(0.02));
            SDVariable b2 = sd.var("b2", Nd4j.zeros(DataType.FLOAT, 10));

            SDVariable hidden = sd.nn().relu(input.mmul(w1).add(b1), 0);
            SDVariable output = sd.nn().softmax("output", hidden.mmul(w2).add(b2));
            sd.loss().softmaxCrossEntropy("loss", label, output, null);

            // Wire Adam8bit as the updater in TrainingConfig.
            // Everything else in the training loop remains identical to standard Adam.
            TrainingConfig config = TrainingConfig.builder()
                    .updater(new Adam8bit(1e-4))       // drop-in replacement for Adam
                    .dataSetFeatureMapping("input")
                    .dataSetLabelMapping("label")
                    .build();

            sd.setTrainingConfig(config);

            System.out.println("  Updater class:  " + config.getUpdater().getClass().getSimpleName());
            System.out.println("  Adam8bit is a drop-in replacement for Adam in any TrainingConfig.");
            System.out.println("  The training API (fit, output, etc.) is identical.");
        }

        // ============================================================
        // 4. GRADIENT ACCUMULATION - Basics
        // ============================================================
        System.out.println("\n=== Gradient Accumulation ===");
        {
            // GradientAccumulator accumulates gradients from N consecutive micro-batches
            // without applying a weight update. After N steps it averages the accumulated
            // gradients and signals that the optimizer should step.
            //
            // This lets you train with an effective batch size of:
            //   effective_batch = micro_batch_size * accumulation_steps
            //
            // without storing all micro-batches in memory simultaneously.
            int accumulationSteps = 4;
            GradientAccumulator accumulator = new GradientAccumulator(accumulationSteps);

            System.out.println("  Accumulation steps: " + accumulationSteps);
            System.out.println("  Is enabled:         " + accumulator.isEnabled());
            System.out.println("  Is ready:           " + accumulator.isReady() + " (need " + accumulationSteps + " steps first)");

            // Simulate micro-batch gradient accumulation loop
            System.out.println("\n  Simulating " + accumulationSteps + " micro-batch gradient steps:");
            for (int step = 0; step < accumulationSteps; step++) {
                // In a real loop: compute forward + backward on a micro-batch,
                // then call accumulate() for each trainable parameter's gradient.
                INDArray microGrad = Nd4j.randn(DataType.FLOAT, 512, 256).mul(0.01);
                accumulator.accumulate("w1", microGrad);

                INDArray microGradB = Nd4j.randn(DataType.FLOAT, 256).mul(0.01);
                accumulator.accumulate("b1", microGradB);

                // step() increments the internal counter
                accumulator.step();
                System.out.println("  Step " + (step + 1) + "/" + accumulationSteps
                        + " - accumulated, ready=" + accumulator.isReady());
            }

            // After N steps, retrieve averaged gradients and reset
            if (accumulator.isReady()) {
                Map<String, INDArray> avgGrads = accumulator.getAndReset();
                System.out.println("\n  Averaged gradient shapes after accumulation:");
                for (Map.Entry<String, INDArray> entry : avgGrads.entrySet()) {
                    System.out.println("    " + entry.getKey() + ": " + entry.getValue().shapeInfoToString());
                }
                System.out.println("  Accumulator reset - ready=" + accumulator.isReady());
            }
        }

        // ============================================================
        // 5. GRADIENT ACCUMULATION IN TRAININGCONFIG
        // ============================================================
        System.out.println("\n=== Gradient Accumulation in TrainingConfig ===");
        {
            SameDiff sd = SameDiff.create();

            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 256);
            SDVariable label  = sd.placeHolder("label",  DataType.FLOAT, -1, 10);
            SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 256, 10).mul(0.02));
            SDVariable output = sd.nn().softmax("output", input.mmul(w));
            sd.loss().softmaxCrossEntropy("loss", label, output, null);

            // gradientAccumulationSteps(N) tells the training loop to accumulate
            // gradients over N calls to fit() before applying an optimizer update.
            //
            // micro_batch_size=32, accum_steps=8 => effective_batch=256
            TrainingConfig config = TrainingConfig.builder()
                    .updater(new Adam(3e-4))
                    .dataSetFeatureMapping("input")
                    .dataSetLabelMapping("label")
                    .gradientAccumulationSteps(8)    // simulate 8x larger batch
                    .build();

            sd.setTrainingConfig(config);

            System.out.println("  Gradient accumulation steps: " + config.getGradientAccumulationSteps());
            System.out.println("  Accumulation enabled:        " + config.isGradientAccumulationEnabled());
            System.out.println("  With micro_batch=32: effective batch = 32 x 8 = 256");
        }

        // ============================================================
        // 6. COMBINING ADAM8BIT + GRADIENT ACCUMULATION
        // ============================================================
        System.out.println("\n=== Combined: Adam8bit + Gradient Accumulation ===");
        {
            SameDiff sd = SameDiff.create();

            // Compact model definition
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1024);
            SDVariable label  = sd.placeHolder("label",  DataType.FLOAT, -1, 128);

            SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, 1024, 512).mul(0.02));
            SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT,  512, 256).mul(0.02));
            SDVariable w3 = sd.var("w3", Nd4j.randn(DataType.FLOAT,  256, 128).mul(0.02));

            SDVariable h1 = sd.nn().relu(input.mmul(w1), 0);
            SDVariable h2 = sd.nn().relu(h1.mmul(w2),   0);
            SDVariable output = sd.nn().softmax("output", h2.mmul(w3));
            sd.loss().softmaxCrossEntropy("loss", label, output, null);

            // Adam8bit saves ~4x optimizer memory.
            // Gradient accumulation with 16 steps simulates a 16x larger batch.
            // Together these are the primary tools for training large models
            // on consumer / single-GPU hardware.
            TrainingConfig config = TrainingConfig.builder()
                    .updater(Adam8bit.builder()
                            .learningRate(1e-4)
                            .blockSize(2048)
                            .build())
                    .dataSetFeatureMapping("input")
                    .dataSetLabelMapping("label")
                    .gradientAccumulationSteps(16)
                    .build();

            sd.setTrainingConfig(config);

            System.out.println("  Optimizer:               Adam8bit (block size 2048)");
            System.out.println("  Gradient accum steps:    " + config.getGradientAccumulationSteps());
            System.out.println("  Effective batch multiplier: 16x");
            System.out.println("  Optimizer memory savings:  ~4x vs standard Adam FP32");
            System.out.println("  Combined memory saving:    enables training models ~4-8x");
            System.out.println("                             larger on the same hardware");
        }

        // ============================================================
        // SUMMARY
        // ============================================================
        System.out.println("\n=== Technique Summary ===");
        System.out.println("  Adam8bit:");
        System.out.println("    - Quantizes optimizer moments to INT8 per block");
        System.out.println("    - ~4x optimizer memory reduction vs FP32 Adam");
        System.out.println("    - Drop-in replacement: same API as Adam");
        System.out.println("    - blockSize=2048 recommended (default)");
        System.out.println("    - Minimal quality loss on most tasks");
        System.out.println();
        System.out.println("  Gradient Accumulation:");
        System.out.println("    - accumulate N micro-batches before each optimizer step");
        System.out.println("    - effective_batch = micro_batch * accumulation_steps");
        System.out.println("    - Trades training speed for memory efficiency");
        System.out.println("    - No quality loss vs true large-batch training");
        System.out.println("    - Use gradientAccumulationSteps(N) in TrainingConfig");
        System.out.println();
        System.out.println("Adam8bit + gradient accumulation example completed successfully.");
    }
}
