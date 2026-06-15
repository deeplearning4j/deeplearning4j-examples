/*
 *
 * This program and the accompanying materials are made available under the
 *  terms of the Apache License, Version 2.0 which is available at
 *  https://www.apache.org/licenses/LICENSE-2.0.
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  License for the specific language governing permissions and limitations
 *  under the License.
 *
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package org.nd4j.examples.samediff.quickstart.training;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.config.FP8TrainingConfig;
import org.nd4j.autodiff.samediff.config.LossScaleConfig;
import org.nd4j.autodiff.samediff.config.SFTConfig;
import org.nd4j.autodiff.samediff.training.GradientAccumulator;
import org.nd4j.autodiff.samediff.training.LossScaler;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

/**
 * Mixed Precision Training in SameDiff - Complete API Reference
 *
 * Mixed precision training uses lower-precision floating point (FP16/BF16/FP8) for
 * compute while maintaining FP32 master weights, achieving faster training with
 * minimal accuracy loss.
 *
 * Topics covered:
 *   1. FP16 Mixed Precision with Dynamic Loss Scaling
 *   2. BF16 Mixed Precision (simplified, no loss scaling needed)
 *   3. FP8 Training Configuration
 *   4. Gradient Accumulation
 *   5. Loss Scaling API (static and dynamic)
 *   6. SFT/Fine-tuning with Mixed Precision
 *   7. Combining all features
 *
 * Key classes:
 *   - TrainingConfig.Builder: mixedPrecision(), mixedPrecisionBfloat16()
 *   - LossScaleConfig: staticScaling(), dynamicScaling()
 *   - LossScaler: runtime loss scale management
 *   - GradientAccumulator: multi-step gradient accumulation
 *   - FP8TrainingConfig: FP8 E4M3/E5M2 configuration
 *   - SFTConfig: supervised fine-tuning defaults
 */
public class MixedPrecisionTrainingExample {

    public static void main(String[] args) {

        // ============================================================
        // 1. FP16 MIXED PRECISION with Dynamic Loss Scaling
        // ============================================================
        System.out.println("=== FP16 Mixed Precision Training ===");
        {
            SameDiff sd = SameDiff.create();

            // Simple model
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 784);
            SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 10);

            SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, 784, 256).mul(0.01));
            SDVariable b1 = sd.var("b1", Nd4j.zeros(DataType.FLOAT, 256));
            SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, 256, 10).mul(0.01));
            SDVariable b2 = sd.var("b2", Nd4j.zeros(DataType.FLOAT, 10));

            SDVariable hidden = sd.nn().relu(input.mmul(w1).add(b1), 0);
            SDVariable output = sd.nn().softmax("output", hidden.mmul(w2).add(b2));
            SDVariable loss = sd.loss().softmaxCrossEntropy("loss", label, output, null);

            // FP16 mixed precision:
            //   - Forward/backward pass computed in FP16 (HALF)
            //   - Master weights maintained in FP32
            //   - Dynamic loss scaling to prevent gradient underflow
            TrainingConfig fp16Config = TrainingConfig.builder()
                    .updater(new Adam(1e-3))
                    .dataSetFeatureMapping("input")
                    .dataSetLabelMapping("label")
                    .mixedPrecision()  // convenience: FLOAT16 compute + FP32 master + dynamic loss scaling
                    .build();

            sd.setTrainingConfig(fp16Config);

            System.out.println("  Compute dtype:      " + fp16Config.getComputeDataType());
            System.out.println("  Master weight dtype: " + fp16Config.getMasterWeightDataType());
            System.out.println("  Mixed precision:     " + fp16Config.isMixedPrecision());
            System.out.println("  Loss scaling:        " + fp16Config.isLossScalingEnabled());
        }

        // ============================================================
        // 2. BF16 MIXED PRECISION (no loss scaling needed)
        // ============================================================
        System.out.println("\n=== BF16 Mixed Precision Training ===");
        {
            SameDiff sd = SameDiff.create();

            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 512);
            SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 10);
            SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 512, 10).mul(0.01));
            SDVariable output = sd.nn().softmax("output", input.mmul(w));
            sd.loss().softmaxCrossEntropy("loss", label, output, null);

            // BF16 mixed precision:
            //   - Same exponent range as FP32 (no underflow issues)
            //   - No loss scaling needed (simpler than FP16)
            //   - Preferred on hardware that supports BF16 (A100+, TPU, Intel AMX)
            TrainingConfig bf16Config = TrainingConfig.builder()
                    .updater(new Adam(1e-3))
                    .dataSetFeatureMapping("input")
                    .dataSetLabelMapping("label")
                    .mixedPrecisionBfloat16()  // BFLOAT16 compute + FP32 master, no loss scaling
                    .build();

            sd.setTrainingConfig(bf16Config);

            System.out.println("  Compute dtype:      " + bf16Config.getComputeDataType());
            System.out.println("  Master weight dtype: " + bf16Config.getMasterWeightDataType());
            System.out.println("  Loss scaling:        " + bf16Config.isLossScalingEnabled() + " (not needed for BF16)");
        }

        // ============================================================
        // 3. MANUAL MIXED PRECISION CONFIGURATION
        // ============================================================
        System.out.println("\n=== Manual Mixed Precision Config ===");
        {
            // Full control over compute dtype, master dtype, and loss scaling
            TrainingConfig manualConfig = TrainingConfig.builder()
                    .updater(new Adam(1e-4))
                    .dataSetFeatureMapping("input")
                    .dataSetLabelMapping("label")
                    .computeDataType(DataType.HALF)                    // FP16 forward/backward
                    .masterWeightDataType(DataType.FLOAT)              // FP32 master weights
                    .lossScaling(LossScaleConfig.dynamicScaling(8192)) // custom initial scale
                    .gradientAccumulationSteps(4)                      // accumulate 4 mini-batches
                    .build();

            System.out.println("  Compute dtype:        " + manualConfig.getComputeDataType());
            System.out.println("  Master weight dtype:  " + manualConfig.getMasterWeightDataType());
            System.out.println("  Loss scaling enabled: " + manualConfig.isLossScalingEnabled());
            System.out.println("  Grad accumulation:    " + manualConfig.isGradientAccumulationEnabled()
                    + " (" + manualConfig.getGradientAccumulationSteps() + " steps)");
        }

        // ============================================================
        // 4. LOSS SCALE CONFIGURATION
        // ============================================================
        System.out.println("\n=== Loss Scale Configuration ===");
        {
            // Static loss scaling: fixed multiplier
            LossScaleConfig staticConfig = LossScaleConfig.staticScaling(1024.0);
            System.out.println("  Static scaling:");
            System.out.println("    Scale: " + staticConfig.getInitialScale());
            System.out.println("    Dynamic: " + staticConfig.isDynamic());

            // Dynamic loss scaling: auto-adjusts scale
            LossScaleConfig dynamicConfig = LossScaleConfig.dynamicScaling();
            System.out.println("\n  Dynamic scaling (defaults):");
            System.out.println("    Initial scale:   " + dynamicConfig.getInitialScale());
            System.out.println("    Growth factor:   " + dynamicConfig.getGrowthFactor());
            System.out.println("    Backoff factor:  " + dynamicConfig.getBackoffFactor());
            System.out.println("    Growth interval: " + dynamicConfig.getGrowthInterval());
            System.out.println("    Min scale:       " + dynamicConfig.getMinScale());
            System.out.println("    Max scale:       " + dynamicConfig.getMaxScale());

            // Dynamic with custom initial scale
            LossScaleConfig customDynamic = LossScaleConfig.dynamicScaling(8192.0);
            System.out.println("\n  Dynamic scaling (custom):");
            System.out.println("    Initial scale: " + customDynamic.getInitialScale());
        }

        // ============================================================
        // 5. LOSS SCALER - Runtime API
        // ============================================================
        System.out.println("\n=== LossScaler Runtime API ===");
        {
            LossScaleConfig config = LossScaleConfig.dynamicScaling();
            LossScaler scaler = new LossScaler(config);

            System.out.println("  Initial scale: " + scaler.getCurrentScale());

            // Scale a loss value
            double rawLoss = 0.5;
            double scaledLoss = scaler.scaleLoss(rawLoss);
            System.out.println("  Raw loss:    " + rawLoss);
            System.out.println("  Scaled loss: " + scaledLoss);

            // Scale an INDArray loss
            INDArray lossArr = Nd4j.scalar(0.5);
            INDArray scaledArr = scaler.scaleLoss(lossArr);
            System.out.println("  Scaled array: " + scaledArr);

            // Unscale gradients and check for overflow
            INDArray gradients = Nd4j.randn(DataType.FLOAT, 100);
            boolean finite = scaler.unscaleGradientsAndCheck(gradients);
            System.out.println("  Gradients finite after unscale: " + finite);

            // Update scale based on gradient health
            scaler.update(finite);
            System.out.println("  Scale after update: " + scaler.getCurrentScale());

            // Simulate overflow (gradients not finite)
            scaler.update(false);
            System.out.println("  Scale after overflow: " + scaler.getCurrentScale() + " (halved)");

            scaler.reset();
            System.out.println("  Scale after reset: " + scaler.getCurrentScale());
        }

        // ============================================================
        // 6. GRADIENT ACCUMULATOR
        // ============================================================
        System.out.println("\n=== Gradient Accumulation ===");
        {
            // Accumulate gradients over 4 steps before applying update
            // Effective batch size = micro_batch * accumulation_steps
            GradientAccumulator accumulator = new GradientAccumulator(4);

            System.out.println("  Accumulation steps: 4");
            System.out.println("  Enabled: " + accumulator.isEnabled());

            // Simulate 4 micro-batch gradient steps
            for (int step = 0; step < 4; step++) {
                INDArray microGrad = Nd4j.randn(DataType.FLOAT, 100).mul(0.1);
                accumulator.accumulate("w1", microGrad);
                accumulator.step();
                System.out.println("  Step " + (step + 1) + ": ready=" + accumulator.isReady());
            }

            // After 4 steps, get averaged gradients
            if (accumulator.isReady()) {
                java.util.Map<String, INDArray> avgGrads = accumulator.getAndReset();
                System.out.println("  Accumulated gradient mean: " + avgGrads.get("w1").meanNumber());
                System.out.println("  Accumulator reset, ready: " + accumulator.isReady());
            }
        }

        // ============================================================
        // 7. FP8 TRAINING CONFIGURATION
        // ============================================================
        System.out.println("\n=== FP8 Training Configuration ===");
        {
            FP8TrainingConfig fp8Config = FP8TrainingConfig.builder()
                    .useE4M3ForForward(true)     // E4M3 for forward (higher precision, range [-448, 448])
                    .perTensorScaling(true)       // per-tensor scaling (vs per-layer)
                    .amaxHistoryLength(16)        // rolling window for amax tracking
                    .build();

            System.out.println("  E4M3 for forward: " + fp8Config.isUseE4M3ForForward());
            System.out.println("  Per-tensor scaling: " + fp8Config.isPerTensorScaling());
            System.out.println("  Amax history: " + fp8Config.getAmaxHistoryLength());

            // Check if operations are eligible for FP8
            System.out.println("\n  FP8-eligible ops:");
            for (String op : new String[]{"matmul", "linear", "dense", "layernorm", "softmax", "attention"}) {
                System.out.println("    " + op + ": " + fp8Config.isEligibleForFP8(op));
            }

            System.out.println("\n  Note: layernorm, rmsnorm, softmax, attention, gelu, silu");
            System.out.println("  are excluded from FP8 by default (need higher precision)");
        }

        // ============================================================
        // 8. SFT CONFIG - Fine-tuning with mixed precision defaults
        // ============================================================
        System.out.println("\n=== SFTConfig Mixed Precision Defaults ===");
        {
            // Default SFT config: BF16 compute, 4-step gradient accumulation
            SFTConfig defaultSft = SFTConfig.defaultSFT();
            System.out.println("  Default SFT:");
            System.out.println("    Compute dtype: " + defaultSft.getComputeDataType());
            System.out.println("    Grad accum steps: " + defaultSft.getGradientAccumulationSteps());

            // LoRA defaults
            SFTConfig loraSft = SFTConfig.loraDefaults(16);
            System.out.println("\n  LoRA SFT (rank=16):");
            System.out.println("    Compute dtype: " + loraSft.getComputeDataType());
            System.out.println("    Grad accum steps: " + loraSft.getGradientAccumulationSteps());

            // QLoRA defaults
            SFTConfig qloraSft = SFTConfig.qloraDefaults();
            System.out.println("\n  QLoRA SFT:");
            System.out.println("    Compute dtype: " + qloraSft.getComputeDataType());
            System.out.println("    Grad accum steps: " + qloraSft.getGradientAccumulationSteps());
        }

        // ============================================================
        // 9. COMPLETE EXAMPLE: BF16 + Gradient Accumulation
        // ============================================================
        System.out.println("\n=== Complete: BF16 + Gradient Accumulation ===");
        {
            SameDiff sd = SameDiff.create();

            // Build a simple MLP
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 100);
            SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 10);

            SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, 100, 64).mul(0.02));
            SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, 64, 10).mul(0.02));

            SDVariable h = sd.nn().relu(input.mmul(w1), 0);
            SDVariable out = sd.nn().softmax("output", h.mmul(w2));
            sd.loss().softmaxCrossEntropy("loss", label, out, null);

            // BF16 mixed precision + gradient accumulation
            // Effective batch = 32 (micro) * 8 (accum) = 256
            TrainingConfig config = TrainingConfig.builder()
                    .updater(new Adam(3e-4))
                    .dataSetFeatureMapping("input")
                    .dataSetLabelMapping("label")
                    .mixedPrecisionBfloat16()
                    .gradientAccumulationSteps(8)
                    .build();

            sd.setTrainingConfig(config);

            System.out.println("  Model configured:");
            System.out.println("    Compute: BF16");
            System.out.println("    Master weights: FP32");
            System.out.println("    Gradient accumulation: 8 steps");
            System.out.println("    Effective batch size multiplier: 8x");
            System.out.println("    Loss scaling: " + (config.isLossScalingEnabled() ? "enabled" : "not needed (BF16)"));
        }

        // ============================================================
        // SUMMARY TABLE
        // ============================================================
        System.out.println("\n=== Mixed Precision Strategy Guide ===");
        System.out.println("  +-----------+------------------+---------------+------------------+");
        System.out.println("  | Precision | Loss Scaling     | Memory Saving | Hardware         |");
        System.out.println("  +-----------+------------------+---------------+------------------+");
        System.out.println("  | FP16      | Required         | ~50%          | V100, A100, RTX  |");
        System.out.println("  | BF16      | Not needed       | ~50%          | A100+, TPU, AMX  |");
        System.out.println("  | FP8       | Per-tensor scale | ~75%          | H100, H200       |");
        System.out.println("  +-----------+------------------+---------------+------------------+");
        System.out.println("  Gradient accumulation: trades compute time for memory");
        System.out.println("  Use accum_steps=N to simulate Nx larger batch size");

        System.out.println("\nAll mixed precision training examples demonstrated successfully.");
    }
}
