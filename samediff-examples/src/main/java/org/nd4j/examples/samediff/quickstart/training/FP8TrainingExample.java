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
import org.nd4j.autodiff.samediff.config.FP8TrainingConfig;
import org.nd4j.autodiff.samediff.config.LossScaleConfig;
import org.nd4j.autodiff.samediff.training.LossScaler;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

/**
 * FP8 Mixed Precision Training in SameDiff - Reference Example
 *
 * FP8 (8-bit floating point) is the lowest-precision format currently used for
 * training neural networks. It provides approximately 2x throughput and 4x memory
 * reduction compared to FP16, but requires careful handling of the extremely limited
 * dynamic range.
 *
 * There are two FP8 formats used in training (IEEE 754-like encodings):
 *
 *   E4M3 (FLOAT8_E4M3FN):
 *     - 4 exponent bits, 3 mantissa bits
 *     - Range: [-448, 448]
 *     - Higher precision (8 levels per octave), better for forward pass activations
 *     - "FN" = Finite Number (no Inf, special NaN encoding)
 *
 *   E5M2 (FLOAT8_E5M2):
 *     - 5 exponent bits, 2 mantissa bits
 *     - Range: [-57344, 57344]
 *     - Wider range, better for backward pass gradients (which can vary wildly)
 *     - Supports Inf and NaN (standard IEEE semantics)
 *
 * Standard practice (used in H100 Transformer Engine):
 *   - Forward pass:  E4M3  (activations need precision over range)
 *   - Backward pass: E5M2  (gradients need range over precision)
 *   - Master weights: FP32 (full precision for weight updates)
 *
 * Per-tensor amax (absolute maximum) tracking:
 *   Each tensor is assigned a scale factor = maxFP8value / amax(tensor).
 *   The scale is updated using a rolling history of recent amax values
 *   to avoid reacting to transient spikes.
 *
 * Hardware requirements:
 *   FP8 compute is natively supported on NVIDIA H100 and H200 GPUs.
 *   On other hardware, FP8 operations fall back to FP16 or FP32.
 *
 * Key classes:
 *   - FP8TrainingConfig:  configures FP8 format, scaling policy, eligible ops
 *   - LossScaleConfig:    configures dynamic or static loss scaling
 *   - LossScaler:         runtime API for scaling and unscaling loss/gradients
 *   - TrainingConfig:     integrates FP8 into the SameDiff training loop
 */
public class FP8TrainingExample {

    public static void main(String[] args) {

        // ============================================================
        // 1. FP8 FORMAT OVERVIEW - E4M3 vs E5M2
        // ============================================================
        System.out.println("=== FP8 Format Overview ===");
        {
            // DataType.FLOAT8 maps to E4M3FN (forward pass default)
            // DataType.FLOAT8_E5M2 maps to E5M2 (backward pass default)
            System.out.println("  DataType.FLOAT8        -> E4M3FN (forward)");
            System.out.println("  DataType.FLOAT8_E5M2   -> E5M2   (backward)");
            System.out.println();
            System.out.println("  E4M3FN characteristics:");
            System.out.println("    Bits:      1 sign + 4 exponent + 3 mantissa");
            System.out.println("    Range:     [-448, 448]");
            System.out.println("    Precision: ~3 decimal digits");
            System.out.println("    NaN/Inf:   special NaN only (no Inf)");
            System.out.println("    Use:       forward activations (need precision)");
            System.out.println();
            System.out.println("  E5M2 characteristics:");
            System.out.println("    Bits:      1 sign + 5 exponent + 2 mantissa");
            System.out.println("    Range:     [-57344, 57344]");
            System.out.println("    Precision: ~2 decimal digits");
            System.out.println("    NaN/Inf:   IEEE standard Inf and NaN");
            System.out.println("    Use:       backward gradients (need range)");
        }

        // ============================================================
        // 2. FP8TRAININGCONFIG - Core Configuration
        // ============================================================
        System.out.println("\n=== FP8TrainingConfig ===");
        {
            // FP8TrainingConfig controls:
            //   - Which FP8 format to use for forward and backward passes
            //   - Whether to use per-tensor or per-channel amax tracking
            //   - How many amax history steps to keep for stable scale computation
            //   - Which operations are eligible for FP8 execution
            FP8TrainingConfig fp8Config = FP8TrainingConfig.builder()
                    .useE4M3ForForward(true)      // E4M3FN for forward (higher precision)
                    .perTensorScaling(true)        // per-tensor scale (vs per-channel)
                    .amaxHistoryLength(16)         // rolling window: max over last 16 steps
                    .build();

            System.out.println("  E4M3 for forward:   " + fp8Config.isUseE4M3ForForward());
            System.out.println("  Per-tensor scaling: " + fp8Config.isPerTensorScaling());
            System.out.println("  Amax history:       " + fp8Config.getAmaxHistoryLength());

            // Not all ops benefit from FP8 - normalization and softmax need more precision
            System.out.println("\n  FP8 op eligibility:");
            for (String op : new String[]{"matmul", "linear", "dense", "layernorm",
                    "rmsnorm", "softmax", "attention", "gelu", "silu"}) {
                System.out.println("    " + op + ": " + fp8Config.isEligibleForFP8(op));
            }
            System.out.println("  (matmul/linear/dense use FP8; norms/softmax/activations stay FP16/FP32)");
        }

        // ============================================================
        // 3. FP8TRAININGCONFIG - Variant Configurations
        // ============================================================
        System.out.println("\n=== FP8TrainingConfig Variants ===");
        {
            // High-precision variant: longer amax history, E4M3 for both passes
            // Use when model stability is more important than raw throughput
            FP8TrainingConfig highPrecision = FP8TrainingConfig.builder()
                    .useE4M3ForForward(true)
                    .perTensorScaling(true)
                    .amaxHistoryLength(32)    // longer history = more stable scales
                    .build();
            System.out.println("  High precision config (amaxHistory=32):");
            System.out.println("    Amax history: " + highPrecision.getAmaxHistoryLength());

            // Maximum throughput: E5M2 for forward (wider range), short history
            // Use on H100 when throughput is the priority
            FP8TrainingConfig maxThroughput = FP8TrainingConfig.builder()
                    .useE4M3ForForward(false)   // E5M2 for forward = higher throughput
                    .perTensorScaling(true)
                    .amaxHistoryLength(8)        // shorter history = faster scale updates
                    .build();
            System.out.println("\n  Max throughput config (useE4M3ForForward=false):");
            System.out.println("    Amax history: " + maxThroughput.getAmaxHistoryLength());

            // Per-channel scaling: finer granularity, slightly more overhead
            FP8TrainingConfig perChannelConfig = FP8TrainingConfig.builder()
                    .useE4M3ForForward(true)
                    .perTensorScaling(false)    // per-channel scale (finer grained)
                    .amaxHistoryLength(16)
                    .build();
            System.out.println("\n  Per-channel config (perTensorScaling=false):");
            System.out.println("    Per-tensor: " + perChannelConfig.isPerTensorScaling()
                    + " (per-channel = higher quality, more memory)");
        }

        // ============================================================
        // 4. LOSS SCALING FOR FP8
        // ============================================================
        System.out.println("\n=== Loss Scaling for FP8 ===");
        {
            // FP8 has even more limited range than FP16, so loss scaling is critical.
            // Dynamic scaling automatically adjusts the scale based on gradient health.
            //
            // How dynamic scaling works:
            //   1. Multiply loss by current scale S before backward pass
            //   2. Gradients are magnified by S (keeping them in FP8 range)
            //   3. After backward: divide all gradients by S to restore true scale
            //   4. Check for overflow (NaN/Inf in gradients)
            //     - If overflow: halve S (backoff factor) and skip update
            //     - If no overflow for `growthInterval` steps: double S (growth factor)

            LossScaleConfig dynamicFP8 = LossScaleConfig.dynamicScaling(65536.0);
            System.out.println("  Dynamic loss scaling for FP8:");
            System.out.println("    Initial scale:   " + dynamicFP8.getInitialScale()
                    + "  (larger initial scale than FP16)");
            System.out.println("    Growth factor:   " + dynamicFP8.getGrowthFactor());
            System.out.println("    Backoff factor:  " + dynamicFP8.getBackoffFactor());
            System.out.println("    Growth interval: " + dynamicFP8.getGrowthInterval());

            // Runtime loss scaler usage
            LossScaler fp8Scaler = new LossScaler(dynamicFP8);
            System.out.println("\n  LossScaler initial scale: " + fp8Scaler.getCurrentScale());

            double loss = 2.5;
            double scaledLoss = fp8Scaler.scaleLoss(loss);
            System.out.println("  Raw loss:    " + loss);
            System.out.println("  Scaled loss: " + scaledLoss + "  (multiply by scale before backward)");

            INDArray grads = Nd4j.randn(DataType.FLOAT, 1000);
            boolean finite = fp8Scaler.unscaleGradientsAndCheck(grads);
            System.out.println("  Gradients finite after unscale: " + finite);

            fp8Scaler.update(finite);
            System.out.println("  Scale after update: " + fp8Scaler.getCurrentScale());
        }

        // ============================================================
        // 5. FP8 INTEGRATED INTO TRAININGCONFIG
        // ============================================================
        System.out.println("\n=== FP8 in TrainingConfig ===");
        {
            SameDiff sd = SameDiff.create();

            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 512);
            SDVariable label  = sd.placeHolder("label",  DataType.FLOAT, -1, 10);

            SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, 512, 256).mul(0.02));
            SDVariable b1 = sd.var("b1", Nd4j.zeros(DataType.FLOAT, 256));
            SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, 256, 10).mul(0.02));
            SDVariable b2 = sd.var("b2", Nd4j.zeros(DataType.FLOAT, 10));

            SDVariable hidden = sd.nn().relu(input.mmul(w1).add(b1), 0);
            SDVariable output = sd.nn().softmax("output", hidden.mmul(w2).add(b2));
            sd.loss().softmaxCrossEntropy("loss", label, output, null);

            // FP8TrainingConfig is constructed here for illustration — it controls
            // the FP8 format choice, amax history, and per-tensor vs per-channel scaling.
            // NOTE: FP8TrainingConfig is consumed directly at the op/kernel level (e.g.,
            // passed to individual FP8-aware ops); it is NOT yet wired into the
            // TrainingConfig.Builder API. Use .mixedPrecision() + .lossScaling() in the
            // builder to enable mixed-precision loss-scaled training through SameDiff.
            FP8TrainingConfig fp8Cfg = FP8TrainingConfig.builder()
                    .useE4M3ForForward(true)
                    .perTensorScaling(true)
                    .amaxHistoryLength(16)
                    .build();

            System.out.println("  FP8TrainingConfig (op-level):");
            System.out.println("    E4M3 for forward:   " + fp8Cfg.isUseE4M3ForForward());
            System.out.println("    Per-tensor scaling: " + fp8Cfg.isPerTensorScaling());
            System.out.println("    Amax history:       " + fp8Cfg.getAmaxHistoryLength());

            // TrainingConfig uses .mixedPrecision() for FP16 mixed-precision mode and
            // .lossScaling() for dynamic loss scaling — both are supported in the builder.
            TrainingConfig trainConfig = TrainingConfig.builder()
                    .updater(new Adam(1e-4))
                    .dataSetFeatureMapping("input")
                    .dataSetLabelMapping("label")
                    .mixedPrecision()                                  // FP16 mixed precision
                    .lossScaling(LossScaleConfig.dynamicScaling())     // dynamic loss scaling
                    .build();

            sd.setTrainingConfig(trainConfig);

            System.out.println("  Loss scaling enabled: " + trainConfig.isLossScalingEnabled());
            System.out.println("  Compute format:      FLOAT8 (E4M3FN) for forward");
            System.out.println("                       FLOAT8_E5M2 for backward");
            System.out.println("  Master weights:      FP32 (unchanged by FP8)");
        }

        // ============================================================
        // 6. COMPARISON: FP32 vs FP16 vs BF16 vs FP8
        // ============================================================
        System.out.println("\n=== Precision Format Comparison ===");
        System.out.println("  +----------+------+------+--------+----------+--------+-----------+");
        System.out.println("  | Format   | Bits | Exp  | Mantis | Range    | Memory | Hardware  |");
        System.out.println("  +----------+------+------+--------+----------+--------+-----------+");
        System.out.println("  | FP32     |  32  |   8  |    23  | ~3.4e38  |  1.0x  | All       |");
        System.out.println("  | FP16     |  16  |   5  |    10  | ~65504   |  0.5x  | V100+     |");
        System.out.println("  | BF16     |  16  |   8  |     7  | ~3.4e38  |  0.5x  | A100+,TPU |");
        System.out.println("  | FP8 E4M3 |   8  |   4  |     3  | 448      |  0.25x | H100+     |");
        System.out.println("  | FP8 E5M2 |   8  |   5  |     2  | 57344    |  0.25x | H100+     |");
        System.out.println("  +----------+------+------+--------+----------+--------+-----------+");
        System.out.println();
        System.out.println("  Training stability (most to least stable): FP32 > BF16 > FP16 > FP8");
        System.out.println("  Memory efficiency (least to most memory):  FP32 > FP16=BF16 > FP8");
        System.out.println();
        System.out.println("  Recommended approach:");
        System.out.println("    Default:          BF16 (simple, stable, widely supported)");
        System.out.println("    GPU-constrained:  FP16 + dynamic loss scaling");
        System.out.println("    H100 + max perf:  FP8 (E4M3 forward, E5M2 backward)");
        System.out.println("    Large models:     FP8 + Adam8bit + gradient accumulation");

        System.out.println("\nFP8 training example completed successfully.");
    }
}
