/* *****************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
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
import org.nd4j.autodiff.samediff.config.ContinuedPretrainingConfig;
import org.nd4j.autodiff.samediff.config.FP8TrainingConfig;
import org.nd4j.autodiff.samediff.config.GradientCheckpointConfig;
import org.nd4j.autodiff.samediff.config.LossScaleConfig;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.training.ContinuedPretrainingWorkflow;
import org.nd4j.autodiff.samediff.training.GradientAccumulator;
import org.nd4j.autodiff.samediff.training.LossScaler;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Adam8bit;
import org.nd4j.linalg.schedule.CosineWarmupSchedule;
import org.nd4j.linalg.schedule.CycleSchedule;
import org.nd4j.linalg.schedule.ExponentialSchedule;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.PolySchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Continued Pre-training for Domain Adaptation - Complete Workflow Example
 *
 * Continued pre-training adapts a general-purpose LLM to a specific domain (medical,
 * legal, code, finance) by training on all tokens in the sequence without masking.
 * This distinguishes it from instruction tuning, which trains only on response tokens.
 *
 * Topics covered:
 *   1. Language model architecture with SameDiff
 *   2. ContinuedPretrainingConfig: all fields including LoRA, BF16, gradient accumulation
 *   3. ContinuedPretrainingWorkflow: buildTrainingConfig and field inspection
 *   4. Gradient checkpointing strategies: sqrtN, everyN, manual, asyncOffload
 *   5. GradientAccumulator: full 12-step accumulation cycle with real tensors
 *   6. LossScaler: dynamic scaling over 20 steps with inf injection
 *   7. Mixed precision training: BFLOAT16 TrainingConfig with computeDataType
 *   8. Adam8bit optimizer: stateSize calculation and block quantization
 *   9. FP8TrainingConfig: op eligibility screening for 10+ op types
 *  10. LR schedule comparison: CosineWarmup, Exponential, Poly, Step, Cycle
 *  11. Complete domain adaptation pipeline wiring all components together
 *  12. DSP-accelerated continued pre-training: compile full training graph, warmup vs steady-state
 *
 * Key classes:
 *   - ContinuedPretrainingConfig: chunk/overlap/LoRA/BF16/gradient accumulation config
 *   - ContinuedPretrainingWorkflow: orchestrates training lifecycle
 *   - GradientCheckpointConfig: sqrtN / everyN / manual / asyncOffload strategies
 *   - GradientAccumulator: accumulates micro-batch gradients before optimizer step
 *   - LossScaler: dynamic FP16 loss scaling with overflow detection
 *   - Adam8bit: 8-bit Adam with block quantization, ~4x optimizer state savings
 *   - FP8TrainingConfig: op eligibility for FP8 E4M3/E5M2 compute
 */
public class ContinuedPretrainingExample {

    private static final Logger log = LoggerFactory.getLogger(ContinuedPretrainingExample.class);

    public static void main(String[] args) {

        System.out.println("=============================================================");
        System.out.println(" Continued Pre-training for Domain Adaptation");
        System.out.println("=============================================================\n");

        section1_buildLanguageModel();
        section2_continuedPretrainingConfig();
        section3_continuedPretrainingWorkflow();
        section4_gradientCheckpointingStrategies();
        section5_gradientAccumulatorDeepDive();
        section6_lossScalerDeepDive();
        section7_mixedPrecisionTraining();
        section8_adam8bitOptimizer();
        section9_fp8TrainingConfig();
        section10_lrScheduleComparison();
        section11_completeDomainAdaptationPipeline();
        section12_dspAcceleratedContinuedPretraining();

        System.out.println("\n=============================================================");
        System.out.println(" All sections complete.");
        System.out.println("=============================================================");
    }

    // -------------------------------------------------------------------------
    // Section 1: Build Language Model
    // -------------------------------------------------------------------------
    static SameDiff section1_buildLanguageModel() {
        System.out.println("--- Section 1: Build Language Model ---");

        SameDiff sd = SameDiff.create();

        // Input placeholder: batch x sequence x embedding (128)
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 128);

        // Hidden layer 1: 128 -> 256
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, 128, 256).muli(0.02));
        SDVariable b1 = sd.var("b1", Nd4j.zeros(DataType.FLOAT, 256));
        SDVariable hidden1 = sd.nn.relu(sd.math.add(sd.mmul(input, w1), b1), 0.0);

        // Hidden layer 2: 256 -> 128
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, 256, 128).muli(0.02));
        SDVariable b2 = sd.var("b2", Nd4j.zeros(DataType.FLOAT, 128));
        SDVariable hidden2 = sd.nn.relu(sd.math.add(sd.mmul(hidden1, w2), b2), 0.0);

        // Output projection: 128 -> 64 (logits over vocabulary subset)
        SDVariable wOut = sd.var("w_out", Nd4j.randn(DataType.FLOAT, 128, 64).muli(0.02));
        SDVariable bOut = sd.var("b_out", Nd4j.zeros(DataType.FLOAT, 64));
        SDVariable logits = sd.math.add(sd.mmul(hidden2, wOut), bOut);
        logits.rename("logits");

        // Loss: labels placeholder + softmax cross-entropy
        SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, -1, 64);
        SDVariable loss = sd.loss.softmaxCrossEntropy("loss", labels, logits, null);

        long numVars = sd.variables().size();
        System.out.printf("  Model variables: %d (input, w1, b1, w2, b2, w_out, b_out, labels, logits, loss)%n", numVars);
        System.out.printf("  Architecture:  input(128) -> hidden1(256) -> hidden2(128) -> output(64)%n");
        System.out.printf("  Trainable params: w1[128x256]=%d, w2[256x128]=%d, w_out[128x64]=%d%n",
                128 * 256, 256 * 128, 128 * 64);
        System.out.printf("  Total param count: %d%n", 128 * 256 + 256 + 256 * 128 + 128 + 128 * 64 + 64);
        System.out.println();

        return sd;
    }

    // -------------------------------------------------------------------------
    // Section 2: ContinuedPretrainingConfig
    // -------------------------------------------------------------------------
    static ContinuedPretrainingConfig section2_continuedPretrainingConfig() {
        System.out.println("--- Section 2: ContinuedPretrainingConfig ---");

        ContinuedPretrainingConfig config = ContinuedPretrainingConfig.builder()
                .chunkSize(2048)
                .chunkOverlap(128)
                .useLoRA(true)
                .loraRank(16)
                .loraAlpha(32)
                .learningRate(5e-5)
                .warmupRatio(0.1)
                .computeDataType(DataType.BFLOAT16)
                .gradientAccumulationSteps(4)
                .weightDecay(0.01)
                .maxGradNorm(1.0)
                .numEpochs(1)
                .build();

        config.validate();

        System.out.println("  ContinuedPretrainingConfig fields:");
        System.out.printf("    chunkSize              = %d%n", config.getChunkSize());
        System.out.printf("    chunkOverlap           = %d%n", config.getChunkOverlap());
        System.out.printf("    useLoRA                = %b%n", config.isUseLoRA());
        System.out.printf("    loraRank               = %d%n", config.getLoraRank());
        System.out.printf("    loraAlpha              = %d%n", config.getLoraAlpha());
        System.out.printf("    loraScaling            = %.4f  (alpha/rank = %d/%d)%n",
                (double) config.getLoraAlpha() / config.getLoraRank(),
                config.getLoraAlpha(), config.getLoraRank());
        System.out.printf("    learningRate           = %.2e%n", config.getLearningRate());
        System.out.printf("    warmupRatio            = %.2f%n", config.getWarmupRatio());
        System.out.printf("    minLearningRate        = %.2e%n", config.getMinLearningRate());
        System.out.printf("    computeDataType        = %s%n", config.getComputeDataType());
        System.out.printf("    gradientAccumSteps     = %d%n", config.getGradientAccumulationSteps());
        System.out.printf("    weightDecay            = %.4f%n", config.getWeightDecay());
        System.out.printf("    maxGradNorm            = %.2f%n", config.getMaxGradNorm());
        System.out.printf("    numEpochs              = %d%n", config.getNumEpochs());
        System.out.printf("    lrSchedule             = %s%n", config.getLrSchedule() == null ? "null (uses CosineWarmup)" : config.getLrSchedule());
        System.out.printf("  Validation: PASSED%n");
        System.out.println();

        return config;
    }

    // -------------------------------------------------------------------------
    // Section 3: ContinuedPretrainingWorkflow
    // -------------------------------------------------------------------------
    static void section3_continuedPretrainingWorkflow() {
        System.out.println("--- Section 3: ContinuedPretrainingWorkflow ---");

        SameDiff model = section1_buildLanguageModel();
        ContinuedPretrainingConfig config = section2_continuedPretrainingConfig();

        // Create workflow - validates config on construction
        ContinuedPretrainingWorkflow workflow = new ContinuedPretrainingWorkflow(model, config);

        // Build TrainingConfig for 1000 total steps
        int totalSteps = 1000;
        TrainingConfig trainingConfig = workflow.buildTrainingConfig(totalSteps);

        System.out.printf("  Workflow built TrainingConfig for %d total steps:%n", totalSteps);
        System.out.printf("    updater class          = %s%n", trainingConfig.getUpdater().getClass().getSimpleName());
        System.out.printf("    computeDataType        = %s%n", trainingConfig.getComputeDataType());
        System.out.printf("    masterWeightDataType   = %s%n", trainingConfig.getMasterWeightDataType());
        System.out.printf("    isMixedPrecision       = %b%n", trainingConfig.isMixedPrecision());
        System.out.printf("    gradientAccumSteps     = %d%n", trainingConfig.getGradientAccumulationSteps());
        System.out.printf("    isGradAccumEnabled     = %b%n", trainingConfig.isGradientAccumulationEnabled());
        System.out.printf("    regularization size    = %d%n", trainingConfig.getRegularization().size());
        System.out.printf("    featureMapping         = %s%n", trainingConfig.getDataSetFeatureMapping());
        System.out.printf("    labelMapping           = %s%n", trainingConfig.getDataSetLabelMapping());

        // Confirm the optimizer is Adam with a schedule
        Adam adam = (Adam) trainingConfig.getUpdater();
        System.out.printf("    Adam.beta1             = %.3f%n", adam.getBeta1());
        System.out.printf("    Adam.beta2             = %.3f%n", adam.getBeta2());
        System.out.printf("    Adam.epsilon           = %.2e%n", adam.getEpsilon());
        System.out.printf("    Adam has schedule      = %b%n", adam.getLearningRateSchedule() != null);

        // Evaluate the built schedule at warmup boundary (10% of 1000 = 100 steps)
        ISchedule schedule = adam.getLearningRateSchedule();
        System.out.printf("    LR at step   0         = %.6f%n", schedule.valueAt(0, 0));
        System.out.printf("    LR at step  50         = %.6f%n", schedule.valueAt(50, 0));
        System.out.printf("    LR at step 100 (peak)  = %.6f%n", schedule.valueAt(100, 0));
        System.out.printf("    LR at step 500 (mid)   = %.6f%n", schedule.valueAt(500, 0));
        System.out.printf("    LR at step 999 (end)   = %.6f%n", schedule.valueAt(999, 0));
        System.out.println();
    }

    // -------------------------------------------------------------------------
    // Section 4: Gradient Checkpointing Strategies
    // -------------------------------------------------------------------------
    static void section4_gradientCheckpointingStrategies() {
        System.out.println("--- Section 4: Gradient Checkpointing Strategies ---");

        int numLayers = 24;

        // Strategy 1: sqrt(N)
        GradientCheckpointConfig sqrtNConfig = GradientCheckpointConfig.sqrtN();
        int sqrtInterval = sqrtNConfig.resolveInterval(numLayers);
        System.out.printf("  sqrtN strategy:%n");
        System.out.printf("    checkpointEveryN       = %d (sentinel -1 = compute at runtime)%n", sqrtNConfig.getCheckpointEveryN());
        System.out.printf("    isSqrtN                = %b%n", sqrtNConfig.isSqrtN());
        System.out.printf("    resolveInterval(%d)    = %d  [sqrt(24) ≈ 4.9 -> 4]%n", numLayers, sqrtInterval);
        System.out.printf("    checkpoints kept       = %d of %d layers%n", numLayers / sqrtInterval, numLayers);

        // Strategy 2: everyN(4)
        GradientCheckpointConfig everyNConfig = GradientCheckpointConfig.everyN(4);
        int everyInterval = everyNConfig.resolveInterval(numLayers);
        System.out.printf("  everyN(4) strategy:%n");
        System.out.printf("    checkpointEveryN       = %d%n", everyNConfig.getCheckpointEveryN());
        System.out.printf("    isSqrtN                = %b%n", everyNConfig.isSqrtN());
        System.out.printf("    resolveInterval(%d)    = %d%n", numLayers, everyInterval);
        System.out.printf("    checkpoints kept       = %d of %d layers%n", numLayers / everyInterval, numLayers);

        // Strategy 3: manual set of named variables
        Set<String> checkpointVars = new HashSet<>(Arrays.asList(
                "layer_0_out", "layer_6_out", "layer_12_out", "layer_18_out", "layer_23_out"));
        GradientCheckpointConfig manualConfig = GradientCheckpointConfig.manual(checkpointVars);
        System.out.printf("  manual strategy:%n");
        System.out.printf("    isManual               = %b%n", manualConfig.isManual());
        System.out.printf("    checkpointVariables    = %s%n", manualConfig.getCheckpointVariables());
        System.out.printf("    resolveInterval(%d)    = %d  (ignored for manual)%n",
                numLayers, manualConfig.resolveInterval(numLayers));

        // Strategy 4: asyncOffload
        GradientCheckpointConfig asyncConfig = GradientCheckpointConfig.asyncOffload();
        System.out.printf("  asyncOffload strategy:%n");
        System.out.printf("    strategy               = %s%n", asyncConfig.getStrategy());
        System.out.printf("    isAsyncOffload         = %b%n", asyncConfig.isAsyncOffload());
        System.out.printf("    isAnyOffload           = %b%n", asyncConfig.isAnyOffload());
        System.out.printf("    pinHostMemory          = %b%n", asyncConfig.isPinHostMemory());
        System.out.printf("    prefetchDistance       = %d%n", asyncConfig.getPrefetchDistance());
        System.out.printf("    checkpointEveryN       = %d%n", asyncConfig.getCheckpointEveryN());
        System.out.printf("    resolveInterval(%d)    = %d%n", numLayers, asyncConfig.resolveInterval(numLayers));
        System.out.println();
    }

    // -------------------------------------------------------------------------
    // Section 5: GradientAccumulator Deep Dive
    // -------------------------------------------------------------------------
    static void section5_gradientAccumulatorDeepDive() {
        System.out.println("--- Section 5: GradientAccumulator Deep Dive ---");

        int accumulationSteps = 4;
        int totalMicroBatches = 12;  // three full accumulation cycles
        GradientAccumulator accumulator = new GradientAccumulator(accumulationSteps);

        System.out.printf("  accumulationSteps      = %d%n", accumulator.getAccumulationSteps());
        System.out.printf("  isEnabled              = %b%n", accumulator.isEnabled());
        System.out.printf("  Simulating %d micro-batches (%d cycles of %d steps each):%n",
                totalMicroBatches, totalMicroBatches / accumulationSteps, accumulationSteps);

        int updateCount = 0;
        for (int microBatch = 0; microBatch < totalMicroBatches; microBatch++) {
            // Simulate gradients for two parameters
            INDArray gradW = Nd4j.randn(DataType.FLOAT, 64, 32).muli(0.01);
            INDArray gradB = Nd4j.randn(DataType.FLOAT, 32).muli(0.001);

            accumulator.accumulate("w_out", gradW);
            accumulator.accumulate("b_out", gradB);
            accumulator.step();

            System.out.printf("    micro-batch %2d: step=%d/%d, numVars=%d, isReady=%b",
                    microBatch + 1,
                    accumulator.getCurrentStep(),
                    accumulationSteps,
                    accumulator.getNumVariables(),
                    accumulator.isReady());

            if (accumulator.isReady()) {
                Map<String, INDArray> averaged = accumulator.getAndReset();
                INDArray avgW = averaged.get("w_out");
                INDArray avgB = averaged.get("b_out");

                double wMean = avgW.meanNumber().doubleValue();
                double wMax  = avgW.maxNumber().doubleValue();
                double bMean = avgB.meanNumber().doubleValue();

                updateCount++;
                System.out.printf(" -> OPTIMIZER STEP #%d: w_out mean=%.6f max=%.6f | b_out mean=%.6f%n",
                        updateCount, wMean, wMax, bMean);

                // Close averaged arrays to free memory
                avgW.close();
                avgB.close();

                gradW.close();
                gradB.close();
            } else {
                System.out.println();
                gradW.close();
                gradB.close();
            }
        }

        System.out.printf("  Total optimizer updates applied: %d (from %d micro-batches)%n",
                updateCount, totalMicroBatches);
        System.out.printf("  Effective batch size multiplier: %dx%n", accumulationSteps);
        System.out.println();
    }

    // -------------------------------------------------------------------------
    // Section 6: LossScaler Deep Dive
    // -------------------------------------------------------------------------
    static void section6_lossScalerDeepDive() {
        System.out.println("--- Section 6: LossScaler Deep Dive (Dynamic Scaling) ---");

        // Configure dynamic loss scaler: start at 256, grow by 2x every 5 steps,
        // backoff by 0.5x on overflow
        LossScaleConfig scaleConfig = LossScaleConfig.builder()
                .mode(LossScaleConfig.Mode.DYNAMIC)
                .initialScale(256.0)
                .minScale(1.0)
                .maxScale(65536.0)
                .growthFactor(2.0)
                .backoffFactor(0.5)
                .growthInterval(5)
                .build();

        LossScaler scaler = new LossScaler(scaleConfig);

        System.out.printf("  Config: mode=%s, initial=%.0f, min=%.0f, max=%.0f, growthInterval=%d%n",
                scaleConfig.getMode(), scaleConfig.getInitialScale(),
                scaleConfig.getMinScale(), scaleConfig.getMaxScale(),
                scaleConfig.getGrowthInterval());
        System.out.println("  Simulating 20 training steps (inf injected at steps 7 and 14):");
        System.out.printf("  %-5s  %-12s  %-10s  %-8s%n", "Step", "Scale", "Finite", "ConsecOK");

        // Steps where we inject an infinite gradient to trigger backoff
        Set<Integer> infSteps = new HashSet<>(Arrays.asList(7, 14));

        for (int step = 0; step < 20; step++) {
            // Simulate a loss value and scale it
            double rawLoss = 2.5 + 0.1 * step;
            double scaledLoss = scaler.scaleLoss(rawLoss);

            // Simulate gradient — inject +Inf at designated steps
            INDArray gradient;
            boolean finite;
            if (infSteps.contains(step)) {
                gradient = Nd4j.create(new float[]{Float.POSITIVE_INFINITY, 1.0f, 2.0f});
                finite = scaler.unscaleGradientsAndCheck(gradient);
            } else {
                gradient = Nd4j.randn(DataType.FLOAT, 3);
                finite = scaler.unscaleGradientsAndCheck(gradient);
            }

            scaler.update(finite);

            System.out.printf("  %-5d  %-12.1f  %-10b  %-8d%n",
                    step,
                    scaler.getCurrentScale(),
                    finite,
                    scaler.getConsecutiveFiniteCount());

            gradient.close();
        }

        System.out.printf("%n  Final scale: %.1f  (grew from 256 on clear steps, backed off on overflow)%n",
                scaler.getCurrentScale());
        System.out.printf("  scaleLoss(1.0) at final scale = %.1f%n", scaler.scaleLoss(1.0));
        System.out.println();
    }

    // -------------------------------------------------------------------------
    // Section 7: Mixed Precision Training
    // -------------------------------------------------------------------------
    static void section7_mixedPrecisionTraining() {
        System.out.println("--- Section 7: Mixed Precision Training ---");

        // BF16 mixed precision: no loss scaling needed (BF16 has FP32 dynamic range)
        TrainingConfig bf16Config = TrainingConfig.builder()
                .updater(new Adam(5e-5))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                .mixedPrecisionBfloat16()
                .build();

        System.out.println("  BFLOAT16 TrainingConfig:");
        System.out.printf("    computeDataType        = %s%n", bf16Config.getComputeDataType());
        System.out.printf("    masterWeightDataType   = %s%n", bf16Config.getMasterWeightDataType());
        System.out.printf("    isMixedPrecision       = %b%n", bf16Config.isMixedPrecision());
        System.out.printf("    isLossScalingEnabled   = %b  (not needed for BF16)%n",
                bf16Config.isLossScalingEnabled());

        // FP16 mixed precision: requires dynamic loss scaling
        TrainingConfig fp16Config = TrainingConfig.builder()
                .updater(new Adam(5e-5))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                .mixedPrecision()
                .build();

        System.out.println("  FLOAT16 TrainingConfig:");
        System.out.printf("    computeDataType        = %s%n", fp16Config.getComputeDataType());
        System.out.printf("    masterWeightDataType   = %s%n", fp16Config.getMasterWeightDataType());
        System.out.printf("    isMixedPrecision       = %b%n", fp16Config.isMixedPrecision());
        System.out.printf("    isLossScalingEnabled   = %b%n", fp16Config.isLossScalingEnabled());
        System.out.printf("    lossScaleMode          = %s%n",
                fp16Config.getLossScaleConfig().getMode());

        // Custom BF16 with explicit computeDataType
        TrainingConfig customBf16 = TrainingConfig.builder()
                .updater(new Adam(5e-5))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                .computeDataType(DataType.BFLOAT16)
                .masterWeightDataType(DataType.FLOAT)
                .gradientAccumulationSteps(4)
                .build();

        System.out.println("  Custom BF16 with gradient accumulation:");
        System.out.printf("    computeDataType        = %s%n", customBf16.getComputeDataType());
        System.out.printf("    masterWeightDataType   = %s%n", customBf16.getMasterWeightDataType());
        System.out.printf("    gradientAccumSteps     = %d%n", customBf16.getGradientAccumulationSteps());
        System.out.printf("    isGradAccumEnabled     = %b%n", customBf16.isGradientAccumulationEnabled());
        System.out.println();
    }

    // -------------------------------------------------------------------------
    // Section 8: Adam8bit Optimizer
    // -------------------------------------------------------------------------
    static void section8_adam8bitOptimizer() {
        System.out.println("--- Section 8: Adam8bit Optimizer ---");

        // Default Adam8bit
        Adam8bit defaultAdam = new Adam8bit();
        System.out.println("  Adam8bit (defaults):");
        System.out.printf("    learningRate           = %.4f%n", defaultAdam.getLearningRate());
        System.out.printf("    beta1                  = %.3f%n", defaultAdam.getBeta1());
        System.out.printf("    beta2                  = %.3f%n", defaultAdam.getBeta2());
        System.out.printf("    epsilon                = %.2e%n", defaultAdam.getEpsilon());
        System.out.printf("    blockSize              = %d%n", defaultAdam.getBlockSize());
        System.out.printf("    pagedOptimizer         = %b%n", defaultAdam.isPagedOptimizer());

        // Custom Adam8bit with a schedule
        CosineWarmupSchedule schedule = CosineWarmupSchedule.fromRatio(5e-5, 1e-6, 0.1, 1000);
        Adam8bit scheduledAdam = Adam8bit.builder()
                .learningRateSchedule(schedule)
                .beta1(0.9)
                .beta2(0.95)
                .epsilon(1e-8)
                .blockSize(2048)
                .pagedOptimizer(false)
                .build();

        System.out.println("  Adam8bit (with cosine schedule, beta2=0.95):");
        System.out.printf("    hasSchedule            = %b%n", scheduledAdam.getLearningRateSchedule() != null);
        System.out.printf("    beta2                  = %.2f%n", scheduledAdam.getBeta2());
        System.out.printf("    blockSize              = %d%n", scheduledAdam.getBlockSize());

        // Memory savings: stateSize for various parameter counts
        System.out.println("  Memory savings from 8-bit quantization:");
        long[] paramCounts = {1_000_000L, 7_000_000_000L, 70_000_000_000L};
        String[] labels = {"1M params (small layer)", "7B params (LLaMA-7B)", "70B params (LLaMA-70B)"};
        for (int i = 0; i < paramCounts.length; i++) {
            long stateElems = scheduledAdam.stateSize(paramCounts[i]);
            long fp32StateBytes = 2L * paramCounts[i] * 4;   // standard Adam: m + v in FP32
            long int8StateBytes = stateElems;                  // 2 * numParams bytes for INT8 m+v
            System.out.printf("    %-32s stateSize=%,d elems  FP32=%,dMB  INT8=%,dMB  saving=%.1fx%n",
                    labels[i], stateElems,
                    fp32StateBytes / (1024 * 1024),
                    int8StateBytes / (1024 * 1024),
                    (double) fp32StateBytes / int8StateBytes);
        }

        // Block quantization info
        int blockSize = scheduledAdam.getBlockSize();
        long numParams1M = 1_000_000L;
        long numBlocks = (numParams1M + blockSize - 1) / blockSize;
        System.out.printf("  Block quantization for 1M params, blockSize=%d:%n", blockSize);
        System.out.printf("    numBlocks              = %d%n", numBlocks);
        System.out.printf("    scales overhead        = %d floats (%d bytes)%n",
                numBlocks * 2, numBlocks * 2 * 4);
        System.out.println();
    }

    // -------------------------------------------------------------------------
    // Section 9: FP8 Training Config
    // -------------------------------------------------------------------------
    static void section9_fp8TrainingConfig() {
        System.out.println("--- Section 9: FP8 Training Config ---");

        FP8TrainingConfig fp8 = FP8TrainingConfig.builder()
                .useE4M3ForForward(true)
                .perTensorScaling(true)
                .amaxHistoryLength(16)
                .fp8EligibleOps(new HashSet<>(Arrays.asList("matmul", "linear", "dense", "conv2d")))
                .excludeFromFP8(new HashSet<>(Arrays.asList(
                        "layernorm", "rmsnorm", "softmax", "attention", "gelu", "silu", "embedding")))
                .build();

        fp8.validate();

        System.out.printf("  FP8TrainingConfig:%n");
        System.out.printf("    useE4M3ForForward      = %b  (E4M3 for activations, E5M2 for gradients)%n",
                fp8.isUseE4M3ForForward());
        System.out.printf("    perTensorScaling       = %b%n", fp8.isPerTensorScaling());
        System.out.printf("    amaxHistoryLength      = %d%n", fp8.getAmaxHistoryLength());

        // Test eligibility for 10+ op types
        String[] opsToTest = {
                "matmul", "linear", "dense", "conv2d",
                "layernorm", "rmsnorm", "softmax", "attention", "gelu", "silu",
                "embedding", "add", "relu", "mul"
        };

        System.out.println("  FP8 eligibility screening:");
        System.out.printf("  %-16s  %-10s%n", "Op Type", "FP8 Eligible");
        System.out.printf("  %-16s  %-10s%n", "--------", "------------");
        int eligible = 0;
        int ineligible = 0;
        for (String op : opsToTest) {
            boolean isEligible = fp8.isEligibleForFP8(op);
            System.out.printf("  %-16s  %b%n", op, isEligible);
            if (isEligible) eligible++;
            else ineligible++;
        }
        System.out.printf("  Summary: %d eligible, %d ineligible of %d ops tested%n",
                eligible, ineligible, opsToTest.length);
        System.out.println();
    }

    // -------------------------------------------------------------------------
    // Section 10: LR Schedule Comparison
    // -------------------------------------------------------------------------
    static void section10_lrScheduleComparison() {
        System.out.println("--- Section 10: LR Schedule Comparison ---");

        double baseLR = 5e-5;
        int totalSteps = 1000;

        // 1. CosineWarmupSchedule: linear warmup (10%) then cosine decay to 1e-6
        CosineWarmupSchedule cosine = CosineWarmupSchedule.fromRatio(baseLR, 1e-6, 0.1, totalSteps);

        // 2. ExponentialSchedule: LR * gamma^step, gamma=0.9999 (very slow decay)
        ExponentialSchedule exponential = new ExponentialSchedule(ScheduleType.ITERATION, baseLR, 0.9999);

        // 3. PolySchedule: polynomial decay with power=-0.5 (sqrt decay)
        PolySchedule poly = new PolySchedule(ScheduleType.ITERATION, baseLR, -0.5, totalSteps);

        // 4. StepSchedule: halve LR every 250 steps
        StepSchedule step = new StepSchedule(ScheduleType.ITERATION, baseLR, 0.5, 250.0);

        // 5. CycleSchedule: 1-cycle policy over 1000 steps
        CycleSchedule cycle = new CycleSchedule(ScheduleType.ITERATION, baseLR, totalSteps);

        // Evaluate each schedule at 10 evenly-spaced steps
        int[] evalSteps = {0, 50, 100, 200, 300, 400, 500, 600, 750, 999};

        System.out.printf("  %-6s  %-12s  %-12s  %-12s  %-12s  %-12s%n",
                "Step", "Cosine", "Exponential", "Poly", "Step", "Cycle");
        System.out.printf("  %-6s  %-12s  %-12s  %-12s  %-12s  %-12s%n",
                "------", "------------", "------------", "------------", "------------", "------------");

        for (int s : evalSteps) {
            System.out.printf("  %-6d  %-12.6f  %-12.6f  %-12.6f  %-12.6f  %-12.6f%n",
                    s,
                    cosine.valueAt(s, 0),
                    exponential.valueAt(s, 0),
                    poly.valueAt(s, 0),
                    step.valueAt(s, 0),
                    cycle.valueAt(s, 0));
        }

        System.out.println("  Notes:");
        System.out.println("    Cosine:  linear warmup 0->100, cosine decay 100->999");
        System.out.println("    Expo:    slow decay, 5e-5 * 0.9999^step");
        System.out.println("    Poly:    fast decay via polynomial, reaches 0 at maxIter");
        System.out.println("    Step:    halves at 250, 500, 750, 1000");
        System.out.println("    Cycle:   1-cycle with warmup ramp, decay, then annealing");
        System.out.println();
    }

    // -------------------------------------------------------------------------
    // Section 11: Complete Domain Adaptation Pipeline
    // -------------------------------------------------------------------------
    static void section11_completeDomainAdaptationPipeline() {
        System.out.println("--- Section 11: Complete Domain Adaptation Pipeline ---");

        // --- Base model ---
        SameDiff baseModel = section1_buildLanguageModel();

        // --- Pretraining config: domain adaptation for medical text ---
        ContinuedPretrainingConfig pretrainingConfig = ContinuedPretrainingConfig.builder()
                .chunkSize(2048)
                .chunkOverlap(128)
                .useLoRA(true)
                .loraRank(16)
                .loraAlpha(32)
                .learningRate(5e-5)
                .warmupRatio(0.1)
                .computeDataType(DataType.BFLOAT16)
                .gradientAccumulationSteps(4)
                .weightDecay(0.01)
                .maxGradNorm(1.0)
                .numEpochs(1)
                .build();
        pretrainingConfig.validate();

        // --- Gradient checkpointing: sqrt(N) strategy ---
        GradientCheckpointConfig checkpointing = GradientCheckpointConfig.sqrtN();

        // --- Loss scaler: dynamic for stability during domain shift ---
        LossScaleConfig lossScaleConfig = LossScaleConfig.builder()
                .mode(LossScaleConfig.Mode.DYNAMIC)
                .initialScale(65536.0)
                .minScale(1.0)
                .maxScale(65536.0)
                .growthFactor(2.0)
                .backoffFactor(0.5)
                .growthInterval(2000)
                .build();
        LossScaler lossScaler = new LossScaler(lossScaleConfig);

        // --- Gradient accumulator: 4 micro-batches ---
        GradientAccumulator accumulator = new GradientAccumulator(pretrainingConfig.getGradientAccumulationSteps());

        // --- LR schedule: cosine warmup (built inside workflow) ---
        int totalSteps = 5000;
        int warmupSteps = (int) (pretrainingConfig.getWarmupRatio() * totalSteps);
        CosineWarmupSchedule lrSchedule = new CosineWarmupSchedule(
                pretrainingConfig.getLearningRate(),
                pretrainingConfig.getMinLearningRate(),
                warmupSteps,
                totalSteps);

        // --- Workflow ---
        ContinuedPretrainingWorkflow workflow = new ContinuedPretrainingWorkflow(baseModel, pretrainingConfig);
        TrainingConfig trainingConfig = workflow.buildTrainingConfig(totalSteps);

        // --- Print full pipeline configuration ---
        System.out.println("  Full Domain Adaptation Pipeline Configuration:");
        System.out.println("  +----------------------------------------------------------+");
        System.out.printf("  | Base model vars:       %-32d |%n", baseModel.variables().size());
        System.out.printf("  | Domain:                %-32s |%n", "Medical text (continued pretraining)");
        System.out.println("  +----------------------------------------------------------+");
        System.out.println("  | ContinuedPretrainingConfig                              |");
        System.out.printf("  |   chunkSize:           %-32d |%n", pretrainingConfig.getChunkSize());
        System.out.printf("  |   chunkOverlap:        %-32d |%n", pretrainingConfig.getChunkOverlap());
        System.out.printf("  |   useLoRA:             %-32b |%n", pretrainingConfig.isUseLoRA());
        System.out.printf("  |   loraRank / alpha:    %d / %-27d |%n",
                pretrainingConfig.getLoraRank(), pretrainingConfig.getLoraAlpha());
        System.out.printf("  |   computeDataType:     %-32s |%n", pretrainingConfig.getComputeDataType());
        System.out.printf("  |   gradAccumSteps:      %-32d |%n", pretrainingConfig.getGradientAccumulationSteps());
        System.out.println("  +----------------------------------------------------------+");
        System.out.println("  | GradientCheckpointConfig                                |");
        System.out.printf("  |   strategy:            %-32s |%n", "sqrtN (auto interval at runtime)");
        System.out.printf("  |   resolveInterval(24): %-32d |%n", checkpointing.resolveInterval(24));
        System.out.println("  +----------------------------------------------------------+");
        System.out.println("  | LossScaler (dynamic)                                    |");
        System.out.printf("  |   initialScale:        %-32.0f |%n", lossScaler.getCurrentScale());
        System.out.printf("  |   growthFactor:        %-32.1f |%n", lossScaleConfig.getGrowthFactor());
        System.out.printf("  |   backoffFactor:       %-32.1f |%n", lossScaleConfig.getBackoffFactor());
        System.out.printf("  |   growthInterval:      %-32d |%n", lossScaleConfig.getGrowthInterval());
        System.out.println("  +----------------------------------------------------------+");
        System.out.println("  | GradientAccumulator                                     |");
        System.out.printf("  |   accumulationSteps:   %-32d |%n", accumulator.getAccumulationSteps());
        System.out.printf("  |   isEnabled:           %-32b |%n", accumulator.isEnabled());
        System.out.println("  +----------------------------------------------------------+");
        System.out.println("  | LR Schedule (CosineWarmup)                              |");
        System.out.printf("  |   totalSteps:          %-32d |%n", totalSteps);
        System.out.printf("  |   warmupSteps:         %-32d |%n", warmupSteps);
        System.out.printf("  |   peakLR (step %d):   %-32.6f |%n", warmupSteps,
                lrSchedule.valueAt(warmupSteps, 0));
        System.out.printf("  |   midLR (step %d):  %-32.6f |%n", totalSteps / 2,
                lrSchedule.valueAt(totalSteps / 2, 0));
        System.out.printf("  |   finalLR (step %d): %-32.6f |%n", totalSteps - 1,
                lrSchedule.valueAt(totalSteps - 1, 0));
        System.out.println("  +----------------------------------------------------------+");
        System.out.println("  | TrainingConfig (from workflow.buildTrainingConfig)       |");
        System.out.printf("  |   updater:             %-32s |%n", trainingConfig.getUpdater().getClass().getSimpleName());
        System.out.printf("  |   computeDataType:     %-32s |%n", trainingConfig.getComputeDataType());
        System.out.printf("  |   masterWeightType:    %-32s |%n", trainingConfig.getMasterWeightDataType());
        System.out.printf("  |   isMixedPrecision:    %-32b |%n", trainingConfig.isMixedPrecision());
        System.out.printf("  |   gradAccumSteps:      %-32d |%n", trainingConfig.getGradientAccumulationSteps());
        System.out.printf("  |   regularization[0]:   %-32s |%n",
                trainingConfig.getRegularization().isEmpty() ? "none"
                        : trainingConfig.getRegularization().get(0).getClass().getSimpleName());
        System.out.println("  +----------------------------------------------------------+");
        System.out.println();
        System.out.println("  Pipeline ready. Call workflow.train(dataIterator, totalSteps) to begin.");
        System.out.println();
    }

    // -------------------------------------------------------------------------
    // Section 12: DSP-Accelerated Continued Pre-training
    // -------------------------------------------------------------------------
    static void section12_dspAcceleratedContinuedPretraining() {
        System.out.println("--- Section 12: DSP-Accelerated Continued Pre-training ---");

        // Build a fresh model (same architecture as section 1)
        SameDiff sd = section1_buildLanguageModel();

        // Enable DSP explicitly (already on by default, but show it)
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        log.info("DSP flags: autoCompile={}, nativeAutoCompile={}",
                sd.isDspAutoCompileEnabled(), sd.isDspNativeAutoCompileEnabled());

        // TrainingConfig with Adam(5e-5) matching CPT learning rates, BF16 compute
        TrainingConfig trainingConfig = TrainingConfig.builder()
                .updater(new Adam(5e-5))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                .computeDataType(DataType.BFLOAT16)
                .masterWeightDataType(DataType.FLOAT)
                .gradientAccumulationSteps(4)
                .build();
        sd.setTrainingConfig(trainingConfig);

        // Synthetic training DataSet: batch=8, features=[8,128], labels=[8,64]
        int batchSize = 8;
        INDArray features = Nd4j.randn(DataType.FLOAT, batchSize, 128).muli(0.1);
        INDArray labels   = Nd4j.randn(DataType.FLOAT, batchSize, 64);
        DataSet ds = new DataSet(features, labels);

        System.out.println("  Running 12 training steps (DSP compiles forward+backward+weight update):");
        System.out.printf("  %-5s  %-10s  %-20s  %-10s  %-10s  %-10s%n",
                "Step", "Time(ms)", "Phase", "Replayed", "SBS", "Total");
        System.out.printf("  %-5s  %-10s  %-20s  %-10s  %-10s  %-10s%n",
                "-----", "----------", "--------------------", "----------", "----------", "----------");

        List<Double> warmupTimes = new ArrayList<>();
        List<Double> steadyTimes = new ArrayList<>();
        int numSteps = 12;

        for (int step = 0; step < numSteps; step++) {
            long t0 = System.nanoTime();
            sd.fit(ds);
            long t1 = System.nanoTime();
            double elapsedMs = (t1 - t0) / 1_000_000.0;

            DspHandle dsp = sd.dsp();
            int phaseCode   = dsp.isCompiled() ? dsp.planPhase() : -1;
            PlanPhase phase = dsp.isCompiled() ? PlanPhase.fromNativeCode(phaseCode) : null;
            int replayed    = dsp.isCompiled() ? dsp.lastExecSegmentsReplayed()   : 0;
            int sbs         = dsp.isCompiled() ? dsp.lastExecSegmentsSlotBySlot() : 0;
            int total       = dsp.isCompiled() ? dsp.lastExecSegmentsTotal()       : 0;

            String phaseName = (phase != null) ? phase.name() : "NOT_COMPILED";
            boolean isSteadyState = (phase == PlanPhase.REPLAYING);

            System.out.printf("  %-5d  %-10s  %-20s  %-10d  %-10d  %-10d%n",
                    step + 1,
                    String.format("%.2f", elapsedMs),
                    phaseName,
                    replayed,
                    sbs,
                    total);

            if (isSteadyState) {
                steadyTimes.add(elapsedMs);
            } else {
                warmupTimes.add(elapsedMs);
            }
        }

        // Warmup vs steady-state comparison
        System.out.println();
        double warmupAvgMs = warmupTimes.isEmpty() ? 0.0
                : warmupTimes.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double steadyAvgMs = steadyTimes.isEmpty() ? 0.0
                : steadyTimes.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

        double warmupSamplesPerSec  = (warmupAvgMs  > 0) ? (batchSize * 1000.0 / warmupAvgMs)  : 0.0;
        double steadySamplesPerSec  = (steadyAvgMs  > 0) ? (batchSize * 1000.0 / steadyAvgMs)  : 0.0;
        double speedup = (warmupAvgMs > 0 && steadyAvgMs > 0) ? (warmupAvgMs / steadyAvgMs) : 1.0;

        System.out.printf("  Warmup   (%2d steps): avg %s ms/step, %s samples/sec%n",
                warmupTimes.size(),
                String.format("%.2f", warmupAvgMs),
                String.format("%.1f", warmupSamplesPerSec));
        System.out.printf("  Steady   (%2d steps): avg %s ms/step, %s samples/sec%n",
                steadyTimes.size(),
                String.format("%.2f", steadyAvgMs),
                String.format("%.1f", steadySamplesPerSec));
        System.out.printf("  Speedup ratio (warmup/steady): %sx%n",
                String.format("%.2f", speedup));

        // DspHandle summary
        System.out.println();
        System.out.println("  DspHandle summary (after 12 steps):");
        if (sd.dsp().isCompiled()) {
            DspHandle dsp = sd.dsp();
            System.out.printf("    isCompiled             = %b%n",  dsp.isCompiled());
            System.out.printf("    totalSlots             = %d%n",  dsp.totalSlots());
            System.out.printf("    numSegments            = %d%n",  dsp.numSegments());
            System.out.printf("    numCapturedGraphSegs   = %d%n",  dsp.numCapturedGraphSegments());
            System.out.printf("    totalGraphReplays      = %d%n",  dsp.totalGraphReplays());
            System.out.printf("    pointersStable         = %b%n",  dsp.pointersStable());
            System.out.printf("    isCompilationSealed    = %b%n",  dsp.isCompilationSealed());
        } else {
            System.out.println("    (plan not compiled — CPU path or DSP inactive)");
        }

        System.out.println();
        System.out.println("  Insight: DSP compiles the full continued pre-training graph including");
        System.out.println("  forward, loss, backward, gradient accumulation, and weight updates.");
        System.out.println("  Once in REPLAYING phase, CUDA graphs replay the entire backward pass");
        System.out.println("  with LoRA adapters and BF16 mixed precision at native kernel speed.");
        System.out.println();

        // Close synthetic arrays
        features.close();
        labels.close();
    }
}
