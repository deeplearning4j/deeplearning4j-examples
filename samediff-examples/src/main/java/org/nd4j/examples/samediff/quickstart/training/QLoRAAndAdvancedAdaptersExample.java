/*
 * ******************************************************************************
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
 *******************************************************************************/

package org.nd4j.examples.samediff.quickstart.training;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.*;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.peft.LoraAdapterCache;
import org.nd4j.autodiff.samediff.peft.PeftModel;
import org.nd4j.autodiff.samediff.TransferLearning;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.adapter.SingletonDataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Comprehensive example covering all advanced PEFT (Parameter-Efficient Fine-Tuning)
 * adapter types available in SameDiff.
 *
 * <p>PEFT methods allow large pre-trained models to be adapted to new tasks by
 * training only a small fraction of the total parameters, dramatically reducing
 * GPU memory requirements and training time.</p>
 *
 * <h3>Sections covered:</h3>
 * <ol>
 *   <li>Base model construction — 3-layer MLP (128→256→64→10)</li>
 *   <li>QLoRA — 4-bit quantized LoRA with NF4 and double quantization</li>
 *   <li>DoRA — Weight-Decomposed LoRA (magnitude + direction)</li>
 *   <li>AdaLoRA — Adaptive rank allocation with rank schedule</li>
 *   <li>IA³ — Infused Adapter (rescaling vectors, fewest params)</li>
 *   <li>LoHa — Low-Rank Hadamard Product (effective rank = dim²)</li>
 *   <li>LoKr — Low-Rank Kronecker Product</li>
 *   <li>VeRA — Vectorized Rank Adaptation (shared frozen matrices)</li>
 *   <li>DyLoRA — Dynamic rank sampling for deploy-time flexibility</li>
 *   <li>LoftQ — LoRA initialized from quantization residual SVD</li>
 *   <li>Prompt Tuning — Learnable soft prompt tokens</li>
 *   <li>Prefix Tuning — Per-layer key/value prefix vectors</li>
 *   <li>Bottleneck Adapters — Classic adapter layers via TransferLearning</li>
 *   <li>LoRA Adapter Cache — Multi-adapter hot-swap with GPU/host tiers</li>
 *   <li>Comparison table — Side-by-side summary of all methods</li>
 *   <li>DSP-Accelerated PEFT Training — LoRA training with DSP plan phase tracing</li>
 * </ol>
 *
 * <p>Run with:</p>
 * <pre>
 *   cd samediff-examples
 *   mvn exec:java -Dexec.mainClass="org.nd4j.examples.samediff.quickstart.training.QLoRAAndAdvancedAdaptersExample"
 * </pre>
 *
 * @author Adam Gibson
 */
public class QLoRAAndAdvancedAdaptersExample {

    private static final Logger log = LoggerFactory.getLogger(QLoRAAndAdvancedAdaptersExample.class);

    // Target module names that match the variables in the base MLP
    private static final List<String> MLP_TARGET_MODULES = Arrays.asList(
            "layer1.weight", "layer2.weight", "layer3.weight");

    public static void main(String[] args) {

        // =====================================================================
        // SECTION 1 — Build the base model: 3-layer MLP (128→256→64→10)
        // =====================================================================
        log.info("=======================================================");
        log.info("SECTION 1: Building Base Model (128->256->64->10 MLP)");
        log.info("=======================================================");

        SameDiff baseModel = buildBaseMLP();

        long totalBaseParams = countParameters(baseModel);
        log.info("Base model variables:");
        for (SDVariable v : baseModel.variables()) {
            if (v.getVariableType() == VariableType.VARIABLE) {
                long[] shape = v.getShape();
                long count = 1;
                if (shape != null) {
                    for (long d : shape) count *= d;
                }
                log.info("  {} shape={} params={}", v.name(), Arrays.toString(shape), count);
            }
        }
        log.info("Total base parameters: {}", totalBaseParams);


        // =====================================================================
        // SECTION 2 — QLoRA: 4-bit NF4 with double quantization
        // =====================================================================
        log.info("");
        log.info("=======================================================");
        log.info("SECTION 2: QLoRA (4-bit NF4 + Double Quantization)");
        log.info("=======================================================");

        QLoraConfig qloraConfig = QLoraConfig.builder()
                .r(16)
                .loraAlpha(32)
                .loraDropout(0.05)
                .bits(4)
                .quantType("nf4")
                .doubleQuant(true)                // quantize the quantization constants
                .computeDataType(DataType.BFLOAT16)
                .loraDataType(DataType.BFLOAT16)
                .blockSize(64)
                .targetModules(MLP_TARGET_MODULES)
                .taskType(TaskType.CAUSAL_LM)
                .build();

        qloraConfig.validate();

        long estimatedBytes = qloraConfig.estimateMemoryBytes(totalBaseParams);
        long estimatedMB    = estimatedBytes / (1024 * 1024);

        log.info("QLoRA config: {}", qloraConfig.getSummary());
        log.info("  Rank (r)          : {}", qloraConfig.getR());
        log.info("  Alpha             : {}", qloraConfig.getLoraAlpha());
        log.info("  Scaling (alpha/r) : {}", String.format("%.4f", qloraConfig.getScaling()));
        log.info("  Quant type        : {}", qloraConfig.getQuantType());
        log.info("  Bits              : {}", qloraConfig.getBits());
        log.info("  Double quant      : {}", qloraConfig.isDoubleQuant());
        log.info("  Block size        : {}", qloraConfig.getBlockSize());
        log.info("  Compute dtype     : {}", qloraConfig.getComputeDataType());
        log.info("  Estimated memory  : {} bytes (~{} MB)", estimatedBytes, estimatedMB);

        PeftModel qloraModel = PeftModel.fromPretrained(baseModel, qloraConfig);
        log.info("QLoRA trainable params: {}", qloraModel.getTrainableParameterCount());
        log.info("QLoRA total params    : {}", qloraModel.getTotalParameterCount());
        log.info("QLoRA trainable %%    : {}%%", String.format("%.4f", qloraModel.getTrainablePercentage()));

        // Also demonstrate the static factory approach
        QLoraConfig qlora4bit = QLoraConfig.default4Bit(MLP_TARGET_MODULES);
        QLoraConfig qlora8bit = QLoraConfig.default8Bit(MLP_TARGET_MODULES);
        log.info("  default4Bit config: {}", qlora4bit.getSummary());
        log.info("  default8Bit config: {}", qlora8bit.getSummary());


        // =====================================================================
        // SECTION 3 — DoRA: Weight-Decomposed LoRA
        // =====================================================================
        log.info("");
        log.info("=======================================================");
        log.info("SECTION 3: DoRA (Weight-Decomposed LoRA)");
        log.info("=======================================================");

        DoraConfig doraConfig = DoraConfig.builder()
                .r(16)
                .loraAlpha(32)
                .loraDropout(0.05)
                .magnitudeInit("pretrained")   // use ||W₀|| column norms
                .magnitudePerRow(false)         // per-column (matches paper)
                .ephemeralGpuOffload(false)
                .targetModules(MLP_TARGET_MODULES)
                .taskType(TaskType.CAUSAL_LM)
                .build();

        doraConfig.validate();

        log.info("DoRA config: {}", doraConfig.getSummary());
        log.info("  r                : {}", doraConfig.getR());
        log.info("  alpha            : {}", doraConfig.getLoraAlpha());
        log.info("  magnitudeInit    : {}", doraConfig.getMagnitudeInit());
        log.info("  magnitudePerRow  : {}", doraConfig.isMagnitudePerRow());
        log.info("  PEFT type        : {}", doraConfig.getPeftType());

        PeftModel doraModel = PeftModel.fromPretrained(baseModel, doraConfig);
        log.info("DoRA trainable params: {}", doraModel.getTrainableParameterCount());
        log.info("DoRA total params    : {}", doraModel.getTotalParameterCount());
        log.info("DoRA trainable %%    : {}%%", String.format("%.4f", doraModel.getTrainablePercentage()));


        // =====================================================================
        // SECTION 4 — AdaLoRA: Adaptive Rank Allocation
        // =====================================================================
        log.info("");
        log.info("=======================================================");
        log.info("SECTION 4: AdaLoRA (Adaptive Low-Rank Adaptation)");
        log.info("=======================================================");

        AdaLoraConfig adaloraConfig = AdaLoraConfig.builder()
                .initRank(12)                   // start rank (higher than target)
                .targetRank(8)                  // final rank after pruning
                .loraAlpha(32)
                .loraDropout(0.0)
                .warmupSteps(100)               // no pruning during warmup
                .totalPruningSteps(1000)        // prune over 1000 steps
                .orthogonalRegularization(true)
                .importanceBeta(0.85)           // EMA decay for importance scores
                .targetModules(MLP_TARGET_MODULES)
                .taskType(TaskType.CAUSAL_LM)
                .build();

        adaloraConfig.validate();

        log.info("AdaLoRA config: {}", adaloraConfig.getSummary());
        log.info("  initRank           : {}", adaloraConfig.getInitRank());
        log.info("  targetRank         : {}", adaloraConfig.getTargetRank());
        log.info("  warmupSteps        : {}", adaloraConfig.getWarmupSteps());
        log.info("  totalPruningSteps  : {}", adaloraConfig.getTotalPruningSteps());
        log.info("  orthogonalReg      : {}", adaloraConfig.isOrthogonalRegularization());
        log.info("  importanceBeta     : {}", adaloraConfig.getImportanceBeta());

        // Show rank schedule progression
        log.info("  Rank schedule (getCurrentRank):");
        int[] checkSteps = {0, 50, 100, 200, 500, 700, 1000, 1500};
        for (int step : checkSteps) {
            int rank = adaloraConfig.getCurrentRank(step);
            log.info("    step={}  rank={}", String.format("%5d", step), rank);
        }

        PeftModel adaloraModel = PeftModel.fromPretrained(baseModel, adaloraConfig);
        log.info("AdaLoRA trainable params: {}", adaloraModel.getTrainableParameterCount());


        // =====================================================================
        // SECTION 5 — IA³: Infused Adapter (rescaling vectors only)
        // =====================================================================
        log.info("");
        log.info("=======================================================");
        log.info("SECTION 5: IA³ (Infused Adapter)");
        log.info("=======================================================");

        IA3Config ia3Config = IA3Config.builder()
                .targetModules(Arrays.asList("layer1.weight", "layer2.weight"))
                .feedforwardModules(Arrays.asList("layer2.weight", "layer3.weight"))
                .initToOne(true)                // start at identity (pretrained behavior)
                .attentionHiddenSize(256)
                .feedforwardHiddenSize(64)
                .taskType(TaskType.CAUSAL_LM)
                .build();

        ia3Config.validate();

        long ia3Params = ia3Config.calculateTrainableParameters(totalBaseParams);

        log.info("IA3 config: {}", ia3Config.getSummary());
        log.info("  targetModules      : {}", ia3Config.getTargetModules());
        log.info("  feedforwardModules : {}", ia3Config.getFeedforwardModules());
        log.info("  initToOne          : {}", ia3Config.isInitToOne());
        log.info("  attnHiddenSize     : {}", ia3Config.getAttentionHiddenSize());
        log.info("  ffHiddenSize       : {}", ia3Config.getFeedforwardHiddenSize());
        log.info("  Estimated trainable params: {}", ia3Params);

        PeftModel ia3Model = PeftModel.fromPretrained(baseModel, ia3Config);
        log.info("IA3 trainable params: {}", ia3Model.getTrainableParameterCount());
        log.info("IA3 total params    : {}", ia3Model.getTotalParameterCount());
        log.info("IA3 trainable %%    : {}%%", String.format("%.4f", ia3Model.getTrainablePercentage()));


        // =====================================================================
        // SECTION 6 — LoHa: Low-Rank Hadamard Product
        // =====================================================================
        log.info("");
        log.info("=======================================================");
        log.info("SECTION 6: LoHa (Low-Rank Hadamard Product)");
        log.info("=======================================================");

        LohaConfig lohaConfig = LohaConfig.builder()
                .dim(8)                         // low-rank dimension; effective rank ≤ dim²=64
                .alpha(1.0)
                .dropout(0.0)
                .useTucker(false)
                .initMethod("kaiming")
                .targetModules(MLP_TARGET_MODULES)
                .taskType(TaskType.CAUSAL_LM)
                .build();

        lohaConfig.validate();

        log.info("LoHa config: {}", lohaConfig.getSummary());
        log.info("  dim (r)          : {}", lohaConfig.getDim());
        log.info("  effectiveMaxRank : {}", lohaConfig.getEffectiveMaxRank());
        log.info("  alpha            : {}", lohaConfig.getAlpha());
        log.info("  useTucker        : {}", lohaConfig.isUseTucker());
        log.info("  initMethod       : {}", lohaConfig.getInitMethod());
        log.info("  PEFT type        : {}", lohaConfig.getPeftType());

        long lohaEstimate = lohaConfig.calculateTrainableParameters(totalBaseParams);
        log.info("  Estimated trainable params: {} (4 matrices per target vs 2 for LoRA)",
                lohaEstimate);

        PeftModel lohaModel = PeftModel.fromPretrained(baseModel, lohaConfig);
        log.info("LoHa trainable params: {}", lohaModel.getTrainableParameterCount());


        // =====================================================================
        // SECTION 7 — LoKr: Low-Rank Kronecker Product
        // =====================================================================
        log.info("");
        log.info("=======================================================");
        log.info("SECTION 7: LoKr (Low-Rank Kronecker Product)");
        log.info("=======================================================");

        LokrConfig lokrConfig = LokrConfig.builder()
                .dim(8)
                .factor(-1)                     // auto-determine Kronecker factor
                .alpha(1.0)
                .dropout(0.0)
                .decomposeKronecker(false)
                .fullMatrix(false)
                .targetModules(MLP_TARGET_MODULES)
                .taskType(TaskType.CAUSAL_LM)
                .build();

        lokrConfig.validate();

        log.info("LoKr config: {}", lokrConfig.getSummary());
        log.info("  dim              : {}", lokrConfig.getDim());
        log.info("  factor           : {} (auto=-1)", lokrConfig.getFactor());
        log.info("  alpha            : {}", lokrConfig.getAlpha());
        log.info("  decomposeKronecker: {}", lokrConfig.isDecomposeKronecker());
        log.info("  PEFT type        : {}", lokrConfig.getPeftType());

        long lokrEstimate = lokrConfig.calculateTrainableParameters(totalBaseParams);
        log.info("  Estimated trainable params: {}", lokrEstimate);

        PeftModel lokrModel = PeftModel.fromPretrained(baseModel, lokrConfig);
        log.info("LoKr trainable params: {}", lokrModel.getTrainableParameterCount());


        // =====================================================================
        // SECTION 8 — VeRA: Vectorized Rank Adaptation
        // =====================================================================
        log.info("");
        log.info("=======================================================");
        log.info("SECTION 8: VeRA (Vectorized Rank Adaptation)");
        log.info("=======================================================");

        // VeRA uses shared frozen random matrices A_shared and B_shared across all
        // layers; only per-layer scaling vectors d and b are trained.
        // Trainable params per layer: outDim + r  (vs r*(outDim+inDim) for LoRA)
        VeraConfig veraConfig = VeraConfig.builder()
                .r(256)                         // VeRA uses higher rank since only scaling is trained
                .sharedSeed(1337L)              // seed for reproducible random frozen matrices
                .lambdaScaling(1.0)
                .targetModules(MLP_TARGET_MODULES)
                .taskType(TaskType.CAUSAL_LM)
                .build();

        veraConfig.validate();

        long veraEstimate = veraConfig.calculateTrainableParameters(totalBaseParams);

        log.info("VeRA config: {}", veraConfig.getSummary());
        log.info("  r (shared rank)  : {}", veraConfig.getR());
        log.info("  sharedSeed       : {}", veraConfig.getSharedSeed());
        log.info("  lambdaScaling    : {}", veraConfig.getLambdaScaling());
        log.info("  targetModules    : {}", veraConfig.getTargetModules());
        log.info("  PEFT type        : {}", veraConfig.getPeftType());
        log.info("  Estimated trainable params: {} (only d+r per layer!)", veraEstimate);
        log.info("  Key insight: shared A/B matrices frozen; only scaling vectors d,b are trained");


        // =====================================================================
        // SECTION 9 — DyLoRA: Dynamic Rank Sampling
        // =====================================================================
        log.info("");
        log.info("=======================================================");
        log.info("SECTION 9: DyLoRA (Dynamic Low-Rank Adaptation)");
        log.info("=======================================================");

        // DyLoRA trains with rank k sampled uniformly from [minRank, r] each step.
        // After training a single adapter works at any rank from minRank to r.
        DyLoraConfig dyloraConfig = DyLoraConfig.builder()
                .r(16)                          // maximum rank
                .minRank(4)                     // minimum rank for sampling
                .loraAlpha(32)
                .loraDropout(0.05)
                .targetModules(MLP_TARGET_MODULES)
                .taskType(TaskType.CAUSAL_LM)
                .build();

        dyloraConfig.validate();

        log.info("DyLoRA config: {}", dyloraConfig.getSummary());
        log.info("  r (max rank)     : {}", dyloraConfig.getR());
        log.info("  minRank          : {}", dyloraConfig.getMinRank());
        log.info("  loraAlpha        : {}", dyloraConfig.getLoraAlpha());
        log.info("  scaling (alpha/r): {}", String.format("%.4f", dyloraConfig.getScaling()));
        log.info("  loraDropout      : {}", dyloraConfig.getLoraDropout());
        log.info("  PEFT type        : {}", dyloraConfig.getPeftType());
        log.info("  Key insight: one trained adapter, deployable at rank 4..16 without retraining");


        // =====================================================================
        // SECTION 10 — LoftQ: LoRA-Fine-Tuning-Aware Quantization Init
        // =====================================================================
        log.info("");
        log.info("=======================================================");
        log.info("SECTION 10: LoftQ (LoRA-Fine-Tuning-Aware Quantization)");
        log.info("=======================================================");

        // LoftQ initializes LoRA adapters by SVD of the quantization residual:
        //   1. Quantize W to get Q
        //   2. Compute residual R = W - Q
        //   3. SVD: R ≈ B @ A  (top-r singular vectors)
        //   4. Initialize LoRA with this B, A
        // At training/inference time it behaves identically to standard LoRA.
        LoftQConfig loftq4bit = LoftQConfig.default4Bit(16);
        LoftQConfig loftq8bit = LoftQConfig.default8Bit(16);

        LoftQConfig loftqCustom = LoftQConfig.builder()
                .r(16)
                .loraAlpha(32)
                .loraDropout(0.05)
                .numIterations(5)               // more iterations = lower residual error
                .quantType("nf4")
                .quantBits(4)
                .blockSize(64)
                .targetModules(MLP_TARGET_MODULES)
                .taskType(TaskType.CAUSAL_LM)
                .build();

        loftqCustom.validate();

        log.info("LoftQ default4Bit config: {}", loftq4bit.getSummary());
        log.info("LoftQ default8Bit config: {}", loftq8bit.getSummary());
        log.info("LoftQ custom config     : {}", loftqCustom.getSummary());
        log.info("  r                : {}", loftqCustom.getR());
        log.info("  numIterations    : {}", loftqCustom.getNumIterations());
        log.info("  quantType        : {}", loftqCustom.getQuantType());
        log.info("  quantBits        : {}", loftqCustom.getQuantBits());
        log.info("  blockSize        : {}", loftqCustom.getBlockSize());
        log.info("  initLoraWeights  : {} (signals LoftQ init path)", loftqCustom.getInitLoraWeights());
        log.info("  getPeftType()    : {} (behaves as LoRA at runtime)", loftqCustom.getPeftType());


        // =====================================================================
        // SECTION 11 — Prompt Tuning: Learnable Soft Prompt Tokens
        // =====================================================================
        log.info("");
        log.info("=======================================================");
        log.info("SECTION 11: Prompt Tuning");
        log.info("=======================================================");

        int embDim = 128;  // match the MLP input dimension

        // Default random initialization
        PromptTuningConfig promptDefault = PromptTuningConfig.defaultConfig(embDim);

        // Text-initialized: embeddings start from tokenized task description
        PromptTuningConfig promptText = PromptTuningConfig.withTextInit(
                "Classify the sentiment of the following review:", embDim);

        // Custom configuration
        PromptTuningConfig promptCustom = PromptTuningConfig.builder()
                .numVirtualTokens(50)
                .promptTuningInit(PromptTuningConfig.PromptTuningInit.KAIMING)
                .tokenEmbeddingDim(embDim)
                .randomSeed(42L)
                .taskType(TaskType.SEQ_CLS)
                .build();

        promptDefault.validate();
        promptText.validate();
        promptCustom.validate();

        log.info("PromptTuning defaultConfig: {}", promptDefault.getSummary());
        log.info("  numVirtualTokens : {}", promptDefault.getNumVirtualTokens());
        log.info("  initMethod       : {}", promptDefault.getPromptTuningInit());
        log.info("  embeddingDim     : {}", promptDefault.getTokenEmbeddingDim());
        log.info("  trainable params : {}", promptDefault.calculateTrainableParameters(totalBaseParams));

        log.info("PromptTuning withTextInit: {}", promptText.getSummary());
        log.info("  initText         : \"{}\"", promptText.getPromptTuningInitText());
        log.info("  initMethod       : {}", promptText.getPromptTuningInit());

        log.info("PromptTuning custom (KAIMING, 50 tokens): {}", promptCustom.getSummary());
        log.info("  trainable params : {}", promptCustom.calculateTrainableParameters(totalBaseParams));

        PeftModel promptModel = PeftModel.fromPretrained(baseModel, promptDefault);
        log.info("PromptTuning model trainable params: {}", promptModel.getTrainableParameterCount());


        // =====================================================================
        // SECTION 12 — Prefix Tuning: Per-Layer Key/Value Prefix Vectors
        // =====================================================================
        log.info("");
        log.info("=======================================================");
        log.info("SECTION 12: Prefix Tuning");
        log.info("=======================================================");

        // Simulate a small transformer: 6 layers, 8 heads, hidden=128
        int numLayers = 6;
        int numHeads  = 8;
        int hiddenSize = 128;

        PrefixTuningConfig prefixConfig = PrefixTuningConfig.forTransformer(
                numLayers, numHeads, hiddenSize);

        // Also construct a custom seq2seq variant
        PrefixTuningConfig prefixSeq2Seq = PrefixTuningConfig.forSeq2Seq(
                numLayers, numHeads, hiddenSize);

        prefixConfig.validate();

        long prefixParams = prefixConfig.calculateTrainableParameters(totalBaseParams);

        log.info("PrefixTuning forTransformer: {}", prefixConfig.getSummary());
        log.info("  numVirtualTokens   : {}", prefixConfig.getNumVirtualTokens());
        log.info("  numLayers          : {}", prefixConfig.getNumLayers());
        log.info("  numHeads           : {}", prefixConfig.getNumHeads());
        log.info("  hiddenSize         : {}", prefixConfig.getHiddenSize());
        log.info("  prefixProjection   : {}", prefixConfig.isPrefixProjection());
        log.info("  encoderHiddenSize  : {}", prefixConfig.getEncoderHiddenSize());
        log.info("  prefixDropout      : {}", prefixConfig.getPrefixDropout());
        log.info("  Estimated trainable params: {}", prefixParams);

        log.info("PrefixTuning seq2seq: {}", prefixSeq2Seq.getSummary());
        log.info("  encoderDecoder     : {}", prefixSeq2Seq.isEncoderDecoder());
        log.info("  taskType           : {}", prefixSeq2Seq.getTaskType());

        PeftModel prefixModel = PeftModel.fromPretrained(baseModel, prefixConfig);
        log.info("PrefixTuning model trainable params: {}", prefixModel.getTrainableParameterCount());
        log.info("PrefixTuning model trainable %%    : {}%%",
                String.format("%.4f", prefixModel.getTrainablePercentage()));


        // =====================================================================
        // SECTION 13 — Bottleneck Adapters via TransferLearning
        // =====================================================================
        log.info("");
        log.info("=======================================================");
        log.info("SECTION 13: Bottleneck Adapters (via TransferLearning)");
        log.info("=======================================================");

        // Classic Houlsby-style adapter: down-project → activation → up-project + residual
        AdapterConfig adapterConfig = AdapterConfig.builder()
                .adapterSize(32)                // bottleneck dimension r
                .adapterActivation("relu")
                .adapterDropout(0.0)
                .adapterScaling(1.0)
                .adapterAfterAttention(true)
                .adapterAfterFeedforward(true)
                .hiddenSize(hiddenSize)
                .numLayers(numLayers)
                .taskType(TaskType.CAUSAL_LM)
                .build();

        adapterConfig.validate();

        long adapterParams = adapterConfig.calculateTrainableParameters(totalBaseParams);

        log.info("AdapterConfig: {}", adapterConfig.getSummary());
        log.info("  adapterSize          : {}", adapterConfig.getAdapterSize());
        log.info("  adapterActivation    : {}", adapterConfig.getAdapterActivation());
        log.info("  adapterAfterAttention: {}", adapterConfig.isAdapterAfterAttention());
        log.info("  adapterAfterFF       : {}", adapterConfig.isAdapterAfterFeedforward());
        log.info("  hiddenSize           : {}", adapterConfig.getHiddenSize());
        log.info("  numLayers            : {}", adapterConfig.getNumLayers());
        log.info("  Estimated trainable params: {}", adapterParams);

        // Apply via TransferLearning.adapterModel factory
        PeftModel adapterModel = TransferLearning.adapterModel(
                baseModel, adapterConfig.getAdapterSize(),
                adapterConfig.getHiddenSize(), adapterConfig.getNumLayers());
        log.info("Adapter model trainable params: {}", adapterModel.getTrainableParameterCount());
        log.info("Adapter model trainable %%    : {}%%",
                String.format("%.4f", adapterModel.getTrainablePercentage()));

        // Alternatively apply directly through PeftModel
        PeftModel adapterModelDirect = PeftModel.fromPretrained(baseModel, adapterConfig);
        log.info("Adapter model (direct) trainable params: {}",
                adapterModelDirect.getTrainableParameterCount());


        // =====================================================================
        // SECTION 14 — LoRA Adapter Cache: Multi-Adapter Hot-Swap
        // =====================================================================
        log.info("");
        log.info("=======================================================");
        log.info("SECTION 14: LoRA Adapter Cache (Multi-Adapter Hot-Swap)");
        log.info("=======================================================");

        // Build a LoRA model to serve as the live model for adapter swapping
        LoraConfig serveLoraConfig = LoraConfig.builder()
                .r(8)
                .loraAlpha(16)
                .loraDropout(0.0)
                .targetModules(MLP_TARGET_MODULES)
                .taskType(TaskType.CAUSAL_LM)
                .build();

        PeftModel serveModel = PeftModel.fromPretrained(baseModel, serveLoraConfig);

        // Create the cache: GPU tier holds up to 3 adapters; host tier holds up to 10
        try (LoraAdapterCache cache = new LoraAdapterCache(3, 10)) {

            log.info("Created LoraAdapterCache: maxGpu={}, maxHost={}",
                    cache.getMaxGpuAdapters(), cache.getMaxHostAdapters());

            // Simulate 5 task-specific adapters by registering random weight maps.
            // In production these would be loaded from saved files via loadAdapter().
            String[] adapterNames = {
                "adapter-task-sentiment",
                "adapter-task-summarize",
                "adapter-task-translate-de",
                "adapter-task-translate-fr",
                "adapter-task-qa"
            };

            for (String name : adapterNames) {
                Map<String, INDArray> weights = buildFakeAdapterWeights(MLP_TARGET_MODULES, 8);
                // Register on host (warm tier) initially — GPU tier fills as adapters are used
                cache.registerAdapter(name, weights, false);
                log.info("  Registered adapter '{}': {} weight tensors", name, weights.size());
            }

            log.info("Cache after registration: gpu={}, host={}",
                    cache.getGpuAdapterCount(), cache.getHostAdapterCount());
            log.info("Cached adapters: {}", cache.getCachedAdapterNames());

            // Check warm/hot status before any swaps
            log.info("Before swaps:");
            for (String name : adapterNames) {
                log.info("  {} hot={} warm={}", name,
                        cache.isHot(name), cache.isWarm(name));
            }

            // Apply adapters in sequence — the first three will be promoted to GPU (hot tier)
            log.info("Applying adapters (first 3 fill the GPU tier):");
            for (String name : adapterNames) {
                cache.applyAdapter(name, serveModel.getModel());
                log.info("  Applied '{}': activeAdapter='{}', gpuCount={}, hostCount={}",
                        name, cache.getActiveAdapterName(),
                        cache.getGpuAdapterCount(), cache.getHostAdapterCount());
            }

            // After filling GPU tier beyond maxGpuAdapters=3, LRU should demote to host
            log.info("After applying all 5 adapters (GPU tier capped at 3):");
            for (String name : adapterNames) {
                log.info("  {} hot={} warm={}",
                        name, cache.isHot(name), cache.isWarm(name));
            }

            // Re-apply an earlier adapter — it will promote from host to GPU (warm swap)
            String reapply = adapterNames[0];
            log.info("Re-applying '{}' (warm swap, host->GPU)...", reapply);
            cache.applyAdapter(reapply, serveModel.getModel());
            log.info("  After re-apply: hot={} warm={}",
                    cache.isHot(reapply), cache.isWarm(reapply));

            // Verify the currently active adapter
            log.info("Active adapter: '{}'", cache.getActiveAdapterName());

            // Print cache performance statistics
            log.info("Cache stats: {}", cache.getStats());

        }  // cache.close() releases all GPU and host memory


        // =====================================================================
        // SECTION 15 — Comparison Table
        // =====================================================================
        log.info("");
        log.info("=======================================================");
        log.info("SECTION 15: PEFT Method Comparison Table");
        log.info("=======================================================");

        // Compute approximate trainable-parameter counts for the 128→256→64→10 base model
        // (2 matrices per LoRA layer × 3 target modules × rank × avg-dim-estimate of 1024)
        long loraEstimate    = computeLoraEstimate(16, MLP_TARGET_MODULES.size());
        long qloraEstimate   = computeLoraEstimate(16, MLP_TARGET_MODULES.size());  // same adapter params
        long doraEstimate    = computeLoraEstimate(16, MLP_TARGET_MODULES.size());  // + magnitude vectors
        long adaloraEstimate = computeLoraEstimate(12, MLP_TARGET_MODULES.size());  // initRank
        long ia3EstimateVal  = ia3Config.calculateTrainableParameters(totalBaseParams);
        long lohaEstimateVal = lohaConfig.calculateTrainableParameters(totalBaseParams);
        long lokrEstimateVal = lokrConfig.calculateTrainableParameters(totalBaseParams);
        long veraEstimateVal = veraConfig.calculateTrainableParameters(totalBaseParams);
        long dyloraEstimate  = computeLoraEstimate(16, MLP_TARGET_MODULES.size());
        long loftqEstimate   = computeLoraEstimate(16, MLP_TARGET_MODULES.size());
        long promptEstimate  = promptDefault.calculateTrainableParameters(totalBaseParams);
        long prefixEstimate  = prefixConfig.calculateTrainableParameters(totalBaseParams);
        long adapterEstimate = adapterConfig.calculateTrainableParameters(totalBaseParams);

        log.info("");
        log.info(String.format("%-18s | %-10s | %-10s | %-20s | %-45s",
                "Method", "Type", "Est.Params", "Key Config", "Description"));
        log.info("-".repeat(115));
        logRow("LoRA",         "Weight",    loraEstimate,    "r=16, α=32",            "Low-rank decomposition W₀+BA");
        logRow("QLoRA",        "Weight",    qloraEstimate,   "4-bit NF4, doubleQuant","Quantized base + LoRA adapters");
        logRow("DoRA",         "Weight",    doraEstimate,    "r=16, magInit=pretrained","Magnitude+direction decomposition");
        logRow("AdaLoRA",      "Weight",    adaloraEstimate, "initR=12, targetR=8",   "Adaptive rank via SVD importance");
        logRow("IA³",          "Activation",ia3EstimateVal,  "feedforward+attn",      "Rescaling vectors (fewest params)");
        logRow("LoHa",         "Weight",    lohaEstimateVal, "dim=8 (effRank=64)",    "Hadamard product of 2 low-rank pairs");
        logRow("LoKr",         "Weight",    lokrEstimateVal, "dim=8, factor=auto",    "Kronecker product structure");
        logRow("VeRA",         "Weight",    veraEstimateVal, "r=256, shared frozen",  "Shared matrices, only scale vectors");
        logRow("DyLoRA",       "Weight",    dyloraEstimate,  "r=16, minR=4",          "Train once, deploy at any rank");
        logRow("LoftQ",        "Weight",    loftqEstimate,   "4-bit NF4, 5 iters",   "SVD residual init for quantized");
        logRow("PromptTuning", "Input",     promptEstimate,  "20 tokens, d=128",      "Soft prompt token embeddings");
        logRow("PrefixTuning", "Attention", prefixEstimate,  "30 tokens, 6 layers",   "Per-layer K/V prefix vectors");
        logRow("Adapters",     "FF/Attn",   adapterEstimate, "r=32, 6 layers",        "Bottleneck down/up projection");
        log.info("-".repeat(115));

        log.info("");
        log.info("Key selection guidance:");
        log.info("  Memory-constrained GPU  -> QLoRA (4-bit base + BF16 adapters)");
        log.info("  Best quality at low r   -> DoRA  (magnitude decomposition helps)");
        log.info("  Budget allocation       -> AdaLoRA (prunes unimportant directions)");
        log.info("  Absolute minimum params -> IA3 or VeRA");
        log.info("  Single-train multi-rank -> DyLoRA");
        log.info("  Serving many tasks      -> LoRA + LoraAdapterCache (hot-swap)");
        log.info("  Multi-task prompting    -> Prompt or Prefix Tuning");
        log.info("  Classic NLP adapters    -> Bottleneck Adapters");


        // =====================================================================
        // SECTION 16 — DSP-Accelerated PEFT Training
        // =====================================================================
        log.info("");
        log.info("=======================================================");
        log.info("SECTION 16: DSP-Accelerated PEFT Training");
        log.info("=======================================================");

        // Build a fresh base model and apply a standard LoRA adapter (rank=8, alpha=16).
        SameDiff dspBase = buildBaseMLP();
        LoraConfig dspLoraConfig = LoraConfig.builder()
                .r(8)
                .loraAlpha(16)
                .loraDropout(0.0)
                .targetModules(MLP_TARGET_MODULES)
                .taskType(TaskType.CAUSAL_LM)
                .build();
        PeftModel dspPeft = PeftModel.fromPretrained(dspBase, dspLoraConfig);

        // The PeftModel modifies the underlying SameDiff in-place; retrieve it.
        SameDiff sd = dspPeft.getModel();

        // Configure training: Adam optimizer, map dataset columns to graph placeholders.
        TrainingConfig trainingConfig = TrainingConfig.builder()
                .updater(new Adam(1e-3))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .build();
        sd.setTrainingConfig(trainingConfig);

        // Log DSP subsystem flags before training begins.
        log.info("DSP flags: dspAutoCompileEnabled={}, dspNativeAutoCompileEnabled={}",
                sd.isDspAutoCompileEnabled(), sd.isDspNativeAutoCompileEnabled());

        // Fixed-batch synthetic dataset: 16 samples, 128 features, 10-class one-hot labels.
        INDArray features = Nd4j.randn(DataType.FLOAT, 16, 128);
        INDArray labels   = Nd4j.zeros(DataType.FLOAT, 16, 10);
        for (int i = 0; i < 16; i++) {
            labels.putScalar(new int[]{i, i % 10}, 1.0f);
        }
        DataSet ds = new DataSet(features, labels);

        log.info("Starting 10 DSP training steps (batch=16, features=128, classes=10)...");

        for (int step = 0; step < 10; step++) {
            long t0 = System.currentTimeMillis();
            sd.fit(new SingletonDataSetIterator(ds));
            long elapsedMs = System.currentTimeMillis() - t0;

            DspHandle dsp = sd.dsp();
            boolean compiled   = dsp != null && dsp.isCompiled();
            PlanPhase phase    = dsp != null ? dsp.planPhase() : null;
            String  phaseName  = phase != null ? phase.name() : "N/A";
            long    replayed   = dsp != null ? dsp.lastExecSegmentsReplayed()  : -1L;
            long    slotBySlot = dsp != null ? dsp.lastExecSegmentsSlotBySlot(): -1L;
            long    total      = dsp != null ? dsp.lastExecSegmentsTotal()      : -1L;

            log.info("  step={} time={}ms compiled={} phase={} segments=[replayed={}, slotBySlot={}, total={}]",
                    String.format("%2d", step),
                    String.format("%5d", elapsedMs),
                    compiled,
                    phaseName,
                    replayed, slotBySlot, total);
        }

        // Final DspHandle metrics after training.
        DspHandle finalDsp = sd.dsp();
        if (finalDsp != null) {
            log.info("");
            log.info("DspHandle metrics after training:");
            log.info("  totalSlots              : {}", finalDsp.totalSlots());
            log.info("  numSegments             : {}", finalDsp.numSegments());
            log.info("  numCapturedGraphSegments: {}", finalDsp.numCapturedGraphSegments());
            log.info("  pointersStable          : {}", finalDsp.pointersStable());
            log.info("  totalGraphReplays       : {}", finalDsp.totalGraphReplays());
        }

        log.info("");
        log.info("DSP handles the full training graph including LoRA adapter layers");
        log.info("Finished.");
    }

    // =========================================================================
    // Helper: build the base 3-layer MLP
    // =========================================================================
    private static SameDiff buildBaseMLP() {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 128);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 10);

        // Layer 1: 128 -> 256
        SDVariable w1 = sd.var("layer1.weight", Nd4j.randn(DataType.FLOAT, 256, 128).muli(0.01));
        SDVariable b1 = sd.var("layer1.bias",   Nd4j.zeros(DataType.FLOAT, 256));
        SDVariable h1 = sd.nn.relu(sd.mmul(input, w1.t()).add(b1), 0);

        // Layer 2: 256 -> 64
        SDVariable w2 = sd.var("layer2.weight", Nd4j.randn(DataType.FLOAT, 64, 256).muli(0.01));
        SDVariable b2 = sd.var("layer2.bias",   Nd4j.zeros(DataType.FLOAT, 64));
        SDVariable h2 = sd.nn.relu(sd.mmul(h1, w2.t()).add(b2), 0);

        // Layer 3: 64 -> 10 (logits)
        SDVariable w3 = sd.var("layer3.weight", Nd4j.randn(DataType.FLOAT, 10, 64).muli(0.01));
        SDVariable b3 = sd.var("layer3.bias",   Nd4j.zeros(DataType.FLOAT, 10));
        SDVariable logits = sd.mmul(h2, w3.t()).add(b3);
        logits.rename("logits");

        // Cross-entropy loss
        SDVariable loss = sd.loss.softmaxCrossEntropy("loss", label, logits, null);

        return sd;
    }

    // =========================================================================
    // Helper: count all VARIABLE parameters
    // =========================================================================
    private static long countParameters(SameDiff sd) {
        long total = 0;
        for (SDVariable v : sd.variables()) {
            if (v.getVariableType() == VariableType.VARIABLE) {
                long[] shape = v.getShape();
                if (shape != null) {
                    long n = 1;
                    for (long d : shape) n *= d;
                    total += n;
                }
            }
        }
        return total;
    }

    // =========================================================================
    // Helper: build fake adapter weight tensors (for cache demo)
    // =========================================================================
    private static Map<String, INDArray> buildFakeAdapterWeights(
            List<String> targetModules, int rank) {
        Map<String, INDArray> weights = new HashMap<>();
        // Approximate the LoRA A and B matrix shapes for each target module
        int[][] approximateShapes = {
            {rank, 128},   // layer1 A: [r, inFeatures]
            {256, rank},   // layer1 B: [outFeatures, r]
            {rank, 256},   // layer2 A
            {64,  rank},   // layer2 B
            {rank, 64},    // layer3 A
            {10,  rank}    // layer3 B
        };
        int idx = 0;
        for (String module : targetModules) {
            String safeName = module.replace(".", "_");
            int[] aShape = approximateShapes[idx * 2];
            int[] bShape = approximateShapes[idx * 2 + 1];
            weights.put(safeName + "_lora_A", Nd4j.randn(DataType.FLOAT, aShape[0], aShape[1]).muli(0.02));
            weights.put(safeName + "_lora_B", Nd4j.zeros(DataType.FLOAT, bShape[0], bShape[1]));
            idx++;
        }
        return weights;
    }

    // =========================================================================
    // Helper: rough LoRA trainable-parameter estimate
    // =========================================================================
    private static long computeLoraEstimate(int rank, int numModules) {
        // Each module: A [r, inDim] + B [outDim, r] ≈ 2 * r * avgDim(1024)
        return (long) numModules * 2L * rank * 1024;
    }

    // =========================================================================
    // Helper: print one comparison row
    // =========================================================================
    private static void logRow(String method, String type, long estParams,
                               String keyConfig, String description) {
        log.info(String.format("%-18s | %-10s | %10d | %-20s | %-45s",
                method, type, estParams, keyConfig, description));
    }
}
