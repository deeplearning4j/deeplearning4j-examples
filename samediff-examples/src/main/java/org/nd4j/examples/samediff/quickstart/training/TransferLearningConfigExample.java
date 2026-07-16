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

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.TransferLearning;
import org.nd4j.autodiff.samediff.config.*;
import org.nd4j.autodiff.samediff.peft.PeftModel;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;

/**
 * Transfer Learning and Fine-Tuning Configuration Examples.
 *
 * This example demonstrates the SameDiff TransferLearning API and the supporting
 * FineTuneConfiguration / VariableGroup classes for discriminative fine-tuning.
 *
 * Topics covered:
 * 1. FineTuneConfiguration — per-variable learning rates and regularization
 * 2. VariableGroup — group variables by prefix/pattern for discriminative fine-tuning
 * 3. TransferLearning.Builder — freeze layers, attach PEFT adapters, build fine-tune models
 * 4. TransferLearning static factories — quick LoRA/prefix/adapter model creation
 * 5. GradientCheckpointConfig — activation checkpointing for memory efficiency
 * 6. ContinuedPretrainingConfig — full pretraining continuation configuration
 * 7. LR Schedule integration — cosine/exponential/step schedules in fine-tuning
 *
 * Key pattern for discriminative fine-tuning:
 *   Lower layers get smaller LR multipliers (e.g., 0.1x)
 *   Upper layers get higher LR multipliers (e.g., 1.0x)
 *   This prevents catastrophic forgetting while allowing task adaptation.
 */
public class TransferLearningConfigExample {
    private static final Logger log = LoggerFactory.getLogger(TransferLearningConfigExample.class);

    public static void main(String[] args) {

        // Build a simple base model for demonstration
        SameDiff baseModel = buildSimpleModel();

        // =====================================================================
        // 1. FineTuneConfiguration — Discriminative Fine-Tuning
        // =====================================================================
        log.info("=== 1. FineTuneConfiguration ===");

        // FineTuneConfiguration allows different learning rates and regularization
        // per variable or variable group. Use it for discriminative fine-tuning
        // where lower layers get smaller LR than upper layers.
        FineTuneConfiguration fineTuneConfig = FineTuneConfiguration.builder()
                // Default updater for all variables not otherwise specified
                .updater(new Adam(1e-4))
                // Per-variable updater overrides
                .updater("output_weights", new Adam(1e-3))     // Output layer: higher LR
                // Per-variable L2 regularization
                .l2(0.0001)
                .l2("output_weights", 0.001)
                // Freeze specific variables (no gradient flow)
                .freeze("embedding_weights")                   // Freeze embeddings entirely
                // Freeze by prefix (all variables starting with "frozen_")
                .freezePrefix("frozen_")
                // Freeze by containing pattern
                .freezeContaining("layer_0")                   // Freeze first layer
                // Freeze by regex pattern
                .freezeMatching("layer_[01]_.*")               // Freeze layers 0 and 1
                .build();

        log.info("  FineTuneConfiguration:");
        log.info("    Default updater: {}", fineTuneConfig.getDefaultUpdater());
        log.info("    Frozen variables: {}", fineTuneConfig.getFrozenVariables());
        log.info("    isFrozen('embedding_weights'): {}", fineTuneConfig.isFrozen("embedding_weights"));
        log.info("    isFrozen('output_weights'): {}", fineTuneConfig.isFrozen("output_weights"));
        log.info("    Updater for 'output_weights': {}", fineTuneConfig.getUpdaterForVariable("output_weights"));

        // =====================================================================
        // 2. VariableGroup — Layer-wise Learning Rate Decay
        // =====================================================================
        log.info("=== 2. VariableGroup (Discriminative Fine-Tuning) ===");

        // VariableGroup groups variables by patterns with a shared LR multiplier.
        // This implements discriminative fine-tuning (different LR per layer group).
        VariableGroup lowerLayers = VariableGroup.builder()
                .name("lower_layers")
                .learningRateMultiplier(0.1)        // 10% of base LR for lower layers
                .addVariablePatterns("layer_[0-3]_.*")   // regex matching layers 0-3
                .build();

        VariableGroup middleLayers = VariableGroup.builder()
                .name("middle_layers")
                .learningRateMultiplier(0.3)        // 30% of base LR
                .addVariablePatterns("layer_[4-7]_.*")
                .build();

        VariableGroup upperLayers = VariableGroup.builder()
                .name("upper_layers")
                .learningRateMultiplier(1.0)        // Full LR for upper layers
                .addVariablePatterns("layer_[89]_.*", "layer_1[0-9]_.*")
                .build();

        // Variable group with per-group updater (overrides base updater entirely for group)
        VariableGroup outputHead = VariableGroup.builder()
                .name("output_head")
                .updater(new Adam(5e-4))             // Completely independent updater for output head
                .addVariablePatterns("output_.*")
                .build();

        log.info("  VariableGroups:");
        log.info("    lower_layers:  LR multiplier = {}", lowerLayers.getLearningRateMultiplier());
        log.info("    middle_layers: LR multiplier = {}", middleLayers.getLearningRateMultiplier());
        log.info("    upper_layers:  LR multiplier = {}", upperLayers.getLearningRateMultiplier());
        log.info("    output_head:   custom updater = {}", outputHead.hasUpdater());
        log.info("    lowerLayers.matches('layer_2_weight'): {}", lowerLayers.matches("layer_2_weight"));
        log.info("    lowerLayers.matches('layer_8_weight'): {}", lowerLayers.matches("layer_8_weight"));

        // Build FineTuneConfiguration with variable groups
        FineTuneConfiguration groupedConfig = FineTuneConfiguration.builder()
                .updater(new Adam(1e-4))
                .variableGroup(lowerLayers)
                .variableGroup(middleLayers)
                .variableGroup(upperLayers)
                .variableGroup(outputHead)
                .freeze("embedding_weights")
                .build();

        log.info("  Grouped config updater for 'layer_2_w': {}",
                groupedConfig.getUpdaterForVariable("layer_2_w"));

        // =====================================================================
        // 3. TransferLearning.Builder — Building Fine-Tune Models
        // =====================================================================
        log.info("=== 3. TransferLearning Builder ===");

        // The TransferLearning.Builder provides a fluent API to:
        //   - Set FineTuneConfiguration (per-variable LR, regularization)
        //   - Freeze/unfreeze specific variables
        //   - Reinitialize specific variables with new values
        //   - Attach PEFT adapters (LoRA, prompt tuning, etc.)
        //   - Build the modified model or a PeftModel wrapper

        // Basic fine-tuning: freeze all but the last layer
        SameDiff fineTuneModel = new TransferLearning.Builder(baseModel)
                .fineTuneConfiguration(
                    FineTuneConfiguration.builder()
                        .updater(new Adam(1e-4))
                        .freeze("w1", "b1")    // Freeze first-layer parameters
                        .build()
                )
                .build();

        log.info("  Built fine-tune model (frozen: w1, b1)");

        // LoRA fine-tuning via TransferLearning.Builder
        PeftModel loraModel = new TransferLearning.Builder(baseModel)
                .lora(16, 32, Arrays.asList("w2"))   // rank=16, alpha=32, target "w2"
                .buildPeft();

        log.info("  LoRA PeftModel:");
        log.info("    Trainable params: {}", loraModel.getTrainableParameterCount());
        log.info("    Total params: {}", loraModel.getTotalParameterCount());
        log.info("    Trainable %: {}", String.format("%.4f%%", loraModel.getTrainablePercentage()));

        // Prompt tuning via TransferLearning.Builder
        // (numVirtualTokens, tokenEmbeddingDim)
        PeftModel promptModel = new TransferLearning.Builder(baseModel)
                .promptTuning(20, 64)
                .buildPeft();
        log.info("  Prompt tuning PeftModel: trainable={}",
                promptModel.getTrainableParameterCount());

        // =====================================================================
        // 4. TransferLearning Static Factories
        // =====================================================================
        log.info("=== 4. TransferLearning Static Factories ===");

        // Quick LoRA model with default transformer config
        PeftModel loraDefault = TransferLearning.loraModel(baseModel, Arrays.asList("w1", "w2"));
        log.info("  loraModel() factory: trainable={}", loraDefault.getTrainableParameterCount());

        // LoRA with custom rank
        PeftModel loraRank32 = TransferLearning.loraModel(baseModel, 32, Arrays.asList("w1", "w2"));
        log.info("  loraModel(rank=32): trainable={}", loraRank32.getTrainableParameterCount());

        // LoRA with full LoraConfig (provides rank AND alpha control)
        LoraConfig customLora = LoraConfig.builder()
                .r(16).loraAlpha(32).loraDropout(0.05)
                .targetModules(Arrays.asList("w2"))
                .taskType(TaskType.CAUSAL_LM)
                .build();
        PeftModel loraCustom = TransferLearning.loraModel(baseModel, customLora);
        log.info("  loraModel(LoraConfig, rank=16, alpha=32): trainable={}", loraCustom.getTrainableParameterCount());

        // Prompt tuning factory (numVirtualTokens, tokenEmbeddingDim)
        PeftModel promptFactory = TransferLearning.promptTuningModel(baseModel, 20, 64);
        log.info("  promptTuningModel(20 tokens, 64 dim): trainable={}",
                promptFactory.getTrainableParameterCount());

        // Prefix tuning factory (numTokens, numLayers, hiddenSize)
        PeftModel prefixFactory = TransferLearning.prefixTuningModel(baseModel, 20, 2, 32);
        log.info("  prefixTuningModel(20 tokens, 2 layers, 32 hidden): trainable={}",
                prefixFactory.getTrainableParameterCount());

        // Adapter factory (adapterSize, hiddenSize, numLayers)
        PeftModel adapterFactory = TransferLearning.adapterModel(baseModel, 16, 64, 2);
        log.info("  adapterModel(adapterSize=16, hiddenSize=64, layers=2): trainable={}",
                adapterFactory.getTrainableParameterCount());

        // =====================================================================
        // 5. GradientCheckpointConfig — Memory-efficient Training
        // =====================================================================
        log.info("=== 5. Gradient Checkpoint Config ===");

        // Gradient checkpointing (activation checkpointing) recomputes activations
        // during backward pass instead of storing them, trading compute for memory.
        // Reduces memory by ~sqrt(N) for sequential models.

        // Sqrt-N checkpointing: checkpoint every sqrt(N) layers automatically
        GradientCheckpointConfig sqrtN = GradientCheckpointConfig.sqrtN();
        log.info("  sqrtN config: strategy={}, isSqrtN={}", sqrtN.getStrategy(), sqrtN.isSqrtN());

        // Checkpoint every N layers manually
        GradientCheckpointConfig everyN = GradientCheckpointConfig.everyN(4);
        log.info("  everyN(4) config: resolveInterval(32 layers)={}", everyN.resolveInterval(32));

        // Manual checkpointing: specify exactly which variables to checkpoint
        java.util.Set<String> checkpointVars = new java.util.HashSet<>(
                Arrays.asList("layer_8_out", "layer_16_out", "layer_24_out")
        );
        GradientCheckpointConfig manual = GradientCheckpointConfig.manual(checkpointVars);
        log.info("  manual config: isManual={}, checkpointVars={}",
                manual.isManual(), manual.getCheckpointVariables());

        // Offload activations to host memory (for very large models)
        GradientCheckpointConfig asyncOffload = GradientCheckpointConfig.asyncOffload(2);
        log.info("  asyncOffload(prefetch=2): strategy={}, isAnyOffload={}",
                asyncOffload.getStrategy(), asyncOffload.isAnyOffload());
        log.info("  asyncOffload config: isAsyncOffload={}", asyncOffload.isAsyncOffload());

        // Full builder API
        GradientCheckpointConfig customCheckpoint = GradientCheckpointConfig.builder()
                .strategy(GradientCheckpointConfig.CheckpointStrategy.RECOMPUTE)
                .checkpointEveryN(4)            // Checkpoint every 4 layers
                .maxCheckpointMemoryMB(4096)    // Abort if >4GB activation memory
                .build();
        log.info("  Custom checkpoint: strategy={}, everyN={}, maxMemMB={}",
                customCheckpoint.getStrategy(),
                customCheckpoint.getCheckpointEveryN(),
                customCheckpoint.getMaxCheckpointMemoryMB());

        // =====================================================================
        // 6. ContinuedPretrainingConfig — Full Pretraining Continuation
        // =====================================================================
        log.info("=== 6. ContinuedPretraining Config ===");

        // ContinuedPretrainingConfig configures resuming pretraining on domain-specific data.
        // Suitable for: domain adaptation, knowledge injection, language extension.
        ContinuedPretrainingConfig contPretrain = ContinuedPretrainingConfig.builder()
                .chunkSize(2048)                // Sequence length per chunk (tokens)
                .chunkOverlap(128)              // Overlap between consecutive chunks
                .learningRate(5e-5)             // Start LR (much lower than from-scratch pretraining)
                .warmupRatio(0.1)               // 10% warmup
                .minLearningRate(1e-6)          // Floor LR at end of cosine decay
                .gradientAccumulationSteps(4)
                .computeDataType(DataType.BFLOAT16)
                .gradientCheckpointConfig(sqrtN) // Memory optimization for long sequences
                .useLoRA(true)                  // Apply LoRA to limit update scope
                .loraRank(16)
                .loraAlpha(32)
                .weightDecay(0.01)
                .maxGradNorm(1.0)
                .numEpochs(1)                   // Typically 1 epoch for continued pretraining
                .build();

        contPretrain.validate();
        log.info("  ContinuedPretraining config:");
        log.info("    Chunk size: {} tokens", contPretrain.getChunkSize());
        log.info("    Chunk overlap: {} tokens", contPretrain.getChunkOverlap());
        log.info("    LR range: {} -> {}", contPretrain.getLearningRate(), contPretrain.getMinLearningRate());
        log.info("    Use LoRA: {}, rank={}, alpha={}",
                contPretrain.isUseLoRA(), contPretrain.getLoraRank(), contPretrain.getLoraAlpha());
        log.info("    Gradient checkpoint: {}", contPretrain.getGradientCheckpointConfig().getStrategy());

        // =====================================================================
        // 7. PeftModel API — Training and Inference
        // =====================================================================
        log.info("=== 7. PeftModel API ===");

        // Use the LoRA model created earlier to demonstrate the PeftModel API
        PeftModel peft = loraModel;

        log.info("  PeftModel summary:\n{}", peft.getSummary());
        peft.printTrainableParameters();

        log.info("  Key PeftModel methods:");
        log.info("    peftModel.mergeAndUnload()   - merge LoRA into base weights, return SameDiff");
        log.info("    peftModel.saveAdapter(file)  - save only adapter weights (small file)");
        log.info("    peftModel.disableAdapter()   - run base model only (no adapter)");
        log.info("    peftModel.enableAdapter()    - re-enable adapter for inference");
        log.info("    peftModel.getAdapters()      - map of named adapters (multi-adapter)");
        log.info("    peftModel.getMergedWeight(v) - get effective weight W = W0 + BA for variable v");

        log.info("**************** Transfer Learning Config Example finished ********************");
    }

    /**
     * Build a minimal SameDiff model for demonstration purposes.
     */
    private static SameDiff buildSimpleModel() {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 64);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 10);

        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, 64, 32).mul(0.02));
        SDVariable b1 = sd.var("b1", Nd4j.zeros(DataType.FLOAT, 32));
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, 32, 10).mul(0.02));
        SDVariable b2 = sd.var("b2", Nd4j.zeros(DataType.FLOAT, 10));

        SDVariable h = sd.nn().relu(input.mmul(w1).add(b1), 0);
        SDVariable out = sd.nn().softmax("output", h.mmul(w2).add(b2));
        sd.loss().softmaxCrossEntropy("loss", label, out, null);

        return sd;
    }
}
