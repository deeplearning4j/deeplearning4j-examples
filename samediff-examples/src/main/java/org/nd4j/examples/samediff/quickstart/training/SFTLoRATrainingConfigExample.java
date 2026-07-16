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

import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.config.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.dataset.curation.format.ChatTemplate;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * SameDiff Training Configuration Examples: SFT, LoRA, GRPO, DPO, and Mixed Precision.
 *
 * This example demonstrates the new SameDiff training infrastructure for fine-tuning
 * and alignment of large language models, including:
 *
 * 1. TrainingConfig - base training configuration with updaters and mixed precision
 * 2. SFTConfig - Supervised Fine-Tuning with response-token masking
 * 3. LoraConfig - Low-Rank Adaptation for parameter-efficient fine-tuning
 * 4. QLoraConfig - Quantized LoRA for memory-efficient fine-tuning
 * 5. GRPOConfig - Group Relative Policy Optimization for RLHF
 * 6. DPOConfig - Direct Preference Optimization for alignment
 *
 * These configurations are builder-pattern objects that define training hyperparameters.
 * They are used with the SFTTrainingPipeline and RLAlignmentPipeline to execute training.
 */
public class SFTLoRATrainingConfigExample {
    private static final Logger log = LoggerFactory.getLogger(SFTLoRATrainingConfigExample.class);

    public static void main(String[] args) {

        // =====================================================================
        // 1. Basic TrainingConfig with mixed precision
        // =====================================================================
        log.info("=== 1. Basic TrainingConfig ===");

        TrainingConfig trainingConfig = TrainingConfig.builder()
                .updater(new Adam(1e-3))
                .l2(0.0001)
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                // Mixed precision: BF16 compute with FP32 master weights
                // Recommended for Ampere+ GPUs; use FLOAT16 for older NVIDIA GPUs
                .mixedPrecisionBfloat16()
                .gradientAccumulationSteps(4)
                .build();

        log.info("  Updater: {}", trainingConfig.getUpdater());
        log.info("  Mixed precision: {}", trainingConfig.isMixedPrecision());
        log.info("  Gradient accumulation: {}", trainingConfig.isGradientAccumulationEnabled());

        // =====================================================================
        // 2. LoRA Configuration (Parameter-Efficient Fine-Tuning)
        // =====================================================================
        log.info("=== 2. LoRA Config ===");

        // LoRA decomposes weight updates into low-rank matrices: W = W0 + BA
        // This reduces trainable parameters by orders of magnitude while
        // maintaining comparable performance to full fine-tuning.
        LoraConfig loraConfig = LoraConfig.builder()
                .r(16)                              // Rank of low-rank matrices (8-64 typical)
                .loraAlpha(32)                      // Scaling factor: effective scale = alpha/r
                .loraDropout(0.05)                  // Dropout on LoRA output
                .targetModules(Arrays.asList(        // Which weight matrices to adapt
                        "q_proj", "k_proj",          // Query and Key projections
                        "v_proj", "o_proj"           // Value and Output projections
                ))
                .taskType(TaskType.CAUSAL_LM)       // Task type (CAUSAL_LM, SEQ_2_SEQ_LM, etc.)
                .bias("none")                       // "none", "all", or "lora_only"
                .build();

        log.info("  LoRA rank: {}", loraConfig.getR());
        log.info("  Scaling factor: {}", loraConfig.getScaling());
        log.info("  Target modules: {}", loraConfig.getTargetModules());
        log.info("  Summary: {}", loraConfig.getSummary());

        // Factory shortcuts for common configurations:
        LoraConfig transformerDefault = LoraConfig.defaultTransformer();  // r=16, alpha=32
        LoraConfig allLinear = LoraConfig.allLinear(32);                  // All linear layers, r=32
        LoraConfig minimal = LoraConfig.minimal();                        // r=4, query+value only

        // =====================================================================
        // 3. SFT Configuration (Supervised Fine-Tuning)
        // =====================================================================
        log.info("=== 3. SFT Config ===");

        // SFT computes loss only on assistant response tokens, not on prompt tokens.
        // The InstructionDataFormatter produces character-level loss masks.
        SFTConfig sftConfig = SFTConfig.builder()
                .chatTemplate(ChatTemplate.CHATML)   // Chat format (CHATML, LLAMA2, etc.)
                .systemMessage("You are a helpful AI assistant.")
                .maxSeqLength(2048)                  // Max token sequence length
                .learningRate(2e-5)                  // Base learning rate
                .minLearningRate(0.0)                // Min LR at end of cosine decay
                .warmupRatio(0.03)                   // Fraction of steps for LR warmup
                .weightDecay(0.01)                   // AdamW weight decay
                .maxGradNorm(1.0)                    // Gradient clipping norm
                .numEpochs(3)
                .gradientAccumulationSteps(4)
                .computeDataType(DataType.BFLOAT16)  // BF16 compute precision
                .peftConfig(loraConfig)               // Attach LoRA (null = full fine-tune)
                .build();

        sftConfig.validate();
        log.info("  Chat template: {}", sftConfig.getChatTemplate());
        log.info("  Learning rate: {}", sftConfig.getLearningRate());
        log.info("  PEFT: {}", sftConfig.getPeftConfig() != null ? "LoRA" : "Full FT");

        // Factory shortcuts:
        SFTConfig defaultSFT = SFTConfig.defaultSFT();           // Full fine-tune defaults
        SFTConfig loraSFT = SFTConfig.loraDefaults(16);           // SFT with LoRA r=16
        SFTConfig qloraSFT = SFTConfig.qloraDefaults();           // SFT with QLoRA (4-bit)

        // =====================================================================
        // 4. QLoRA Configuration (Quantized LoRA)
        // =====================================================================
        log.info("=== 4. QLoRA Config ===");

        // QLoRA quantizes the base model to 4-bit (NF4) while applying LoRA adapters
        // in full precision, dramatically reducing memory usage.
        QLoraConfig qloraConfig = QLoraConfig.builder()
                .r(64)                               // Higher rank compensates for quantization
                .loraAlpha(16)
                .loraDropout(0.1)
                .targetModules(Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj"))
                .taskType(TaskType.CAUSAL_LM)
                .build();

        log.info("  QLoRA rank: {}", qloraConfig.getR());
        log.info("  PEFT type: {}", qloraConfig.getPeftType());

        // =====================================================================
        // 5. GRPO Configuration (Group Relative Policy Optimization)
        // =====================================================================
        log.info("=== 5. GRPO Config (RLHF) ===");

        // GRPO generates multiple completions per prompt, scores them with a reward
        // function, and uses z-score normalized advantages for policy gradient updates.
        // No separate reward model training needed (unlike PPO).
        GRPOConfig grpoConfig = GRPOConfig.builder()
                .groupSize(8)                        // Completions per prompt for advantage estimation
                .clipEpsilon(0.2)                    // PPO-style surrogate clipping range
                .klPenalty(0.01)                     // KL divergence penalty vs reference policy
                .maxNewTokens(256)                   // Max generation length per completion
                .build();

        log.info("  Group size: {}", grpoConfig.getGroupSize());
        log.info("  Clip epsilon: {}", grpoConfig.getClipEpsilon());
        log.info("  KL penalty: {}", grpoConfig.getKlPenalty());
        log.info("  Method: {}", grpoConfig.getMethodName());

        // =====================================================================
        // 6. DPO Configuration (Direct Preference Optimization)
        // =====================================================================
        log.info("=== 6. DPO Config ===");

        // DPO directly optimizes the policy from preference pairs (chosen vs rejected)
        // without training a separate reward model. Simpler than RLHF/PPO.
        DPOConfig dpoConfig = DPOConfig.builder()
                .beta(0.1)                           // KL constraint strength
                .labelSmoothing(0.0)                 // For robust DPO variant
                .build();

        log.info("  DPO beta: {}", dpoConfig.getBeta());
        log.info("  Method: {}", dpoConfig.getMethodName());

        log.info("**************** Training Config Examples finished ********************");
    }
}
