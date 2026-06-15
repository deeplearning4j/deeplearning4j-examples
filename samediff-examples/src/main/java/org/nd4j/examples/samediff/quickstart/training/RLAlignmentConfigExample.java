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

import org.nd4j.autodiff.samediff.config.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * RL Alignment Configuration Examples.
 *
 * This example covers the full set of RL-based alignment training methods
 * available through the RLAlignmentConfig hierarchy in SameDiff:
 *
 * Methods demonstrated:
 * 1.  DPO     - Direct Preference Optimization (standard, IPO, RDPO variants)
 * 2.  GRPO    - Group Relative Policy Optimization (DeepSeek-R1 style)
 * 3.  PPO     - Proximal Policy Optimization (classic RLHF)
 * 4.  KTO     - Kahneman-Tversky Optimization (unpaired preferences)
 * 5.  ORPO    - Odds Ratio Preference Optimization (no reference model)
 * 6.  SimPO   - Simple Preference Optimization (no reference model)
 * 7.  DAPO    - Decoupled clip and dynamic sampling Policy Optimization
 * 8.  GSPO    - Group Stable Policy Optimization
 * 9.  DrGRPO  - Doctor GRPO (length-normalized, bias-subtracted)
 * 10. RewardModel - Bradley-Terry reward model training
 * 11. RLPipelineConfig - outer training loop configuration
 *
 * All configs extend RLAlignmentConfig and share common fields:
 *   - klCoefficient (KL divergence regularization weight)
 *   - useReferenceModel (frozen reference policy for KL)
 *   - normalizeAdvantages (z-score normalization)
 *   - maxGradNorm (gradient clipping)
 *   - learningRate
 *   - vocabSize / policyLogitVariable
 */
public class RLAlignmentConfigExample {
    private static final Logger log = LoggerFactory.getLogger(RLAlignmentConfigExample.class);

    public static void main(String[] args) {

        // =====================================================================
        // 1. DPO — Direct Preference Optimization
        // =====================================================================
        log.info("=== 1. DPO Config ===");

        // Standard DPO: uses paired (chosen, rejected) preference data.
        // Directly optimizes log-ratio of chosen vs rejected probabilities.
        // No reward model needed — simpler than PPO-based RLHF.
        DPOConfig dpoStandard = DPOConfig.builder()
                .policyLogitVariable("logits")      // Required: SameDiff variable name for logits
                .chosenVariable("chosen_ids")       // Token IDs for preferred response
                .rejectedVariable("rejected_ids")   // Token IDs for rejected response
                .beta(0.1)                          // KL regularization strength (higher = stay closer to reference)
                .labelSmoothing(0.0)                // 0.0 = standard DPO; >0 = robust DPO
                .klCoefficient(0.1)                 // Shared KL penalty weight
                .useReferenceModel(true)            // Frozen reference model for KL
                .normalizeAdvantages(true)
                .maxGradNorm(1.0)
                .learningRate(5e-7)
                .vocabSize(32000)
                .build();

        log.info("  Standard DPO:");
        log.info("    Method: {}", dpoStandard.getMethodName());
        log.info("    Beta: {}", dpoStandard.getBeta());
        log.info("    Variant: {}", dpoStandard.getVariant());

        // IPO variant: Identity Preference Optimization (more conservative)
        DPOConfig dpoIPO = DPOConfig.builder()
                .policyLogitVariable("logits")
                .chosenVariable("chosen_ids")
                .rejectedVariable("rejected_ids")
                .beta(0.5)                          // IPO uses higher beta typically
                .variant(DPOConfig.DPOVariant.IPO)  // IPO loss formulation
                .build();
        log.info("  IPO variant: {}", dpoIPO.getMethodName());

        // Static factory shortcut
        DPOConfig dpoFromFactory = DPOConfig.standard("logits", "chosen_ids", "rejected_ids");
        log.info("  From factory: {}", dpoFromFactory.getMethodName());

        // =====================================================================
        // 2. GRPO — Group Relative Policy Optimization
        // =====================================================================
        log.info("=== 2. GRPO Config ===");

        // GRPO generates G completions per prompt, scores them with a reward function,
        // then normalizes rewards within the group (z-score) as advantages.
        // Used in DeepSeek-R1 and similar reasoning models.
        // No separate value network needed (unlike PPO).
        GRPOConfig grpoConfig = GRPOConfig.builder()
                .policyLogitVariable("logits")
                .groupSize(8)               // G completions per prompt (DeepSeek-R1 uses 8-16)
                .clipEpsilon(0.2)           // PPO-style clipping range for surrogate objective
                .klPenalty(0.04)            // KL divergence penalty coefficient
                .maxNewTokens(512)          // Max tokens to generate per completion
                .klCoefficient(0.1)
                .useReferenceModel(true)
                .normalizeAdvantages(true)  // z-score normalize within group
                .learningRate(1e-6)
                .build();

        log.info("  GRPO:");
        log.info("    Method: {}", grpoConfig.getMethodName());
        log.info("    Group size: {}", grpoConfig.getGroupSize());
        log.info("    Clip epsilon: {}", grpoConfig.getClipEpsilon());

        // Factory shortcut
        GRPOConfig grpoSimple = GRPOConfig.standard("logits", 8);
        log.info("  GRPO from factory: group={}", grpoSimple.getGroupSize());

        // =====================================================================
        // 3. PPO — Proximal Policy Optimization
        // =====================================================================
        log.info("=== 3. PPO Config ===");

        // Classic PPO with actor-critic architecture.
        // Requires a separately trained reward model and a value head.
        // Most powerful but most complex RLHF method.
        PPOConfig ppoConfig = PPOConfig.builder()
                .policyLogitVariable("logits")
                .valueVariable("value_head")    // Value function output for baseline estimation
                .clipEpsilon(0.2)               // Surrogate objective clip range
                .valueLossCoeff(0.5)            // Weight for value function loss
                .entropyCoeff(0.01)             // Entropy bonus to encourage exploration
                .ppoEpochs(4)                   // Optimization epochs per PPO batch
                .gaeLambda(0.95)                // GAE (Generalized Advantage Estimation) lambda
                .maxNewTokens(256)
                .klCoefficient(0.05)
                .useReferenceModel(true)
                .normalizeAdvantages(true)
                .maxGradNorm(1.0)
                .learningRate(1e-6)
                .build();

        log.info("  PPO:");
        log.info("    Method: {}", ppoConfig.getMethodName());
        log.info("    Value loss coefficient: {}", ppoConfig.getValueLossCoeff());
        log.info("    Entropy coefficient: {}", ppoConfig.getEntropyCoeff());
        log.info("    PPO epochs: {}", ppoConfig.getPpoEpochs());
        log.info("    GAE lambda: {}", ppoConfig.getGaeLambda());

        // =====================================================================
        // 4. KTO — Kahneman-Tversky Optimization
        // =====================================================================
        log.info("=== 4. KTO Config ===");

        // KTO works with UNPAIRED preference data (only needs good OR bad labels,
        // not matched chosen/rejected pairs). Based on prospect theory loss aversion.
        // Useful when paired preferences are hard to collect.
        KTOConfig ktoConfig = KTOConfig.builder()
                .policyLogitVariable("logits")
                .desirabilityVariable("desirable")  // 1=good/desirable, 0=bad/undesirable
                .betaDesirable(0.1)                 // KL constraint for desired responses
                .betaUndesirable(0.1)               // KL constraint for undesired responses
                .lossAversion(1.5)                  // Penalize bad responses more than rewarding good ones (>1)
                .klCoefficient(0.1)
                .useReferenceModel(true)
                .normalizeAdvantages(false)         // KTO uses its own normalization
                .learningRate(5e-7)
                .build();

        log.info("  KTO:");
        log.info("    Method: {}", ktoConfig.getMethodName());
        log.info("    Loss aversion: {}", ktoConfig.getLossAversion());
        log.info("    Beta desirable: {}", ktoConfig.getBetaDesirable());

        // Factory shortcut
        KTOConfig ktoSimple = KTOConfig.standard("logits", "desirable");
        log.info("  KTO from factory: {}", ktoSimple.getMethodName());

        // =====================================================================
        // 5. ORPO — Odds Ratio Preference Optimization
        // =====================================================================
        log.info("=== 5. ORPO Config ===");

        // ORPO combines SFT loss and preference alignment in one stage.
        // No reference model needed — uses current policy as implicit reference.
        // Simpler than DPO (single training stage) but may be less stable.
        ORPOConfig orpoConfig = ORPOConfig.builder()
                .policyLogitVariable("logits")
                .chosenVariable("chosen_ids")
                .rejectedVariable("rejected_ids")
                .orpoLambda(0.5)            // Weight of odds ratio loss vs SFT loss
                .useReferenceModel(false)   // No reference model needed
                .normalizeAdvantages(false)
                .learningRate(5e-7)
                .build();

        log.info("  ORPO:");
        log.info("    Method: {}", orpoConfig.getMethodName());
        log.info("    Lambda: {}", orpoConfig.getOrpoLambda());
        log.info("    Uses reference model: {}", orpoConfig.isUseReferenceModel());

        // =====================================================================
        // 6. SimPO — Simple Preference Optimization
        // =====================================================================
        log.info("=== 6. SimPO Config ===");

        // SimPO uses sequence-average log probabilities (length-normalized).
        // No reference model. Often matches DPO quality at lower complexity.
        SimPOConfig simpoConfig = SimPOConfig.builder()
                .policyLogitVariable("logits")
                .chosenVariable("chosen_ids")
                .rejectedVariable("rejected_ids")
                .beta(2.5)                  // Temperature for reward scaling
                .gamma(1.0)                 // Target reward margin between chosen and rejected
                .useReferenceModel(false)
                .normalizeAdvantages(false)
                .learningRate(1e-6)
                .build();

        log.info("  SimPO:");
        log.info("    Method: {}", simpoConfig.getMethodName());
        log.info("    Beta: {}", simpoConfig.getBeta());
        log.info("    Gamma: {}", simpoConfig.getGamma());

        // Variant with explicit vocab size for log prob computation
        SimPOConfig simpoWithVocab = SimPOConfig.standard("logits", "chosen_ids", "rejected_ids", 32000);
        log.info("  SimPO (with vocabSize={}): {}", simpoWithVocab.getVocabSize(), simpoWithVocab.getMethodName());

        // =====================================================================
        // 7. DAPO — Decoupled clip and dynamic sampling Policy Optimization
        // =====================================================================
        log.info("=== 7. DAPO Config ===");

        // DAPO improves on GRPO with:
        //   - Asymmetric clipping (different bounds for high/low probability ratios)
        //   - Dynamic sampling to filter trivial examples
        //   - Overlong filtering to skip sequences exceeding max length
        DAPOConfig dapoConfig = DAPOConfig.builder()
                .policyLogitVariable("logits")
                .clipEpsilonLow(0.1)        // Lower clip bound (tighter for high-ratio responses)
                .clipEpsilonHigh(0.28)       // Upper clip bound (looser for exploration)
                .tokenLevelKL(true)          // Per-token KL divergence (vs sequence-level)
                .dynamicSampling(true)       // Filter easy/hard examples dynamically
                .overlongFiltering(true)     // Exclude sequences over maxNewTokens
                .groupSize(8)
                .maxNewTokens(512)
                .klCoefficient(0.04)
                .useReferenceModel(true)
                .normalizeAdvantages(true)
                .learningRate(1e-6)
                .build();

        log.info("  DAPO:");
        log.info("    Method: {}", dapoConfig.getMethodName());
        log.info("    Clip range: [{}, {}]", dapoConfig.getClipEpsilonLow(), dapoConfig.getClipEpsilonHigh());
        log.info("    Token-level KL: {}", dapoConfig.isTokenLevelKL());
        log.info("    Dynamic sampling: {}", dapoConfig.isDynamicSampling());

        // =====================================================================
        // 8. GSPO — Group Stable Policy Optimization
        // =====================================================================
        log.info("=== 8. GSPO Config ===");

        // GSPO adds stability improvements to GRPO:
        //   - Importance-weighted advantage estimation
        //   - Stability coefficient for numerical robustness
        GRPOConfig gspoBase = GRPOConfig.builder()
                .policyLogitVariable("logits")
                .groupSize(8)
                .clipEpsilon(0.2)
                .klPenalty(0.04)
                .maxNewTokens(512)
                .build();

        GSPOConfig gspoConfig = GSPOConfig.builder()
                .policyLogitVariable("logits")
                .stabilityCoeff(0.1)                // Numerical stability for advantage normalization
                .importanceWeightedAdvantage(true)   // Weight advantages by policy-reference ratio
                .groupSize(8)
                .clipEpsilon(0.2)
                .maxNewTokens(512)
                .klCoefficient(0.04)
                .useReferenceModel(true)
                .normalizeAdvantages(true)
                .learningRate(1e-6)
                .build();

        log.info("  GSPO:");
        log.info("    Method: {}", gspoConfig.getMethodName());
        log.info("    Stability coefficient: {}", gspoConfig.getStabilityCoeff());
        log.info("    Importance weighting: {}", gspoConfig.isImportanceWeightedAdvantage());

        // =====================================================================
        // 9. DrGRPO — Doctor GRPO
        // =====================================================================
        log.info("=== 9. DrGRPO Config ===");

        // DrGRPO (Dr. GRPO) fixes reward over-optimization issues in GRPO via:
        //   - Length normalization (divide by sequence length to prevent length gaming)
        //   - Baseline subtraction (subtract mean reward for variance reduction)
        DrGRPOConfig drGrpoConfig = DrGRPOConfig.builder()
                .policyLogitVariable("logits")
                .lengthNormalization(true)   // Divide reward by sequence length (prevent length hacking)
                .baselineSubtraction(true)   // Subtract mean group reward (reduce variance)
                .groupSize(8)
                .clipEpsilon(0.2)
                .maxNewTokens(512)
                .klCoefficient(0.04)
                .useReferenceModel(true)
                .normalizeAdvantages(true)
                .learningRate(1e-6)
                .build();

        log.info("  DrGRPO:");
        log.info("    Method: {}", drGrpoConfig.getMethodName());
        log.info("    Length normalization: {}", drGrpoConfig.isLengthNormalization());
        log.info("    Baseline subtraction: {}", drGrpoConfig.isBaselineSubtraction());

        // =====================================================================
        // 10. Reward Model Training
        // =====================================================================
        log.info("=== 10. Reward Model Config ===");

        // Train a Bradley-Terry reward model from preference pairs.
        // The reward model scores (chosen, rejected) pairs and is used in PPO.
        RewardModelConfig rewardConfig = RewardModelConfig.builder()
                .policyLogitVariable("logits")
                .chosenVariable("chosen_ids")
                .rejectedVariable("rejected_ids")
                .rewardOutputVariable("reward_score")
                .rewardType(RewardModelConfig.RewardType.BRADLEY_TERRY)
                .margin(0.5)                // Minimum reward margin between chosen and rejected
                .useReferenceModel(false)   // Reward model doesn't need a reference
                .learningRate(1e-5)
                .build();

        log.info("  Reward Model:");
        log.info("    Method: {}", rewardConfig.getMethodName());
        log.info("    Type: {}", rewardConfig.getRewardType());
        log.info("    Margin: {}", rewardConfig.getMargin());

        RewardModelConfig rewardFromFactory = RewardModelConfig.bradleyTerry(
                "logits", "chosen_ids", "rejected_ids", "reward_score");
        log.info("  From factory: {}", rewardFromFactory.getMethodName());

        // =====================================================================
        // 11. RLPipelineConfig — Outer Training Loop
        // =====================================================================
        log.info("=== 11. RLPipelineConfig (Outer Loop) ===");

        // RLPipelineConfig wraps any RLAlignmentConfig and configures the outer
        // training loop: epochs, gradient accumulation, logging, etc.
        LoraConfig peftAdapter = LoraConfig.defaultTransformer();

        RLPipelineConfig pipelineConfig = RLPipelineConfig.builder()
                .numEpochs(1)
                .learningRate(5e-7)         // Typically lower than SFT
                .warmupRatio(0.1)           // 10% of steps for LR warmup
                .gradientAccumulationSteps(2)
                .computeDataType(DataType.BFLOAT16)
                .weightDecay(0.0)           // No weight decay for alignment (common practice)
                .peftConfig(peftAdapter)    // Attach LoRA adapter (optional)
                .logEveryNSteps(10)
                .evaluateEveryNSteps(100)
                .maxSteps(-1)               // -1 = run full epochs
                .build();

        pipelineConfig.validate();
        log.info("  RLPipelineConfig:");
        log.info("    Epochs: {}", pipelineConfig.getNumEpochs());
        log.info("    Learning rate: {}", pipelineConfig.getLearningRate());
        log.info("    Warmup ratio: {}", pipelineConfig.getWarmupRatio());
        log.info("    Gradient accumulation: {}", pipelineConfig.getGradientAccumulationSteps());
        log.info("    PEFT: {}", pipelineConfig.getPeftConfig() != null ? pipelineConfig.getPeftConfig().getPeftType() : "none");

        RLPipelineConfig defaultPipeline = RLPipelineConfig.defaults();
        log.info("  Default pipeline LR: {}", defaultPipeline.getLearningRate());

        // =====================================================================
        // SUMMARY: Choosing an RL Alignment Method
        // =====================================================================
        log.info("=== RL Alignment Method Selection Guide ===");
        log.info("  +----------+-------------+------------------+----------------+");
        log.info("  | Method   | Ref. Model  | Data Type        | Complexity     |");
        log.info("  +----------+-------------+------------------+----------------+");
        log.info("  | DPO      | Yes         | Paired prefs     | Low            |");
        log.info("  | GRPO     | Yes         | Prompts + scorer | Medium         |");
        log.info("  | PPO      | Yes         | Prompts + RM     | High           |");
        log.info("  | KTO      | Yes         | Unpaired prefs   | Low            |");
        log.info("  | ORPO     | No          | Paired prefs     | Lowest         |");
        log.info("  | SimPO    | No          | Paired prefs     | Low            |");
        log.info("  | DAPO     | Yes         | Prompts + scorer | Medium         |");
        log.info("  | GSPO     | Yes         | Prompts + scorer | Medium         |");
        log.info("  | DrGRPO   | Yes         | Prompts + scorer | Medium         |");
        log.info("  +----------+-------------+------------------+----------------+");
        log.info("  RM = Reward Model, prefs = preference pairs (chosen/rejected)");
        log.info("**************** RL Alignment Config Example finished ********************");
    }
}
