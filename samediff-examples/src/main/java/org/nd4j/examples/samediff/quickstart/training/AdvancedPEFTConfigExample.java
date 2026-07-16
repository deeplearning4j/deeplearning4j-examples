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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * Advanced PEFT (Parameter-Efficient Fine-Tuning) Configuration Examples.
 *
 * This example covers the full range of PEFT methods available in SameDiff beyond
 * basic LoRA, including:
 *
 * 1. AdaLoRA — Adaptive LoRA that dynamically allocates rank budget
 * 2. DoRA — Weight-Decomposed LoRA (magnitude + direction decomposition)
 * 3. IA3 — Infused Adapter by Inhibiting and Amplifying Inner Activations
 * 4. PrefixTuning — Prepend learnable prefix tokens to attention
 * 5. PromptTuning — Prepend learnable soft prompt embeddings
 *
 * Each method trades off trainable parameter count, memory usage, and performance.
 *
 * <h3>PEFT Method Comparison:</h3>
 * <pre>
 * Method         | Trainable Params | Memory  | Quality | Use Case
 * ---------------|-----------------|---------|---------|-------------------
 * Full Fine-Tune | 100%            | High    | Best    | Unlimited compute
 * LoRA           | ~0.1-1%         | Low     | Good    | General PEFT
 * QLoRA          | ~0.1-1%         | Lowest  | Good    | Memory-constrained
 * AdaLoRA        | ~0.1-1%         | Low     | Better  | Adaptive rank
 * DoRA           | ~0.1-1%         | Low     | Better  | LoRA + magnitude
 * IA3            | ~0.01%          | Lowest  | Fair    | Few-shot, minimal
 * PrefixTuning   | ~0.1%           | Low     | Good    | Multi-task
 * PromptTuning   | ~0.01%          | Lowest  | Fair    | Simple adaptation
 * </pre>
 *
 * <h3>RL Alignment Methods:</h3>
 * <pre>
 * Method | Description
 * -------|-----------------------------------------------------------
 * GRPO   | Group Relative Policy Optimization (DeepSeek-style)
 * DPO    | Direct Preference Optimization (no reward model needed)
 * PPO    | Proximal Policy Optimization (classic RLHF)
 * KTO    | Kahneman-Tversky Optimization (unpaired preferences)
 * ORPO   | Odds Ratio Preference Optimization
 * SimPO  | Simple Preference Optimization
 * </pre>
 */
public class AdvancedPEFTConfigExample {
    private static final Logger log = LoggerFactory.getLogger(AdvancedPEFTConfigExample.class);

    public static void main(String[] args) {

        // =====================================================================
        // 1. AdaLoRA — Adaptive Low-Rank Adaptation
        // =====================================================================
        log.info("=== 1. AdaLoRA Config ===");

        // AdaLoRA starts with a higher rank and prunes unimportant singular values
        // during training, effectively allocating rank budget where it's needed most.
        // More efficient than fixed-rank LoRA for heterogeneous layers.
        AdaLoraConfig adaLoraConfig = AdaLoraConfig.builder()
                .initRank(64)                        // Initial rank (will be pruned)
                .loraAlpha(32)
                .loraDropout(0.05)
                .targetModules(Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj"))
                .taskType(TaskType.CAUSAL_LM)
                .targetRank(16)                      // Target rank after pruning
                .warmupSteps(200)                    // Warmup steps before pruning
                .totalPruningSteps(1000)             // Steps to finalize pruning
                .build();

        log.info("  AdaLoRA initial rank: {}", adaLoraConfig.getInitRank());
        log.info("  Target rank after pruning: {}", adaLoraConfig.getTargetRank());
        log.info("  PEFT type: {}", adaLoraConfig.getPeftType());

        // =====================================================================
        // 2. DoRA — Weight-Decomposed LoRA
        // =====================================================================
        log.info("=== 2. DoRA Config ===");

        // DoRA decomposes weight updates into magnitude and direction components:
        //   W = m * (W0 + BA) / ||W0 + BA||
        // This captures both the direction and magnitude of weight changes,
        // often outperforming standard LoRA at similar parameter counts.
        DoraConfig doraConfig = DoraConfig.builder()
                .r(16)
                .loraAlpha(32)
                .loraDropout(0.05)
                .targetModules(Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj"))
                .taskType(TaskType.CAUSAL_LM)
                .build();

        log.info("  DoRA rank: {}", doraConfig.getR());
        log.info("  PEFT type: {}", doraConfig.getPeftType());

        // =====================================================================
        // 3. IA3 — Infused Adapter by Inhibiting and Amplifying
        // =====================================================================
        log.info("=== 3. IA3 Config ===");

        // IA3 adds learned scaling vectors to key, value, and FFN activations.
        // Extremely parameter-efficient: only 3 vectors per layer (no matrices).
        // Best for few-shot scenarios where minimal adaptation is sufficient.
        IA3Config ia3Config = IA3Config.builder()
                .targetModules(Arrays.asList("k_proj", "v_proj", "down_proj"))
                .taskType(TaskType.CAUSAL_LM)
                .feedforwardModules(Arrays.asList("down_proj"))  // Which modules are FFN
                .build();

        log.info("  IA3 target modules: {}", ia3Config.getTargetModules());
        log.info("  IA3 feedforward modules: {}", ia3Config.getFeedforwardModules());
        log.info("  PEFT type: {}", ia3Config.getPeftType());

        // =====================================================================
        // 4. PrefixTuning — Learnable Attention Prefixes
        // =====================================================================
        log.info("=== 4. PrefixTuning Config ===");

        // PrefixTuning prepends learnable key-value pairs to the attention layers.
        // The prefix is reparameterized through an MLP for stable training.
        // Good for multi-task learning (different prefix per task, shared base model).
        PrefixTuningConfig prefixConfig = PrefixTuningConfig.builder()
                .numVirtualTokens(20)           // Number of prefix tokens
                .encoderHiddenSize(512)          // MLP hidden size for reparameterization
                .prefixProjection(true)          // Use MLP projection (recommended)
                .taskType(TaskType.CAUSAL_LM)
                .build();

        log.info("  Prefix virtual tokens: {}", prefixConfig.getNumVirtualTokens());
        log.info("  Prefix projection: {}", prefixConfig.isPrefixProjection());
        log.info("  PEFT type: {}", prefixConfig.getPeftType());

        // =====================================================================
        // 5. PromptTuning — Soft Prompt Embeddings
        // =====================================================================
        log.info("=== 5. PromptTuning Config ===");

        // PromptTuning prepends learnable embedding vectors to the input.
        // Simpler than PrefixTuning (no reparameterization MLP).
        // Very few trainable parameters but may need longer prompts for good results.
        PromptTuningConfig promptConfig = PromptTuningConfig.builder()
                .numVirtualTokens(50)           // Number of soft prompt tokens
                .taskType(TaskType.CAUSAL_LM)
                .build();

        log.info("  Prompt virtual tokens: {}", promptConfig.getNumVirtualTokens());
        log.info("  PEFT type: {}", promptConfig.getPeftType());

        // =====================================================================
        // 6. RL Alignment Methods
        // =====================================================================
        log.info("=== 6. RL Alignment Methods ===");

        // PPO — Classic RLHF (Proximal Policy Optimization)
        // Requires a trained reward model. More complex but well-studied.
        PPOConfig ppoConfig = PPOConfig.builder()
                .policyLogitVariable("logits")     // Required: name of policy output variable
                .valueVariable("value_head")       // Required: name of value head output
                .clipEpsilon(0.2)
                .klCoefficient(0.01)               // KL divergence penalty weight
                .valueLossCoeff(0.5)               // Value function loss weight
                .entropyCoeff(0.01)                // Entropy bonus for exploration
                .ppoEpochs(4)                      // Optimization epochs per batch
                .gaeLambda(0.95)                   // GAE lambda for advantage estimation
                .build();
        log.info("  PPO: clip={}, kl={}", ppoConfig.getClipEpsilon(), ppoConfig.getKlCoefficient());

        // KTO — Kahneman-Tversky Optimization
        // Works with unpaired preference data (only needs good OR bad examples,
        // not paired chosen/rejected). Based on prospect theory loss aversion.
        KTOConfig ktoConfig = KTOConfig.builder()
                .policyLogitVariable("logits")
                .desirabilityVariable("desirable")  // Required: 1=good, 0=bad labels
                .betaDesirable(0.1)                 // KL constraint for desirable samples
                .betaUndesirable(0.1)               // KL constraint for undesirable samples
                .lossAversion(1.0)                  // Penalize undesirable responses more (>1)
                .build();
        log.info("  KTO: betaDesirable={}, betaUndesirable={}", ktoConfig.getBetaDesirable(), ktoConfig.getBetaUndesirable());

        // ORPO — Odds Ratio Preference Optimization
        // Combines SFT and alignment in a single training stage.
        // No need for a reference model (simpler than DPO).
        ORPOConfig orpoConfig = ORPOConfig.builder()
                .policyLogitVariable("logits")
                .chosenVariable("chosen_ids")       // Required: chosen response token IDs
                .rejectedVariable("rejected_ids")   // Required: rejected response token IDs
                .orpoLambda(1.0)                    // Weight of the odds ratio loss
                .useReferenceModel(false)           // ORPO doesn't need a reference model
                .build();
        log.info("  ORPO: lambda={}", orpoConfig.getOrpoLambda());

        // SimPO — Simple Preference Optimization
        // Simplified DPO variant using sequence-average log probabilities.
        // No reference model needed. Often matches or exceeds DPO quality.
        SimPOConfig simpoConfig = SimPOConfig.builder()
                .policyLogitVariable("logits")
                .chosenVariable("chosen_ids")
                .rejectedVariable("rejected_ids")
                .beta(2.5)
                .gamma(0.5)
                .useReferenceModel(false)
                .build();
        log.info("  SimPO: beta={}, gamma={}", simpoConfig.getBeta(), simpoConfig.getGamma());

        log.info("**************** Advanced PEFT Config Example finished ********************");
    }
}
