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
import org.nd4j.autodiff.samediff.config.DAPOConfig;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.config.DPOConfig;
import org.nd4j.autodiff.samediff.config.DrGRPOConfig;
import org.nd4j.autodiff.samediff.config.GRPOConfig;
import org.nd4j.autodiff.samediff.config.GSPOConfig;
import org.nd4j.autodiff.samediff.config.KTOConfig;
import org.nd4j.autodiff.samediff.config.ORPOConfig;
import org.nd4j.autodiff.samediff.config.PPOConfig;
import org.nd4j.autodiff.samediff.config.RLPipelineConfig;
import org.nd4j.autodiff.samediff.config.RewardModelConfig;
import org.nd4j.autodiff.samediff.config.SimPOConfig;
import org.nd4j.autodiff.samediff.rl.DPOTrainer;
import org.nd4j.autodiff.samediff.rl.GRPOTrainer;
import org.nd4j.autodiff.samediff.rl.KTOTrainer;
import org.nd4j.autodiff.samediff.rl.ORPOTrainer;
import org.nd4j.autodiff.samediff.rl.PPOTrainer;
import org.nd4j.autodiff.samediff.rl.RewardFunction;
import org.nd4j.autodiff.samediff.rl.RewardModelTrainer;
import org.nd4j.autodiff.samediff.rl.SamplingStrategy;
import org.nd4j.autodiff.samediff.training.PreferencePair;
import org.nd4j.autodiff.samediff.training.RLAlignmentPipeline;
import org.nd4j.autodiff.samediff.training.TrainingResult;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Comprehensive RL Alignment Training Example.
 *
 * Demonstrates all RL-based alignment methods available in SameDiff with real
 * executable code. Each section builds working trainers, runs a single training
 * step on synthetic data, and prints the resulting loss value.
 *
 * Methods covered:
 *  1.  Build policy and reference models (128->64->10 MLP)
 *  2.  DPO  - Direct Preference Optimization (STANDARD + IPO variant)
 *  3.  GRPO - Group Relative Policy Optimization
 *  4.  PPO  - Proximal Policy Optimization
 *  5.  KTO  - Kahneman-Tversky Optimization
 *  6.  ORPO - Odds Ratio Preference Optimization (no reference model)
 *  7.  SimPO - Simple Preference Optimization (config overview)
 *  8.  DAPO  - Decoupled Alignment Policy Optimization (config overview)
 *  9.  DrGRPO - De-biased Reward GRPO (config overview)
 * 10.  GSPO  - Group Stable Policy Optimization (config overview)
 * 11.  Reward Model Training (Bradley-Terry objective)
 * 12.  RLPipelineConfig - outer training loop configuration
 * 13.  RLAlignmentPipeline - high-level pipeline with PreferencePair list
 * 14.  Comparison summary table
 *
 * All models use a small 128->64->10 MLP so the example runs on CPU without
 * GPU hardware. Vocab size is set to 10 to match the output dimension.
 */
public class RLAlignmentTrainingExample {

    private static final Logger log = LoggerFactory.getLogger(RLAlignmentTrainingExample.class);

    // Shared dimensions for all examples in this file.
    private static final int INPUT_DIM   = 128;
    private static final int HIDDEN_DIM  = 64;
    private static final int VOCAB_SIZE  = 10;   // output dimension = vocabulary size
    private static final int BATCH_SIZE  = 4;
    private static final int SEQ_LEN     = 8;

    // Variable name conventions used across all trainers.
    private static final String INPUT_VAR   = "input";
    private static final String LOGIT_VAR   = "policy_logits";
    private static final String CHOSEN_VAR  = "chosen";
    private static final String REJECTED_VAR = "rejected";

    // -----------------------------------------------------------------------
    // Entry point
    // -----------------------------------------------------------------------

    public static void main(String[] args) throws Exception {
        log.info("=== RL Alignment Training Example ===");
        log.info("VOCAB_SIZE={}, BATCH_SIZE={}, SEQ_LEN={}", VOCAB_SIZE, BATCH_SIZE, SEQ_LEN);

        // Each section gets its own fresh policy/reference pair so that one trainer's
        // loss-graph nodes don't leak into another trainer's backward pass.

        // Section 2: DPO
        demoDPO(buildMLP("dpo_policy"), buildMLP("dpo_reference"));

        // Section 3: GRPO
        demoGRPO(buildMLP("grpo_policy"), buildMLP("grpo_reference"));

        // Section 4: PPO
        demoPPO(buildMLP("ppo_policy"), buildMLP("ppo_reference"));

        // Section 5: KTO
        demoKTO(buildMLP("kto_policy"), buildMLP("kto_reference"));

        // Section 6: ORPO
        demoORPO(buildMLP("orpo_policy"));

        // Section 7-10: config-only overviews
        demoSimPOConfig();
        demoDAPOConfig();
        demoDrGRPOConfig();
        demoGSPOConfig();

        // Section 11: Reward Model Training
        demoRewardModelTraining();

        // Section 12: RLPipelineConfig
        demoRLPipelineConfig();

        // Section 13: RLAlignmentPipeline
        demoRLAlignmentPipeline(buildMLP("pipeline_policy"), buildMLP("pipeline_reference"));

        // Section 14: Comparison table
        printComparisonTable();

        // Section 15: DSP-Accelerated RL Training
        demoDspAcceleratedDPO();

        log.info("=== RL Alignment Training Example complete ===");
    }

    // -----------------------------------------------------------------------
    // Section 1: Build 128 -> 64 -> VOCAB_SIZE MLP
    // -----------------------------------------------------------------------

    /**
     * Build a minimal 3-layer MLP suitable for all RL alignment trainers.
     * Input shape: [batch, INPUT_DIM]
     * Output shape: [batch, VOCAB_SIZE]  (named LOGIT_VAR)
     */
    private static SameDiff buildMLP(String name) {
        SameDiff sd = SameDiff.create();

        // Input placeholder: [batch, INPUT_DIM]
        SDVariable input = sd.placeHolder(INPUT_VAR, DataType.FLOAT, -1, INPUT_DIM);

        // Layer 1: [INPUT_DIM -> HIDDEN_DIM]
        SDVariable w1 = sd.var(name + "_w1",
                Nd4j.randn(DataType.FLOAT, INPUT_DIM, HIDDEN_DIM).muli(0.01));
        SDVariable b1 = sd.var(name + "_b1",
                Nd4j.zeros(DataType.FLOAT, HIDDEN_DIM));
        SDVariable h1 = sd.nn().relu(sd.mmul(input, w1).add(b1), 0.0);

        // Layer 2: [HIDDEN_DIM -> VOCAB_SIZE]
        SDVariable w2 = sd.var(name + "_w2",
                Nd4j.randn(DataType.FLOAT, HIDDEN_DIM, VOCAB_SIZE).muli(0.01));
        SDVariable b2 = sd.var(name + "_b2",
                Nd4j.zeros(DataType.FLOAT, VOCAB_SIZE));
        // Named output variable used by all trainers as policyLogitVariable
        sd.mmul(h1, w2).add(b2).rename(LOGIT_VAR);

        log.info("Built {} MLP: {}->{}->{}  output='{}'",
                name, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, LOGIT_VAR);
        return sd;
    }

    // -----------------------------------------------------------------------
    // Section 2: DPO — Direct Preference Optimization
    // -----------------------------------------------------------------------

    private static void demoDPO(SameDiff policyModel, SameDiff referenceModel) {
        log.info("\n--- Section 2: DPO (Direct Preference Optimization) ---");

        // --- 2a. Standard DPO ---
        DPOConfig dpoConfig = DPOConfig.standard(LOGIT_VAR, CHOSEN_VAR, REJECTED_VAR);
        dpoConfig.setLogits2D(true);  // toy MLP produces 2D logits [batch, vocab]
        log.info("DPO config: beta={}, variant={}", dpoConfig.getBeta(), dpoConfig.getVariant());

        DPOTrainer dpoTrainer = new DPOTrainer(policyModel, referenceModel, dpoConfig);

        // Input map: chosen and rejected token sequences.
        // DPOTrainer.prepareInputs concatenates them and handles reference logprob computation.
        Map<String, INDArray> dpoInputs = new HashMap<>();
        dpoInputs.put(CHOSEN_VAR,   Nd4j.randn(DataType.FLOAT, BATCH_SIZE, INPUT_DIM));
        dpoInputs.put(REJECTED_VAR, Nd4j.randn(DataType.FLOAT, BATCH_SIZE, INPUT_DIM));

        double dpoLoss = dpoTrainer.trainStep(dpoInputs);
        log.info("DPO (STANDARD) trainStep loss: {}", String.format("%.6f", dpoLoss));

        // --- 2b. IPO variant ---
        // IPO needs its own fresh models: sharing policyModel would conflict because
        // buildLossGraph already added _dpo_* placeholders for the standard trainer.
        SameDiff ipoPolicyModel    = buildMLP("ipo_policy");
        SameDiff ipoReferenceModel = buildMLP("ipo_reference");
        DPOConfig ipoConfig = DPOConfig.builder()
                .policyLogitVariable(LOGIT_VAR)
                .chosenVariable(CHOSEN_VAR)
                .rejectedVariable(REJECTED_VAR)
                .beta(0.1)
                .variant(DPOConfig.DPOVariant.IPO)
                .logits2D(true)  // toy MLP produces 2D logits [batch, vocab]
                .build();
        log.info("IPO config: beta={}, variant={}", ipoConfig.getBeta(), ipoConfig.getVariant());

        DPOTrainer ipoTrainer = new DPOTrainer(ipoPolicyModel, ipoReferenceModel, ipoConfig);
        double ipoLoss = ipoTrainer.trainStep(dpoInputs);
        log.info("DPO (IPO) trainStep loss: {}", String.format("%.6f", ipoLoss));

        // --- 2c. RDPO variant with label smoothing ---
        DPOConfig rdpoConfig = DPOConfig.builder()
                .policyLogitVariable(LOGIT_VAR)
                .chosenVariable(CHOSEN_VAR)
                .rejectedVariable(REJECTED_VAR)
                .beta(0.1)
                .labelSmoothing(0.1)
                .variant(DPOConfig.DPOVariant.RDPO)
                .build();
        log.info("RDPO config: beta={}, labelSmoothing={}, variant={}",
                rdpoConfig.getBeta(), rdpoConfig.getLabelSmoothing(), rdpoConfig.getVariant());
    }

    // -----------------------------------------------------------------------
    // Section 3: GRPO — Group Relative Policy Optimization
    // -----------------------------------------------------------------------

    private static void demoGRPO(SameDiff policyModel, SameDiff referenceModel) {
        log.info("\n--- Section 3: GRPO (Group Relative Policy Optimization) ---");

        GRPOConfig grpoConfig = GRPOConfig.builder()
                .policyLogitVariable(LOGIT_VAR)
                .groupSize(4)          // number of completions per prompt
                .clipEpsilon(0.2)
                .klPenalty(0.01)
                .maxNewTokens(SEQ_LEN)
                .logits2D(true)  // toy MLP produces 2D logits [batch, vocab]
                .build();

        log.info("GRPO config: groupSize={}, clipEpsilon={}, klPenalty={}, maxNewTokens={}",
                grpoConfig.getGroupSize(), grpoConfig.getClipEpsilon(),
                grpoConfig.getKlPenalty(), grpoConfig.getMaxNewTokens());

        // SamplingStrategy: for demo purposes, return zero arrays of the right shape.
        // A real implementation calls the model autoregressively.
        SamplingStrategy sampler = new SamplingStrategy() {
            @Override
            public INDArray generate(SameDiff model, INDArray prompts,
                                     int maxNewTokens, String logitVariable) {
                long batch = prompts.size(0);
                return Nd4j.zeros(DataType.FLOAT, batch, INPUT_DIM);
            }

            @Override
            public INDArray generateMultiple(SameDiff model, INDArray prompts,
                                              int numCompletions, int maxNewTokens,
                                              String logitVariable) {
                long batch = prompts.size(0);
                // Returns [batch * numCompletions, INPUT_DIM]
                return Nd4j.zeros(DataType.FLOAT, batch * numCompletions, INPUT_DIM);
            }
        };

        // RewardFunction: return random scores in [0, 1] for each completion.
        RewardFunction rewardFn = (prompts, completions) ->
                Nd4j.rand(DataType.FLOAT, (int) completions.size(0));

        GRPOTrainer grpoTrainer = new GRPOTrainer(policyModel, referenceModel,
                grpoConfig, sampler, rewardFn);

        // GRPO expects a "prompts" key in the input map.
        Map<String, INDArray> grpoInputs = new HashMap<>();
        grpoInputs.put("prompts", Nd4j.randn(DataType.FLOAT, BATCH_SIZE, INPUT_DIM));

        double grpoLoss = grpoTrainer.trainStep(grpoInputs);
        log.info("GRPO trainStep loss: {}", String.format("%.6f", grpoLoss));
    }

    // -----------------------------------------------------------------------
    // Section 4: PPO — Proximal Policy Optimization
    // -----------------------------------------------------------------------

    private static void demoPPO(SameDiff policyModel, SameDiff referenceModel) {
        log.info("\n--- Section 4: PPO (Proximal Policy Optimization) ---");

        // PPO requires a value model (separate SameDiff) with a scalar value head.
        SameDiff valueModel = buildValueModel();

        PPOConfig ppoConfig = PPOConfig.builder()
                .policyLogitVariable(LOGIT_VAR)
                .valueVariable("value_output")
                .clipEpsilon(0.2)
                .valueLossCoeff(0.5)
                .entropyCoeff(0.01)
                .ppoEpochs(2)
                .gaeLambda(0.95)
                .maxNewTokens(SEQ_LEN)
                .logits2D(true)  // toy MLP produces 2D logits [batch, vocab]
                .build();

        log.info("PPO config: clipEpsilon={}, valueLossCoeff={}, entropyCoeff={}, ppoEpochs={}, gaeLambda={}",
                ppoConfig.getClipEpsilon(), ppoConfig.getValueLossCoeff(),
                ppoConfig.getEntropyCoeff(), ppoConfig.getPpoEpochs(), ppoConfig.getGaeLambda());

        SamplingStrategy sampler = new SamplingStrategy() {
            @Override
            public INDArray generate(SameDiff model, INDArray prompts,
                                     int maxNewTokens, String logitVariable) {
                long batch = prompts.size(0);
                return Nd4j.zeros(DataType.FLOAT, batch, INPUT_DIM);
            }

            @Override
            public INDArray generateMultiple(SameDiff model, INDArray prompts,
                                              int numCompletions, int maxNewTokens,
                                              String logitVariable) {
                long batch = prompts.size(0);
                return Nd4j.zeros(DataType.FLOAT, batch * numCompletions, INPUT_DIM);
            }
        };

        RewardFunction rewardFn = (prompts, completions) ->
                Nd4j.rand(DataType.FLOAT, (int) completions.size(0));

        PPOTrainer ppoTrainer = new PPOTrainer(
                policyModel, referenceModel, ppoConfig,
                sampler, rewardFn, valueModel);

        Map<String, INDArray> ppoInputs = new HashMap<>();
        ppoInputs.put("prompts", Nd4j.randn(DataType.FLOAT, BATCH_SIZE, INPUT_DIM));

        double ppoLoss = ppoTrainer.trainStep(ppoInputs);
        log.info("PPO trainStep loss (avg over {} ppoEpochs): {}",
                ppoConfig.getPpoEpochs(), String.format("%.6f", ppoLoss));
    }

    /** Build a small value-head model with a scalar output per batch item. */
    private static SameDiff buildValueModel() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder(INPUT_VAR, DataType.FLOAT, -1, INPUT_DIM);
        SDVariable w = sd.var("value_w",
                Nd4j.randn(DataType.FLOAT, INPUT_DIM, 1).muli(0.01));
        SDVariable b = sd.var("value_b", Nd4j.zeros(DataType.FLOAT, 1));
        // Output: [batch, 1] — squeeze to [batch] for GAE
        sd.mmul(input, w).add(b).reshape(-1).rename("value_output");
        return sd;
    }

    // -----------------------------------------------------------------------
    // Section 5: KTO — Kahneman-Tversky Optimization
    // -----------------------------------------------------------------------

    private static void demoKTO(SameDiff policyModel, SameDiff referenceModel) {
        log.info("\n--- Section 5: KTO (Kahneman-Tversky Optimization) ---");

        String desirabilityVar = "desirability";
        KTOConfig ktoConfig = KTOConfig.builder()
                .policyLogitVariable(LOGIT_VAR)
                .desirabilityVariable(desirabilityVar)
                .betaDesirable(0.1)
                .betaUndesirable(0.1)
                .lossAversion(1.5)       // penalise undesirable responses 1.5x harder
                .logits2D(true)  // toy MLP produces 2D logits [batch, vocab]
                .build();

        log.info("KTO config: betaD={}, betaU={}, lossAversion={}",
                ktoConfig.getBetaDesirable(), ktoConfig.getBetaUndesirable(),
                ktoConfig.getLossAversion());

        KTOTrainer ktoTrainer = new KTOTrainer(policyModel, referenceModel, ktoConfig);

        // Desirability labels: 1 = good response, 0 = bad response.
        float[] labels = {1.0f, 0.0f, 1.0f, 0.0f};
        Map<String, INDArray> ktoInputs = new HashMap<>();
        ktoInputs.put(INPUT_VAR,          Nd4j.randn(DataType.FLOAT, BATCH_SIZE, INPUT_DIM));
        ktoInputs.put(desirabilityVar,    Nd4j.createFromArray(labels));

        double ktoLoss = ktoTrainer.trainStep(ktoInputs);
        log.info("KTO trainStep loss: {}", String.format("%.6f", ktoLoss));
    }

    // -----------------------------------------------------------------------
    // Section 6: ORPO — Odds Ratio Preference Optimization
    // -----------------------------------------------------------------------

    private static void demoORPO(SameDiff policyModel) {
        log.info("\n--- Section 6: ORPO (Odds Ratio Preference Optimization) ---");

        // ORPO does NOT require a reference model.
        ORPOConfig orpoConfig = ORPOConfig.standard(LOGIT_VAR, CHOSEN_VAR, REJECTED_VAR);
        orpoConfig.setLogits2D(true);  // toy MLP produces 2D logits [batch, vocab]
        log.info("ORPO config: orpoLambda={}, useReferenceModel={}",
                orpoConfig.getOrpoLambda(), orpoConfig.isUseReferenceModel());

        // ORPOTrainer constructor takes only policyModel (no reference).
        ORPOTrainer orpoTrainer = new ORPOTrainer(policyModel, orpoConfig);

        Map<String, INDArray> orpoInputs = new HashMap<>();
        orpoInputs.put(CHOSEN_VAR,   Nd4j.randn(DataType.FLOAT, BATCH_SIZE, INPUT_DIM));
        orpoInputs.put(REJECTED_VAR, Nd4j.randn(DataType.FLOAT, BATCH_SIZE, INPUT_DIM));

        double orpoLoss = orpoTrainer.trainStep(orpoInputs);
        log.info("ORPO trainStep loss: {}", String.format("%.6f", orpoLoss));
    }

    // -----------------------------------------------------------------------
    // Section 7: SimPO — Simple Preference Optimization (config overview)
    // -----------------------------------------------------------------------

    private static void demoSimPOConfig() {
        log.info("\n--- Section 7: SimPO (Simple Preference Optimization) ---");

        // SimPO eliminates the reference model by using length-normalised log probs
        // and a target reward margin (gamma).
        SimPOConfig simpoConfig = SimPOConfig.standard(LOGIT_VAR, CHOSEN_VAR, REJECTED_VAR);
        log.info("SimPO config: beta={}, gamma={}, useReferenceModel={}",
                simpoConfig.getBeta(), simpoConfig.getGamma(),
                simpoConfig.isUseReferenceModel());

        // Custom config with explicit vocab size
        SimPOConfig simpoCustom = SimPOConfig.builder()
                .policyLogitVariable(LOGIT_VAR)
                .chosenVariable(CHOSEN_VAR)
                .rejectedVariable(REJECTED_VAR)
                .beta(2.5)      // recommended range: 2.0-2.5
                .gamma(1.0)     // target reward margin
                .vocabSize(VOCAB_SIZE)
                .useReferenceModel(false)
                .build();
        log.info("SimPO custom: beta={}, gamma={}, vocabSize={}",
                simpoCustom.getBeta(), simpoCustom.getGamma(), simpoCustom.getVocabSize());
    }

    // -----------------------------------------------------------------------
    // Section 8: DAPO — Decoupled Alignment Policy Optimization (config overview)
    // -----------------------------------------------------------------------

    private static void demoDAPOConfig() {
        log.info("\n--- Section 8: DAPO (Decoupled Alignment Policy Optimization) ---");

        // DAPO extends GRPO with asymmetric clipping [1-low, 1+high] and
        // token-level KL divergence for finer-grained regularisation.
        DAPOConfig dapoConfig = DAPOConfig.builder()
                .policyLogitVariable(LOGIT_VAR)
                .clipEpsilonLow(0.1)   // lower clip bound
                .clipEpsilonHigh(0.28) // upper clip bound (allows more upward updates)
                .tokenLevelKL(true)
                .dynamicSampling(true)     // skip groups with uniform rewards
                .overlongFiltering(true)
                .groupSize(8)
                .maxNewTokens(256)
                .build();

        log.info("DAPO config: clipLow={}, clipHigh={}, tokenLevelKL={}, dynamicSampling={}, overlongFiltering={}",
                dapoConfig.getClipEpsilonLow(), dapoConfig.getClipEpsilonHigh(),
                dapoConfig.isTokenLevelKL(), dapoConfig.isDynamicSampling(),
                dapoConfig.isOverlongFiltering());
    }

    // -----------------------------------------------------------------------
    // Section 9: DrGRPO — De-biased Reward GRPO (config overview)
    // -----------------------------------------------------------------------

    private static void demoDrGRPOConfig() {
        log.info("\n--- Section 9: DrGRPO (De-biased Reward GRPO) ---");

        // DrGRPO removes length bias by normalising rewards by completion length
        // and subtracting the mean reward baseline before computing advantages.
        DrGRPOConfig drgrpoConfig = DrGRPOConfig.builder()
                .policyLogitVariable(LOGIT_VAR)
                .lengthNormalization(true)   // divide reward by completion length
                .baselineSubtraction(true)   // subtract mean before advantages
                .groupSize(8)
                .clipEpsilon(0.2)
                .maxNewTokens(256)
                .build();

        log.info("DrGRPO config: lengthNormalization={}, baselineSubtraction={}, groupSize={}, clipEpsilon={}",
                drgrpoConfig.isLengthNormalization(), drgrpoConfig.isBaselineSubtraction(),
                drgrpoConfig.getGroupSize(), drgrpoConfig.getClipEpsilon());
    }

    // -----------------------------------------------------------------------
    // Section 10: GSPO — Group Stable Policy Optimization (config overview)
    // -----------------------------------------------------------------------

    private static void demoGSPOConfig() {
        log.info("\n--- Section 10: GSPO (Group Stable Policy Optimization) ---");

        // GSPO uses importance-weighted advantages and a stability coefficient
        // to prevent catastrophic updates during training.
        GSPOConfig gspoConfig = GSPOConfig.builder()
                .policyLogitVariable(LOGIT_VAR)
                .stabilityCoeff(0.1)               // penalty to prevent large updates
                .importanceWeightedAdvantage(true)  // reduces gradient variance
                .groupSize(8)
                .clipEpsilon(0.2)
                .maxNewTokens(256)
                .build();

        log.info("GSPO config: stabilityCoeff={}, importanceWeighted={}, groupSize={}, clipEpsilon={}",
                gspoConfig.getStabilityCoeff(), gspoConfig.isImportanceWeightedAdvantage(),
                gspoConfig.getGroupSize(), gspoConfig.getClipEpsilon());
    }

    // -----------------------------------------------------------------------
    // Section 11: Reward Model Training
    // -----------------------------------------------------------------------

    private static void demoRewardModelTraining() {
        log.info("\n--- Section 11: Reward Model Training (Bradley-Terry) ---");

        // RewardModelConfig trains a scalar reward head on preference pairs.
        // The model's reward output variable must produce a scalar per batch item.
        String rewardOutputVar = "reward_output";
        SameDiff rewardModel = buildRewardModel(rewardOutputVar);

        RewardModelConfig rmConfig = RewardModelConfig.bradleyTerry(
                rewardOutputVar, CHOSEN_VAR, REJECTED_VAR, rewardOutputVar);

        log.info("RewardModel config: rewardType={}, margin={}, useReferenceModel={}",
                rmConfig.getRewardType(), rmConfig.getMargin(),
                rmConfig.isUseReferenceModel());

        RewardModelTrainer rmTrainer = new RewardModelTrainer(rewardModel, rmConfig);

        Map<String, INDArray> rmInputs = new HashMap<>();
        rmInputs.put(CHOSEN_VAR,   Nd4j.randn(DataType.FLOAT, BATCH_SIZE, INPUT_DIM));
        rmInputs.put(REJECTED_VAR, Nd4j.randn(DataType.FLOAT, BATCH_SIZE, INPUT_DIM));

        double rmLoss = rmTrainer.trainStep(rmInputs);
        log.info("RewardModel (Bradley-Terry) trainStep loss: {}", String.format("%.6f", rmLoss));

        // Contrastive variant
        RewardModelConfig contrastiveConfig = RewardModelConfig.builder()
                .policyLogitVariable(rewardOutputVar)
                .chosenVariable(CHOSEN_VAR)
                .rejectedVariable(REJECTED_VAR)
                .rewardOutputVariable(rewardOutputVar)
                .rewardType(RewardModelConfig.RewardType.CONTRASTIVE)
                .margin(0.5)
                .useReferenceModel(false)
                .build();
        log.info("Contrastive RewardModel config: margin={}", contrastiveConfig.getMargin());
    }

    /** Reward model: same MLP as policy but output variable renamed to rewardOutputVar. */
    private static SameDiff buildRewardModel(String rewardOutputVar) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder(INPUT_VAR, DataType.FLOAT, -1, INPUT_DIM);
        SDVariable w1 = sd.var("rm_w1", Nd4j.randn(DataType.FLOAT, INPUT_DIM, HIDDEN_DIM).muli(0.01));
        SDVariable b1 = sd.var("rm_b1", Nd4j.zeros(DataType.FLOAT, HIDDEN_DIM));
        SDVariable h1 = sd.nn().relu(sd.mmul(input, w1).add(b1), 0.0);
        // Scalar reward head: [batch, HIDDEN_DIM] -> [batch, 1] -> [batch]
        SDVariable wr = sd.var("rm_wr", Nd4j.randn(DataType.FLOAT, HIDDEN_DIM, 1).muli(0.01));
        SDVariable br = sd.var("rm_br", Nd4j.zeros(DataType.FLOAT, 1));
        sd.mmul(h1, wr).add(br).reshape(-1).rename(rewardOutputVar);
        return sd;
    }

    // -----------------------------------------------------------------------
    // Section 12: RLPipelineConfig
    // -----------------------------------------------------------------------

    private static void demoRLPipelineConfig() {
        log.info("\n--- Section 12: RLPipelineConfig ---");

        // Default pipeline config (1 epoch, lr=5e-7, BFLOAT16, no PEFT)
        RLPipelineConfig defaults = RLPipelineConfig.defaults();
        log.info("Default pipeline: numEpochs={}, lr={}, computeDataType={}, gradAccum={}, warmupRatio={}",
                defaults.getNumEpochs(), defaults.getLearningRate(),
                defaults.getComputeDataType(), defaults.getGradientAccumulationSteps(),
                defaults.getWarmupRatio());

        // Custom pipeline config for a longer run
        RLPipelineConfig custom = RLPipelineConfig.builder()
                .numEpochs(3)
                .learningRate(1e-6)
                .warmupRatio(0.05)
                .gradientAccumulationSteps(4)     // effective batch = 4 * micro-batch
                .computeDataType(DataType.FLOAT)  // use FLOAT for CPU debugging
                .weightDecay(0.01)
                .logEveryNSteps(50)
                .evaluateEveryNSteps(200)
                .maxSteps(1000)
                .build();

        log.info("Custom pipeline: numEpochs={}, lr={}, gradAccum={}, logEvery={}, evalEvery={}, maxSteps={}",
                custom.getNumEpochs(), custom.getLearningRate(),
                custom.getGradientAccumulationSteps(), custom.getLogEveryNSteps(),
                custom.getEvaluateEveryNSteps(), custom.getMaxSteps());
    }

    // -----------------------------------------------------------------------
    // Section 13: RLAlignmentPipeline
    // -----------------------------------------------------------------------

    private static void demoRLAlignmentPipeline(SameDiff policyModel, SameDiff referenceModel) {
        log.info("\n--- Section 13: RLAlignmentPipeline ---");

        // Build a DPO config and a pipeline-level config.
        DPOConfig dpoConfig = DPOConfig.standard(LOGIT_VAR, CHOSEN_VAR, REJECTED_VAR);
        dpoConfig.setLogits2D(true);  // toy MLP produces 2D logits [batch, vocab]
        RLPipelineConfig pipelineConfig = RLPipelineConfig.builder()
                .numEpochs(1)
                .learningRate(5e-7)
                .logEveryNSteps(1)   // log every step for the demo
                .build();

        // Create the pipeline via the static factory (automatically builds DPOTrainer).
        RLAlignmentPipeline pipeline = RLAlignmentPipeline.create(
                policyModel, referenceModel, dpoConfig, pipelineConfig);

        log.info("Pipeline created. Method: {}", pipeline.getTrainer().getConfig().getMethodName());

        // Build a small list of preference pairs (raw text; production code tokenises these).
        List<PreferencePair> pairs = new ArrayList<>();
        pairs.add(PreferencePair.builder()
                .prompt("Explain quantum entanglement.")
                .chosen("Quantum entanglement is a phenomenon where two particles become correlated.")
                .rejected("I do not know.")
                .build());
        pairs.add(PreferencePair.builder()
                .prompt("What is backpropagation?")
                .chosen("Backpropagation computes gradients by applying the chain rule through the network.")
                .rejected("It is a type of neural network.")
                .build());

        // trainDPO runs the outer epoch/step loop and returns a TrainingResult.
        TrainingResult result = pipeline.trainDPO(pairs);

        log.info("Pipeline training complete:");
        log.info("  totalSteps={}, totalEpochs={}", result.getTotalSteps(), result.getTotalEpochs());
        log.info("  finalLoss={}", Double.isNaN(result.getFinalLoss()) ? "N/A (no steps logged)" : result.getFinalLoss());
        log.info("  trainingTimeMs={}", result.getTrainingTimeMs());
        log.info("  summary: {}", result.summary());

        // The trained policy model is accessible for further use or serialisation.
        SameDiff trainedModel = pipeline.getTrainedModel();
        log.info("  Trained model variable count: {}", trainedModel.variables().size());
    }

    // -----------------------------------------------------------------------
    // Section 14: Comparison of all methods
    // -----------------------------------------------------------------------

    private static void printComparisonTable() {
        log.info("\n--- Section 14: RL Alignment Methods Comparison ---");
        log.info("+---------------+-------------------+-----------------+-------------------+------------------------+");
        log.info("| Method        | Reference Model   | Input Type      | Key Hyperparams   | Key Advantage          |");
        log.info("+---------------+-------------------+-----------------+-------------------+------------------------+");
        log.info("| DPO           | Required          | Pref. pairs     | beta              | Stable, simple         |");
        log.info("| IPO           | Required          | Pref. pairs     | beta              | Avoids overfit         |");
        log.info("| RDPO          | Required          | Pref. pairs     | beta, smoothing   | Noise-robust           |");
        log.info("| KTO           | Required          | Labeled samples | betaD, betaU      | Unpaired labels        |");
        log.info("| ORPO          | Not needed        | Pref. pairs     | orpoLambda        | Memory-efficient       |");
        log.info("| SimPO         | Not needed        | Pref. pairs     | beta, gamma       | Length-normalised      |");
        log.info("| GRPO          | Required          | Prompts+rewards | groupSize, clipEps| DeepSeek-R1 style      |");
        log.info("| DrGRPO        | Required          | Prompts+rewards | lengthNorm        | Removes length bias    |");
        log.info("| DAPO          | Required          | Prompts+rewards | clipLow, clipHigh | Asymmetric clipping    |");
        log.info("| GSPO          | Required          | Prompts+rewards | stabilityCoeff    | Importance weighting   |");
        log.info("| PPO           | Required          | Prompts+rewards | clipEps, ppoEpoch | Classic RLHF           |");
        log.info("| RewardModel   | Not needed        | Pref. pairs     | rewardType, margin| Trains reward signal   |");
        log.info("+---------------+-------------------+-----------------+-------------------+------------------------+");
        log.info("");
        log.info("Config class hierarchy:");
        log.info("  RLAlignmentConfig (abstract base)");
        log.info("  ├── DPOConfig         (beta, labelSmoothing, variant)");
        log.info("  ├── KTOConfig         (betaDesirable, betaUndesirable, lossAversion)");
        log.info("  ├── ORPOConfig        (orpoLambda)");
        log.info("  ├── SimPOConfig       (beta, gamma)");
        log.info("  ├── GRPOConfig        (groupSize, clipEpsilon, klPenalty)");
        log.info("  ├── DrGRPOConfig      (lengthNormalization, baselineSubtraction)");
        log.info("  ├── DAPOConfig        (clipEpsilonLow, clipEpsilonHigh, tokenLevelKL)");
        log.info("  ├── GSPOConfig        (stabilityCoeff, importanceWeightedAdvantage)");
        log.info("  ├── PPOConfig         (clipEpsilon, valueLossCoeff, entropyCoeff, gaeLambda)");
        log.info("  └── RewardModelConfig (rewardType, margin, rewardOutputVariable)");
        log.info("");
        log.info("Pipeline entry points:");
        log.info("  RLAlignmentPipeline.create(policy, rlConfig, pipelineConfig)           // no ref model");
        log.info("  RLAlignmentPipeline.create(policy, reference, rlConfig, pipelineConfig) // with ref model");
        log.info("  RLAlignmentPipeline.create(policy, ref, rlConfig, pipeline, sampler, rewardFn) // online");
        log.info("  pipeline.trainDPO(preferencePairs)   // offline preference training");
        log.info("  pipeline.trainOnline(prompts, reward) // online RL training");
        log.info("  pipeline.train(datasetIterator, maxSteps) // generic iterator-based training");
    }

    // -----------------------------------------------------------------------
    // Section 15: DSP-Accelerated RL Training
    // -----------------------------------------------------------------------

    private static void demoDspAcceleratedDPO() throws Exception {
        log.info("\n--- Section 15: DSP-Accelerated RL Training (DPO) ---");

        // Build fresh policy and reference models for this section.
        SameDiff policyModel    = buildMLP("dsp_policy");
        SameDiff referenceModel = buildMLP("dsp_reference");

        // DSP is enabled by default — log its state before training begins.
        log.info("DSP state: dspAutoCompileEnabled={}, dspNativeAutoCompileEnabled={}",
                policyModel.isDspAutoCompileEnabled(),
                policyModel.isDspNativeAutoCompileEnabled());

        // The RLAlignmentTrainer manages its own Adam updater via dpoConfig.learningRate.
        // (TrainingConfig is not needed here — setting it would require a DataSet feature mapping.)
        DPOConfig dpoConfig = DPOConfig.standard(LOGIT_VAR, CHOSEN_VAR, REJECTED_VAR);
        dpoConfig.setLogits2D(true);     // toy MLP produces 2D logits [batch, vocab]
        dpoConfig.setLearningRate(5e-7); // learning rate for the trainer's internal Adam updater
        DPOTrainer trainer  = new DPOTrainer(policyModel, referenceModel, dpoConfig);

        log.info("Running 10 DPO training steps with DSP plan tracking...");
        for (int step = 0; step < 10; step++) {
            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put(CHOSEN_VAR,   Nd4j.randn(DataType.FLOAT, BATCH_SIZE, INPUT_DIM));
            inputs.put(REJECTED_VAR, Nd4j.randn(DataType.FLOAT, BATCH_SIZE, INPUT_DIM));

            long t0   = System.currentTimeMillis();
            double loss = trainer.trainStep(inputs);
            long ms   = System.currentTimeMillis() - t0;

            // Inspect DSP plan state after this step.
            DspHandle dsp = policyModel.dsp();
            boolean compiled = dsp.isCompiled();
            int phaseOrdinal = compiled ? dsp.planPhase() : -1;
            PlanPhase phase  = compiled ? PlanPhase.fromNativeCode(phaseOrdinal) : null;
            int replayed     = compiled ? dsp.lastExecSegmentsReplayed()  : 0;
            int slotBySlot   = compiled ? dsp.lastExecSegmentsSlotBySlot() : 0;
            int total        = compiled ? dsp.lastExecSegmentsTotal()      : 0;

            log.info("step={} loss={} time={}ms | compiled={} phase={} segs(replay={} sbs={} total={})",
                    step,
                    String.format("%.6f", loss),
                    ms,
                    compiled,
                    phase != null ? phase.name() : "N/A",
                    replayed, slotBySlot, total);
        }

        // Print DspHandle metrics after the training loop.
        DspHandle dsp = policyModel.dsp();
        if (dsp.isCompiled()) {
            log.info("DspHandle metrics after training:");
            log.info("  executeCount={}", dsp.executeCount());
            log.info("  numSegments={}", dsp.numSegments());
            log.info("  totalGraphReplays={}", dsp.totalGraphReplays());
            log.info("  numCapturedGraphSegments={}", dsp.numCapturedGraphSegments());
            log.info("  captureStats={}", dsp.captureStats());
            log.info("  isCompilationSealed={}", dsp.isCompilationSealed());
        } else {
            log.info("DspHandle: plan not yet compiled (CPU-only path or no warmup completed)");
        }

        log.info("Insight: DSP compiles the full DPO training graph including policy forward, "
                + "reference forward, KL divergence, and gradient updates");
    }
}
