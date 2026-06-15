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

import org.nd4j.autodiff.samediff.config.SFTConfig;
import org.nd4j.autodiff.samediff.config.ContinuedPretrainingConfig;
import org.nd4j.linalg.schedule.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Learning Rate Schedule Configuration Examples.
 *
 * SameDiff uses the ISchedule interface to represent learning rate schedules.
 * Every ISchedule implementation provides:
 *   double valueAt(int iteration, int epoch)
 *   ISchedule clone()
 *
 * Schedules can be ITERATION-based (step every mini-batch) or
 * EPOCH-based (step every epoch).
 *
 * Schedules covered:
 * 1. CosineWarmupSchedule - linear warmup then cosine decay (LLM standard)
 * 2. ExponentialSchedule  - exponential decay: LR * gamma^i
 * 3. StepSchedule         - piecewise constant decay
 * 4. PolySchedule         - polynomial decay (linear decay when power=1)
 * 5. SigmoidSchedule      - sigmoid-shaped smooth transition
 * 6. CycleSchedule        - 1-cycle / cosine annealing with warm restarts
 * 7. InverseSchedule      - 1 / (1 + gamma*i)^power decay
 * 8. RampSchedule         - linear warmup wrapper around any base schedule
 * 9. MapSchedule          - arbitrary step function from a key-value map
 * 10. FixedSchedule       - constant LR (no schedule)
 *
 * How schedules integrate with training:
 *   - Set via TrainingConfig.Builder.updater(new Adam(schedule))
 *   - Or via SFTConfig.builder().lrSchedule(schedule)
 *   - Or via ContinuedPretrainingConfig.builder().lrSchedule(schedule)
 */
public class LRScheduleConfigExample {
    private static final Logger log = LoggerFactory.getLogger(LRScheduleConfigExample.class);

    public static void main(String[] args) {

        int totalSteps = 10000;
        int warmupSteps = 500;

        // =====================================================================
        // 1. CosineWarmupSchedule — Industry standard for LLM training
        // =====================================================================
        log.info("=== 1. CosineWarmupSchedule ===");

        // Linear warmup from 0 to maxLR over warmupSteps,
        // then cosine decay from maxLR to minLR over remaining steps.
        // Formula:
        //   if i < warmupSteps: LR = maxLR * (i / warmupSteps)
        //   else: LR = minLR + (maxLR - minLR) * 0.5 * (1 + cos(pi * (i - warmup) / (total - warmup)))
        CosineWarmupSchedule cosineWarmup = new CosineWarmupSchedule(
                3e-4,           // maxLR
                0.0,            // minLR (decay to 0)
                warmupSteps,    // warmup steps
                totalSteps      // total training steps
        );
        printScheduleSamples("CosineWarmup(maxLR=3e-4, minLR=0, warmup=500, total=10000)",
                cosineWarmup, totalSteps);

        // With non-zero min LR (common in LLM fine-tuning)
        CosineWarmupSchedule cosineWithFloor = new CosineWarmupSchedule(
                2e-4,   // maxLR
                1e-5,   // minLR = 5% of maxLR (prevents LR from going to zero)
                warmupSteps,
                totalSteps
        );
        log.info("  At step 0:    {}", cosineWithFloor.valueAt(0, 0));
        log.info("  At step 500:  {}", cosineWithFloor.valueAt(warmupSteps, 0));
        log.info("  At step 5000: {}", cosineWithFloor.valueAt(5000, 0));
        log.info("  At step 10000: {}", cosineWithFloor.valueAt(totalSteps, 0));

        // From warmup ratio (common in HuggingFace-style configs)
        // warmupRatio = 0.05 means 5% of totalSteps are warmup
        CosineWarmupSchedule cosineFromRatio = CosineWarmupSchedule.fromRatio(
                3e-4,   // maxLR
                0.0,    // minLR
                0.05,   // warmupRatio (5% warmup)
                totalSteps
        );
        log.info("  fromRatio(warmupRatio=0.05): warmupSteps={}",
                (int)(totalSteps * 0.05));

        // =====================================================================
        // 2. ExponentialSchedule — Multiplicative decay
        // =====================================================================
        log.info("=== 2. ExponentialSchedule ===");

        // LR = initialValue * gamma^i
        // Commonly used for CNNs and smaller models.
        ExponentialSchedule exponential = new ExponentialSchedule(
                ScheduleType.ITERATION,
                1e-3,   // initialValue
                0.9995  // gamma (per-step decay)
        );
        printScheduleSamples("Exponential(LR=1e-3, gamma=0.9995, per-iter)",
                exponential, totalSteps);

        // Epoch-based: gamma applied once per epoch
        ExponentialSchedule epochExponential = new ExponentialSchedule(
                ScheduleType.EPOCH,
                1e-3,   // initial LR
                0.5     // halve LR every epoch
        );
        log.info("  Epoch-based: epoch 0={}, epoch 1={}, epoch 2={}, epoch 3={}",
                epochExponential.valueAt(0, 0),
                epochExponential.valueAt(0, 1),
                epochExponential.valueAt(0, 2),
                epochExponential.valueAt(0, 3));

        // =====================================================================
        // 3. StepSchedule — Piecewise constant (step function)
        // =====================================================================
        log.info("=== 3. StepSchedule ===");

        // LR = initialValue * decayRate^floor(i / step)
        // Classic multi-step LR from classical deep learning.
        StepSchedule stepSchedule = new StepSchedule(
                ScheduleType.ITERATION,
                1e-3,   // initialValue
                0.5,    // decayRate (halve LR at each step)
                2000    // step (decay every 2000 iterations)
        );
        log.info("  StepSchedule(LR=1e-3, halve every 2000 iters):");
        log.info("    iter=0:    {}", stepSchedule.valueAt(0, 0));
        log.info("    iter=1999: {}", stepSchedule.valueAt(1999, 0));
        log.info("    iter=2000: {}", stepSchedule.valueAt(2000, 0));   // halved
        log.info("    iter=4000: {}", stepSchedule.valueAt(4000, 0));   // halved again

        // =====================================================================
        // 4. PolySchedule — Polynomial decay
        // =====================================================================
        log.info("=== 4. PolySchedule ===");

        // LR = initialValue * (1 + i/maxIter)^power
        // power=1: linear decay.  power>1: faster decay.  Returns 0 at i >= maxIter.
        PolySchedule linearDecay = new PolySchedule(
                ScheduleType.ITERATION,
                1e-3,   // initialValue
                1.0,    // power (linear decay)
                totalSteps
        );
        log.info("  Linear decay (power=1.0):");
        log.info("    iter=0:    {}", linearDecay.valueAt(0, 0));
        log.info("    iter=5000: {}", linearDecay.valueAt(5000, 0));
        log.info("    iter=10000: {} (returns 0 at maxIter)", linearDecay.valueAt(totalSteps, 0));

        PolySchedule squareDecay = new PolySchedule(ScheduleType.ITERATION, 1e-3, 2.0, totalSteps);
        log.info("  Square decay (power=2.0): iter=5000 => {}", squareDecay.valueAt(5000, 0));

        // =====================================================================
        // 5. SigmoidSchedule — Smooth transition
        // =====================================================================
        log.info("=== 5. SigmoidSchedule ===");

        // LR = initialValue / (1 + exp(-gamma * (i - stepSize)))
        // Creates an S-shaped transition. Less commonly used but smooth.
        SigmoidSchedule sigmoidSchedule = new SigmoidSchedule(
                ScheduleType.ITERATION,
                1e-3,       // initialValue
                0.001,      // gamma (controls steepness of transition)
                5000        // stepSize (center of sigmoid)
        );
        log.info("  SigmoidSchedule(gamma=0.001, center=5000):");
        log.info("    iter=0:    {}", sigmoidSchedule.valueAt(0, 0));
        log.info("    iter=2500: {}", sigmoidSchedule.valueAt(2500, 0));
        log.info("    iter=5000: {}", sigmoidSchedule.valueAt(5000, 0));
        log.info("    iter=7500: {}", sigmoidSchedule.valueAt(7500, 0));

        // =====================================================================
        // 6. CycleSchedule — 1-Cycle / Warm Restarts
        // =====================================================================
        log.info("=== 6. CycleSchedule ===");

        // Triangle/cosine waves with an annealing tail.
        // Inspired by 1-Cycle LR policy (Smith 2018).
        CycleSchedule cycleSchedule = new CycleSchedule(
                ScheduleType.ITERATION,
                1e-5,       // initialLearningRate (bottom of cycle)
                3e-4,       // maxLearningRate (peak of cycle)
                4000,       // cycleLength (steps per cycle)
                1000,       // annealingLength (final annealing steps)
                0.1         // annealingDecay (decay fraction during annealing)
        );
        log.info("  CycleSchedule: maxLR={}", 3e-4);

        // Simple constructor: just specify max LR and cycle length
        CycleSchedule simpleCycle = new CycleSchedule(
                ScheduleType.ITERATION,
                3e-4,   // maxLearningRate
                4000    // cycleLength
        );
        log.info("  Simple cycle: at step 0={}, 2000={}, 4000={}",
                simpleCycle.valueAt(0, 0),
                simpleCycle.valueAt(2000, 0),
                simpleCycle.valueAt(4000, 0));

        // =====================================================================
        // 7. InverseSchedule — Inverse decay
        // =====================================================================
        log.info("=== 7. InverseSchedule ===");

        // LR = initialValue / (1 + gamma*i)^power
        // Used in the original Transformer paper (Attention Is All You Need).
        InverseSchedule inverseSchedule = new InverseSchedule(
                ScheduleType.ITERATION,
                1e-3,   // initialValue
                1e-4,   // gamma (smaller = slower decay)
                0.5     // power (0.5 = square root inverse)
        );
        log.info("  InverseSchedule(LR=1e-3, gamma=1e-4, power=0.5):");
        log.info("    iter=0:    {}", inverseSchedule.valueAt(0, 0));
        log.info("    iter=1000: {}", inverseSchedule.valueAt(1000, 0));
        log.info("    iter=5000: {}", inverseSchedule.valueAt(5000, 0));
        log.info("    iter=10000: {}", inverseSchedule.valueAt(10000, 0));

        // =====================================================================
        // 8. RampSchedule — Linear warmup wrapper
        // =====================================================================
        log.info("=== 8. RampSchedule ===");

        // Wraps any base schedule with a linear ramp-up over numIter steps.
        // After numIter steps, delegates to the base schedule.
        // Useful when base schedule doesn't have built-in warmup.
        RampSchedule rampedExponential = new RampSchedule(
                exponential,    // base schedule (will be used after warmup)
                500             // numIter (ramp up over first 500 steps)
        );
        log.info("  RampSchedule wrapping Exponential (500 warmup steps):");
        log.info("    iter=0:   {}", rampedExponential.valueAt(0, 0));
        log.info("    iter=250: {}", rampedExponential.valueAt(250, 0));
        log.info("    iter=500: {}", rampedExponential.valueAt(500, 0));
        log.info("    iter=501: {}", rampedExponential.valueAt(501, 0));   // base schedule takes over

        // =====================================================================
        // 9. MapSchedule — Arbitrary step function
        // =====================================================================
        log.info("=== 9. MapSchedule ===");

        // Map schedule: specify exact LR values at specific iterations/epochs.
        // Lookup returns the nearest lower key's value (piecewise constant).
        // MUST contain a value at key 0.
        MapSchedule mapSchedule = new MapSchedule.Builder(ScheduleType.EPOCH)
                .add(0, 1e-3)   // Epoch 0-2: LR = 1e-3
                .add(3, 1e-4)   // Epoch 3-6: LR = 1e-4 (reduce 10x)
                .add(7, 1e-5)   // Epoch 7+:  LR = 1e-5 (reduce 10x again)
                .build();

        log.info("  MapSchedule(epoch-based):");
        log.info("    epoch 0: {}", mapSchedule.valueAt(0, 0));
        log.info("    epoch 2: {}", mapSchedule.valueAt(0, 2));
        log.info("    epoch 3: {}", mapSchedule.valueAt(0, 3));   // steps down
        log.info("    epoch 5: {}", mapSchedule.valueAt(0, 5));
        log.info("    epoch 7: {}", mapSchedule.valueAt(0, 7));   // steps down again
        log.info("    epoch 10: {}", mapSchedule.valueAt(0, 10));

        // =====================================================================
        // 10. FixedSchedule — Constant LR (no schedule)
        // =====================================================================
        log.info("=== 10. FixedSchedule ===");

        FixedSchedule fixedSchedule = new FixedSchedule(1e-4);
        log.info("  Fixed LR={}: always returns {}", 1e-4, fixedSchedule.valueAt(9999, 99));

        // =====================================================================
        // Integrating schedules with SFT and Pretraining configs
        // =====================================================================
        log.info("=== Schedules in SFTConfig ===");

        CosineWarmupSchedule sftSchedule = CosineWarmupSchedule.fromRatio(2e-5, 0.0, 0.03, 5000);

        SFTConfig sftWithSchedule = SFTConfig.builder()
                .learningRate(2e-5)         // Ignored when lrSchedule is set
                .lrSchedule(sftSchedule)    // Cosine warmup with 3% warmup
                .warmupRatio(0.03)          // Informational (actual schedule controls LR)
                .numEpochs(3)
                .build();
        log.info("  SFTConfig with cosine schedule: warmup={} steps",
                (int)(5000 * 0.03));

        log.info("=== Schedules in ContinuedPretrainingConfig ===");

        MapSchedule pretrainSchedule = new MapSchedule.Builder(ScheduleType.EPOCH)
                .add(0, 5e-5)
                .add(1, 2e-5)
                .build();

        ContinuedPretrainingConfig pretrain = ContinuedPretrainingConfig.builder()
                .lrSchedule(pretrainSchedule)
                .numEpochs(2)
                .build();
        log.info("  ContinuedPretraining with map schedule: epoch0={}, epoch1={}",
                pretrainSchedule.valueAt(0, 0), pretrainSchedule.valueAt(0, 1));

        // =====================================================================
        // SUMMARY TABLE
        // =====================================================================
        log.info("=== LR Schedule Quick Reference ===");
        log.info("  +---------------------+-------------------------+--------------------------------+");
        log.info("  | Schedule            | Formula                 | Best Use Case                  |");
        log.info("  +---------------------+-------------------------+--------------------------------+");
        log.info("  | CosineWarmup        | warmup + cosine decay   | LLM fine-tuning (default)      |");
        log.info("  | Exponential         | LR * gamma^i            | CNN training, smooth decay     |");
        log.info("  | Step                | LR * rate^(i/step)      | Classic multi-step decay       |");
        log.info("  | Poly (power=1)      | Linear decay            | BERT pre-training              |");
        log.info("  | Poly (power>1)      | Polynomial decay        | Fast initial decay             |");
        log.info("  | Sigmoid             | S-curve transition      | Smooth warmup/cooldown         |");
        log.info("  | Cycle               | Triangular / 1-cycle    | CLR / super-convergence        |");
        log.info("  | Inverse (power=0.5) | Transformer LR formula  | Transformers from scratch      |");
        log.info("  | Ramp + base         | Linear warmup + any     | Add warmup to any schedule     |");
        log.info("  | Map                 | Step function (arbitrary)| Curriculum, manual changes     |");
        log.info("  | Fixed               | Constant                | Simple experiments             |");
        log.info("  +---------------------+-------------------------+--------------------------------+");
        log.info("  ScheduleType.ITERATION: step at every mini-batch (default for most)");
        log.info("  ScheduleType.EPOCH:     step at every epoch (coarser, simpler)");
        log.info("**************** LR Schedule Config Example finished ********************");
    }

    private static void printScheduleSamples(String name, ISchedule schedule, int totalSteps) {
        int[] sampleSteps = {0, totalSteps / 20, totalSteps / 4, totalSteps / 2,
                             3 * totalSteps / 4, totalSteps - 1};
        log.info("  {}:", name);
        for (int step : sampleSteps) {
            log.info("    step {:6d}: {}", step, schedule.valueAt(step, 0));
        }
    }
}
