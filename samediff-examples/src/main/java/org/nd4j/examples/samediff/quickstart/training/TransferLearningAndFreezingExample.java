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
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.*;
import org.nd4j.autodiff.samediff.peft.PeftModel;
import org.nd4j.autodiff.samediff.training.GradientAccumulator;
import org.nd4j.autodiff.samediff.training.LossScaler;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.schedule.CosineWarmupSchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Transfer Learning, Variable Freezing, PeftModel, and Training Utilities Example.
 *
 * Transfer learning adapts a pre-trained model to a new task by selectively
 * freezing/unfreezing layers and optionally applying parameter-efficient
 * fine-tuning (PEFT) techniques.
 *
 * <h3>Topics Covered:</h3>
 * <ul>
 *   <li>Variable freezing/unfreezing by name, pattern, and prefix</li>
 *   <li>{@link PeftModel} — Apply LoRA/QLoRA/DoRA/IA3 adapters to any SameDiff model</li>
 *   <li>{@link ContinuedPretrainingConfig} — Configure continued pre-training (domain adaptation)</li>
 *   <li>{@link GradientCheckpointConfig} — Memory-efficient training via activation recomputation</li>
 *   <li>{@link GradientAccumulator} — Accumulate gradients across micro-batches</li>
 *   <li>{@link LossScaler} — Dynamic/static loss scaling for mixed-precision training</li>
 * </ul>
 *
 * Run with:
 *   cd samediff-examples
 *   mvn exec:java -Dexec.mainClass="org.nd4j.examples.samediff.quickstart.training.TransferLearningAndFreezingExample"
 */
public class TransferLearningAndFreezingExample {
    private static final Logger log = LoggerFactory.getLogger(TransferLearningAndFreezingExample.class);

    public static void main(String[] args) {

        // =====================================================================
        // 1. Build a model to demonstrate freezing
        // =====================================================================
        log.info("=== 1. Building a Multi-Layer Model ===");

        SameDiff sd = SameDiff.create();

        // Simulate a 4-layer network with named weight variables
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 128);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 10);

        // Encoder layers
        SDVariable encW1 = sd.var("encoder.layer.0.weight", Nd4j.randn(128, 256).muli(0.01));
        SDVariable encB1 = sd.var("encoder.layer.0.bias", Nd4j.zeros(256));
        SDVariable encW2 = sd.var("encoder.layer.1.weight", Nd4j.randn(256, 128).muli(0.01));
        SDVariable encB2 = sd.var("encoder.layer.1.bias", Nd4j.zeros(128));

        // Decoder layers
        SDVariable decW1 = sd.var("decoder.layer.0.weight", Nd4j.randn(128, 64).muli(0.01));
        SDVariable decB1 = sd.var("decoder.layer.0.bias", Nd4j.zeros(64));

        // Classification head
        SDVariable headW = sd.var("head.weight", Nd4j.randn(64, 10).muli(0.01));
        SDVariable headB = sd.var("head.bias", Nd4j.zeros(10));

        SDVariable h1 = sd.nn.relu(input.mmul(encW1).add(encB1), 0);
        SDVariable h2 = sd.nn.relu(h1.mmul(encW2).add(encB2), 0);
        SDVariable h3 = sd.nn.relu(h2.mmul(decW1).add(decB1), 0);
        SDVariable logits = h3.mmul(headW).add(headB);
        SDVariable output = sd.nn.softmax("output", logits, -1);
        SDVariable loss = sd.loss.softmaxCrossEntropy("loss", label, logits, null);

        // Count trainable variables
        long trainableCount = sd.variables().stream()
                .filter(v -> v.getVariableType() == VariableType.VARIABLE)
                .count();
        log.info("Total trainable variables: {}", trainableCount);

        // =====================================================================
        // 2. Freeze Variables by Name
        // =====================================================================
        log.info("\n=== 2. Freeze Variables by Name ===");

        // Freeze specific variables (VARIABLE → CONSTANT)
        // Frozen variables are not updated during training.
        sd.freezeVariables("encoder.layer.0.weight", "encoder.layer.0.bias");
        log.info("Frozen encoder.layer.0.weight and encoder.layer.0.bias");

        // Verify — frozen variables become CONSTANT
        log.info("  encoder.layer.0.weight type: {}",
                sd.getVariable("encoder.layer.0.weight").getVariableType());
        log.info("  encoder.layer.1.weight type: {}",
                sd.getVariable("encoder.layer.1.weight").getVariableType());

        // =====================================================================
        // 3. Freeze Variables by Pattern (regex)
        // =====================================================================
        log.info("\n=== 3. Freeze by Pattern ===");

        // Freeze all encoder layers using regex
        // NOTE: freezeMatching uses Pattern.matches() which requires FULL string match
        sd.freezeMatching("encoder\\.layer\\..*");
        log.info("Frozen all variables matching 'encoder\\.layer\\..*'");

        long remainingTrainable = sd.variables().stream()
                .filter(v -> v.getVariableType() == VariableType.VARIABLE)
                .count();
        log.info("Remaining trainable variables: {}", remainingTrainable);

        // =====================================================================
        // 4. Freeze Variables by Prefix
        // =====================================================================
        log.info("\n=== 4. Freeze by Prefix ===");

        // Freeze all decoder variables
        sd.freezePrefix("decoder.");
        log.info("Frozen all variables with prefix 'decoder.'");

        remainingTrainable = sd.variables().stream()
                .filter(v -> v.getVariableType() == VariableType.VARIABLE)
                .count();
        log.info("Remaining trainable: {} (only head.weight and head.bias)", remainingTrainable);

        // =====================================================================
        // 5. Unfreeze Variables
        // =====================================================================
        log.info("\n=== 5. Unfreeze Variables ===");

        // Unfreeze specific variables (CONSTANT → VARIABLE)
        sd.unfreezeVariables("encoder.layer.1.weight", "encoder.layer.1.bias");
        log.info("Unfrozen encoder.layer.1.weight and encoder.layer.1.bias");

        // Unfreeze by pattern
        sd.unfreezeMatching("decoder\\.layer\\..*");
        log.info("Unfrozen all decoder variables");

        remainingTrainable = sd.variables().stream()
                .filter(v -> v.getVariableType() == VariableType.VARIABLE)
                .count();
        log.info("Trainable after unfreeze: {}", remainingTrainable);

        // Unfreeze everything (convertConstantsToVariables)
        sd.convertConstantsToVariables();
        log.info("All constants converted back to variables");

        remainingTrainable = sd.variables().stream()
                .filter(v -> v.getVariableType() == VariableType.VARIABLE)
                .count();
        log.info("Total trainable now: {}", remainingTrainable);

        // Freeze entire model (all VARIABLE → CONSTANT)
        SameDiff frozenCopy = sd.freeze(false);  // false = return a copy, don't modify original
        long frozenCount = frozenCopy.variables().stream()
                .filter(v -> v.getVariableType() == VariableType.VARIABLE)
                .count();
        log.info("Frozen copy trainable: {}", frozenCount);
        log.info("Original still trainable: {}", sd.variables().stream()
                .filter(v -> v.getVariableType() == VariableType.VARIABLE).count());

        // =====================================================================
        // 6. PeftModel — Parameter-Efficient Fine-Tuning
        // =====================================================================
        log.info("\n=== 6. PeftModel ===");

        // PeftModel wraps a SameDiff model with a PEFT adapter (LoRA, QLoRA, etc.)
        // It automatically freezes the base model and adds trainable adapter parameters.

        // Create a LoRA config
        LoraConfig loraConfig = LoraConfig.builder()
                .r(8)                      // LoRA rank
                .loraAlpha(16)             // Scaling factor
                .loraDropout(0.05)         // Dropout on LoRA layers
                .targetModules(Arrays.asList("encoder.layer.0.weight",
                        "encoder.layer.1.weight",
                        "decoder.layer.0.weight"))
                .build();

        // Apply LoRA to the model
        PeftModel peftModel = PeftModel.fromPretrained(sd, loraConfig);

        log.info("PeftModel created:");
        log.info("  Total parameters: {}", peftModel.getTotalParameterCount());
        log.info("  Trainable parameters: {}", peftModel.getTrainableParameterCount());
        log.info("  Trainable percentage: {}%",
                String.format("%.2f", peftModel.getTrainablePercentage()));
        peftModel.printTrainableParameters();

        // Get model summary
        String summary = peftModel.getSummary();
        log.info("  Summary: {}", summary.substring(0, Math.min(200, summary.length())));

        // Run inference through PeftModel
        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.randn(8, 128));
        Map<String, INDArray> peftOutput = peftModel.output(ph, "output");
        log.info("  PeftModel output shape: {}",
                Arrays.toString(peftOutput.get("output").shape()));

        // Merge LoRA weights back into base model (for deployment)
        SameDiff mergedModel = peftModel.mergeAndUnload();
        log.info("  Merged model: LoRA weights absorbed into base weights");

        // Get a specific merged weight
        INDArray mergedW = peftModel.getMergedWeight("encoder.layer.0.weight");
        log.info("  Merged encoder.layer.0.weight shape: {}",
                Arrays.toString(mergedW.shape()));

        // Disable/enable adapter (toggle between original and adapted behavior)
        peftModel.disableAdapter();
        log.info("  Adapter disabled — using original base weights");
        peftModel.enableAdapter();
        log.info("  Adapter re-enabled — using adapted weights");

        // =====================================================================
        // 7. PeftModel with Other Adapter Types
        // =====================================================================
        log.info("\n=== 7. Other Adapter Types ===");

        // QLoRA (quantized base + LoRA adapters)
        QLoraConfig qloraConfig = QLoraConfig.builder()
                .r(8)
                .loraAlpha(16)
                .quantType("nf4")            // NormalFloat4 quantization
                .bits(4)                     // 4-bit quantization
                .doubleQuant(true)           // Double quantization for extra compression
                .computeDataType(DataType.BFLOAT16)
                .targetModules(Arrays.asList("encoder.layer.0.weight"))
                .build();
        log.info("QLoRA: quantType={}, bits={}, doubleQuant={}",
                qloraConfig.getQuantType(), qloraConfig.getBits(), qloraConfig.isDoubleQuant());

        // DoRA (Weight-Decomposed LoRA)
        DoraConfig doraConfig = DoraConfig.builder()
                .r(8)
                .loraAlpha(16)
                .magnitudeInit("pretrained")  // "pretrained" or "unit"
                .targetModules(Arrays.asList("encoder.layer.0.weight"))
                .build();
        log.info("DoRA: magnitudeInit={}", doraConfig.getMagnitudeInit());

        // IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
        IA3Config ia3Config = IA3Config.builder()
                .targetModules(Arrays.asList("encoder.layer.0.weight"))
                .feedforwardModules(Arrays.asList("up_proj", "down_proj"))
                .initToOne(true)
                .build();
        log.info("IA3: initToOne={}", ia3Config.isInitToOne());

        // LoRA preset for transformers
        LoraConfig transformerLora = LoraConfig.defaultTransformer();
        log.info("LoRA defaultTransformer preset: r={}, alpha={}, dropout={}",
                transformerLora.getR(), transformerLora.getLoraAlpha(),
                transformerLora.getLoraDropout());

        // =====================================================================
        // 8. Continued Pre-Training Configuration
        // =====================================================================
        log.info("\n=== 8. Continued Pre-Training ===");

        // Continued pre-training adapts a language model to a new domain
        // using unsupervised language modeling on domain-specific text.
        ContinuedPretrainingConfig cptConfig = ContinuedPretrainingConfig.builder()
                .chunkSize(2048)                  // Text chunk size in tokens
                .chunkOverlap(128)                // Overlap between chunks
                .learningRate(5e-5)               // Lower LR for continued pre-training
                .warmupRatio(0.1)                 // 10% warmup
                .minLearningRate(1e-6)            // Minimum LR for cosine decay
                .gradientAccumulationSteps(4)     // Effective batch = batch * 4
                .computeDataType(DataType.BFLOAT16)
                .useLoRA(true)                    // Use LoRA for efficiency
                .loraRank(16)
                .loraAlpha(32)
                .weightDecay(0.01)
                .maxGradNorm(1.0)
                .numEpochs(1)
                .build();

        cptConfig.validate();
        log.info("Continued pre-training config:");
        log.info("  chunkSize={}, overlap={}", cptConfig.getChunkSize(), cptConfig.getChunkOverlap());
        log.info("  LR={}, warmupRatio={}", cptConfig.getLearningRate(), cptConfig.getWarmupRatio());
        log.info("  LoRA: rank={}, alpha={}", cptConfig.getLoraRank(), cptConfig.getLoraAlpha());
        log.info("  computeDataType={}", cptConfig.getComputeDataType());

        // =====================================================================
        // 9. Gradient Checkpointing
        // =====================================================================
        log.info("\n=== 9. Gradient Checkpointing ===");

        // Gradient checkpointing trades compute for memory by recomputing
        // intermediate activations during the backward pass instead of storing them.

        // Strategy 1: sqrt(N) checkpointing — optimal for N-layer networks
        GradientCheckpointConfig sqrtNConfig = GradientCheckpointConfig.sqrtN();
        log.info("sqrt(N) checkpointing:");
        log.info("  isSqrtN={}", sqrtNConfig.isSqrtN());
        log.info("  resolveInterval(24 layers) = {} layers between checkpoints",
                sqrtNConfig.resolveInterval(24));

        // Strategy 2: Every N layers
        GradientCheckpointConfig everyNConfig = GradientCheckpointConfig.everyN(4);
        log.info("Every 4 layers checkpointing");

        // Strategy 3: Manual checkpoint placement
        Set<String> checkpointVars = new HashSet<>(Arrays.asList(
                "encoder.layer.0.output", "encoder.layer.1.output"));
        GradientCheckpointConfig manualConfig = GradientCheckpointConfig.manual(checkpointVars);
        log.info("Manual checkpointing: {}", checkpointVars);

        // Strategy 4: Async host offloading (offload checkpoints to CPU RAM)
        GradientCheckpointConfig asyncConfig = GradientCheckpointConfig.asyncOffload();
        log.info("Async offload checkpointing:");
        log.info("  strategy={}", asyncConfig.getStrategy());
        log.info("  isAsyncOffload={}", asyncConfig.isAsyncOffload());
        log.info("  isAnyOffload={}", asyncConfig.isAnyOffload());

        // Async with custom prefetch distance
        GradientCheckpointConfig asyncPrefetch = GradientCheckpointConfig.asyncOffload(3);
        log.info("  Async with prefetchDistance=3");

        // Checkpoint strategies
        log.info("\nCheckpoint strategies:");
        log.info("  RECOMPUTE        — recompute activations during backward (default)");
        log.info("  OFFLOAD_HOST     — offload activations to host memory");
        log.info("  OFFLOAD_HOST_ASYNC — async prefetch from host to GPU");

        // =====================================================================
        // 10. GradientAccumulator — Effective Large Batch Training
        // =====================================================================
        log.info("\n=== 10. GradientAccumulator ===");

        // Accumulates gradients over N micro-batches before applying an update.
        // Effective batch size = micro_batch_size * accumulation_steps
        GradientAccumulator accumulator = new GradientAccumulator(4);  // 4 accumulation steps
        log.info("GradientAccumulator: {} accumulation steps", accumulator.getAccumulationSteps());
        log.info("  isEnabled={} (true when steps > 1)", accumulator.isEnabled());

        // Simulate accumulation
        for (int step = 0; step < 8; step++) {
            // Simulate gradients from a micro-batch
            Map<String, INDArray> grads = new HashMap<>();
            grads.put("w1", Nd4j.randn(128, 64).muli(0.01));
            grads.put("w2", Nd4j.randn(64, 10).muli(0.01));

            accumulator.accumulate(grads);
            accumulator.step();

            if (accumulator.isReady()) {
                // Gradients are averaged and ready for the optimizer
                Map<String, INDArray> avgGrads = accumulator.getAndReset();
                log.info("  Step {}: optimizer update (accumulated {} micro-batches)",
                        step, accumulator.getAccumulationSteps());
                log.info("    Averaged gradient keys: {}", avgGrads.keySet());
            } else {
                log.info("  Step {}: accumulating... ({}/{})",
                        step, accumulator.getCurrentStep(), accumulator.getAccumulationSteps());
            }
        }

        // Check if a specific gradient has been accumulated
        accumulator.accumulate("test_var", Nd4j.zeros(10));
        log.info("Has gradient for 'test_var': {}", accumulator.hasGradient("test_var"));
        log.info("Number of accumulated variables: {}", accumulator.getNumVariables());
        accumulator.reset();

        // =====================================================================
        // 11. LossScaler — Mixed-Precision Loss Scaling
        // =====================================================================
        log.info("\n=== 11. LossScaler ===");

        // Dynamic loss scaling for FP16/BF16 training:
        // Scales loss up before backward pass, scales gradients down before update.
        // Prevents gradient underflow in FP16 training.

        // Dynamic scaling (auto-adjusts scale based on gradient stability)
        LossScaleConfig dynamicConfig = LossScaleConfig.dynamicScaling();
        LossScaler dynamicScaler = new LossScaler(dynamicConfig);
        log.info("Dynamic loss scaling:");
        log.info("  mode={}", dynamicConfig.getMode());
        log.info("  initialScale={}", dynamicConfig.getInitialScale());
        log.info("  growthFactor={}", dynamicConfig.getGrowthFactor());
        log.info("  backoffFactor={}", dynamicConfig.getBackoffFactor());
        log.info("  growthInterval={}", dynamicConfig.getGrowthInterval());

        // Scale a loss value
        INDArray lossValue = Nd4j.scalar(0.5);
        INDArray scaledLoss = dynamicScaler.scaleLoss(lossValue);
        log.info("  Original loss: {}, Scaled loss: {}", lossValue, scaledLoss);
        log.info("  Current scale: {}", dynamicScaler.getCurrentScale());

        // Unscale gradients and check for overflow
        INDArray gradients = Nd4j.randn(128, 64);
        boolean gradientsOk = dynamicScaler.unscaleGradientsAndCheck(gradients);
        log.info("  Gradients finite after unscaling: {}", gradientsOk);

        // Update scaler based on gradient check
        dynamicScaler.update(gradientsOk);
        log.info("  Scale after update: {}", dynamicScaler.getCurrentScale());

        // Simulate overflow detection
        INDArray infGradients = Nd4j.ones(10).muli(Double.POSITIVE_INFINITY);
        boolean infCheck = dynamicScaler.areGradientsFinite(infGradients);
        log.info("  Inf gradients finite: {}", infCheck);
        dynamicScaler.update(infCheck);  // false — scale will back off
        log.info("  Scale after overflow backoff: {}", dynamicScaler.getCurrentScale());

        // Static scaling (fixed scale, no auto-adjustment)
        LossScaleConfig staticConfig = LossScaleConfig.staticScaling(1024.0);
        LossScaler staticScaler = new LossScaler(staticConfig);
        log.info("\nStatic loss scaling:");
        log.info("  mode={}, scale={}", staticConfig.getMode(), staticConfig.getInitialScale());

        // Dynamic with custom initial scale
        LossScaleConfig customDynamic = LossScaleConfig.dynamicScaling(32768.0);
        log.info("\nCustom dynamic scaling: initialScale={}", customDynamic.getInitialScale());

        // Full custom config
        LossScaleConfig fullConfig = LossScaleConfig.builder()
                .mode(LossScaleConfig.Mode.DYNAMIC)
                .initialScale(65536.0)
                .minScale(1.0)
                .maxScale(65536.0)
                .growthFactor(2.0)       // Double scale after growthInterval stable steps
                .backoffFactor(0.5)      // Halve scale on overflow
                .growthInterval(2000)    // Steps between scale increases
                .build();
        log.info("Full config: enabled={}, isDynamic={}", fullConfig.isEnabled(), fullConfig.isDynamic());

        // =====================================================================
        // 12. Complete Transfer Learning Workflow
        // =====================================================================
        log.info("\n=== 12. Complete Transfer Learning Workflow ===");
        log.info("");
        log.info("  // 1. Load pre-trained model");
        log.info("  SameDiff baseModel = SameDiff.load(pretrainedFile, false);");
        log.info("");
        log.info("  // 2. Freeze encoder, keep head trainable");
        log.info("  baseModel.freezePrefix(\"encoder.\");");
        log.info("");
        log.info("  // 3. Optionally replace the head for a new task");
        log.info("  // (add new variables and ops)");
        log.info("");
        log.info("  // 4. Apply LoRA for efficient fine-tuning");
        log.info("  PeftModel peft = PeftModel.fromPretrained(baseModel,");
        log.info("      LoraConfig.defaultTransformer());");
        log.info("");
        log.info("  // 5. Configure training with gradient checkpointing");
        log.info("  peft.setTrainingConfig(TrainingConfig.builder()");
        log.info("      .updater(new Adam(1e-4))");
        log.info("      .computeDataType(DataType.BFLOAT16)");
        log.info("      .gradientAccumulationSteps(4)");
        log.info("      .build());");
        log.info("");
        log.info("  // 6. Train");
        log.info("  peft.fit(trainData, numEpochs);");
        log.info("");
        log.info("  // 7. Merge and export for deployment");
        log.info("  SameDiff deployModel = peft.mergeAndUnload();");
        log.info("  deployModel.save(outputFile, false);");
        log.info("");
        log.info("  // 8. Or save just the adapter (smaller file)");
        log.info("  peft.saveAdapter(adapterDir);");
        log.info("  // Later: PeftModel.fromPretrained(baseModel, adapterDir);");

        log.info("\n**************** Transfer Learning and Freezing Example finished ********************");
    }
}
