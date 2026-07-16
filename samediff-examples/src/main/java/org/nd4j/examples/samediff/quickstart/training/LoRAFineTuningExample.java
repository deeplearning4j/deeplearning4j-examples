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
import org.nd4j.autodiff.samediff.TransferLearning;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.LoraConfig;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.peft.PeftModel;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.adapter.SingletonDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * End-to-End LoRA Fine-Tuning Example using SameDiff and PeftModel.
 *
 * <p>Low-Rank Adaptation (LoRA) enables parameter-efficient fine-tuning by injecting
 * trainable low-rank decompositions into frozen base model weights:
 * <pre>
 *   W_effective = W_frozen + (alpha/r) * B @ A
 *
 *   where:
 *     W_frozen  is the original pre-trained weight [d_out, d_in]
 *     A          is the up-projection matrix      [r,     d_in]  (Kaiming init)
 *     B          is the down-projection matrix    [d_out, r]     (zero init)
 *     r          is the rank (r << min(d_out, d_in))
 *     alpha      is the scaling factor
 * </pre>
 *
 * <p>Because B is zero-initialized the adapter starts as an identity transform —
 * the model behaves identically to the base model at step 0.</p>
 *
 * <h3>Topics Covered:</h3>
 * <ol>
 *   <li>Build a 4-layer MLP base model with named encoder/decoder layers</li>
 *   <li>Create a {@link LoraConfig} and wrap the model with {@link PeftModel}</li>
 *   <li>Forward pass through PeftModel — shapes and values</li>
 *   <li>Training with PeftModel.fit() using a synthetic DataSet</li>
 *   <li>Merge adapters and verify the deployed model</li>
 *   <li>Save / reload adapter weights from disk</li>
 *   <li>TransferLearning.Builder pattern for freeze + LoRA in one chain</li>
 *   <li>LoRA preset factories: defaultTransformer, allLinear, minimal</li>
 *   <li>Toggle adapter on / off and observe output changes</li>
 *   <li>LoRA scaling math: standard (alpha/r) vs rsLoRA (alpha/sqrt(r))</li>
 * </ol>
 *
 * <p>Run with:</p>
 * <pre>
 *   cd samediff-examples
 *   mvn exec:java -Dexec.mainClass="org.nd4j.examples.samediff.quickstart.training.LoRAFineTuningExample"
 * </pre>
 */
public class LoRAFineTuningExample {
    private static final Logger log = LoggerFactory.getLogger(LoRAFineTuningExample.class);

    // -------------------------------------------------------------------------
    // Shared model dimensions used throughout the example
    // -------------------------------------------------------------------------
    private static final int INPUT_DIM  = 128;
    private static final int H1_DIM     = 256;
    private static final int H2_DIM     = 128;
    private static final int H3_DIM     = 64;
    private static final int OUTPUT_DIM = 10;
    private static final int BATCH_SIZE = 16;

    public static void main(String[] args) throws Exception {

        // =====================================================================
        // 1. Build a Base Model
        //    4-layer MLP: 128 -> 256 -> 128 -> 64 -> 10
        //    Layers named "encoder.layer.0.*", "encoder.layer.1.*",
        //                  "decoder.layer.0.*", "head.*"
        // =====================================================================
        log.info("=== 1. Build Base Model (4-layer MLP: 128→256→128→64→10) ===");

        SameDiff sd = buildBaseModel();

        // Count trainable (VARIABLE) and total parameters
        long trainableParams = sd.variables().stream()
                .filter(v -> v.getVariableType() == VariableType.VARIABLE)
                .mapToLong(v -> v.getArr() != null ? v.getArr().length() : 0L)
                .sum();
        long trainableVarCount = sd.variables().stream()
                .filter(v -> v.getVariableType() == VariableType.VARIABLE)
                .count();

        log.info("  Variables in model: {}", trainableVarCount);
        log.info("  Total trainable parameters: {}", trainableParams);
        log.info("  Variable names:");
        sd.variables().stream()
                .filter(v -> v.getVariableType() == VariableType.VARIABLE)
                .forEach(v -> log.info("    {} shape={}", v.name(), Arrays.toString(v.getShape())));

        // Quick sanity-check forward pass on the raw base model
        Map<String, INDArray> rawInputs = new HashMap<>();
        rawInputs.put("input", Nd4j.randn(DataType.FLOAT, BATCH_SIZE, INPUT_DIM).muli(0.1f));
        Map<String, INDArray> rawOut = sd.output(rawInputs, "output");
        log.info("  Base model output shape: {}", Arrays.toString(rawOut.get("output").shape()));
        log.info("  Base model output[0:3]: {}", rawOut.get("output").getRow(0).get(
                org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 3)));

        // =====================================================================
        // 2. LoRA Basics — LoraConfig + PeftModel.fromPretrained()
        //    Rank=8, alpha=16, applied to all weight matrices in encoder and decoder
        // =====================================================================
        log.info("\n=== 2. LoRA Basics (r=8, alpha=16) ===");

        // LoRA targets: the weight matrices we want to adapt.
        // Pattern matching uses regex, so "encoder.layer.0.weight" is matched literally.
        LoraConfig loraConfig = LoraConfig.builder()
                .r(8)
                .loraAlpha(16)
                .loraDropout(0.05)
                .targetModules(Arrays.asList(
                        "encoder.layer.0.weight",
                        "encoder.layer.1.weight",
                        "decoder.layer.0.weight",
                        "head.weight"
                ))
                .build();

        log.info("  LoRA config: r={}, alpha={}, scaling={}",
                loraConfig.getR(), loraConfig.getLoraAlpha(), loraConfig.getScaling());
        log.info("  Summary: {}", loraConfig.getSummary());

        // Wrap the base model — base weights are frozen, LoRA matrices are trainable
        PeftModel peftModel = PeftModel.fromPretrained(sd, loraConfig);

        long trainable = peftModel.getTrainableParameterCount();
        long total     = peftModel.getTotalParameterCount();
        log.info("  Trainable parameters: {} / {} ({}%)",
                trainable, total, String.format("%.4f", peftModel.getTrainablePercentage()));
        peftModel.printTrainableParameters();

        // Full summary shows LoRA config + parameter breakdown
        String summary = peftModel.getSummary();
        log.info("  PeftModel summary:\n{}", summary);

        // =====================================================================
        // 3. Forward Pass Through PeftModel
        //    W_effective = W_frozen + (alpha/r) * B @ A  for each target module.
        //    At init B=0 so output matches the base model exactly.
        // =====================================================================
        log.info("\n=== 3. Forward Pass Through PeftModel ===");

        Map<String, INDArray> placeholders = new HashMap<>();
        INDArray featuresBatch = Nd4j.randn(DataType.FLOAT, BATCH_SIZE, INPUT_DIM).muli(0.1f);
        placeholders.put("input", featuresBatch);

        Map<String, INDArray> peftOutput = peftModel.output(placeholders, "output");
        INDArray outArr = peftOutput.get("output");

        log.info("  PeftModel output shape: {}", Arrays.toString(outArr.shape()));
        log.info("  Output[0] (first sample, all 10 classes): {}", outArr.getRow(0));
        log.info("  Output[0] sum (softmax should sum to ~1.0): {}",
                String.format("%.6f", outArr.getRow(0).sumNumber().doubleValue()));

        // Show the LoRA-injected merged weight for one target module
        INDArray mergedW0 = peftModel.getMergedWeight("encoder.layer.0.weight");
        if (mergedW0 != null) {
            log.info("  Merged encoder.layer.0.weight shape: {}", Arrays.toString(mergedW0.shape()));
            log.info("  Merged weight norm: {}", String.format("%.6f", mergedW0.norm2Number().doubleValue()));
        }

        // =====================================================================
        // 4. Training With PeftModel
        //    - Only LoRA matrices (A and B) are trainable; base weights are frozen.
        //    - Use a synthetic DataSet with Nd4j.randn features and one-hot labels.
        //    - Wrap in SingletonDataSetIterator and call peftModel.fit().
        // =====================================================================
        log.info("\n=== 4. Training With PeftModel (2 epochs, synthetic data) ===");

        // Build a synthetic training batch:  features [16, 128], labels [16, 10] one-hot
        INDArray features    = Nd4j.randn(DataType.FLOAT, BATCH_SIZE, INPUT_DIM);
        INDArray labels      = makeOneHot(BATCH_SIZE, OUTPUT_DIM);
        DataSet  trainingSet = new DataSet(features, labels);

        // SingletonDataSetIterator wraps a single DataSet; reset() lets it be reused
        DataSetIterator iterator = new SingletonDataSetIterator(trainingSet);

        // TrainingConfig: map DataSet features→"input", labels→"label"; use Adam
        TrainingConfig trainingConfig = TrainingConfig.builder()
                .updater(new Adam(1e-3))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .build();

        peftModel.setTrainingConfig(trainingConfig);

        log.info("  Starting training: 2 epochs, batch={}, lr=1e-3", BATCH_SIZE);
        peftModel.fit(iterator, 2);
        log.info("  Training complete.");

        // Verify output after training — values should have shifted
        Map<String, INDArray> afterTrainOut = peftModel.output(placeholders, "output");
        log.info("  Post-training output[0]: {}", afterTrainOut.get("output").getRow(0));
        log.info("  Post-training output[0] sum: {}",
                String.format("%.6f", afterTrainOut.get("output").getRow(0).sumNumber().doubleValue()));

        // =====================================================================
        // 5. Merge and Deploy
        //    peftModel.mergeAndUnload() absorbs B @ A into W for each target module,
        //    returning a plain SameDiff model with no adapter overhead.
        // =====================================================================
        log.info("\n=== 5. Merge and Deploy ===");

        SameDiff mergedModel = peftModel.mergeAndUnload();

        log.info("  Merged model variable count: {}", mergedModel.variables().size());

        // Forward pass through merged model — should produce identical output to peftModel
        Map<String, INDArray> mergedOut = mergedModel.output(placeholders, "output");
        log.info("  Merged model output shape: {}", Arrays.toString(mergedOut.get("output").shape()));
        log.info("  Merged model output[0]: {}", mergedOut.get("output").getRow(0));

        // Compare shapes of merged vs base model weights for one layer
        INDArray baseW0  = sd.getVariable("encoder.layer.0.weight").getArr();
        INDArray mergedW = mergedModel.getVariable("encoder.layer.0.weight").getArr();
        log.info("  Base    encoder.layer.0.weight shape: {}", Arrays.toString(baseW0.shape()));
        log.info("  Merged  encoder.layer.0.weight shape: {}", Arrays.toString(mergedW.shape()));
        log.info("  Weights differ (LoRA was applied): {}",
                !baseW0.equalsWithEps(mergedW, 1e-6));

        // =====================================================================
        // 6. Save / Load Adapter
        //    saveAdapter() writes adapter_weight_N.npy files to a directory.
        //    PeftModel.fromPretrained(baseModel, adapterDir) reloads from disk.
        //    NOTE: config serialization is not yet fully implemented; the reload
        //    path with fromPretrained(File) requires a complete adapter_config.json.
        //    This section shows the save path and the equivalent config-driven reload.
        // =====================================================================
        log.info("\n=== 6. Save / Load Adapter ===");

        File adapterDir = new File(System.getProperty("java.io.tmpdir"), "lora_adapter_example");
        adapterDir.mkdirs();
        log.info("  Saving adapter to: {}", adapterDir.getAbsolutePath());

        peftModel.saveAdapter(adapterDir);

        File[] savedFiles = adapterDir.listFiles();
        if (savedFiles != null) {
            log.info("  Saved {} adapter file(s):", savedFiles.length);
            for (File f : savedFiles) {
                log.info("    {} ({} bytes)", f.getName(), f.length());
            }
        }

        // Reload: create a fresh base model and apply the same LoraConfig.
        // In production, peftModel.saveAdapter() would also write adapter_config.json,
        // and PeftModel.fromPretrained(freshBase, adapterDir) would deserialize it.
        // For now we demonstrate the equivalent config-driven approach:
        SameDiff freshBase     = buildBaseModel();
        PeftModel reloadedPeft = PeftModel.fromPretrained(freshBase, loraConfig);
        log.info("  Reloaded PeftModel trainable params: {}",
                reloadedPeft.getTrainableParameterCount());

        // Verify the reloaded model runs inference
        Map<String, INDArray> reloadOut = reloadedPeft.output(placeholders, "output");
        log.info("  Reloaded model output shape: {}", Arrays.toString(reloadOut.get("output").shape()));

        // =====================================================================
        // 7. TransferLearning.Builder Pattern
        //    freeze() a prefix then attach LoRA via .lora() and call .buildPeft().
        //    The Builder also supports reinitialize(), fineTuneConfiguration(), etc.
        // =====================================================================
        log.info("\n=== 7. TransferLearning.Builder (freeze prefix + LoRA) ===");

        // Build a fresh base for the builder demo
        SameDiff builderBase = buildBaseModel();

        // Chain: freeze encoder, add LoRA to decoder weight and head weight only
        PeftModel builderPeft = new TransferLearning.Builder(builderBase)
                .freezePrefix("encoder.")               // freeze all encoder.* variables
                .lora(8, 16, Arrays.asList(             // r=8, alpha=16
                        "decoder.layer.0.weight",
                        "head.weight"))
                .buildPeft();

        log.info("  Builder PeftModel trainable params: {}",
                builderPeft.getTrainableParameterCount());
        log.info("  Builder PeftModel total params: {}",
                builderPeft.getTotalParameterCount());
        log.info("  Builder PeftModel trainable %: {}%",
                String.format("%.4f", builderPeft.getTrainablePercentage()));

        // Forward pass through builder-constructed PeftModel
        Map<String, INDArray> builderOut = builderPeft.output(placeholders, "output");
        log.info("  Builder model output shape: {}", Arrays.toString(builderOut.get("output").shape()));
        log.info("  Builder model output[0][0:3]: {}",
                builderOut.get("output").getRow(0).get(
                        org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 3)));

        // =====================================================================
        // 8. LoRA Preset Factories
        //    defaultTransformer() — r=16, alpha=32, standard transformer targets
        //    allLinear(rank)      — all linear projection layers, alpha=2*rank
        //    minimal()            — r=4, alpha=8, query+value only, no dropout
        // =====================================================================
        log.info("\n=== 8. LoRA Preset Factories ===");

        LoraConfig preset1 = LoraConfig.defaultTransformer();
        log.info("  defaultTransformer(): r={}, alpha={}, dropout={}, targets={}",
                preset1.getR(), preset1.getLoraAlpha(),
                preset1.getLoraDropout(), preset1.getTargetModules());

        LoraConfig preset2 = LoraConfig.allLinear(32);
        log.info("  allLinear(32):        r={}, alpha={}, dropout={}, targets={}",
                preset2.getR(), preset2.getLoraAlpha(),
                preset2.getLoraDropout(), preset2.getTargetModules());

        LoraConfig preset3 = LoraConfig.minimal();
        log.info("  minimal():            r={}, alpha={}, dropout={}, targets={}",
                preset3.getR(), preset3.getLoraAlpha(),
                preset3.getLoraDropout(), preset3.getTargetModules());

        // Apply the minimal preset to our base model and print trainable count
        SameDiff presetBase  = buildBaseModel();
        // minimal() targets "query" and "value" — not present in our MLP names,
        // so no LoRA matrices are injected (0 trainable PEFT params).
        // Demonstrate the expected outcome explicitly.
        LoraConfig mlpMinimal = LoraConfig.builder()
                .r(4)
                .loraAlpha(8)
                .loraDropout(0.0)
                .targetModules(Arrays.asList("encoder.layer.0.weight", "head.weight"))
                .build();
        PeftModel minimalPeft = PeftModel.fromPretrained(presetBase, mlpMinimal);
        log.info("  MLP-minimal preset: trainable={}, total={}",
                minimalPeft.getTrainableParameterCount(),
                minimalPeft.getTotalParameterCount());

        // =====================================================================
        // 9. Adapter Enable / Disable
        //    disableAdapter() zeros out the B matrices → output equals base model.
        //    enableAdapter() re-enables (logs; user must re-run forward pass).
        //    Here we record output before and after to show the change.
        // =====================================================================
        log.info("\n=== 9. Adapter Enable / Disable ===");

        // Use the trained peftModel from section 4 — its B matrices are non-zero
        Map<String, INDArray> probeInputs = new HashMap<>();
        probeInputs.put("input", Nd4j.randn(DataType.FLOAT, 4, INPUT_DIM).muli(0.1f));

        // Output WITH adapter active (B matrices have been updated by training)
        Map<String, INDArray> outWithAdapter = peftModel.output(probeInputs, "output");
        log.info("  Output WITH adapter (trained B matrices):");
        log.info("    sample[0]: {}", outWithAdapter.get("output").getRow(0));

        // Disable adapter: sets all B matrices to zero → W_effective = W_frozen + 0 = W_frozen
        peftModel.disableAdapter();
        Map<String, INDArray> outNoAdapter = peftModel.output(probeInputs, "output");
        log.info("  Output WITHOUT adapter (B=0, pure base weights):");
        log.info("    sample[0]: {}", outNoAdapter.get("output").getRow(0));

        // Confirm the two outputs differ (adapter was non-trivial after training)
        double diff = outWithAdapter.get("output").getRow(0)
                .distance2(outNoAdapter.get("output").getRow(0));
        log.info("  L2 distance between adapter-on vs adapter-off: {}", String.format("%.6f", diff));

        // Re-enable and verify output returns to adapter-active values
        peftModel.enableAdapter();
        log.info("  Adapter re-enabled (active adapter: {})", peftModel.getActiveAdapter());

        // =====================================================================
        // 10. LoRA Scaling Math
        //     Standard:  scale = alpha / r
        //     rsLoRA:    scale = alpha / sqrt(r)
        //     rsLoRA maintains stable gradient norms when increasing r.
        // =====================================================================
        log.info("\n=== 10. LoRA Scaling Math ===");

        int[] ranks  = {4, 8, 16, 32, 64};
        int   alpha  = 16;

        log.info("  Standard LoRA scaling (alpha={}, scale = alpha/r):", alpha);
        for (int r : ranks) {
            LoraConfig std = LoraConfig.builder()
                    .r(r).loraAlpha(alpha).useRsLora(false)
                    .targetModules(Arrays.asList("encoder.layer.0.weight"))
                    .build();
            log.info("    r={}  scale = {}", String.format("%2d", r), String.format("%.6f", std.getScaling()));
        }

        log.info("  rsLoRA scaling (alpha={}, scale = alpha/sqrt(r)):", alpha);
        for (int r : ranks) {
            LoraConfig rs = LoraConfig.builder()
                    .r(r).loraAlpha(alpha).useRsLora(true)
                    .targetModules(Arrays.asList("encoder.layer.0.weight"))
                    .build();
            log.info("    r={}  scale = {}  (rsLoRA)", String.format("%2d", r), String.format("%.6f", rs.getScaling()));
        }

        log.info("  Observation: standard scaling decreases as r grows (unstable at high rank).");
        log.info("  rsLoRA keeps scaling in a tighter range, improving training stability.");

        // Side-by-side for r=8 and r=64
        LoraConfig stdR8  = LoraConfig.builder().r(8).loraAlpha(16).useRsLora(false)
                .targetModules(Arrays.asList("encoder.layer.0.weight")).build();
        LoraConfig rsR8   = LoraConfig.builder().r(8).loraAlpha(16).useRsLora(true)
                .targetModules(Arrays.asList("encoder.layer.0.weight")).build();
        LoraConfig stdR64 = LoraConfig.builder().r(64).loraAlpha(16).useRsLora(false)
                .targetModules(Arrays.asList("encoder.layer.0.weight")).build();
        LoraConfig rsR64  = LoraConfig.builder().r(64).loraAlpha(16).useRsLora(true)
                .targetModules(Arrays.asList("encoder.layer.0.weight")).build();

        log.info("  r=8,  std scaling: {}  rs scaling: {}",
                String.format("%.4f", stdR8.getScaling()),  String.format("%.4f", rsR8.getScaling()));
        log.info("  r=64, std scaling: {}  rs scaling: {}",
                String.format("%.4f", stdR64.getScaling()), String.format("%.4f", rsR64.getScaling()));

        // =====================================================================
        // 11. DSP-Accelerated LoRA Training
        //     DSP (Dynamic Shape Plan) compiles the full training graph
        //     (forward + backward + updater) into a flat-slot dispatch plan
        //     for minimal per-step overhead. DSP is enabled by default —
        //     both dspAutoCompileEnabled and dspNativeAutoCompileEnabled
        //     default to true. Fixed batch size is required for steady-state.
        // =====================================================================
        log.info("\n=== 11. DSP-Accelerated LoRA Training ===");

        // Build a fresh base model to avoid entanglement with sections 1-10
        SameDiff dspSd = buildBaseModel();

        // Apply LoRA (r=8, alpha=16) to all weight matrices
        LoraConfig dspLoraConfig = LoraConfig.builder()
                .r(8)
                .loraAlpha(16)
                .loraDropout(0.0)
                .targetModules(Arrays.asList(
                        "encoder.layer.0.weight",
                        "encoder.layer.1.weight",
                        "decoder.layer.0.weight",
                        "head.weight"
                ))
                .build();

        PeftModel dspPeft = PeftModel.fromPretrained(dspSd, dspLoraConfig);

        // Configure training: Adam lr=1e-3, map DataSet feature/label names
        TrainingConfig dspTrainingConfig = TrainingConfig.builder()
                .updater(new Adam(1e-3))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .build();
        dspPeft.setTrainingConfig(dspTrainingConfig);

        // Confirm DSP is enabled on the underlying SameDiff — both flags default true
        log.info("  dspAutoCompileEnabled:       {}", dspSd.isDspAutoCompileEnabled());
        log.info("  dspNativeAutoCompileEnabled: {}", dspSd.isDspNativeAutoCompileEnabled());
        log.info("  Trainable LoRA params: {}", dspPeft.getTrainableParameterCount());

        // Fixed-batch DataSet — a constant shape is REQUIRED for DSP steady-state
        INDArray dspFeatures = Nd4j.randn(DataType.FLOAT, BATCH_SIZE, INPUT_DIM);
        INDArray dspLabels   = makeOneHot(BATCH_SIZE, OUTPUT_DIM);
        DataSet  ds          = new DataSet(dspFeatures, dspLabels);

        // Timing arrays for warmup vs steady-state comparison
        long[] stepTimeMs = new long[10];

        log.info("  Running 10 training steps (fixed batch={}, lr=1e-3):", BATCH_SIZE);
        log.info("  Step  | Time(ms) | Compiled | Phase           | Segs | Replayed | PtrsStable");
        log.info("  ------+----------+----------+-----------------+------+----------+-----------");

        for (int step = 0; step < 10; step++) {
            long t0 = System.nanoTime();
            dspPeft.fit(new SingletonDataSetIterator(ds), 1);
            long t1 = System.nanoTime();
            stepTimeMs[step] = (t1 - t0) / 1_000_000L;

            DspHandle h = dspSd.dsp();
            boolean compiled    = h.isCompiled();
            int     numSegs     = compiled ? h.numSegments()          : -1;
            int     replayed    = compiled ? h.lastExecSegmentsReplayed() : -1;
            boolean ptrsStable  = compiled && h.pointersStable();
            PlanPhase phase     = compiled ? PlanPhase.fromNativeCode(h.planPhase()) : null;
            String  phaseStr    = phase != null ? phase.name() : "NOT_COMPILED";

            log.info("  {} | {} | {} | {} | {} | {} | {}",
                    String.format("%5d", step),
                    String.format("%8d", stepTimeMs[step]),
                    String.format("%8s", compiled ? "YES" : "NO"),
                    String.format("%-15s", phaseStr),
                    String.format("%4d", numSegs),
                    String.format("%8d", replayed),
                    ptrsStable);
        }

        // Post-training DSP plan summary
        DspHandle finalHandle = dspSd.dsp();
        if (finalHandle.isCompiled()) {
            log.info("\n  DSP Plan Summary after 10 training steps:");
            log.info("    totalSlots:              {}", finalHandle.totalSlots());
            log.info("    numSegments:             {}", finalHandle.numSegments());
            log.info("    numCapturedGraphSegs:    {}", finalHandle.numCapturedGraphSegments());
            log.info("    totalGraphReplays:       {}", finalHandle.totalGraphReplays());
            log.info("    pointersStable:          {}", finalHandle.pointersStable());
            log.info("    planPhase:               {}",
                    PlanPhase.fromNativeCode(finalHandle.planPhase()));
            log.info("    executeCount:            {}", finalHandle.executeCount());
        } else {
            log.info("  DSP plan not compiled (CPU backend or DSP disabled).");
        }

        // Warmup vs steady-state timing comparison
        // Steps 0-1 are warmup (SLOT_BY_SLOT/SHAPES_FROZEN), step 9 is steady-state
        double warmupAvgMs = (stepTimeMs[0] + stepTimeMs[1]) / 2.0;
        double steadyAvgMs = 0.0;
        for (int i = 7; i < 10; i++) steadyAvgMs += stepTimeMs[i];
        steadyAvgMs /= 3.0;
        log.info("\n  Warmup average  (steps 0-1): {} ms", String.format("%.1f", warmupAvgMs));
        log.info("  Steady average  (steps 7-9): {} ms", String.format("%.1f", steadyAvgMs));
        if (warmupAvgMs > 0) {
            log.info("  Speedup (warmup/steady):     {}x",
                    String.format("%.2f", warmupAvgMs / steadyAvgMs));
        }

        log.info("\n**************** LoRA Fine-Tuning Example finished ********************");
    }

    // =========================================================================
    // Helper: build the 4-layer base MLP
    // Architecture: 128 → 256 → 128 → 64 → 10 (softmax output)
    // Named to mirror a real encoder-decoder naming convention.
    // =========================================================================
    private static SameDiff buildBaseModel() {
        SameDiff sd = SameDiff.create();

        // Placeholders — dynamic batch size (-1)
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, INPUT_DIM);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, OUTPUT_DIM);

        // Encoder layer 0: 128 → 256
        SDVariable encW0 = sd.var("encoder.layer.0.weight",
                Nd4j.randn(DataType.FLOAT, INPUT_DIM, H1_DIM).muli(0.02f));
        SDVariable encB0 = sd.var("encoder.layer.0.bias",
                Nd4j.zeros(DataType.FLOAT, H1_DIM));

        // Encoder layer 1: 256 → 128
        SDVariable encW1 = sd.var("encoder.layer.1.weight",
                Nd4j.randn(DataType.FLOAT, H1_DIM, H2_DIM).muli(0.02f));
        SDVariable encB1 = sd.var("encoder.layer.1.bias",
                Nd4j.zeros(DataType.FLOAT, H2_DIM));

        // Decoder layer 0: 128 → 64
        SDVariable decW0 = sd.var("decoder.layer.0.weight",
                Nd4j.randn(DataType.FLOAT, H2_DIM, H3_DIM).muli(0.02f));
        SDVariable decB0 = sd.var("decoder.layer.0.bias",
                Nd4j.zeros(DataType.FLOAT, H3_DIM));

        // Head (classification): 64 → 10
        SDVariable headW = sd.var("head.weight",
                Nd4j.randn(DataType.FLOAT, H3_DIM, OUTPUT_DIM).muli(0.02f));
        SDVariable headB = sd.var("head.bias",
                Nd4j.zeros(DataType.FLOAT, OUTPUT_DIM));

        // Forward pass
        SDVariable h0     = sd.nn.relu(input.mmul(encW0).add(encB0), 0);
        SDVariable h1     = sd.nn.relu(h0.mmul(encW1).add(encB1), 0);
        SDVariable h2     = sd.nn.relu(h1.mmul(decW0).add(decB0), 0);
        SDVariable logits = h2.mmul(headW).add(headB);
        sd.nn.softmax("output", logits, -1);

        // Loss (required for training)
        sd.setLossVariables(
                sd.loss.softmaxCrossEntropy("loss", label, logits, null));

        return sd;
    }

    // =========================================================================
    // Helper: create a one-hot label matrix [batchSize, numClasses]
    // Each row has a single 1.0 at a random class index.
    // =========================================================================
    private static INDArray makeOneHot(int batchSize, int numClasses) {
        INDArray labels = Nd4j.zeros(DataType.FLOAT, batchSize, numClasses);
        for (int i = 0; i < batchSize; i++) {
            int cls = (int) (Math.random() * numClasses);
            labels.putScalar(new int[]{i, cls}, 1.0f);
        }
        return labels;
    }
}
