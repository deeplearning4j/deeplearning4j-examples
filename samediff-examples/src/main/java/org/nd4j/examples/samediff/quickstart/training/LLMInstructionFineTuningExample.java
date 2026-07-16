/* *****************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.nd4j.examples.samediff.quickstart.training;

import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.LLMModel;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.QuantType;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.config.LoraConfig;
import org.nd4j.autodiff.samediff.config.SFTConfig;
import org.nd4j.autodiff.samediff.peft.PeftModel;
import org.nd4j.autodiff.samediff.training.SFTTrainingPipeline;
import org.nd4j.ggml.GGMLModelExport;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SimpleListMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.schedule.CosineWarmupSchedule;
import org.nd4j.weightinit.impl.XavierInitScheme;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.file.Files;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Instruction fine-tuning (SFT) and LoRA adaptation of a causal language model using the
 * REAL LLM data pipeline: a production BPE tokenizer, a real chat template, and exact
 * token-level response masking — not the synthetic random-vector data of the earlier
 * training examples.
 *
 * <p>What runs here, end to end:</p>
 * <ol>
 *   <li><b>Real tokenizer + chat template.</b> The Qwen3.5 tokenizer (151k-token BPE,
 *       ChatML template) is pulled from the GGUF model cache via {@link LLMModelDownloader}
 *       and used through the same {@link Tokenizer} API the generation pipeline uses.</li>
 *   <li><b>Exact response masking.</b> Each conversation is rendered twice with
 *       {@link Tokenizer#applyChatTemplate}: once up to the assistant header (prompt only)
 *       and once in full. The longest common token prefix separates prompt tokens from
 *       response tokens, giving an exact token-level loss mask — the production alternative
 *       to the {@code charsPerToken} heuristic in
 *       {@link SFTTrainingPipeline#convertCharMaskToTokenMask(int[], int)}.</li>
 *   <li><b>A real transformer.</b> The model is a compact LLaMA-style decoder built from
 *       the same fused ops the inference stack executes — {@code rms_norm},
 *       {@code dot_product_attention_v2} with causal masking, SwiGLU MLPs — with
 *       HuggingFace-style parameter names ({@code model.layers.N.self_attn.q_proj.weight}),
 *       so PEFT target-module matching behaves exactly as it does on imported models.</li>
 *   <li><b>Three layers of the training API.</b> Full fine-tune via
 *       {@link TrainingConfig} + {@link SameDiff#fit} (masked causal-LM loss, AdamW,
 *       cosine warmup schedule), LoRA adaptation via {@link PeftModel} (frozen base
 *       verified, adapter toggling, merge-and-unload, adapter save), and the
 *       {@link SFTTrainingPipeline} orchestration API driven by the same real-tokenized
 *       iterator.</li>
 *   <li><b>Real evaluation.</b> Held-out masked NLL/perplexity before and after training,
 *       plus greedy generations from the model at each stage.</li>
 * </ol>
 *
 * <p><b>Scope note.</b> The base model here starts from random weights and trains on a
 * small built-in corpus, so generations demonstrate learned <i>format and domain tokens</i>
 * rather than general knowledge — the pipeline, not the checkpoint, is the subject. The
 * identical code fine-tunes any SameDiff causal LM exposing {@code input_ids} /
 * {@code labels} / {@code loss_mask} placeholders. GGUF models imported with
 * {@code GGMLModelImport} expose KV-cache/positional placeholders for incremental decoding
 * and are therefore driven through {@code GenerationPipeline} for inference; their
 * fine-tuning story is PEFT adapters over the imported weights rather than {@code fit()}
 * on the decode graph.</p>
 *
 * <p>System properties: {@code -Dexample.train.epochs=6}, {@code -Dexample.adapt.epochs=4},
 * {@code -Dexample.gen.tokens=24}.</p>
 *
 * <p><b>CPU throughput note.</b> If training runs on a single core (some packaged
 * nd4j-native builds pin OpenBLAS to one thread), export
 * {@code OPENBLAS_NUM_THREADS=<physical cores>} in the environment before launching —
 * on a 12-core box that is roughly a 10x speedup for the vocab-sized matmuls here.
 * On the CUDA backend this example's training steps take milliseconds.</p>
 */
public class LLMInstructionFineTuningExample {

    private static final Logger log = LoggerFactory.getLogger(LLMInstructionFineTuningExample.class);

    // Model geometry: compact on purpose (~31M params, dominated by the 151k-vocab
    // embedding) so full fine-tuning runs in minutes on CPU.
    private static final int SEQ_LEN = 96;
    private static final int BATCH = 4;
    private static final int HIDDEN = 192;
    private static final int LAYERS = 4;
    private static final int HEADS = 4;
    private static final int FFN = 512;
    private static final long SEED = 12345;

    private static final String SYSTEM_MESSAGE =
            "You are a concise assistant for the ND4J and SameDiff libraries.";

    /** Primary SFT corpus: instruction/response pairs about the ND4J/SameDiff stack. */
    private static final String[][] TRAIN_PAIRS = {
            {"What is SameDiff?", "SameDiff is ND4J's automatic differentiation engine for defining and training computation graphs."},
            {"How do I create an INDArray of zeros?", "Call Nd4j.zeros(rows, cols) to allocate a zero-filled INDArray."},
            {"Which class imports a GGUF model?", "GGMLModelImport.importModel(file) imports a GGUF model into a SameDiff graph."},
            {"How do I run a forward pass in SameDiff?", "Call sd.output(placeholderMap, outputName) to execute the graph and fetch outputs."},
            {"What does TrainingConfig control?", "TrainingConfig sets the updater, data mappings, regularization and precision used by SameDiff.fit."},
            {"How do I enable LoRA fine-tuning?", "Wrap the model with PeftModel.fromPretrained(model, loraConfig) so only adapter weights train."},
            {"What optimizer is recommended for fine-tuning?", "AdamW with a cosine warmup schedule is the standard choice for fine-tuning."},
            {"How do I save a trained model?", "Call sd.save(file, true) to persist the graph together with its updater state."},
            {"What is a placeholder in SameDiff?", "A placeholder is a graph input whose value is supplied at execution time."},
            {"How do I compute gradients manually?", "Use sd.calculateGradients(placeholders, variableNames) to get gradients per variable."},
            {"What does markAsLoss do?", "markAsLoss registers a scalar variable as the training loss minimized by fit."},
            {"Which op fuses RMS norm and matmul?", "The rms_norm_linear fused op combines RMS normalization with a linear projection."},
            {"How do I tokenize text for an LLM?", "Use HuggingFaceTokenizer.fromFile(tokenizerJson) and call encode(text, addSpecialTokens)."},
            {"What is the GenerationPipeline?", "GenerationPipeline wraps a decoder, tokenizer and sampling config for autoregressive text generation."},
            {"How do I sample with temperature?", "Build a SamplingConfig with temperature, topK and topP, then pass it to the pipeline config."},
            {"What is the DSP in ND4J?", "The dynamic shape plan compiles SameDiff graphs into reusable execution plans with frozen shapes."},
            {"How do I freeze base weights?", "Use PEFT adapters or TransferLearning.freezePrefix so only selected parameters update."},
            {"What does the GraphOptimizer do?", "GraphOptimizer rewrites SameDiff graphs with fusion, folding and elimination passes before execution."},
            {"How do I evaluate perplexity?", "PerplexityEvaluator.evaluate runs sliding windows over text and reports perplexity and bits per byte."},
            {"What is gradient accumulation?", "Gradient accumulation sums gradients over several micro-batches before applying one optimizer step."},
            {"How do I merge LoRA adapters?", "Call peftModel.mergeAndUnload() to bake adapter deltas into the base weights."},
            {"What dtype should master weights use?", "Keep master weights in FLOAT while compute may run in FLOAT16 or BFLOAT16."},
            {"How do I export to GGUF?", "GGMLModelExport.exportModel(model, file) writes a SameDiff model to the GGUF format."},
            {"What is response masking?", "Response masking restricts the loss to assistant tokens so the model does not train on prompts."},
    };

    /** Adaptation corpus for the LoRA phase: a new domain (GPU/runtime configuration). */
    private static final String[][] ADAPT_PAIRS = {
            {"How do I select the CUDA backend?", "Set the nd4j.backend Maven property to nd4j-cuda-12.9-platform and rebuild."},
            {"How do I check GPU memory use?", "Call Nd4j.getEnvironment() and inspect the workspace and memory statistics it exposes."},
            {"What enables TF32 matmuls?", "Set the nd4j.cublas.tf32 system property to true before the first CUDA operation."},
            {"How do I pin execution to one GPU?", "Configure the affinity manager or set the device via Nd4j.getAffinityManager()."},
            {"What is CUDA graph capture?", "CUDA graph capture records a kernel sequence once and replays it with near-zero launch overhead."},
            {"How do I enable op timing?", "Set nd4j.op.timing to true to collect per-op execution timing statistics."},
            {"What does cudaMallocAsync provide?", "Stream-ordered allocation that avoids device-wide synchronization on memory operations."},
            {"How do I clear the Triton cache?", "Remove the compiled module cache directory; it repopulates on the next compilation."},
    };

    /** Held-out pairs (same domain as TRAIN_PAIRS) for before/after evaluation. */
    private static final String[][] EVAL_PAIRS = {
            {"How do I load a saved SameDiff model?", "Use SameDiff.load(file, true) to restore the graph and its updater state."},
            {"What does sd.fit return?", "fit returns a History object containing the loss curve and evaluation records."},
            {"How do I list model variables?", "Iterate sd.variables() and filter by VariableType to inspect parameters."},
            {"What is a frozen plan?", "A frozen plan has fixed shapes and stable pointers so replays skip recompilation."},
    };

    public static void main(String[] args) throws Exception {
        int trainEpochs = Integer.getInteger("example.train.epochs", 6);
        int adaptEpochs = Integer.getInteger("example.adapt.epochs", 4);
        int genTokens = Integer.getInteger("example.gen.tokens", 24);

        // ============================================================
        // Section 1: real tokenizer, real chat template, exact masking
        // ============================================================
        log.info("=== Section 1: Tokenizer, chat template and response masking ===");
        File gguf = LLMModelDownloader.download(LLMModel.QWEN35_0_8B, QuantType.Q4_K_M).getModelFile();
        Tokenizer tokenizer = HuggingFaceTokenizer.fromDirectory(gguf.getParentFile());
        int vocabSize = tokenizer.getVocabSize();
        log.info("Qwen3.5 tokenizer loaded: vocab={}, eosTokenId={}", vocabSize, tokenizer.getEosTokenId());

        List<TokenizedExample> trainData = tokenizeConversations(tokenizer, TRAIN_PAIRS);
        List<TokenizedExample> adaptData = tokenizeConversations(tokenizer, ADAPT_PAIRS);
        List<TokenizedExample> evalData = tokenizeConversations(tokenizer, EVAL_PAIRS);

        TokenizedExample sample = trainData.get(0);
        log.info("Masking check for: \"{}\"", TRAIN_PAIRS[0][0]);
        log.info("  total tokens={}, prompt tokens (mask=0)={}, response tokens (mask=1)={}",
                sample.length, sample.promptLength, sample.length - sample.promptLength);

        List<MultiDataSet> trainBatches = toBatches(trainData);
        List<MultiDataSet> adaptBatches = toBatches(adaptData);
        List<MultiDataSet> evalBatches = toBatches(evalData);
        log.info("Batches: train={}, adapt={}, eval={} (batch={}, seqLen={})",
                trainBatches.size(), adaptBatches.size(), evalBatches.size(), BATCH, SEQ_LEN);

        // ============================================================
        // Section 2: the base model + before-training metrics
        // ============================================================
        log.info("=== Section 2: Base model ===");
        SameDiff model = buildCausalLm(SEED, vocabSize);
        log.info("Built {}-layer causal LM: hidden={}, heads={}, ffn={}, params={}",
                LAYERS, HIDDEN, HEADS, FFN, String.format("%,d", countParams(model)));

        double[] before = heldOutNll(model, evalBatches);
        log.info("BEFORE training: held-out masked NLL={} perplexity={}",
                String.format("%.4f", before[0]), String.format("%.1f", before[1]));
        String beforeGen = greedyGenerate(model, tokenizer, EVAL_PAIRS[0][0], genTokens);
        log.info("BEFORE generation for \"{}\": \"{}\"", EVAL_PAIRS[0][0], beforeGen);

        // ============================================================
        // Section 3: full supervised fine-tune (TrainingConfig + fit)
        // ============================================================
        log.info("=== Section 3: Full SFT with masked causal-LM loss ===");
        int totalSteps = trainBatches.size() * trainEpochs;
        Adam adamW = Adam.builder()
                .learningRateSchedule(CosineWarmupSchedule.fromRatio(3e-4, 3e-5, 0.1, totalSteps))
                .beta1(0.9).beta2(0.999).epsilon(1e-8)
                .build();
        TrainingConfig trainingConfig = new TrainingConfig.Builder()
                .updater(adamW)
                .dataSetFeatureMapping("input_ids")
                .dataSetLabelMapping("labels")
                .dataSetFeatureMaskMapping("loss_mask")
                .build();
        model.setTrainingConfig(trainingConfig);

        MultiDataSetIterator trainIter = new SimpleListMultiDataSetIterator(trainBatches);
        long t0 = System.currentTimeMillis();
        History history = model.fit(trainIter, trainEpochs);
        long trainMs = System.currentTimeMillis() - t0;

        INDArray lossValues = history.getLossCurve().getLossValues();
        log.info("Trained {} steps in {}ms ({} steps/s)", totalSteps, trainMs,
                String.format("%.2f", totalSteps * 1000.0 / trainMs));
        log.info("Loss curve: first={} mid={} last={}",
                String.format("%.4f", lossValues.getDouble(0)),
                String.format("%.4f", lossValues.getDouble(lossValues.length() / 2)),
                String.format("%.4f", lossValues.getDouble(lossValues.length() - 1)));

        // ============================================================
        // Section 4: after-training metrics
        // ============================================================
        log.info("=== Section 4: After full SFT ===");
        double[] after = heldOutNll(model, evalBatches);
        log.info("AFTER training: held-out masked NLL={} perplexity={} (before: NLL={} ppl={})",
                String.format("%.4f", after[0]), String.format("%.1f", after[1]),
                String.format("%.4f", before[0]), String.format("%.1f", before[1]));
        for (String[] pair : new String[][]{EVAL_PAIRS[0], EVAL_PAIRS[1]}) {
            String gen = greedyGenerate(model, tokenizer, pair[0], genTokens);
            log.info("AFTER generation for \"{}\": \"{}\"", pair[0], gen);
        }

        // ============================================================
        // Section 5: LoRA adaptation to a new domain (PeftModel)
        // ============================================================
        log.info("=== Section 5: LoRA adaptation on the GPU-configuration domain ===");
        LoraConfig loraConfig = LoraConfig.builder()
                .r(8)
                .loraAlpha(16)
                .loraDropout(0.0)
                .targetModules(Arrays.asList("q_proj", "v_proj"))
                .build();
        PeftModel peft = PeftModel.fromPretrained(model, loraConfig);
        log.info("LoRA r={} alpha={} scaling={} targets={}", loraConfig.getR(),
                loraConfig.getLoraAlpha(), loraConfig.getScaling(), loraConfig.getTargetModules());
        log.info("Trainable params: {}/{} ({}%)",
                String.format("%,d", peft.getTrainableParameterCount()),
                String.format("%,d", peft.getTotalParameterCount()),
                String.format("%.3f", peft.getTrainablePercentage()));

        String frozenProbe = "model.layers.0.self_attn.q_proj.weight";
        INDArray baseWeightBefore = peft.getModel().getVariable(frozenProbe).getArr().dup();

        Adam adapterAdam = Adam.builder().learningRate(1e-3).beta1(0.9).beta2(0.999).epsilon(1e-8).build();
        peft.setTrainingConfig(new TrainingConfig.Builder()
                .updater(adapterAdam)
                .dataSetFeatureMapping("input_ids")
                .dataSetLabelMapping("labels")
                .dataSetFeatureMaskMapping("loss_mask")
                .build());
        peft.fit(new SimpleListMultiDataSetIterator(adaptBatches), adaptEpochs);

        INDArray baseWeightAfter = peft.getModel().getVariable(frozenProbe).getArr();
        double baseDrift = baseWeightBefore.sub(baseWeightAfter).norm2Number().doubleValue();
        log.info("Frozen-base check: L2 drift of {} after adapter training = {} (must be 0.0)",
                frozenProbe, baseDrift);

        // Adapter contribution: same batch, adapter off vs on.
        MultiDataSet probeBatch = adaptBatches.get(0);
        Map<String, INDArray> probeInput = Collections.singletonMap("input_ids", probeBatch.getFeatures(0));
        peft.disableAdapter();
        INDArray logitsBase = peft.output(probeInput, "logits").get("logits").dup();
        peft.enableAdapter();
        INDArray logitsAdapted = peft.output(probeInput, "logits").get("logits");
        double adapterDelta = logitsAdapted.sub(logitsBase).norm2Number().doubleValue();
        log.info("Adapter toggle: L2(logits_on - logits_off) = {} (non-zero => adapter active)",
                String.format("%.4f", adapterDelta));

        File adapterDir = Files.createTempDirectory("lora-adapter").toFile();
        peft.saveAdapter(adapterDir);
        log.info("Adapter weights saved to {}", adapterDir.getAbsolutePath());

        SameDiff merged = peft.mergeAndUnload();
        double mergedDrift = merged.getVariable(frozenProbe).getArr().sub(baseWeightBefore).norm2Number().doubleValue();
        log.info("mergeAndUnload: {} changed by L2={} (adapter delta baked into base weights)",
                frozenProbe, String.format("%.4f", mergedDrift));
        String adaptedGen = greedyGenerate(merged, tokenizer, ADAPT_PAIRS[0][0], genTokens);
        log.info("Merged-model generation for \"{}\": \"{}\"", ADAPT_PAIRS[0][0], adaptedGen);

        // ============================================================
        // Section 6: the SFTTrainingPipeline orchestration API
        // ============================================================
        log.info("=== Section 6: SFTTrainingPipeline (official orchestration API) ===");
        // The pipeline owns TrainingConfig construction (AdamW + cosine warmup from
        // SFTConfig), PEFT wrapping, and epoch looping. We feed it the SAME
        // real-tokenized iterator used above — production callers supply their own
        // tokenized MultiDataSets exactly like this.
        SFTConfig sftConfig = SFTConfig.builder()
                .chatTemplate(org.nd4j.linalg.dataset.curation.format.ChatTemplate.CHATML)
                .systemMessage(SYSTEM_MESSAGE)
                .maxSeqLength(SEQ_LEN)
                .learningRate(2e-4)
                .warmupRatio(0.1)
                .numEpochs(1)
                .gradientAccumulationSteps(1)
                .computeDataType(DataType.FLOAT)
                .peftConfig(LoraConfig.builder()
                        .r(4).loraAlpha(8)
                        .targetModules(Arrays.asList("q_proj", "v_proj"))
                        .build())
                .build();
        SameDiff freshBase = buildCausalLm(SEED, vocabSize);
        SFTTrainingPipeline sftPipeline = new SFTTrainingPipeline(freshBase, sftConfig);
        log.info("Pipeline model trainable summary: {}", sftPipeline.getPeftModel().getSummary());
        sftPipeline.train(new SimpleListMultiDataSetIterator(trainBatches), trainBatches.size());
        SameDiff sftMerged = sftPipeline.mergeAndExport();
        log.info("SFTTrainingPipeline finished; merged model has {} params",
                String.format("%,d", countParams(sftMerged)));

        // ============================================================
        // Section 7: persistence + GGUF export check
        // ============================================================
        log.info("=== Section 7: Saving the fine-tuned model ===");
        File outDir = Files.createTempDirectory("llm-finetune-example").toFile();
        File savedModel = new File(outDir, "finetuned-model.sd");
        model.save(savedModel, true);
        log.info("SameDiff checkpoint (graph + updater state): {} ({} MB)",
                savedModel.getAbsolutePath(), savedModel.length() / (1024 * 1024));

        if (GGMLModelExport.canExport(model)) {
            String arch = GGMLModelExport.detectArchitecture(model);
            File ggufOut = new File(outDir, "finetuned-f16.gguf");
            GGMLModelExport.exportModel(model, ggufOut);
            log.info("GGUF export ({} architecture): {} ({} MB)", arch,
                    ggufOut.getAbsolutePath(), ggufOut.length() / (1024 * 1024));
        } else {
            log.info("GGUF export not available for this graph; validation notes: {}",
                    GGMLModelExport.validateForExport(model));
        }

        log.info("=== Done ===");
        log.info("Pipeline demonstrated: real tokenizer -> exact response masking -> masked");
        log.info("causal-LM loss -> full SFT -> held-out eval -> LoRA adapters (frozen base,");
        log.info("toggle, merge, save) -> SFTTrainingPipeline -> checkpoint/GGUF export.");
        tokenizer.close();
    }

    // ====================================================================
    // Model builder: compact LLaMA-style decoder from production fused ops
    // ====================================================================

    /**
     * Builds a causal LM with HuggingFace-style parameter names so PEFT target modules
     * ({@code q_proj}, {@code v_proj}, ...) match exactly as they would on an imported
     * model. Uses the fused {@code rms_norm} and {@code dot_product_attention_v2}
     * (with internal causal masking) ops — the same kernels the LLM inference stack runs.
     *
     * <p>Placeholders: {@code input_ids} [batch, {@value #SEQ_LEN}] INT32,
     * {@code labels} [batch, seq] INT32 (next-token ids), {@code loss_mask} [batch, seq]
     * FLOAT (1 = train on this position). Output: {@code logits} [batch, seq, vocab].
     * Loss: masked mean of per-position sparse softmax cross-entropy.</p>
     */
    private static SameDiff buildCausalLm(long seed, int vocabSize) {
        Nd4j.getRandom().setSeed(seed);
        SameDiff sd = SameDiff.create();

        SDVariable inputIds = sd.placeHolder("input_ids", DataType.INT32, -1, SEQ_LEN);
        SDVariable labels = sd.placeHolder("labels", DataType.INT32, -1, SEQ_LEN);
        SDVariable lossMask = sd.placeHolder("loss_mask", DataType.FLOAT, -1, SEQ_LEN);

        SDVariable embedTable = sd.var("model.embed_tokens.weight",
                new XavierInitScheme('c', vocabSize, HIDDEN), DataType.FLOAT, vocabSize, HIDDEN);
        SDVariable x = sd.gather("token_embeddings", embedTable, inputIds, 0);   // [B,S,H]

        int headDim = HIDDEN / HEADS;
        for (int layer = 0; layer < LAYERS; layer++) {
            String p = "model.layers." + layer + ".";

            // --- attention block (pre-norm, residual) ---
            SDVariable attnGamma = sd.var(p + "input_layernorm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN));
            SDVariable xNorm = rmsNorm(sd, p + "attn_norm", x, attnGamma);

            SDVariable wq = sd.var(p + "self_attn.q_proj.weight",
                    new XavierInitScheme('c', HIDDEN, HIDDEN), DataType.FLOAT, HIDDEN, HIDDEN);
            SDVariable wk = sd.var(p + "self_attn.k_proj.weight",
                    new XavierInitScheme('c', HIDDEN, HIDDEN), DataType.FLOAT, HIDDEN, HIDDEN);
            SDVariable wv = sd.var(p + "self_attn.v_proj.weight",
                    new XavierInitScheme('c', HIDDEN, HIDDEN), DataType.FLOAT, HIDDEN, HIDDEN);
            SDVariable wo = sd.var(p + "self_attn.o_proj.weight",
                    new XavierInitScheme('c', HIDDEN, HIDDEN), DataType.FLOAT, HIDDEN, HIDDEN);

            // Projections run as rank-2 matmuls ([B*S, H] x [H, H]), then heads are
            // folded into the batch dimension ([B*heads, S, headDim]) so causal
            // attention runs through dot_product_attention_v2's 3D form — the exact
            // configuration the op's gradient checks validate (useCausalMask=true,
            // training=true).
            SDVariable xNormFlat = sd.reshape(xNorm, -1, HIDDEN);
            SDVariable q = toHeads(sd, xNormFlat.mmul(wq));
            SDVariable k = toHeads(sd, xNormFlat.mmul(wk));
            SDVariable v = toHeads(sd, xNormFlat.mmul(wv));
            SDVariable attn = sd.nn.dotProductAttentionV2(p + "attention",
                    q, v, k, null, null,
                    1.0 / Math.sqrt(headDim), 0.0, true, true);
            SDVariable attnMerged = fromHeads(sd, attn);           // [B*S, H]
            x = x.add(p + "attn_residual",
                    sd.reshape(attnMerged.mmul(wo), -1, SEQ_LEN, HIDDEN));

            // --- SwiGLU MLP block (pre-norm, residual) ---
            SDVariable mlpGamma = sd.var(p + "post_attention_layernorm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN));
            SDVariable xNorm2 = rmsNorm(sd, p + "mlp_norm", x, mlpGamma);
            SDVariable wGate = sd.var(p + "mlp.gate_proj.weight",
                    new XavierInitScheme('c', HIDDEN, FFN), DataType.FLOAT, HIDDEN, FFN);
            SDVariable wUp = sd.var(p + "mlp.up_proj.weight",
                    new XavierInitScheme('c', HIDDEN, FFN), DataType.FLOAT, HIDDEN, FFN);
            SDVariable wDown = sd.var(p + "mlp.down_proj.weight",
                    new XavierInitScheme('c', FFN, HIDDEN), DataType.FLOAT, FFN, HIDDEN);
            SDVariable xNorm2Flat = sd.reshape(xNorm2, -1, HIDDEN);
            SDVariable mlpFlat = sd.nn.swish(xNorm2Flat.mmul(wGate)).mul(xNorm2Flat.mmul(wUp)).mmul(wDown);
            x = x.add(p + "mlp_residual", sd.reshape(mlpFlat, -1, SEQ_LEN, HIDDEN));
        }

        SDVariable finalGamma = sd.var("model.norm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN));
        SDVariable xFinal = rmsNorm(sd, "final_norm", x, finalGamma);

        // Weight-tied LM head as a rank-2 matmul: [B*S, H] @ [H, V]. The rank-3 view
        // "logits" [B,S,V] is what inference/eval callers consume.
        SDVariable logitsFlat = sd.reshape(xFinal, -1, HIDDEN).mmul(embedTable.permute(1, 0));
        SDVariable logits = sd.reshape("logits", logitsFlat, -1, SEQ_LEN, vocabSize);

        // Masked causal-LM loss: per-position CE (labels already shifted host-side),
        // averaged over positions where loss_mask == 1. The native sparse CE op wants
        // rank-2 logits [N, vocab] + rank-1 labels [N].
        SDVariable labelsFlat = sd.reshape(labels, -1);
        SDVariable maskFlat = sd.reshape(lossMask, -1);
        SDVariable perPosition = sd.loss.sparseSoftmaxCrossEntropy("ce_per_position", logitsFlat, labelsFlat);
        SDVariable maskedSum = perPosition.mul(maskFlat).sum();
        SDVariable maskCount = maskFlat.sum().add(1e-6);
        SDVariable loss = maskedSum.div("loss", maskCount);
        loss.markAsLoss();
        return sd;
    }

    /**
     * Primitive-op RMS norm — the exact recipe the GGUF importer builds:
     * {@code x * rsqrt(mean(x^2) + eps) * gamma}. Built from primitives so autodiff
     * covers every parameter; the GraphOptimizer's NormalizationFusionOptimizations
     * pass recognizes this pattern and fuses it into the fused rms_norm op for
     * inference.
     */
    private static SDVariable rmsNorm(SameDiff sd, String name, SDVariable x, SDVariable gamma) {
        SDVariable meanSquared = x.mul(x).mean(true, -1);
        SDVariable rms = sd.math.sqrt(meanSquared.add(1e-5));
        return x.div(rms).mul(name, gamma);
    }

    /** [B*S, H] -> [B*heads, S, headDim]: fold attention heads into the batch dim. */
    private static SDVariable toHeads(SameDiff sd, SDVariable x2d) {
        int headDim = HIDDEN / HEADS;
        SDVariable bshd = sd.reshape(x2d, -1, SEQ_LEN, HEADS, headDim);   // [B,S,nH,dh]
        SDVariable bhsd = bshd.permute(0, 2, 1, 3);                       // [B,nH,S,dh]
        return sd.reshape(bhsd, -1, SEQ_LEN, headDim);                    // [B*nH,S,dh]
    }

    /** [B*heads, S, headDim] -> [B*S, H]: merge heads back into the hidden dim. */
    private static SDVariable fromHeads(SameDiff sd, SDVariable heads3d) {
        int headDim = HIDDEN / HEADS;
        SDVariable bhsd = sd.reshape(heads3d, -1, HEADS, SEQ_LEN, headDim); // [B,nH,S,dh]
        SDVariable bshd = bhsd.permute(0, 2, 1, 3);                         // [B,S,nH,dh]
        return sd.reshape(bshd, -1, HIDDEN);                                // [B*S,H]
    }

    // ====================================================================
    // Data pipeline: chat formatting + exact token-level response masking
    // ====================================================================

    /** One tokenized conversation, padded to SEQ_LEN, with the exact response mask. */
    private static final class TokenizedExample {
        final int[] tokens = new int[SEQ_LEN];
        final int[] labels = new int[SEQ_LEN];
        final float[] mask = new float[SEQ_LEN];
        int length;
        int promptLength;
    }

    /**
     * Formats each (instruction, response) pair with the tokenizer's chat template and
     * derives the EXACT response mask: the conversation is encoded once with just the
     * generation prompt and once in full; the longest common token prefix marks where
     * prompt ends and trainable response tokens begin.
     */
    private static List<TokenizedExample> tokenizeConversations(Tokenizer tokenizer, String[][] pairs) {
        List<TokenizedExample> out = new ArrayList<>(pairs.length);
        for (String[] pair : pairs) {
            List<ChatTemplate.Message> promptMsgs = Arrays.asList(
                    ChatTemplate.Message.system(SYSTEM_MESSAGE),
                    ChatTemplate.Message.user(pair[0]));
            List<ChatTemplate.Message> fullMsgs = Arrays.asList(
                    ChatTemplate.Message.system(SYSTEM_MESSAGE),
                    ChatTemplate.Message.user(pair[0]),
                    ChatTemplate.Message.assistant(pair[1]));

            int[] promptIds = tokenizer.encode(
                    tokenizer.applyChatTemplate(promptMsgs, true), false).getIds();
            int[] fullIds = tokenizer.encode(
                    tokenizer.applyChatTemplate(fullMsgs, false), false).getIds();

            int common = 0;
            while (common < promptIds.length && common < fullIds.length
                    && promptIds[common] == fullIds[common]) {
                common++;
            }

            TokenizedExample ex = new TokenizedExample();
            ex.length = Math.min(fullIds.length, SEQ_LEN);
            ex.promptLength = Math.min(common, ex.length);
            for (int t = 0; t < ex.length; t++) {
                ex.tokens[t] = fullIds[t];
            }
            // Next-token labels; positions predicting a response token get mask 1.
            for (int t = 0; t < ex.length - 1; t++) {
                ex.labels[t] = ex.tokens[t + 1];
                ex.mask[t] = (t + 1 >= ex.promptLength) ? 1f : 0f;
            }
            out.add(ex);
        }
        return out;
    }

    /** Packs tokenized examples into [BATCH, SEQ_LEN] MultiDataSets. */
    private static List<MultiDataSet> toBatches(List<TokenizedExample> examples) {
        List<MultiDataSet> batches = new ArrayList<>();
        for (int start = 0; start + 1 <= examples.size(); start += BATCH) {
            int rows = Math.min(BATCH, examples.size() - start);
            int[][] ids = new int[rows][];
            int[][] labels = new int[rows][];
            float[][] mask = new float[rows][];
            for (int r = 0; r < rows; r++) {
                TokenizedExample ex = examples.get(start + r);
                ids[r] = ex.tokens;
                labels[r] = ex.labels;
                mask[r] = ex.mask;
            }
            batches.add(new org.nd4j.linalg.dataset.MultiDataSet(
                    new INDArray[]{Nd4j.createFromArray(ids)},
                    new INDArray[]{Nd4j.createFromArray(labels)},
                    new INDArray[]{Nd4j.createFromArray(mask)},
                    null));
        }
        return batches;
    }

    // ====================================================================
    // Evaluation + generation helpers
    // ====================================================================

    /**
     * Masked next-token NLL and perplexity over held-out batches, computed from raw
     * logits the same way {@code PerplexityEvaluator} does (max-shifted log-sum-exp).
     */
    private static double[] heldOutNll(SameDiff model, List<MultiDataSet> evalBatches) {
        double totalNll = 0.0;
        long count = 0;
        for (MultiDataSet batch : evalBatches) {
            INDArray inputIds = batch.getFeatures(0);
            INDArray labels = batch.getLabels(0);
            INDArray mask = batch.getFeaturesMaskArray(0);
            INDArray logits = model.output(
                    Collections.singletonMap("input_ids", inputIds), "logits").get("logits");
            long rows = inputIds.size(0);
            for (long b = 0; b < rows; b++) {
                for (int t = 0; t < SEQ_LEN; t++) {
                    if (mask.getFloat(b, t) < 0.5f) continue;
                    INDArray position = logits.get(NDArrayIndex.point(b), NDArrayIndex.point(t), NDArrayIndex.all());
                    double max = position.maxNumber().doubleValue();
                    double logSumExp = Math.log(org.nd4j.linalg.ops.transforms.Transforms
                            .exp(position.sub(max), false).sumNumber().doubleValue()) + max;
                    double logProb = position.getDouble(labels.getInt((int) b, t)) - logSumExp;
                    totalNll -= logProb;
                    count++;
                }
            }
        }
        double avg = count > 0 ? totalNll / count : Double.NaN;
        return new double[]{avg, Math.exp(avg)};
    }

    /**
     * Greedy decoding against the fixed-shape training graph: the context lives in a
     * [1, SEQ_LEN] buffer (causal attention ignores the zero padding to the right of the
     * current position) and each step reads the logits at the last real position.
     */
    private static String greedyGenerate(SameDiff model, Tokenizer tokenizer, String question, int maxNewTokens) {
        List<ChatTemplate.Message> msgs = Arrays.asList(
                ChatTemplate.Message.system(SYSTEM_MESSAGE),
                ChatTemplate.Message.user(question));
        int[] promptIds = tokenizer.encode(tokenizer.applyChatTemplate(msgs, true), false).getIds();

        List<Integer> context = new ArrayList<>();
        for (int id : promptIds) {
            if (context.size() < SEQ_LEN - 1) context.add(id);
        }
        List<Integer> generated = new ArrayList<>();
        int eos = tokenizer.getEosTokenId();

        for (int step = 0; step < maxNewTokens && context.size() < SEQ_LEN; step++) {
            int[] buffer = new int[SEQ_LEN];
            for (int i = 0; i < context.size(); i++) buffer[i] = context.get(i);
            INDArray input = Nd4j.createFromArray(new int[][]{buffer});
            INDArray logits = model.output(
                    Collections.singletonMap("input_ids", input), "logits").get("logits");
            INDArray lastPosition = logits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(context.size() - 1), NDArrayIndex.all());
            int next = lastPosition.argMax(0).getInt(0);
            if (next == eos) break;
            context.add(next);
            generated.add(next);
        }
        int[] genIds = generated.stream().mapToInt(Integer::intValue).toArray();
        return tokenizer.decode(genIds, true).trim();
    }

    private static long countParams(SameDiff sd) {
        long total = 0;
        for (SDVariable v : sd.variables()) {
            if (v.getVariableType() == org.nd4j.autodiff.samediff.VariableType.VARIABLE
                    && v.getArr() != null) {
                total += v.getArr().length();
            }
        }
        return total;
    }
}
