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
import org.eclipse.deeplearning4j.llm.eval.PerplexityEvaluator;
import org.eclipse.deeplearning4j.llm.generation.DecoderInputBuilder;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.samediff.DistillationTrainer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.DistillationConfig;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SimpleListMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.schedule.CosineWarmupSchedule;
import org.nd4j.weightinit.impl.XavierInitScheme;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Knowledge distillation from a REAL pretrained teacher: Qwen3.5-0.8B (imported from GGUF)
 * teaches a compact from-scratch student on real WikiText-2 data.
 *
 * <p>This replaces the random-weight/random-data setup of
 * {@link DistillationTrainingPipelineExample} with the actual production recipe:</p>
 * <ol>
 *   <li><b>Real teacher.</b> The Qwen3.5-0.8B GGUF is downloaded (or served from
 *       {@code ~/.cache/dl4j-llm-models}), imported with
 *       {@code GGMLModelImport.importModel(file, ConversionOptions.forInference())} and
 *       driven with a COMPLETE prefill input map — imported decoders declare per-layer
 *       state placeholders (KV caches; for this hybrid GDN/SSM architecture also
 *       recurrent states) that all must be fed. See {@code teacherLogitsFor()} for the
 *       production recipe ({@code DecoderInputBuilder} + zero recurrent states +
 *       position scalars).</li>
 *   <li><b>Real data.</b> WikiText-2 test text (auto-downloaded by
 *       {@link PerplexityEvaluator#loadWikiText2()}) is tokenized with the teacher's own
 *       248k-token BPE tokenizer and cut into fixed [batch, seq] windows.</li>
 *   <li><b>Offline teacher targets.</b> Teacher logits are precomputed once per training
 *       window and cached in FLOAT16 — the standard offline-distillation trick that avoids
 *       re-running the 0.8B teacher every epoch. (Production distillers often go further
 *       and store only top-K logits per position.)</li>
 *   <li><b>In-graph distillation loss.</b> The student graph carries the full Hinton loss:
 *       {@code alpha * T^2 * KL(softmax(teacher/T) || softmax(student/T))
 *       + (1-alpha) * CE(labels)}, masked and averaged, trained with AdamW + cosine
 *       warmup through the standard {@link SameDiff#fit} path — weights actually update,
 *       unlike {@link DistillationTrainer#trainStep} which only <i>measures</i> the loss.</li>
 *   <li><b>Real evaluation.</b> Student masked NLL/perplexity before vs after training;
 *       a teacher reference computed on the SAME held-out windows from the cached
 *       logits (apples-to-apples); teacher-student top-1 agreement; and side-by-side
 *       greedy continuations — the teacher's through a real {@link GenerationPipeline}.</li>
 * </ol>
 *
 * <p><b>Resources.</b> The FP32-dequantized teacher needs ~3.5GB RAM, and student
 * training peaks around 17-20GB: full-vocab soft targets mean the KD backward pass
 * materializes several [batch, seq, 248k] tensors while the teacher stays resident
 * (production distillers store only top-K teacher logits per position to avoid exactly
 * this). Teacher-target precomputation runs once per window (~2s each on CPU once the
 * DSP plan is warm). Shrink {@code -Dexample.distill.batches} for a faster pass, or
 * switch the module's {@code nd4j.backend} to CUDA.</p>
 *
 * <p>System properties: {@code -Dexample.distill.batches=6} (train batches of
 * {@value #BATCH}x{@value #SEQ_LEN} tokens), {@code -Dexample.distill.epochs=6},
 * {@code -Dexample.gen.tokens=24}.</p>
 *
 * <p><b>CPU throughput note.</b> If the teacher forward passes or student training run
 * on a single core (some packaged nd4j-native builds pin OpenBLAS to one thread), export
 * {@code OPENBLAS_NUM_THREADS=<physical cores>} in the environment before launching.</p>
 */
public class QwenToStudentDistillationExample {

    private static final Logger log = LoggerFactory.getLogger(QwenToStudentDistillationExample.class);

    // Distillation window geometry. Full-vocab soft targets are cached per window, so
    // batch/seq are kept small; the student itself is ~49M params (~20x smaller than
    // the 1.0B-parameter teacher — most of both budgets is the 248k-vocab embedding).
    private static final int SEQ_LEN = 64;
    private static final int BATCH = 2;
    private static final int HIDDEN = 192;
    private static final int LAYERS = 4;
    private static final int HEADS = 4;
    private static final int FFN = 512;
    private static final long SEED = 20260702;

    // Hinton distillation hyperparameters (baked into the student graph).
    private static final double TEMPERATURE = 2.0;
    private static final double ALPHA = 0.5;   // weight of the KD term vs hard-label CE

    public static void main(String[] args) throws Exception {
        int trainBatchCount = Integer.getInteger("example.distill.batches", 6);
        int evalBatchCount = 2;
        int epochs = Integer.getInteger("example.distill.epochs", 6);
        int genTokens = Integer.getInteger("example.gen.tokens", 24);

        // ============================================================
        // Section 1: the teacher — import Qwen3.5-0.8B from GGUF
        // ============================================================
        log.info("=== Section 1: Teacher (Qwen3.5-0.8B) ===");
        File ggufFile = LLMModelDownloader.download(LLMModel.QWEN35_0_8B, QuantType.Q4_K_M).getModelFile();
        GGMLMetadata metadata = GGMLModelImport.inspectModel(ggufFile);
        log.info("GGUF: architecture={}, layers={}, hidden={}, context={}",
                metadata.getArchitecture(), metadata.getNumLayers(), metadata.getHiddenSize(),
                metadata.getContextLength());

        long t0 = System.currentTimeMillis();
        SameDiff teacher = GGMLModelImport.importModel(ggufFile.getAbsolutePath(),
                ConversionOptions.forInference());
        log.info("Teacher imported in {}ms ({} ops, {} variables)",
                System.currentTimeMillis() - t0, teacher.ops().length, teacher.variables().size());
        long teacherParams = countParams(teacher);
        log.info("Teacher parameter count: {}", String.format("%,d", teacherParams));

        Tokenizer tokenizer = HuggingFaceTokenizer.fromDirectory(ggufFile.getParentFile());
        int vocabSize = tokenizer.getVocabSize();
        // Architecture builders differ on the logits variable name: LLaMA-family graphs
        // expose "logits", the Qwen3.5 hybrid (GDN/SSM) builder exposes "lm_logits".
        String teacherLogitsName = teacher.hasVariable("logits") ? "logits" : "lm_logits";
        log.info("Tokenizer vocab={}, teacher logits variable='{}'", vocabSize, teacherLogitsName);

        // ============================================================
        // Section 2: real corpus — WikiText-2, tokenized into windows
        // ============================================================
        log.info("=== Section 2: WikiText-2 distillation corpus ===");
        String wikiText = PerplexityEvaluator.loadWikiText2();
        int neededTokens = (trainBatchCount + evalBatchCount) * BATCH * (SEQ_LEN + 1) + 64;
        // Tokenize just enough text (~4 chars/token headroom) instead of the whole set.
        String slice = wikiText.substring(0, Math.min(wikiText.length(), neededTokens * 8));
        int[] corpusIds = tokenizer.encode(slice, false).getIds();
        log.info("Tokenized {} chars -> {} tokens (need {})", slice.length(), corpusIds.length, neededTokens);

        List<int[][]> trainWindows = cutWindows(corpusIds, 0, trainBatchCount);
        List<int[][]> evalWindows = cutWindows(corpusIds, trainBatchCount * BATCH * (SEQ_LEN + 1), evalBatchCount);
        log.info("Windows: {} train batches, {} eval batches of [{} x {}]",
                trainWindows.size(), evalWindows.size(), BATCH, SEQ_LEN);

        // ============================================================
        // Section 3: offline teacher targets (cached FP16 logits)
        // ============================================================
        log.info("=== Section 3: Precomputing teacher logits (offline distillation) ===");
        t0 = System.currentTimeMillis();
        List<INDArray> trainTeacherLogits = teacherLogitsFor(teacher, teacherLogitsName, trainWindows,
                vocabSize, metadata.getHiddenSize());
        List<INDArray> evalTeacherLogits = teacherLogitsFor(teacher, teacherLogitsName, evalWindows,
                vocabSize, metadata.getHiddenSize());
        log.info("Teacher targets ready in {}ms — {} MB cached in FP16",
                System.currentTimeMillis() - t0,
                (trainTeacherLogits.size() + evalTeacherLogits.size())
                        * BATCH * SEQ_LEN * (long) vocabSize * 2 / (1024 * 1024));

        // ============================================================
        // Section 4: the student, with the distillation loss in-graph
        // ============================================================
        log.info("=== Section 4: Student model ===");
        SameDiff student = buildStudentLm(SEED, vocabSize);
        long studentParams = countParams(student);
        log.info("Student: {} layers, hidden={}, params={} — compression {}x vs teacher",
                LAYERS, HIDDEN, String.format("%,d", studentParams),
                String.format("%.1f", (double) teacherParams / studentParams));
        log.info("Loss = {} * T^2 * KL(teacher/T || student/T) + {} * CE(labels), T={}",
                ALPHA, 1 - ALPHA, TEMPERATURE);

        List<MultiDataSet> trainBatches = toDistillationBatches(trainWindows, trainTeacherLogits);
        List<MultiDataSet> evalBatches = toDistillationBatches(evalWindows, evalTeacherLogits);

        double[] studentBefore = studentHeldOutNll(student, evalBatches);
        double agreementBefore = topOneAgreement(student, evalBatches);
        log.info("BEFORE: student held-out NLL={} ppl={} | teacher top-1 agreement={}%",
                String.format("%.4f", studentBefore[0]), String.format("%.1f", studentBefore[1]),
                String.format("%.1f", agreementBefore * 100));

        // ============================================================
        // Section 5: train the student
        // ============================================================
        log.info("=== Section 5: Distillation training ({} epochs x {} batches) ===",
                epochs, trainBatches.size());
        int totalSteps = trainBatches.size() * epochs;
        Adam adamW = Adam.builder()
                .learningRateSchedule(CosineWarmupSchedule.fromRatio(3e-4, 3e-5, 0.1, totalSteps))
                .beta1(0.9).beta2(0.999).epsilon(1e-8)
                .build();
        student.setTrainingConfig(new TrainingConfig.Builder()
                .updater(adamW)
                .dataSetFeatureMapping("input_ids", "teacher_logits", "loss_mask")
                .dataSetLabelMapping("labels")
                .build());

        t0 = System.currentTimeMillis();
        History history = student.fit(new SimpleListMultiDataSetIterator(trainBatches), epochs);
        INDArray lossValues = history.getLossCurve().getLossValues();
        log.info("Trained {} steps in {}ms; loss first={} mid={} last={}",
                totalSteps, System.currentTimeMillis() - t0,
                String.format("%.4f", lossValues.getDouble(0)),
                String.format("%.4f", lossValues.getDouble(lossValues.length() / 2)),
                String.format("%.4f", lossValues.getDouble(lossValues.length() - 1)));

        // ============================================================
        // Section 6: evaluation — NLL, agreement, teacher reference
        // ============================================================
        log.info("=== Section 6: Evaluation ===");
        double[] studentAfter = studentHeldOutNll(student, evalBatches);
        double agreementAfter = topOneAgreement(student, evalBatches);
        log.info("AFTER: student held-out NLL={} ppl={} (before ppl={}) | top-1 agreement {}% -> {}%",
                String.format("%.4f", studentAfter[0]), String.format("%.1f", studentAfter[1]),
                String.format("%.1f", studentBefore[1]),
                String.format("%.1f", agreementBefore * 100),
                String.format("%.1f", agreementAfter * 100));

        // Teacher reference on the SAME held-out windows, computed from the cached eval
        // logits — a direct apples-to-apples bound for the student's numbers. (For
        // standard attention-KV models, PerplexityEvaluator.evaluate is the general
        // sliding-window API; it feeds only input_ids, which hybrid-state architectures
        // like Qwen3.5 reject — they need the full state map, see teacherLogitsFor().)
        double[] teacherRef = nllFromCachedLogits(evalTeacherLogits, evalWindows);
        log.info("Teacher reference on the same windows: NLL={} perplexity={}",
                String.format("%.4f", teacherRef[0]), String.format("%.1f", teacherRef[1]));

        // ============================================================
        // Section 7: DistillationConfig / DistillationTrainer APIs
        // ============================================================
        log.info("=== Section 7: DistillationConfig + DistillationTrainer ===");
        // DistillationConfig also models progressive distillation (temperature
        // annealing). The schedule below is what a multi-epoch run would apply.
        DistillationConfig kdConfig = DistillationConfig.builder()
                .distillationType(DistillationConfig.DistillationType.LOGIT_KD)
                .temperature(4.0)
                .alpha(ALPHA)
                .studentLogitVariable("logits")
                .teacherLogitVariable("logits")
                .temperatureAnnealing(true)
                .initialTemperature(4.0)
                .finalTemperature(1.0)
                .build();
        for (double progress : new double[]{0.0, 0.25, 0.5, 0.75, 1.0}) {
            log.info("  annealing: progress={} -> effective temperature={}",
                    progress, String.format("%.2f", kdConfig.getEffectiveTemperature(progress)));
        }
        // DistillationTrainer wraps the DistillationKLLoss native op, which supports
        // rank-2 [batch, numClasses] logits — CLASSIFIER distillation. LM-shaped rank-3
        // [batch, seq, vocab] logits are not supported by that op, which is exactly why
        // this example builds the LLM logit-KD loss in-graph (Section 4) and trains it
        // through fit(). The trainer's EMA self-distillation utilities are still useful
        // for LM students: snapshot the student as its own teacher and blend
        // periodically during training.
        DistillationTrainer selfDistill = DistillationTrainer.selfDistillation(student, kdConfig);
        selfDistill.refreshTeacherEMA(0.999);
        log.info("selfDistillation(): teacher = student snapshot; refreshTeacherEMA(0.999):");
        log.info("  teacher <- 0.999*teacher + 0.001*student (call every N training steps)");
        log.info("(DistillationTrainer.trainStep = rank-2 classifier logit-KD; the LLM-shaped");
        log.info(" rank-3 KD in this example runs in-graph through fit() instead)");

        // ============================================================
        // Section 8: persist the student (before the bonus generation demos —
        // the trained artifact is the deliverable)
        // ============================================================
        log.info("=== Section 8: Persist the student ===");
        File outDir = Files.createTempDirectory("qwen-distillation").toFile();
        File studentFile = new File(outDir, "distilled-student.sd");
        student.save(studentFile, true);
        log.info("Student saved: {} ({} MB)", studentFile.getAbsolutePath(),
                studentFile.length() / (1024 * 1024));

        // ============================================================
        // Section 9: side-by-side generation (teacher via GenerationPipeline)
        // ============================================================
        log.info("=== Section 9: Teacher vs student continuations ===");
        String prompt = "The history of the region shows that";
        String studentGen = greedyGenerate(student, tokenizer, prompt, genTokens);
        log.info("Student ({}M params): \"{} {}\"", studentParams / 1_000_000, prompt, studentGen);

        // The teacher decodes through the real production path: GenerationPipeline with
        // KV cache, DSP plan compilation and the GGUF chat template.
        GenerationPipeline teacherPipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(teacher)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(genTokens)
                .build());
        try {
            GenerationResult teacherGen = teacherPipeline.generate(prompt, genTokens);
            log.info("Teacher (0.8B, GenerationPipeline): \"{}{}\" [{} tok/s]",
                    prompt, teacherGen.getText(),
                    String.format("%.2f", teacherGen.getTokensPerSecond()));
        } finally {
            teacherPipeline.close();
        }

        log.info("=== Done: teacher import -> real corpus -> offline targets -> in-graph KD loss");
        log.info("    -> fit -> agreement/perplexity eval -> checkpoint -> pipeline generation ===");
        tokenizer.close();
    }

    // ====================================================================
    // Teacher forward passes (PerplexityEvaluator-style invocation)
    // ====================================================================

    /**
     * Runs the imported teacher window-by-window at [1, SEQ_LEN] and stacks each batch's
     * logits into [BATCH, SEQ_LEN, vocab], cached as FLOAT16 to halve memory.
     *
     * <p>Imported decoder graphs declare per-layer state placeholders (attention KV
     * caches, and for hybrid architectures like Qwen3.5 also GDN/conv recurrent state) —
     * ALL of them must be fed. {@link DecoderInputBuilder#buildDecoderInputMap} is the
     * production helper that assembles the complete prefill input map (empty caches,
     * zero states, positions, masks) from the decoder's declared inputs; it is the same
     * code the GenerationPipeline uses internally.</p>
     *
     * <p>GGUF embedding tables are often padded past the tokenizer vocabulary (Qwen3.5:
     * 248320 rows vs 248070 real tokens), so the logits are sliced to the tokenizer
     * vocab — the padding ids are never produced by the tokenizer and carry no
     * probability mass worth distilling.</p>
     */
    private static List<INDArray> teacherLogitsFor(SameDiff teacher, String logitsName,
                                                   List<int[][]> batches, int vocabSize, long hiddenSize) {
        List<String> decoderInputNames = teacher.inputs();
        // Hybrid architectures (Qwen3.5 GDN/SSM, LFM2 conv) also declare per-layer
        // recurrent-state placeholders; zero state = "no prior history" for prefill.
        List<ModelIOConfig.RecurrentStatePair> recurrentStates =
                ModelIOConfig.findRecurrentStatePairs(teacher, ModelIOConfig.builder().build());
        List<INDArray> out = new ArrayList<>(batches.size());
        int done = 0;
        for (int[][] windows : batches) {
            INDArray[] perWindow = new INDArray[windows.length];
            for (int w = 0; w < windows.length; w++) {
                // Windows carry SEQ_LEN+1 tokens (the +1 supplies shifted labels); the
                // teacher sees exactly the SEQ_LEN inputs the student will see.
                INDArray inputIds = Nd4j.createFromArray(
                        new int[][]{Arrays.copyOf(windows[w], SEQ_LEN)});
                // Prefill-style forward: past length 0, dynamic (non-static) KV mode.
                Map<String, INDArray> feeds = DecoderInputBuilder.buildDecoderInputMap(
                        decoderInputNames, teacher, null, inputIds,
                        0, SEQ_LEN, null, SEQ_LEN, 0, false, hiddenSize);
                for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                    if (!feeds.containsKey(pair.inputName) && teacher.hasVariable(pair.inputName)) {
                        long[] stateShape = GenerationPipeline.deriveRecurrentStateShape(teacher, pair.inputName);
                        if (stateShape != null) {
                            feeds.put(pair.inputName, Nd4j.zeros(
                                    teacher.getVariable(pair.inputName).dataType(), stateShape));
                        }
                    }
                }
                // GGUF in-graph-KV scalar positions: a fresh prefill starts at position 0.
                for (String scalarName : new String[]{"position_offset", "cache_position"}) {
                    if (teacher.hasVariable(scalarName) && !feeds.containsKey(scalarName)) {
                        feeds.put(scalarName, Nd4j.scalar(DataType.INT64, 0));
                    }
                }
                INDArray logits = teacher.output(feeds, logitsName).get(logitsName);
                if (logits.rank() == 2) {
                    logits = logits.reshape(1, logits.size(0), logits.size(1));
                }
                if (logits.size(2) > vocabSize) {
                    logits = logits.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.interval(0, vocabSize));
                }
                perWindow[w] = logits.castTo(DataType.FLOAT16);
            }
            out.add(Nd4j.concat(0, perWindow));
            done++;
            log.info("  teacher targets: batch {}/{} done", done, batches.size());
        }
        return out;
    }

    // ====================================================================
    // Student model: compact LLaMA-style decoder with the KD loss in-graph
    // ====================================================================

    /**
     * Same architecture family as the teacher (RMS-norm, causal multi-head attention via
     * {@code dot_product_attention_v2}, SwiGLU MLP, tied embeddings) at a fraction of the
     * size, plus the distillation loss:
     *
     * <pre>
     * teacher_lp = logSoftmax(teacher_logits / T)     (teacher_logits fed as a feature)
     * student_lp = logSoftmax(logits / T)
     * kd  = sum(exp(teacher_lp) * (teacher_lp - student_lp), vocab) * T^2   per position
     * ce  = sparseSoftmaxCrossEntropy(logits, labels)                        per position
     * loss = maskedMean(ALPHA * kd + (1-ALPHA) * ce)
     * </pre>
     */
    private static SameDiff buildStudentLm(long seed, int vocabSize) {
        Nd4j.getRandom().setSeed(seed);
        SameDiff sd = SameDiff.create();

        SDVariable inputIds = sd.placeHolder("input_ids", DataType.INT32, -1, SEQ_LEN);
        SDVariable labels = sd.placeHolder("labels", DataType.INT32, -1, SEQ_LEN);
        SDVariable lossMask = sd.placeHolder("loss_mask", DataType.FLOAT, -1, SEQ_LEN);
        // FP16 teacher cache is cast up once inside the graph.
        SDVariable teacherLogitsFp16 = sd.placeHolder("teacher_logits", DataType.FLOAT16, -1, SEQ_LEN, vocabSize);

        SDVariable embedTable = sd.var("model.embed_tokens.weight",
                new XavierInitScheme('c', vocabSize, HIDDEN), DataType.FLOAT, vocabSize, HIDDEN);
        SDVariable x = sd.gather("token_embeddings", embedTable, inputIds, 0);

        int headDim = HIDDEN / HEADS;
        for (int layer = 0; layer < LAYERS; layer++) {
            String p = "model.layers." + layer + ".";
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

            // Rank-2 projections ([B*S, H]), heads folded into the batch dim
            // ([B*heads, S, headDim]) so causal attention runs through
            // dot_product_attention_v2's gradient-checked 3D form.
            SDVariable xNormFlat = sd.reshape(xNorm, -1, HIDDEN);
            SDVariable q = toHeads(sd, xNormFlat.mmul(wq));
            SDVariable k = toHeads(sd, xNormFlat.mmul(wk));
            SDVariable v = toHeads(sd, xNormFlat.mmul(wv));
            SDVariable attn = sd.nn.dotProductAttentionV2(p + "attention",
                    q, v, k, null, null,
                    1.0 / Math.sqrt(headDim), 0.0, true, true);
            x = x.add(p + "attn_residual",
                    sd.reshape(fromHeads(sd, attn).mmul(wo), -1, SEQ_LEN, HIDDEN));

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
        // Weight-tied LM head as a rank-2 matmul; "logits" [B,S,V] is the public view.
        SDVariable logitsFlat = sd.reshape(xFinal, -1, HIDDEN).mmul(embedTable.permute(1, 0));
        SDVariable logits = sd.reshape("logits", logitsFlat, -1, SEQ_LEN, vocabSize);

        // ---- Distillation loss ----
        SDVariable teacherLogits = teacherLogitsFp16.castTo(DataType.FLOAT);
        SDVariable teacherLp = sd.nn.logSoftmax("teacher_lp", teacherLogits.div(TEMPERATURE), 2);
        SDVariable studentLp = sd.nn.logSoftmax("student_lp", logits.div(TEMPERATURE), 2);
        SDVariable kdPerPosition = sd.math.exp(teacherLp).mul(teacherLp.sub(studentLp))
                .sum("kd_per_position", 2)
                .mul(TEMPERATURE * TEMPERATURE);                                // [B,S]
        // The native sparse CE op wants rank-2 logits [N, vocab] + rank-1 labels [N].
        SDVariable labelsFlat = sd.reshape(labels, -1);
        SDVariable maskFlat = sd.reshape(lossMask, -1);
        SDVariable cePerPosition = sd.loss.sparseSoftmaxCrossEntropy("ce_per_position", logitsFlat, labelsFlat);

        SDVariable perPosition = sd.reshape(kdPerPosition, -1).mul(ALPHA)
                .add(cePerPosition.mul(1.0 - ALPHA));
        SDVariable maskedSum = perPosition.mul(maskFlat).sum();
        SDVariable maskCount = maskFlat.sum().add(1e-6);
        SDVariable loss = maskedSum.div("loss", maskCount);
        loss.markAsLoss();
        return sd;
    }

    /**
     * Primitive-op RMS norm — the exact recipe the GGUF importer builds:
     * {@code x * rsqrt(mean(x^2) + eps) * gamma}. Primitives keep the whole student
     * differentiable; the GraphOptimizer fuses the pattern for inference.
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
    // Data windows + batch assembly
    // ====================================================================

    /** Cuts contiguous [BATCH][SEQ_LEN+1] token windows starting at {@code tokenOffset}. */
    private static List<int[][]> cutWindows(int[] corpusIds, int tokenOffset, int batchCount) {
        List<int[][]> batches = new ArrayList<>(batchCount);
        int cursor = tokenOffset;
        for (int b = 0; b < batchCount; b++) {
            int[][] windows = new int[BATCH][];
            for (int w = 0; w < BATCH; w++) {
                windows[w] = Arrays.copyOfRange(corpusIds, cursor, cursor + SEQ_LEN + 1);
                cursor += SEQ_LEN + 1;
            }
            batches.add(windows);
        }
        return batches;
    }

    /**
     * Builds MultiDataSets: features = [input_ids [B,S], teacher_logits [B,S,V] fp16],
     * labels = next-token ids, feature mask = loss positions (all but the last, which has
     * no next token inside the window).
     */
    private static List<MultiDataSet> toDistillationBatches(List<int[][]> windows, List<INDArray> teacherLogits) {
        List<MultiDataSet> out = new ArrayList<>(windows.size());
        for (int b = 0; b < windows.size(); b++) {
            int[][] batchWindows = windows.get(b);
            int[][] ids = new int[BATCH][SEQ_LEN];
            int[][] labels = new int[BATCH][SEQ_LEN];
            float[][] mask = new float[BATCH][SEQ_LEN];
            for (int w = 0; w < BATCH; w++) {
                for (int t = 0; t < SEQ_LEN; t++) {
                    ids[w][t] = batchWindows[w][t];
                    labels[w][t] = batchWindows[w][t + 1];
                    mask[w][t] = (t < SEQ_LEN - 1) ? 1f : 0f;
                }
            }
            // The loss mask travels as a third FEATURE (mapped straight to its
            // placeholder) — feature-mask slots must align 1:1 with feature arrays,
            // and a null slot for teacher_logits trips validation.
            out.add(new org.nd4j.linalg.dataset.MultiDataSet(
                    new INDArray[]{Nd4j.createFromArray(ids), teacherLogits.get(b),
                            Nd4j.createFromArray(mask)},
                    new INDArray[]{Nd4j.createFromArray(labels)},
                    null,
                    null));
        }
        return out;
    }

    // ====================================================================
    // Evaluation helpers
    // ====================================================================

    /** Masked next-token NLL + perplexity of the student over held-out batches. */
    private static double[] studentHeldOutNll(SameDiff student, List<MultiDataSet> evalBatches) {
        double totalNll = 0.0;
        long count = 0;
        for (MultiDataSet batch : evalBatches) {
            Map<String, INDArray> feeds = new HashMap<>();
            feeds.put("input_ids", batch.getFeatures(0));
            INDArray logits = student.output(feeds, "logits").get("logits");
            INDArray labels = batch.getLabels(0);
            INDArray mask = batch.getFeatures(2);   // loss_mask travels as feature 2
            for (int b = 0; b < BATCH; b++) {
                for (int t = 0; t < SEQ_LEN; t++) {
                    if (mask.getFloat(b, t) < 0.5f) continue;
                    INDArray position = logits.get(NDArrayIndex.point(b), NDArrayIndex.point(t), NDArrayIndex.all());
                    double max = position.maxNumber().doubleValue();
                    double logSumExp = Math.log(Transforms.exp(position.sub(max), false)
                            .sumNumber().doubleValue()) + max;
                    totalNll -= position.getDouble(labels.getInt(b, t)) - logSumExp;
                    count++;
                }
            }
        }
        double avg = count > 0 ? totalNll / count : Double.NaN;
        return new double[]{avg, Math.exp(avg)};
    }

    /** Teacher NLL/perplexity computed from the cached [B,S,V] eval logits. */
    private static double[] nllFromCachedLogits(List<INDArray> cachedLogits, List<int[][]> windows) {
        double totalNll = 0.0;
        long count = 0;
        for (int b = 0; b < windows.size(); b++) {
            INDArray logits = cachedLogits.get(b).castTo(DataType.FLOAT);
            int[][] batchWindows = windows.get(b);
            for (int w = 0; w < BATCH; w++) {
                for (int t = 0; t < SEQ_LEN - 1; t++) {
                    INDArray position = logits.get(NDArrayIndex.point(w), NDArrayIndex.point(t), NDArrayIndex.all());
                    double max = position.maxNumber().doubleValue();
                    double logSumExp = Math.log(Transforms.exp(position.sub(max), false)
                            .sumNumber().doubleValue()) + max;
                    totalNll -= position.getDouble(batchWindows[w][t + 1]) - logSumExp;
                    count++;
                }
            }
        }
        double avg = count > 0 ? totalNll / count : Double.NaN;
        return new double[]{avg, Math.exp(avg)};
    }

    /** Fraction of masked positions where student argmax == teacher argmax. */
    private static double topOneAgreement(SameDiff student, List<MultiDataSet> evalBatches) {
        long agree = 0, total = 0;
        for (MultiDataSet batch : evalBatches) {
            Map<String, INDArray> feeds = new HashMap<>();
            feeds.put("input_ids", batch.getFeatures(0));
            INDArray studentLogits = student.output(feeds, "logits").get("logits");
            INDArray teacherLogits = batch.getFeatures(1);   // fp16 cache
            INDArray mask = batch.getFeatures(2);   // loss_mask travels as feature 2
            INDArray studentTop = studentLogits.argMax(2);   // [B,S]
            INDArray teacherTop = teacherLogits.argMax(2);   // [B,S]
            for (int b = 0; b < BATCH; b++) {
                for (int t = 0; t < SEQ_LEN; t++) {
                    if (mask.getFloat(b, t) < 0.5f) continue;
                    total++;
                    if (studentTop.getInt(b, t) == teacherTop.getInt(b, t)) agree++;
                }
            }
        }
        return total > 0 ? (double) agree / total : 0.0;
    }

    /** Greedy decoding against the fixed-shape student graph (see the SFT example). */
    private static String greedyGenerate(SameDiff model, Tokenizer tokenizer, String prompt, int maxNewTokens) {
        int[] promptIds = tokenizer.encode(prompt, false).getIds();
        List<Integer> context = new ArrayList<>();
        for (int id : promptIds) {
            if (context.size() < SEQ_LEN - 1) context.add(id);
        }
        List<Integer> generated = new ArrayList<>();
        int eos = tokenizer.getEosTokenId();
        for (int step = 0; step < maxNewTokens && context.size() < SEQ_LEN; step++) {
            int[] buffer = new int[SEQ_LEN];
            for (int i = 0; i < context.size(); i++) buffer[i] = context.get(i);
            INDArray logits = model.output(Collections.singletonMap("input_ids",
                    Nd4j.createFromArray(new int[][]{buffer})), "logits").get("logits");
            INDArray last = logits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(context.size() - 1), NDArrayIndex.all());
            int next = last.argMax(0).getInt(0);
            if (next == eos) break;
            context.add(next);
            generated.add(next);
        }
        return tokenizer.decode(generated.stream().mapToInt(Integer::intValue).toArray(), true).trim();
    }

    private static long countParams(SameDiff sd) {
        long total = 0;
        for (SDVariable v : sd.variables()) {
            if (v.getVariableType() == VariableType.VARIABLE && v.getArr() != null) {
                total += v.getArr().length();
            }
        }
        return total;
    }
}
