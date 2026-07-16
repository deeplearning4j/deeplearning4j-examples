/*
 *
 * This program and the accompanying materials are made available under the
 *  terms of the Apache License, Version 2.0 which is available at
 *  https://www.apache.org/licenses/LICENSE-2.0.
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  License for the specific language governing permissions and limitations
 *  under the License.
 *
 *  SPDX-License-Identifier: Apache-2.0
 *
 */
package org.nd4j.examples.samediff.quickstart.modeling.llm;

import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.LLMModel;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.QuantType;
import org.eclipse.deeplearning4j.llm.generation.DecoderInputBuilder;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * The GraphOptimizer applied to a REAL imported LLM — Qwen3.5-0.8B — rather than the
 * hand-built 4-element toy graphs of
 * {@code org.nd4j.examples.samediff.quickstart.modeling.GraphOptimizerExample}.
 *
 * What this demonstrates on a production 0.8B-parameter transformer graph:
 *   1. Importing the GGUF WITHOUT optimization and profiling the raw graph
 *      (op count + per-op-type histogram of the actual decoder).
 *   2. Running {@link GraphOptimizer#optimize(SameDiff, String...)} and measuring what
 *      the passes did: how many ops were removed, and which FUSED op types appeared
 *      (rms_norm folding, swish fusion, matmul/bias fusion, CSE across 24 layers...).
 *   3. Verifying CORRECTNESS the way the platform tests do: identical prompt through the
 *      raw and optimized graphs, comparing max|logit difference| and the top-5
 *      next-token predictions.
 *   4. Pass-set control: {@link GraphOptimizer#defaultOptimizations()} vs
 *      {@link GraphOptimizer#defaultCorrectnessOptimizations()} (the latter excludes
 *      precision-changing passes) via the pass-list overload.
 *   5. FP16 weight pre-cast ({@code -Dnd4j.optimizer.fp16=true}): halves matmul weight
 *      memory; the example measures parameter bytes before/after and re-checks logits
 *      at a relaxed tolerance.
 *   6. Forward-pass wall-clock, raw vs optimized.
 *   7. The integration point everyone actually uses: {@link GenerationPipeline} runs the
 *      optimizer on its decoder during {@code create()} (builder flag
 *      {@code graphOptimizerEnabled}, default true) — a short greedy generation shows
 *      the optimized model producing text through the production decode path.
 *
 * Resources: the FP32-dequantized model is ~3.2GB; optimize() works on a duplicate, so
 * peak RAM is roughly two copies (~7GB). First run downloads ~500MB of GGUF into
 * ~/.cache/dl4j-llm-models.
 *
 * Note on {@code nd4j.optimizer.enabled}: that property gates the AUTOMATIC optimizer
 * runs inside model-loading paths (ONNX import cache, generation pipeline). Direct calls
 * to {@code GraphOptimizer.optimize()} — as below — always execute.
 *
 * System properties: -Dexample.prompt="The capital of France is"
 *                    -Dexample.gen.tokens=24 -Dexample.timing.iters=3
 */
public class LLMGraphOptimizerExample {

    /**
     * Logits output name of the imported graph. LLaMA-family builders expose "logits";
     * the Qwen3.5 hybrid (GDN/SSM) builder exposes "lm_logits". Resolved after import.
     */
    private static String logitsName = "logits";

    /** Model hidden size from GGUF metadata; used when assembling forward-pass inputs. */
    private static long hiddenSize = 0;

    public static void main(String[] args) throws Exception {
        String prompt = System.getProperty("example.prompt", "The capital of France is");
        int genTokens = Integer.getInteger("example.gen.tokens", 24);
        int timingIters = Integer.getInteger("example.timing.iters", 3);

        // ============================================================
        // 1. RAW IMPORT + GRAPH PROFILE
        // ============================================================
        System.out.println("=== 1. Importing Qwen3.5-0.8B (raw, no optimization) ===");
        File ggufFile = LLMModelDownloader.download(LLMModel.QWEN35_0_8B, QuantType.Q4_K_M).getModelFile();
        GGMLMetadata metadata = GGMLModelImport.inspectModel(ggufFile);
        System.out.println("  " + metadata.getArchitecture() + ", "
                + metadata.getNumLayers() + " layers, hidden " + metadata.getHiddenSize()
                + ", vocab " + metadata.getVocabSize()
                + ", context " + metadata.getContextLength());

        long t0 = System.currentTimeMillis();
        SameDiff raw = GGMLModelImport.importModel(ggufFile.getAbsolutePath(),
                ConversionOptions.forInference());
        System.out.println("  Imported in " + (System.currentTimeMillis() - t0) + "ms");

        logitsName = raw.hasVariable("logits") ? "logits" : "lm_logits";
        hiddenSize = metadata.getHiddenSize();
        System.out.println("  Logits output variable: " + logitsName);

        Map<String, Integer> rawHistogram = opHistogram(raw);
        int rawOpCount = raw.getOps().size();
        System.out.println("  Raw graph: " + rawOpCount + " ops, "
                + raw.variables().size() + " variables");
        System.out.println("  Top op types (raw):");
        printTopOps(rawHistogram, 12);

        Tokenizer tokenizer = HuggingFaceTokenizer.fromDirectory(ggufFile.getParentFile());

        // ============================================================
        // 2. REFERENCE FORWARD PASS (raw graph)
        // ============================================================
        System.out.println("\n=== 2. Reference forward pass ===");
        int[] promptIds = tokenizer.encode(prompt, false).getIds();
        INDArray inputIds = Nd4j.createFromArray(promptIds).reshape(1, promptIds.length);
        System.out.println("  Prompt: \"" + prompt + "\" (" + promptIds.length + " tokens)");

        t0 = System.currentTimeMillis();
        INDArray rawLogits = lastPositionLogits(raw, inputIds);
        System.out.println("  Raw forward: " + (System.currentTimeMillis() - t0) + "ms");
        int[] rawTop = topK(rawLogits, 5);
        System.out.println("  Raw top-5 next tokens: " + decodeTokens(tokenizer, rawTop));

        // ============================================================
        // 3. DEFAULT OPTIMIZATION PASSES
        // ============================================================
        System.out.println("\n=== 3. GraphOptimizer.optimize (default passes) ===");
        // optimize() duplicates the graph internally — `raw` is never modified. The
        // required-output list ("logits") also lets dead-code elimination drop anything
        // not needed for that output.
        t0 = System.currentTimeMillis();
        SameDiff optimized = GraphOptimizer.optimize(raw, logitsName);
        long optimizeMs = System.currentTimeMillis() - t0;

        int optOpCount = optimized.getOps().size();
        Map<String, Integer> optHistogram = opHistogram(optimized);
        System.out.println("  Optimization time: " + optimizeMs + "ms");
        System.out.println("  Ops: " + rawOpCount + " -> " + optOpCount
                + "  (" + String.format("%.1f", 100.0 * (rawOpCount - optOpCount) / rawOpCount)
                + "% removed)");

        System.out.println("  Fused/new op types introduced by the optimizer:");
        boolean anyNew = false;
        for (Map.Entry<String, Integer> e : optHistogram.entrySet()) {
            int before = rawHistogram.getOrDefault(e.getKey(), 0);
            if (e.getValue() > before) {
                System.out.println("    + " + e.getKey() + ": " + before + " -> " + e.getValue());
                anyNew = true;
            }
        }
        if (!anyNew) {
            System.out.println("    (none — fusions may already be present in this import path)");
        }
        System.out.println("  Op types reduced the most:");
        rawHistogram.entrySet().stream()
                .map(e -> new AbstractEntry(e.getKey(), e.getValue() - optHistogram.getOrDefault(e.getKey(), 0)))
                .filter(e -> e.delta > 0)
                .sorted((a, b) -> Integer.compare(b.delta, a.delta))
                .limit(8)
                .forEach(e -> System.out.println("    - " + e.name + ": removed " + e.delta));

        // ============================================================
        // 4. CORRECTNESS: same logits, same predictions
        // ============================================================
        System.out.println("\n=== 4. Correctness check (raw vs optimized) ===");
        INDArray optLogits = lastPositionLogits(optimized, inputIds);
        double maxAbsDiff = Transforms.abs(rawLogits.sub(optLogits)).maxNumber().doubleValue();
        int[] optTop = topK(optLogits, 5);
        System.out.println("  max|logit_raw - logit_optimized| = " + String.format("%.6e", maxAbsDiff));
        System.out.println("  Optimized top-5 next tokens: " + decodeTokens(tokenizer, optTop));
        boolean sameTop1 = rawTop[0] == optTop[0];
        System.out.println("  Top-1 prediction identical: " + sameTop1);
        if (!sameTop1) {
            throw new IllegalStateException("Optimizer changed the top-1 prediction — this is a bug");
        }
        optimized.close();   // each optimize() dup holds ~3GB off-heap — release deterministically
        optimized = null;

        // ============================================================
        // 5. PASS-SET CONTROL
        // ============================================================
        System.out.println("\n=== 5. Choosing pass sets ===");
        List<OptimizerSet> allPasses = GraphOptimizer.defaultOptimizations();
        List<OptimizerSet> correctnessPasses = GraphOptimizer.defaultCorrectnessOptimizations();
        System.out.println("  defaultOptimizations():            " + allPasses.size() + " pass sets");
        System.out.println("  defaultCorrectnessOptimizations(): " + correctnessPasses.size()
                + " pass sets (precision-changing passes excluded)");
        for (OptimizerSet set : allPasses) {
            System.out.println("    - " + set.getClass().getSimpleName());
        }

        SameDiff conservativelyOptimized = GraphOptimizer.optimize(raw,
                Collections.singletonList(logitsName), correctnessPasses);
        System.out.println("  Correctness-only pass run: " + rawOpCount + " -> "
                + conservativelyOptimized.getOps().size() + " ops");
        conservativelyOptimized.close();
        conservativelyOptimized = null;

        // ============================================================
        // 6. FP16 WEIGHT PRE-CAST
        // ============================================================
        System.out.println("\n=== 6. FP16 weight pre-cast (nd4j.optimizer.fp16) ===");
        long bytesBefore = parameterBytes(raw);
        System.setProperty("nd4j.optimizer.fp16", "true");
        SameDiff fp16Optimized;
        try {
            fp16Optimized = GraphOptimizer.optimize(raw, logitsName);
        } finally {
            System.clearProperty("nd4j.optimizer.fp16");
        }
        long bytesAfter = parameterBytes(fp16Optimized);
        System.out.println("  Parameter memory: " + (bytesBefore / (1024 * 1024)) + " MB -> "
                + (bytesAfter / (1024 * 1024)) + " MB");

        INDArray fp16Logits = lastPositionLogits(fp16Optimized, inputIds);
        double fp16Diff = Transforms.abs(rawLogits.sub(fp16Logits)).maxNumber().doubleValue();
        // Note: nd4j max reductions do not propagate NaN — scan the vector explicitly.
        if (hasNaN(fp16Logits)) {
            // A real result, not an error: this hybrid architecture upcasts to FP32
            // in-graph precisely because its large hidden dot products overflow HALF.
            // Pre-casting those weights to FP16 reintroduces the overflow. This is why
            // fp16 pre-cast is a per-architecture decision — verified by this check.
            System.out.println("  FP16 pre-cast produced NaN logits for this architecture:");
            System.out.println("  the importer's in-graph FP32 upcasts exist because hidden-size");
            System.out.println("  dot products overflow HALF. Memory savings above are real, but");
            System.out.println("  fp16 pre-cast must be validated per model — exactly like this.");
        } else {
            int[] fp16Top = topK(fp16Logits, 5);
            System.out.println("  max|logit diff| vs FP32: " + String.format("%.4f", fp16Diff)
                    + " (half precision — expect small drift)");
            System.out.println("  FP16 top-5 next tokens: " + decodeTokens(tokenizer, fp16Top));
            System.out.println("  Top-1 prediction identical: " + (rawTop[0] == fp16Top[0]));
        }
        fp16Optimized.close();
        fp16Optimized = null;

        // ============================================================
        // 7. FORWARD-PASS WALL CLOCK
        // ============================================================
        System.out.println("\n=== 7. Forward-pass timing (" + timingIters + " iterations) ===");
        SameDiff timedOptimized = GraphOptimizer.optimize(raw, logitsName);
        double rawMs = timeForward(raw, inputIds, timingIters);
        double optMs = timeForward(timedOptimized, inputIds, timingIters);
        System.out.println(String.format("  raw:       %.0f ms/forward", rawMs));
        System.out.println(String.format("  optimized: %.0f ms/forward  (%.2fx)", optMs, rawMs / optMs));
        timedOptimized.close();
        timedOptimized = null;

        // ============================================================
        // 8. THE PRODUCTION INTEGRATION: GenerationPipeline
        // ============================================================
        System.out.println("\n=== 8. GenerationPipeline (optimizer on by default) ===");
        // create() runs the GraphOptimizer over the decoder before compiling the DSP
        // plan — .graphOptimizerEnabled(false) opts out. This is the path every LLM
        // benchmark and the SmolDocling pipeline take.
        GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(raw)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(genTokens)
                .graphOptimizerEnabled(true)
                .build());
        try {
            GenerationResult gen = pipeline.generate(prompt, genTokens);
            System.out.println("  \"" + prompt + gen.getText() + "\"");
            System.out.println(String.format(
                    "  %d tokens | %.2f tok/s overall | %.2f tok/s steady | first token %dms",
                    gen.getGeneratedTokenCount(), gen.getTokensPerSecond(),
                    gen.getSteadyStateTokensPerSecond(), gen.getFirstTokenLatencyMs()));
        } finally {
            pipeline.close();
        }

        // ============================================================
        // 9. PROPERTY REFERENCE
        // ============================================================
        System.out.println("\n=== 9. Optimizer control properties ===");
        System.out.println("  nd4j.optimizer.enabled=true|false   gate AUTOMATIC runs (import cache, pipeline)");
        System.out.println("  nd4j.optimizer.fp16=true            FP16 matmul-weight pre-cast");
        System.out.println("  nd4j.optimizer.bf16=true            BF16 pre-cast (wins over fp16)");
        System.out.println("  nd4j.optimizer.skip=Pass1,Pass2     skip pass sets by class name");
        System.out.println("  nd4j.optimizer.maxIterations=3      fixed-point iteration cap");
        System.out.println("  nd4j.optimizer.logApplied=true      log each applied optimization");

        tokenizer.close();
        System.out.println("\nLLM GraphOptimizer example completed.");
    }

    // ================================================================
    // Helpers
    // ================================================================

    private static final class AbstractEntry {
        final String name;
        final int delta;
        AbstractEntry(String name, int delta) {
            this.name = name;
            this.delta = delta;
        }
    }

    /** Histogram of op types in the graph (opName -> count), insertion-ordered by name. */
    private static Map<String, Integer> opHistogram(SameDiff sd) {
        Map<String, Integer> histogram = new TreeMap<>();
        for (SameDiffOp op : sd.getOps().values()) {
            String name = op.getOp() != null ? op.getOp().opName() : "<null>";
            histogram.merge(name, 1, Integer::sum);
        }
        return new LinkedHashMap<>(histogram);
    }

    private static void printTopOps(Map<String, Integer> histogram, int limit) {
        histogram.entrySet().stream()
                .sorted((a, b) -> Integer.compare(b.getValue(), a.getValue()))
                .limit(limit)
                .forEach(e -> System.out.println("    " + String.format("%-24s", e.getKey()) + e.getValue()));
    }

    /**
     * Builds the COMPLETE prefill input map for one forward pass. Imported decoder
     * graphs declare per-layer state placeholders (attention KV caches; for hybrid
     * architectures like Qwen3.5 also GDN/conv recurrent state) plus scalar position
     * inputs — all must be fed. DecoderInputBuilder covers ids/masks/positions/KV; the
     * recurrent states are zero-filled via GenerationPipeline.deriveRecurrentStateShape.
     */
    private static Map<String, INDArray> buildForwardFeeds(SameDiff model, INDArray inputIds) {
        Map<String, INDArray> feeds = DecoderInputBuilder.buildDecoderInputMap(
                model.inputs(), model, null, inputIds,
                0, inputIds.size(1), null, inputIds.size(1), 0, false, hiddenSize);
        for (ModelIOConfig.RecurrentStatePair pair :
                ModelIOConfig.findRecurrentStatePairs(model, ModelIOConfig.builder().build())) {
            if (!feeds.containsKey(pair.inputName) && model.hasVariable(pair.inputName)) {
                long[] stateShape = GenerationPipeline.deriveRecurrentStateShape(model, pair.inputName);
                if (stateShape != null) {
                    feeds.put(pair.inputName,
                            Nd4j.zeros(model.getVariable(pair.inputName).dataType(), stateShape));
                }
            }
        }
        for (String scalarName : new String[]{"position_offset", "cache_position"}) {
            if (model.hasVariable(scalarName) && !feeds.containsKey(scalarName)) {
                feeds.put(scalarName, Nd4j.scalar(DataType.INT64, 0));
            }
        }
        return feeds;
    }

    /** Runs the model on [1, seqLen] input_ids and returns the last position's logits [vocab]. */
    private static INDArray lastPositionLogits(SameDiff model, INDArray inputIds) {
        INDArray logits = model.output(buildForwardFeeds(model, inputIds), logitsName)
                .get(logitsName);
        long lastPosition = inputIds.size(1) - 1;
        if (logits.rank() == 3) {
            return logits.get(NDArrayIndex.point(0), NDArrayIndex.point(lastPosition), NDArrayIndex.all())
                    .dup();
        }
        return logits.get(NDArrayIndex.point(lastPosition), NDArrayIndex.all()).dup();
    }

    /** Host-side top-K over a [vocab] logits vector (NaN entries are ignored). */
    private static int[] topK(INDArray logitsVector, int k) {
        float[] values = logitsVector.dup().data().asFloat();
        int[] top = new int[k];
        for (int i = 0; i < k; i++) {
            int best = -1;
            float bestValue = Float.NEGATIVE_INFINITY;
            for (int v = 0; v < values.length; v++) {
                if (!Float.isNaN(values[v]) && values[v] > bestValue) {
                    bestValue = values[v];
                    best = v;
                }
            }
            if (best < 0) {
                return Arrays.copyOf(top, i);   // fewer than k finite values
            }
            top[i] = best;
            values[best] = Float.NEGATIVE_INFINITY;
        }
        return top;
    }

    private static boolean hasNaN(INDArray vector) {
        for (float value : vector.dup().data().asFloat()) {
            if (Float.isNaN(value)) return true;
        }
        return false;
    }

    private static String decodeTokens(Tokenizer tokenizer, int[] tokenIds) {
        List<String> decoded = new ArrayList<>(tokenIds.length);
        for (int id : tokenIds) {
            // GGUF embedding tables can be padded past the tokenizer vocab; guard ids.
            if (id >= tokenizer.getVocabSize()) {
                decoded.add("<pad:" + id + ">");
            } else {
                decoded.add("\"" + tokenizer.decode(new int[]{id}, true).replace("\n", "\\n") + "\"");
            }
        }
        return String.join(", ", decoded);
    }

    private static double timeForward(SameDiff model, INDArray inputIds, int iterations) {
        // one untimed warmup, then average
        lastPositionLogits(model, inputIds);
        long start = System.currentTimeMillis();
        for (int i = 0; i < iterations; i++) {
            lastPositionLogits(model, inputIds);
        }
        return (System.currentTimeMillis() - start) / (double) iterations;
    }

    /** Total bytes of all parameter arrays (variables + constants). */
    private static long parameterBytes(SameDiff sd) {
        long bytes = 0;
        for (SDVariable v : sd.variables()) {
            INDArray arr = v.getArr();
            // Empty arrays carry no data buffer; dtype width avoids touching it.
            if (arr != null && !arr.isEmpty()) {
                bytes += arr.length() * arr.dataType().width();
            }
        }
        return bytes;
    }
}
