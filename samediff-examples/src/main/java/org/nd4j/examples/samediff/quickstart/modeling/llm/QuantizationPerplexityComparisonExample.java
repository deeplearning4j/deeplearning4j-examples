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
import org.eclipse.deeplearning4j.llm.eval.PerplexityEvaluator;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.SameDiffMemoryUtils;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * How much quality does quantization actually cost? This example answers with REAL
 * measurements instead of folklore: it evaluates the SAME model (Qwen3.5-0.8B) at two
 * quantization levels on the SAME WikiText-2 text and reports perplexity, file size,
 * import time and a sample generation side by side.
 *
 *   Q4_K_M — 4.5-bit k-quant "medium": the standard deployment choice (~508MB)
 *   Q8_0   — 8-bit: near-lossless reference                          (~774MB)
 *
 * Methodology notes worth stealing for your own evals:
 *   - Perplexity via {@link PerplexityEvaluator#evaluate} runs true sliding-window
 *     scoring over the imported graph. It assembles the COMPLETE decoder input map
 *     (KV caches, recurrent GDN/conv states, position scalars) internally through
 *     {@code DecoderInputBuilder.buildScoringInputMap} — Qwen3.5 is a hybrid
 *     attention/SSM architecture and a bare {@code {input_ids}} feed does not run.
 *   - Both quant levels dequantize to FP32 at import ({@code ConversionOptions
 *     .forInference()}), so the perplexity difference isolates exactly the
 *     information lost by the 4-bit vs 8-bit WEIGHT encoding — same compute path,
 *     same dtype, same kernels.
 *   - Models are evaluated SEQUENTIALLY with a full close (~3.5GB each dequantized);
 *     the teardown between them exercises the DSP plan release path on a real
 *     generation-scale plan.
 *   - Windows are deliberately few for a runnable example; production comparisons
 *     want the full test set ({@link PerplexityEvaluator#evaluateWikiText2}) and
 *     matched context/stride to published numbers.
 *
 * Expected shape of the result: Q8_0 perplexity a few percent better than Q4_K_M at
 * ~1.5x the bytes; both generations coherent. If Q4 ever comes out dramatically worse,
 * the quantization of a specific tensor class (usually attention or embedding) is the
 * suspect — that is exactly what this harness is for.
 *
 * First run downloads whichever GGUF is not cached (~508MB + ~774MB, under
 * ~/.cache/dl4j-llm-models). System properties:
 *   -Dexample.ppl.chars=2500   characters of WikiText-2 to score (~600 tokens)
 *   -Dexample.ctx=64           context window   -Dexample.stride=64
 *   -Dexample.gen.tokens=12    sample generation length (0 to skip)
 */
public class QuantizationPerplexityComparisonExample {

    /** One quant level's measurements. */
    private static final class QuantRun {
        final String name;
        final long fileBytes;
        long importMs;
        double perplexity;
        double bitsPerByte;
        int tokens;
        long evalMs;
        String sample;
        double sampleTokPerSec;

        QuantRun(String name, long fileBytes) {
            this.name = name;
            this.fileBytes = fileBytes;
        }
    }

    public static void main(String[] args) throws Exception {
        int pplChars = Integer.getInteger("example.ppl.chars", 2500);
        int ctx = Integer.getInteger("example.ctx", 64);
        int stride = Integer.getInteger("example.stride", 64);
        int genTokens = Integer.getInteger("example.gen.tokens", 12);
        String prompt = "The most important property of a good compression algorithm is";

        // ============================================================
        // 1. THE EVALUATION TEXT — identical for every quant level
        // ============================================================
        System.out.println("=== 1. Evaluation corpus ===");
        String wikiText = PerplexityEvaluator.loadWikiText2();
        String evalText = wikiText.substring(0, Math.min(wikiText.length(), pplChars));
        System.out.println("  WikiText-2 test slice: " + evalText.length() + " chars, context="
                + ctx + ", stride=" + stride);

        // ============================================================
        // 2. EVALUATE EACH QUANT LEVEL SEQUENTIALLY
        // ============================================================
        QuantType[] levels = {QuantType.Q4_K_M, QuantType.Q8_0};
        List<QuantRun> runs = new ArrayList<>();

        for (QuantType quant : levels) {
            System.out.println("\n=== 2. " + quant + " ===");
            File gguf = LLMModelDownloader.download(LLMModel.QWEN35_0_8B, quant).getModelFile();
            QuantRun run = new QuantRun(quant.name(), gguf.length());
            System.out.println("  File: " + gguf.getName() + " ("
                    + (run.fileBytes / (1024 * 1024)) + " MB)");

            long t0 = System.currentTimeMillis();
            SameDiff model = GGMLModelImport.importModel(gguf.getAbsolutePath(),
                    ConversionOptions.forInference());
            run.importMs = System.currentTimeMillis() - t0;
            System.out.println("  Imported (dequantized to FP32) in " + run.importMs + "ms");

            Tokenizer tokenizer = HuggingFaceTokenizer.fromDirectory(gguf.getParentFile());

            // --- Perplexity: identical text, window and stride for both levels ---
            PerplexityEvaluator.PerplexityResult ppl =
                    PerplexityEvaluator.evaluate(model, tokenizer, evalText, ctx, stride);
            run.perplexity = ppl.getPerplexity();
            run.bitsPerByte = ppl.getBitsPerByte();
            run.tokens = ppl.getNumTokens();
            run.evalMs = ppl.getEvaluationTimeMs();
            System.out.println(String.format(
                    "  Perplexity %.3f | bits/byte %.4f | %d tokens in %dms",
                    run.perplexity, run.bitsPerByte, run.tokens, run.evalMs));

            // --- Sample generation through the production pipeline ---
            if (genTokens > 0) {
                try (GenerationPipeline pipeline = GenerationPipeline.create(
                        GenerationPipelineConfig.builder()
                                .decoder(model)
                                .tokenizer(tokenizer)
                                .samplingConfig(SamplingConfig.greedy())
                                .maxNewTokens(genTokens)
                                .build())) {
                    GenerationResult gen = pipeline.generate(prompt, genTokens);
                    run.sample = gen.getText().replace("\n", " ");
                    run.sampleTokPerSec = gen.getTokensPerSecond();
                    System.out.println(String.format("  Sample (%.2f tok/s): \"%s%s\"",
                            run.sampleTokPerSec, prompt, run.sample));
                }
            }

            // --- Full teardown before the next level (~3.5GB reclaimed) ---
            // Supported order: close the SameDiff (native DSP plan release) first,
            // THEN free the model arrays.
            model.close();
            SameDiffMemoryUtils.freeModelArrays(model);
            tokenizer.close();
            runs.add(run);
        }

        // ============================================================
        // 3. SIDE-BY-SIDE VERDICT
        // ============================================================
        System.out.println("\n=== 3. Comparison ===");
        System.out.println("  Level   |  MB  | import ms | perplexity | bits/byte | eval tok");
        for (QuantRun run : runs) {
            System.out.println(String.format("  %-7s | %4d | %9d | %10.3f | %9.4f | %8d",
                    run.name, run.fileBytes / (1024 * 1024), run.importMs,
                    run.perplexity, run.bitsPerByte, run.tokens));
        }
        if (runs.size() == 2 && runs.get(1).perplexity > 0) {
            QuantRun q4 = runs.get(0);
            QuantRun q8 = runs.get(1);
            double pplDeltaPct = 100.0 * (q4.perplexity - q8.perplexity) / q8.perplexity;
            double sizeRatio = (double) q8.fileBytes / q4.fileBytes;
            System.out.println(String.format(
                    "\n  Q4_K_M costs %+.2f%% perplexity vs Q8_0 and saves %.2fx the bytes.",
                    pplDeltaPct, sizeRatio));
            System.out.println("  (Positive delta = Q4 worse. Small single-digit percentages are the");
            System.out.println("   typical k-quant tax; blowups indicate a tensor class that should");
            System.out.println("   be excluded from aggressive quantization.)");
        }

        System.out.println("\nQuantization perplexity comparison completed.");
    }
}
