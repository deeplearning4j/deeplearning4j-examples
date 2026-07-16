/*
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.nd4j.examples.samediff.advanced.generation;

import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Demonstrates the LLM generation pipeline APIs by constructing real SamplingConfig
 * objects, building GenerationPipelineConfig with a SameDiff model, and showing
 * how the pipeline configuration connects models, tokenizers, and sampling strategies.
 */
public class LLMGenerationPipelineExample {

    public static void main(String[] args) throws Exception {

        // ================================================================
        // 1. SamplingConfig presets and custom configs
        // ================================================================
        System.out.println("=== 1. SamplingConfig presets ===");

        SamplingConfig greedy = SamplingConfig.greedy();
        SamplingConfig precise = SamplingConfig.precise();
        SamplingConfig creative = SamplingConfig.creative();
        SamplingConfig defaults = SamplingConfig.defaultConfig();
        SamplingConfig llamaCpp = SamplingConfig.llamaCppDefaults();

        SamplingConfig[] presets = {greedy, precise, creative, defaults, llamaCpp};
        String[] names = {"greedy", "precise", "creative", "defaultConfig", "llamaCppDefaults"};

        System.out.printf("  %-18s %-8s %-6s %-6s %-6s %-10s%n",
                "Preset", "temp", "topK", "topP", "doSamp", "repPenalty");
        System.out.println("  " + "-".repeat(60));
        for (int i = 0; i < presets.length; i++) {
            SamplingConfig c = presets[i];
            System.out.printf("  %-18s %-8.2f %-6d %-6.2f %-6s %-10.2f%n",
                    names[i],
                    c.getTemperature(),
                    c.getTopK(),
                    c.getTopP(),
                    c.isDoSample(),
                    c.getRepetitionPenalty());
        }

        // Custom config via builder
        SamplingConfig custom = SamplingConfig.builder()
                .temperature(0.7)
                .topK(40)
                .topP(0.95)
                .repetitionPenalty(1.1)
                .doSample(true)
                .build();

        System.out.println("\n  Custom: temp=" + custom.getTemperature()
                + " topK=" + custom.getTopK()
                + " topP=" + custom.getTopP()
                + " repPenalty=" + custom.getRepetitionPenalty()
                + " doSample=" + custom.isDoSample());

        // ================================================================
        // 2. Build a SameDiff model for pipeline config
        // ================================================================
        System.out.println("\n=== 2. Build model for pipeline ===");

        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, 32, 64).muli(0.05));
        SDVariable b1 = sd.var("b1", Nd4j.zeros(DataType.FLOAT, 64));
        SDVariable hidden = sd.nn().gelu("hidden", input.mmul(w1).add(b1));
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, 64, 32).muli(0.05));
        SDVariable output = sd.nn().softmax("output", hidden.mmul(w2), -1);

        // Verify model works
        INDArray testInput = Nd4j.rand(DataType.FLOAT, 2, 32);
        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", testInput);
        INDArray testOutput = sd.outputSingle(ph, "output");

        System.out.println("  Model: [?,32] -> GELU(64) -> softmax(32)");
        System.out.println("  Variables: " + sd.variableNames().size());
        System.out.println("  Output shape: " + Arrays.toString(testOutput.shape()));
        System.out.println("  Output row 0 sum: " + testOutput.getRow(0).sumNumber());

        // ================================================================
        // 3. GenerationPipelineConfig
        // ================================================================
        System.out.println("\n=== 3. GenerationPipelineConfig ===");

        GenerationPipelineConfig config = GenerationPipelineConfig.builder()
                .decoder(sd)
                .samplingConfig(greedy)
                .maxNewTokens(200)
                .maxKvCacheLength(4096)
                .build();

        System.out.println("  maxNewTokens:      " + config.getMaxNewTokens());
        System.out.println("  maxKvCacheLength:  " + config.getMaxKvCacheLength());
        System.out.println("  decoder set:       " + (config.getDecoder() != null));
        System.out.println("  samplingConfig:    temp=" + config.getSamplingConfig().getTemperature());

        // Config with speculative decoding parameters
        GenerationPipelineConfig specConfig = GenerationPipelineConfig.builder()
                .decoder(sd)
                .samplingConfig(precise)
                .maxNewTokens(500)
                .maxKvCacheLength(8192)
                .maxSpeculativeTokens(5)
                .build();

        System.out.println("\n  Speculative config:");
        System.out.println("    maxNewTokens:          " + specConfig.getMaxNewTokens());
        System.out.println("    maxKvCacheLength:      " + specConfig.getMaxKvCacheLength());
        System.out.println("    maxSpeculativeTokens:  " + specConfig.getMaxSpeculativeTokens());

        // Config with creative sampling and prefill settings
        GenerationPipelineConfig creativeConfig = GenerationPipelineConfig.builder()
                .decoder(sd)
                .samplingConfig(creative)
                .maxNewTokens(300)
                .maxKvCacheLength(2048)
                .maxPrefillLength(1024)
                .build();

        System.out.println("\n  Creative config:");
        System.out.println("    maxPrefillLength: " + creativeConfig.getMaxPrefillLength());
        System.out.println("    samplingConfig:   temp=" + creativeConfig.getSamplingConfig().getTemperature()
                + " topK=" + creativeConfig.getSamplingConfig().getTopK());

        // ================================================================
        // 4. Sampling strategy comparison
        // ================================================================
        System.out.println("\n=== 4. Sampling strategy comparison ===");

        // Simulate how different temperatures affect a logit distribution
        INDArray logits = Nd4j.create(new double[]{2.0, 1.0, 0.5, 0.1, -0.5, -1.0});
        System.out.println("  Raw logits: " + logits);

        double[] temps = {0.1, 0.5, 0.7, 1.0, 1.5, 2.0};
        for (double temp : temps) {
            INDArray scaled = logits.div(Math.max(temp, 1e-8));
            INDArray expScaled = Nd4j.math().exp(scaled);
            INDArray probs = expScaled.div(expScaled.sumNumber());
            double maxProb = probs.maxNumber().doubleValue();
            double entropy = 0;
            for (int i = 0; i < probs.length(); i++) {
                double p = probs.getDouble(i);
                if (p > 0) entropy -= p * Math.log(p);
            }
            System.out.printf("  temp=%.1f: max_prob=%.4f entropy=%.3f probs=%s%n",
                    temp, maxProb, entropy, probs);
        }

        // ================================================================
        // 5. Top-K filtering demonstration
        // ================================================================
        System.out.println("\n=== 5. Top-K filtering ===");

        INDArray probDist = Nd4j.create(new double[]{0.3, 0.25, 0.15, 0.12, 0.08, 0.05, 0.03, 0.02});
        System.out.println("  Full distribution: " + probDist);

        int[] topKValues = {1, 3, 5, 8};
        for (int k : topKValues) {
            // Simulate top-k: zero out everything below top-k, renormalize
            INDArray sorted = probDist.dup();
            double[] vals = new double[(int) sorted.length()];
            for (int i = 0; i < sorted.length(); i++) vals[i] = sorted.getDouble(i);
            Arrays.sort(vals);
            double threshold = vals[vals.length - k];
            INDArray filtered = probDist.dup();
            for (int i = 0; i < filtered.length(); i++) {
                if (filtered.getDouble(i) < threshold) filtered.putScalar(i, 0);
            }
            double sum = filtered.sumNumber().doubleValue();
            if (sum > 0) filtered.divi(sum);
            System.out.printf("  topK=%d: %s (sum=%.2f)%n", k, filtered, filtered.sumNumber());
        }

        // ================================================================
        // 6. Top-P (nucleus) filtering demonstration
        // ================================================================
        System.out.println("\n=== 6. Top-P nucleus sampling ===");

        double[] topPValues = {0.5, 0.8, 0.9, 0.95, 1.0};
        for (double p : topPValues) {
            // Simulate nucleus sampling: keep smallest set of tokens whose cumulative prob >= p
            double cumProb = 0;
            int tokensKept = 0;
            for (int i = 0; i < probDist.length(); i++) {
                cumProb += probDist.getDouble(i);
                tokensKept++;
                if (cumProb >= p) break;
            }
            System.out.printf("  topP=%.2f: keep %d of %d tokens (cum_prob=%.3f)%n",
                    p, tokensKept, probDist.length(), cumProb);
        }

        // ================================================================
        // 7. Model variable summary
        // ================================================================
        System.out.println("\n=== 7. Model variable summary ===");

        long totalParams = 0;
        for (String varName : sd.variableNames()) {
            SDVariable v = sd.getVariable(varName);
            INDArray arr = v.getArr();
            if (arr != null) {
                long numParams = arr.length();
                totalParams += numParams;
                System.out.println("  " + varName + ": " + Arrays.toString(arr.shape())
                        + " (" + numParams + " params, " + arr.dataType() + ")");
            }
        }
        System.out.println("  Total parameters: " + totalParams);
        System.out.printf("  Estimated size (FP32): %.2f KB%n", totalParams * 4.0 / 1024);
        System.out.printf("  Estimated size (FP16): %.2f KB%n", totalParams * 2.0 / 1024);

        System.out.println("\nLLMGenerationPipelineExample complete.");
    }
}
