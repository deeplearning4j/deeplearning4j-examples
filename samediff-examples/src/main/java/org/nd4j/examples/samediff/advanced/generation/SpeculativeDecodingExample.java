/*
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.nd4j.examples.samediff.advanced.generation;

import org.eclipse.deeplearning4j.llm.speculative.NgramSpeculator;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;

import java.util.Arrays;
import java.util.Random;

/**
 * Demonstrates speculative decoding concepts by building and using NgramSpeculators
 * with real token sequences, creating SamplingConfig presets, and computing the
 * mathematical relationship between acceptance rate and speedup.
 */
public class SpeculativeDecodingExample {

    public static void main(String[] args) throws Exception {

        // ================================================================
        // 1. NgramSpeculator — build and configure
        // ================================================================
        System.out.println("=== 1. NgramSpeculator construction ===");

        NgramSpeculator trigram = NgramSpeculator.builder()
                .ngramSize(3)
                .maxDraftTokens(5)
                .minFrequency(1)
                .build();

        System.out.println("  ngramSize:      " + trigram.getNgramSize());
        System.out.println("  maxDraftTokens: " + trigram.getMaxDraftTokens());
        System.out.println("  minFrequency:   " + trigram.getMinFrequency());

        NgramSpeculator bigram = NgramSpeculator.builder()
                .ngramSize(2)
                .maxDraftTokens(3)
                .minFrequency(2)
                .build();

        System.out.println("  Bigram speculator: size=" + bigram.getNgramSize()
                + " maxDraft=" + bigram.getMaxDraftTokens()
                + " minFreq=" + bigram.getMinFrequency());

        NgramSpeculator fivegram = NgramSpeculator.builder()
                .ngramSize(5)
                .maxDraftTokens(8)
                .minFrequency(1)
                .build();

        System.out.println("  5-gram speculator: size=" + fivegram.getNgramSize()
                + " maxDraft=" + fivegram.getMaxDraftTokens());

        // ================================================================
        // 2. Feed context and observe n-gram table behavior
        // ================================================================
        System.out.println("\n=== 2. N-gram context feeding ===");

        // Simulate a repetitive token sequence (like code or structured text)
        // Tokens: [10, 20, 30, 10, 20, 30, 10, 20, 30, 40, 50]
        int[] repetitiveContext = {10, 20, 30, 10, 20, 30, 10, 20, 30, 40, 50};

        System.out.println("  Context tokens: " + Arrays.toString(repetitiveContext));
        System.out.println("  Pattern: [10,20,30] repeats 3 times");
        System.out.println();

        // With a trigram speculator:
        //   (10,20) -> 30 (seen 3 times)
        //   (20,30) -> 10 (seen 2 times)
        //   (30,10) -> 20 (seen 2 times)
        // This is exactly the pattern that makes speculative decoding effective:
        // structured, repetitive text gives high acceptance rates.

        trigram.updateContext(repetitiveContext);
        int[] drafted = trigram.draft(new int[]{10, 20});
        System.out.println("  After feeding context, draft from [10,20]: " +
                (drafted != null ? Arrays.toString(drafted) : "null (no match)"));

        int[] drafted2 = trigram.draft(new int[]{20, 30});
        System.out.println("  Draft from [20,30]: " +
                (drafted2 != null ? Arrays.toString(drafted2) : "null (no match)"));

        // Non-repetitive context should yield fewer/no drafts
        int[] randomContext = new int[20];
        Random rng = new Random(42);
        for (int i = 0; i < randomContext.length; i++) {
            randomContext[i] = rng.nextInt(1000);
        }
        NgramSpeculator fresh = NgramSpeculator.builder()
                .ngramSize(3).maxDraftTokens(5).minFrequency(1).build();
        fresh.updateContext(randomContext);
        int[] drafted3 = fresh.draft(new int[]{randomContext[0], randomContext[1]});
        System.out.println("  Random context draft: " +
                (drafted3 != null ? Arrays.toString(drafted3) : "null (no match)"));

        // ================================================================
        // 3. SamplingConfig presets
        // ================================================================
        System.out.println("\n=== 3. SamplingConfig presets ===");

        SamplingConfig greedy = SamplingConfig.greedy();
        SamplingConfig topK50 = SamplingConfig.topK(50);
        SamplingConfig topP90 = SamplingConfig.topP(0.9);
        SamplingConfig precise = SamplingConfig.precise();

        System.out.println("  Greedy:  temp=" + greedy.getTemperature()
                + " topK=" + greedy.getTopK() + " topP=" + greedy.getTopP());
        System.out.println("  TopK50:  temp=" + topK50.getTemperature()
                + " topK=" + topK50.getTopK() + " topP=" + topK50.getTopP());
        System.out.println("  TopP90:  temp=" + topP90.getTemperature()
                + " topK=" + topP90.getTopK() + " topP=" + topP90.getTopP());
        System.out.println("  Precise: temp=" + precise.getTemperature()
                + " topK=" + precise.getTopK() + " topP=" + precise.getTopP());

        SamplingConfig custom = SamplingConfig.builder()
                .temperature(0.7)
                .topK(40)
                .topP(0.95)
                .repetitionPenalty(1.1)
                .build();

        System.out.println("  Custom:  temp=" + custom.getTemperature()
                + " topK=" + custom.getTopK()
                + " topP=" + custom.getTopP()
                + " repPenalty=" + custom.getRepetitionPenalty());

        // ================================================================
        // 4. Acceptance rate vs speedup — computed table
        // ================================================================
        System.out.println("\n=== 4. Acceptance rate vs speedup ===");
        System.out.println();
        System.out.println("  Speedup formula: (1 + K*alpha) / (1 + overhead)");
        System.out.println("  where K=draft tokens per step, alpha=acceptance rate");
        System.out.println();

        int[] draftCounts = {3, 5, 8};
        double[] acceptRates = {0.3, 0.5, 0.6, 0.7, 0.8, 0.9};
        double overhead = 0.1;

        System.out.printf("  %-8s", "alpha");
        for (int K : draftCounts) {
            System.out.printf("  K=%-6d", K);
        }
        System.out.println();
        System.out.println("  " + "-".repeat(35));

        for (double alpha : acceptRates) {
            System.out.printf("  %-8.1f", alpha);
            for (int K : draftCounts) {
                double speedup = (1.0 + K * alpha) / (1.0 + overhead);
                System.out.printf("  %-8.2fx", speedup);
            }
            System.out.println();
        }

        // ================================================================
        // 5. Expected tokens per step
        // ================================================================
        System.out.println("\n=== 5. Expected accepted tokens per step ===");
        System.out.println();

        // E[accepted] = sum_{i=1}^{K} alpha^i + 1 (the +1 is the fallback token)
        // For greedy with good draft: alpha ~ 0.8
        System.out.println("  E[tokens per step] = 1 + alpha + alpha^2 + ... + alpha^K");
        System.out.println();

        for (double alpha : new double[]{0.5, 0.7, 0.8, 0.9}) {
            int K = 5;
            double expected = 0;
            for (int i = 0; i <= K; i++) {
                expected += Math.pow(alpha, i);
            }
            // Adjusted: expected = (1 - alpha^(K+1)) / (1 - alpha) for geometric series
            double expectedGeo = (1 - Math.pow(alpha, K + 1)) / (1 - alpha);
            System.out.printf("  alpha=%.1f, K=%d: E[tokens]=%.2f%n", alpha, K, expectedGeo);
        }

        // ================================================================
        // 6. Ngram size vs draft quality tradeoff
        // ================================================================
        System.out.println("\n=== 6. N-gram size comparison ===");

        int[] ngramSizes = {2, 3, 4, 5};
        // Longer n-grams match more context but have fewer table hits
        for (int n : ngramSizes) {
            NgramSpeculator spec = NgramSpeculator.builder()
                    .ngramSize(n)
                    .maxDraftTokens(5)
                    .minFrequency(1)
                    .build();
            spec.updateContext(repetitiveContext);

            // Try drafting from the last (n-1) tokens of the repetitive pattern
            int[] lastN = new int[n - 1];
            System.arraycopy(repetitiveContext, 0, lastN, 0, Math.min(n - 1, repetitiveContext.length));
            int[] draft = spec.draft(lastN);

            System.out.println("  " + n + "-gram: ngramSize=" + spec.getNgramSize()
                    + " draft=" + (draft != null ? Arrays.toString(draft) : "none")
                    + " (context_key_len=" + (n - 1) + ")");
        }

        // ================================================================
        // 7. Use case recommendations based on config
        // ================================================================
        System.out.println("\n=== 7. Configuration recommendations ===");

        // Build configs for different use cases and show their settings
        NgramSpeculator codeConfig = NgramSpeculator.builder()
                .ngramSize(4).maxDraftTokens(8).minFrequency(1).build();
        NgramSpeculator chatConfig = NgramSpeculator.builder()
                .ngramSize(3).maxDraftTokens(4).minFrequency(2).build();
        NgramSpeculator structuredConfig = NgramSpeculator.builder()
                .ngramSize(5).maxDraftTokens(10).minFrequency(1).build();

        System.out.println("  Code generation:   ngram=" + codeConfig.getNgramSize()
                + " maxDraft=" + codeConfig.getMaxDraftTokens()
                + " minFreq=" + codeConfig.getMinFrequency()
                + " (high repetition -> aggressive drafting)");
        System.out.println("  Chat/instruction:  ngram=" + chatConfig.getNgramSize()
                + " maxDraft=" + chatConfig.getMaxDraftTokens()
                + " minFreq=" + chatConfig.getMinFrequency()
                + " (diverse text -> conservative)");
        System.out.println("  Structured output: ngram=" + structuredConfig.getNgramSize()
                + " maxDraft=" + structuredConfig.getMaxDraftTokens()
                + " minFreq=" + structuredConfig.getMinFrequency()
                + " (JSON/tables -> long drafts)");

        System.out.println("\nSpeculativeDecodingExample complete.");
    }
}
