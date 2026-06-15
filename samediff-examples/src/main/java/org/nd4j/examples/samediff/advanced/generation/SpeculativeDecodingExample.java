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

package org.nd4j.examples.samediff.advanced.generation;

import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.NgramSpeculator;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;
import org.eclipse.deeplearning4j.llm.generation.SpeculativeDecodeLoop;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Speculative Decoding for Faster LLM Inference — Working API Reference.
 *
 * <p>Speculative decoding reduces autoregressive generation latency by using a
 * fast "draft" model (or heuristic) to propose several candidate tokens at
 * once, then verifying all of them in a single forward pass of the large
 * "target" model.  When the draft tokens are accepted they are all emitted
 * in one step; rejected tokens cause a fall-back to the target model's
 * single-token output.  In practice this yields 2–4× speedups with no
 * change in output quality.
 *
 * <h3>Two speculator variants:</h3>
 * <ul>
 *   <li><b>NgramSpeculator</b> — draft tokens by matching recent context
 *       against an n-gram lookup table built from the prompt.  No extra
 *       model required; best for repetitive or structured text.</li>
 *   <li><b>DraftModelSpeculator</b> — uses a small language model (e.g.
 *       the 0.6B version of the same model family) to propose tokens.
 *       Higher acceptance rate; requires a compatible draft model.</li>
 * </ul>
 *
 * <h3>Key classes (all in {@code org.eclipse.deeplearning4j.llm.generation}):</h3>
 * <ul>
 *   <li>{@link NgramSpeculator}          — heuristic draft via n-gram matching</li>
 *   <li>{@link SpeculativeDecodeLoop}    — core loop: draft → verify → emit</li>
 *   <li>{@link GenerationPipelineConfig} — config (includes draftModelPath /
 *                                          maxSpeculativeTokens / speculator)</li>
 *   <li>{@link SamplingConfig}           — temperature / top-k / top-p settings</li>
 * </ul>
 *
 * <p>Performance notes:
 * <ul>
 *   <li>Acceptance rate depends on how well the draft matches the target model.</li>
 *   <li>Greedy sampling (temperature=0) gives the highest acceptance rates.</li>
 *   <li>With temperature &gt; 0, speculative decoding applies a corrected distribution
 *       so output statistics are identical to standard sampling.</li>
 *   <li>The speedup is roughly proportional to mean accepted draft length.</li>
 * </ul>
 *
 * <p>Run with:
 * <pre>
 *   cd samediff-examples
 *   mvn exec:java \
 *     -Dexec.mainClass="org.nd4j.examples.samediff.advanced.generation.SpeculativeDecodingExample"
 * </pre>
 */
public class SpeculativeDecodingExample {

    public static void main(String[] args) throws Exception {

        System.out.println("=== Speculative Decoding API Reference ===");
        System.out.println();
        System.out.println("Speculative decoding accelerates generation by drafting multiple");
        System.out.println("tokens ahead, then verifying them all in one target-model forward pass.");
        System.out.println("Output distribution is mathematically equivalent to standard sampling.");
        System.out.println();

        // ============================================================
        // 1. NGRAM SPECULATOR — construct and exercise the n-gram table
        // ============================================================
        System.out.println("=== 1. NgramSpeculator — heuristic draft, no extra model ===");
        System.out.println();

        // NgramSpeculator maintains a dynamic n-gram table of the tokens seen
        // so far in the context window.  When proposing draft tokens it looks up
        // the last (ngramSize-1) tokens as a key and suggests the most frequent
        // continuation.  Simple, fast, and surprisingly effective for repetitive
        // or formulaic text (code, structured output, legal boilerplate, etc.).
        //
        // Constructor: NgramSpeculator(int ngramSize, int maxSpeculativeTokens)
        //   ngramSize            — window size (3 = trigrams)
        //   maxSpeculativeTokens — max candidates to propose per verify step

        NgramSpeculator ngramSpeculator = new NgramSpeculator(
                3,   // ngramSize:            trigrams (look back 2 tokens, predict 1)
                5    // maxSpeculativeTokens: up to 5 candidates per verification step
        );

        System.out.println("  NgramSpeculator created:");
        System.out.println("    ngramSize:            " + ngramSpeculator.getNgramSize()
                + "  (look back ngramSize-1=2 tokens to predict next)");
        System.out.println("    maxSpeculativeTokens: " + ngramSpeculator.getMaxSpeculativeTokens()
                + "  (propose up to 5 candidates per step)");
        System.out.println();

        // Feed a repetitive token sequence to build the internal n-gram table.
        // The sequence [10, 20, 30] repeats three times: after seeing (10,20)
        // three times followed by 30, the trigram (10,20)->30 is the dominant entry.
        //
        // speculate(List<Integer>) learns from every prefix within the list and
        // returns draft continuations for the tail of the given context.
        List<Integer> context = Arrays.asList(10, 20, 30, 10, 20, 30, 10, 20, 30);
        System.out.println("  Feeding context to build n-gram table: " + context);
        System.out.println("  (sequence [10,20,30] repeats 3 times — trigram 10,20->30 will dominate)");
        System.out.println();

        // Call speculate on successive sliding windows so the table is populated
        // before we query it for predictions.
        int[] contextArr = context.stream().mapToInt(Integer::intValue).toArray();
        for (int start = 0; start <= contextArr.length - ngramSpeculator.getNgramSize(); start++) {
            List<Integer> window = new ArrayList<>();
            for (int i = start; i < Math.min(start + ngramSpeculator.getNgramSize() + 4,
                    contextArr.length); i++) {
                window.add(contextArr[i]);
            }
            ngramSpeculator.speculate(window);
        }

        // Query: given prefix [10, 20], what tokens does the speculator propose?
        List<Integer> queryPrefix = Arrays.asList(10, 20);
        int[] drafted = ngramSpeculator.speculate(queryPrefix);
        System.out.print("  speculate([10, 20]) returned: ");
        if (drafted == null || drafted.length == 0) {
            System.out.println("[] (n-gram table sparse — need more context)");
        } else {
            System.out.println(Arrays.toString(drafted));
            if (drafted[0] == 30) {
                System.out.println("    -> first draft token is 30 (correct: 10,20->30 seen 3x)");
            }
        }
        System.out.println();

        // Different speculator sizes
        NgramSpeculator bigram   = new NgramSpeculator(2, 3);
        NgramSpeculator fivegram = new NgramSpeculator(5, 8);
        System.out.println("  Bigram speculator:   ngramSize=" + bigram.getNgramSize()
                + "  maxSpeculativeTokens=" + bigram.getMaxSpeculativeTokens());
        System.out.println("  5-gram speculator:   ngramSize=" + fivegram.getNgramSize()
                + "  maxSpeculativeTokens=" + fivegram.getMaxSpeculativeTokens());
        System.out.println();

        // Verify static helpers:
        //   verifySpeculation(int[] draftTokens, float[][] targetLogits)
        //     Returns how many draft tokens are accepted under greedy verification.
        //   getCorrectionToken(float[] logits)
        //     Returns the argmax correction token after a rejection event.
        int[] draftTokens = {30, 10, 20};
        float[][] fakeLogits = new float[3][];
        fakeLogits[0] = new float[50];  fakeLogits[0][30] = 10.0f;  // target agrees: 30
        fakeLogits[1] = new float[50];  fakeLogits[1][10] = 10.0f;  // target agrees: 10
        fakeLogits[2] = new float[50];  fakeLogits[2][20] = 10.0f;  // target agrees: 20
        int accepted = NgramSpeculator.verifySpeculation(draftTokens, fakeLogits);
        System.out.printf("  verifySpeculation(%s, targetLogits) -> %d token(s) accepted%n",
                Arrays.toString(draftTokens), accepted);

        float[] correctionLogits = new float[50];
        correctionLogits[7]  = 3.0f;
        correctionLogits[42] = 5.0f;  // highest logit -> correction token drawn near 42
        int correctionToken = NgramSpeculator.getCorrectionToken(correctionLogits);
        System.out.println("  getCorrectionToken(logits[42]=5.0 highest) -> token " + correctionToken);
        System.out.println("  (used after a rejection to sample from the corrected target distribution)");
        System.out.println();

        System.out.println("  When to use NgramSpeculator:");
        System.out.println("    - No draft model available");
        System.out.println("    - Generating repetitive or structured text (code, tables, JSON)");
        System.out.println("    - Minimal memory overhead required");
        System.out.println("    - Quick prototyping / benchmarking");
        System.out.println();

        // ============================================================
        // 2. SAMPLING CONFIGS — all presets and a custom build
        // ============================================================
        System.out.println("=== 2. SamplingConfig — all presets and custom config ===");
        System.out.println();

        SamplingConfig greedy     = SamplingConfig.greedy();
        SamplingConfig defaultCfg = SamplingConfig.defaultConfig();
        SamplingConfig precise    = SamplingConfig.precise();
        SamplingConfig creative   = SamplingConfig.creative();
        SamplingConfig llamaCpp   = SamplingConfig.llamaCppDefaults();

        // Custom config via builder
        SamplingConfig custom = SamplingConfig.builder()
                .temperature(0.7)
                .topK(40)
                .topP(0.95)
                .repetitionPenalty(1.1)
                .maxNewTokens(200)
                .doSample(true)
                .build();

        System.out.printf("  %-22s  temp=%.2f  topK=%-4d  topP=%.2f  doSample=%b%n",
                "greedy()",
                greedy.getTemperature(), greedy.getTopK(), greedy.getTopP(),
                greedy.isDoSample());
        System.out.printf("  %-22s  temp=%.2f  topK=%-4d  topP=%.2f  doSample=%b%n",
                "defaultConfig()",
                defaultCfg.getTemperature(), defaultCfg.getTopK(), defaultCfg.getTopP(),
                defaultCfg.isDoSample());
        System.out.printf("  %-22s  temp=%.2f  topK=%-4d  topP=%.2f  doSample=%b%n",
                "precise()",
                precise.getTemperature(), precise.getTopK(), precise.getTopP(),
                precise.isDoSample());
        System.out.printf("  %-22s  temp=%.2f  topK=%-4d  topP=%.2f  doSample=%b%n",
                "creative()",
                creative.getTemperature(), creative.getTopK(), creative.getTopP(),
                creative.isDoSample());
        System.out.printf("  %-22s  temp=%.2f  topK=%-4d  topP=%.2f  doSample=%b%n",
                "llamaCppDefaults()",
                llamaCpp.getTemperature(), llamaCpp.getTopK(), llamaCpp.getTopP(),
                llamaCpp.isDoSample());
        System.out.printf("  %-22s  temp=%.2f  topK=%-4d  topP=%.2f  doSample=%b  "
                + "repPenalty=%.2f  maxNew=%d%n",
                "custom(0.7/40/0.95)",
                custom.getTemperature(), custom.getTopK(), custom.getTopP(),
                custom.isDoSample(), custom.getRepetitionPenalty(), custom.getMaxNewTokens());
        System.out.println();

        System.out.println("  Boolean helper queries:");
        System.out.println("    greedy.isGreedy():   " + greedy.isGreedy());
        System.out.println("    creative.isGreedy(): " + creative.isGreedy());
        System.out.println("    custom.hasTopK():    " + custom.hasTopK());
        System.out.println("    custom.hasTopP():    " + custom.hasTopP());
        System.out.println("    custom.hasRepetitionPenalty(): " + custom.hasRepetitionPenalty());
        System.out.println();

        // ============================================================
        // 3. SPECULATIVE DECODE LOOP — construct with NgramSpeculator
        // ============================================================
        System.out.println("=== 3. SpeculativeDecodeLoop — construct with NgramSpeculator ===");
        System.out.println();

        // SpeculativeDecodeLoop wraps an NgramSpeculator and drives the
        // draft -> verify -> emit loop.  It also tracks statistics about how
        // many draft tokens were accepted vs. attempted across all steps.
        //
        // The step() method requires a live SameDiff target model and KV-cache
        // arrays.  We show construction and stat inspection here; actual decode
        // steps need a loaded model (see QwenTextGenerationExample for end-to-end).

        SpeculativeDecodeLoop decodeLoop = new SpeculativeDecodeLoop(ngramSpeculator);

        System.out.println("  SpeculativeDecodeLoop(ngramSpeculator) constructed.");
        System.out.println("  Initial statistics (no steps run yet):");
        System.out.println("    getTotalAccepted():    " + decodeLoop.getTotalAccepted());
        System.out.println("    getTotalAttempted():   " + decodeLoop.getTotalAttempted());
        System.out.println("    getSpeculativeSteps(): " + decodeLoop.getSpeculativeSteps());
        System.out.println("    getNormalSteps():      " + decodeLoop.getNormalSteps());
        System.out.println("    isDisabled():          " + decodeLoop.isDisabled());
        System.out.println("    getStats():            " + decodeLoop.getStats());
        System.out.println("    getTimingStats():      " + decodeLoop.getTimingStats());
        System.out.println();

        // The no-arg constructor starts with a null speculator; speculative steps
        // are disabled until probeMultiTokenSupport() succeeds on a real model.
        SpeculativeDecodeLoop defaultLoop = new SpeculativeDecodeLoop();
        System.out.println("  Default SpeculativeDecodeLoop() (no speculator):");
        System.out.println("    isDisabled(): " + defaultLoop.isDisabled()
                + "  (dynamically disabled if acceptance rate collapses)");
        System.out.println();

        System.out.println("  Runtime usage pattern (requires loaded SameDiff target model):");
        System.out.println("    // 1. Probe multi-token decode support:");
        System.out.println("    boolean ok = decodeLoop.probeMultiTokenSupport(targetSd,");
        System.out.println("        inputNames, logitsName, kvCacheNames, ngramSize, seqLen);");
        System.out.println("    // 2. Run one speculative step:");
        System.out.println("    SpeculativeDecodeLoop.SpeculativeStepResult r =");
        System.out.println("        decodeLoop.step(tokenIds, position, targetSd, logitsName,");
        System.out.println("            inputNames, embedSd, embedInputNames, embedOutput,");
        System.out.println("            kvCacheNames, kvMap, hiddenSize, maxLen, seqLen, stopIds);");
        System.out.println("    // 3. Inspect the step result:");
        System.out.println("    int[] confirmedTokens = r.getAcceptedTokens();");
        System.out.println("    int   positionsAdvanced = r.getNewPositions();");
        System.out.println("    boolean hitEos = r.hitEos();");
        System.out.println("    // 4. After generation, read cumulative stats:");
        System.out.println("    long accepted  = decodeLoop.getTotalAccepted();");
        System.out.println("    long attempted = decodeLoop.getTotalAttempted();");
        System.out.println("    double alpha = (double) accepted / Math.max(1, attempted);");
        System.out.println();

        // ============================================================
        // 4. GENERATIONPIPELINECONFIG WITH SPECULATIVE SETTINGS
        // ============================================================
        System.out.println("=== 4. GenerationPipelineConfig with speculative settings ===");
        System.out.println();

        // Option A: n-gram speculative decoding using maxSpeculativeTokens.
        // The pipeline uses NgramSpeculator internally; feed the token stream
        // to the SpeculativeDecodeLoop (section 3) for context-aware drafts.
        // Here we configure the pipeline-level knob: how many tokens to draft per step.
        GenerationPipelineConfig ngramPipelineCfg = GenerationPipelineConfig.builder()
                .maxSpeculativeTokens(5)            // draft up to 5 tokens per verify step
                .samplingConfig(greedy)             // greedy -> highest acceptance rate
                .maxNewTokens(200)
                .decoderPath("/models/llama3-8b-decoder.sdz")
                .embedTokensPath("/models/llama3-8b-embed.sdz")
                .build();

        System.out.println("  Option A — n-gram speculation via maxSpeculativeTokens:");
        System.out.println("    maxSpeculativeTokens:  " + ngramPipelineCfg.getMaxSpeculativeTokens());
        System.out.println("    samplingConfig.isGreedy(): "
                + ngramPipelineCfg.getSamplingConfig().isGreedy());
        System.out.println("    maxNewTokens:          " + ngramPipelineCfg.getMaxNewTokens());
        System.out.println("    decoderPath:           " + ngramPipelineCfg.getDecoderPath());
        System.out.println("    (wire a SpeculativeDecodeLoop with NgramSpeculator into the");
        System.out.println("     pipeline for context-aware n-gram drafts — see section 3)");
        System.out.println();

        // Option B: load the small draft model from a file path.
        // The pipeline instantiates a DraftModelSpeculator automatically.
        GenerationPipelineConfig draftModelCfg = GenerationPipelineConfig.builder()
                .draftModelPath("/models/llama3-0.5b-decoder.sdz")   // small draft model
                .maxSpeculativeTokens(5)                              // draft 5 tokens per step
                .samplingConfig(greedy)
                .maxNewTokens(200)
                .decoderPath("/models/llama3-8b-decoder.sdz")
                .embedTokensPath("/models/llama3-8b-embed.sdz")
                .build();

        System.out.println("  Option B — DraftModelSpeculator via .draftModelPath():");
        System.out.println("    draftModelPath:        " + draftModelCfg.getDraftModelPath());
        System.out.println("    maxSpeculativeTokens:  " + draftModelCfg.getMaxSpeculativeTokens());
        System.out.println("    samplingConfig.isGreedy(): "
                + draftModelCfg.getSamplingConfig().isGreedy());
        System.out.println("    maxNewTokens:          " + draftModelCfg.getMaxNewTokens());
        System.out.println();

        // Option C: precise sampling with draft model — lower acceptance rate than
        // greedy but correct stochastic output.
        GenerationPipelineConfig preciseCfg = GenerationPipelineConfig.builder()
                .draftModelPath("/models/qwen2-0.5b.sdz")
                .maxSpeculativeTokens(5)
                .samplingConfig(precise)
                .maxNewTokens(512)
                .decoderPath("/models/qwen2-7b.sdz")
                .build();
        System.out.println("  Option C — precise sampling (temp=" + precise.getTemperature()
                + ") with draft model:");
        System.out.println("    draftModelPath:        " + preciseCfg.getDraftModelPath());
        System.out.println("    samplingConfig.temp:   " + preciseCfg.getSamplingConfig().getTemperature());
        System.out.println("    maxNewTokens:          " + preciseCfg.getMaxNewTokens());
        System.out.println();

        System.out.println("  To run generation (requires real model files):");
        System.out.println("    GenerationPipeline pipeline = GenerationPipeline.create(ngramPipelineCfg);");
        System.out.println("    GenerationResult result = pipeline.generate(\"Explain quantum entanglement:\");");
        System.out.println("    pipeline.close();");
        System.out.println();

        // ============================================================
        // 5. ACCEPTANCE RATE AND SPEEDUP — actual computed values
        // ============================================================
        System.out.println("=== 5. Acceptance rate and expected speedup ===");
        System.out.println();

        System.out.println("  Formula:  speedup = (1 + K * alpha) / (1 + overhead_factor)");
        System.out.println("    alpha          = fraction of draft tokens accepted (0..1)");
        System.out.println("    K              = draft tokens per step (maxSpeculativeTokens)");
        System.out.println("    overhead_factor ~ 0.1 (cost of drafting relative to verifying)");
        System.out.println();

        double overheadFactor = 0.1;
        int K = 5;
        double[] acceptRates = {0.5, 0.6, 0.7, 0.8, 0.9};

        System.out.printf("  %-10s  %-5s  %s%n", "alpha", "K", "speedup");
        System.out.println("  " + "-".repeat(30));
        for (double alpha : acceptRates) {
            double speedup = (1.0 + K * alpha) / (1.0 + overheadFactor);
            System.out.printf("  alpha=%.1f    K=%d  ->  speedup=%.2fx%n", alpha, K, speedup);
        }
        System.out.println();

        // Vary K at a fixed realistic acceptance rate of 0.75
        double fixedAlpha = 0.75;
        System.out.printf("  Fixed alpha=%.2f, varying K:%n", fixedAlpha);
        System.out.println("  " + "-".repeat(30));
        for (int draftK : new int[]{1, 2, 3, 5, 8, 10}) {
            double speedup = (1.0 + draftK * fixedAlpha) / (1.0 + overheadFactor);
            System.out.printf("  K=%-2d  ->  speedup=%.2fx%n", draftK, speedup);
        }
        System.out.println();

        System.out.println("  Practical benchmarks (8B target, 0.5B draft, K=5):");
        System.out.println("    Code generation (high repetition):    alpha~0.85 -> "
                + String.format("%.2f", (1.0 + 5 * 0.85) / (1.0 + overheadFactor)) + "x");
        System.out.println("    Instruction-following (medium):       alpha~0.70 -> "
                + String.format("%.2f", (1.0 + 5 * 0.70) / (1.0 + overheadFactor)) + "x");
        System.out.println("    Open-ended creative (diverse):        alpha~0.55 -> "
                + String.format("%.2f", (1.0 + 5 * 0.55) / (1.0 + overheadFactor)) + "x");
        System.out.println("    N-gram only (structured text):        alpha~0.45 -> "
                + String.format("%.2f", (1.0 + 5 * 0.45) / (1.0 + overheadFactor)) + "x");
        System.out.println();

        // ============================================================
        // 6. SAMPLING AND OUTPUT EQUIVALENCE
        // ============================================================
        System.out.println("=== 6. Sampling modes and output equivalence ===");
        System.out.println();

        System.out.println("  Greedy decoding (temperature=0, doSample=false):");
        System.out.printf("    greedy.getTemperature() = %.1f%n",  greedy.getTemperature());
        System.out.printf("    greedy.isDoSample()     = %b%n",    greedy.isDoSample());
        System.out.printf("    greedy.isGreedy()       = %b%n",    greedy.isGreedy());
        System.out.println("    -> draft token accepted iff it matches target argmax.");
        System.out.println("    -> output bit-for-bit identical to standard greedy decoding.");
        System.out.println();

        System.out.println("  Stochastic decoding (temperature > 0, doSample=true):");
        System.out.printf("    precise.getTemperature()  = %.2f%n", precise.getTemperature());
        System.out.printf("    creative.getTemperature() = %.2f%n", creative.getTemperature());
        System.out.println("    -> draft token i accepted with p = min(1, p_target(xi) / p_draft(xi)).");
        System.out.println("    -> rejected tokens replaced by a corrected sample from target distribution.");
        System.out.println("    -> output distribution provably identical to standard sampling.");
        System.out.println("    -> acceptance rates lower than greedy, but correctness guaranteed.");
        System.out.println();

        System.out.println("SpeculativeDecodingExample completed.");
        System.out.println("See QwenTextGenerationExample for a runnable end-to-end generation example.");
        System.out.println("Set draftModelPath / maxSpeculativeTokens in GenerationPipelineConfig");
        System.out.println("to enable speculative decoding in your own pipelines.");
    }
}
