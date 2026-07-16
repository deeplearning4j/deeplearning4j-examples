/*
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.nd4j.examples.samediff.advanced.generation;

import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader.VLMModel;
import org.eclipse.deeplearning4j.vlm.model.loading.OnnxModelCache;
import org.nd4j.autodiff.samediff.SameDiff;

import java.io.File;

/**
 * REAL draft-model speculative decoding: a 135M-parameter draft proposes tokens, the
 * 1.7B-parameter target verifies them in one batched forward pass, and every accepted
 * token skips a full target decode step.
 *
 * Where {@link SpeculativeDecodingExample} demonstrates the concepts and configs on toy
 * data, this example runs the production pipeline end to end with the pairing the
 * platform benchmark itself uses ({@code -Dvlm.speculative.draft=true}):
 *
 *   Target: SmolDocling's decoder — SmolLM2-1.7B (ONNX, cached as .sdz)
 *   Draft:  SmolLM2-135M-Instruct decoder (ONNX)
 *
 * Both are SmolLM2-family models sharing one tokenizer/vocabulary — the requirement for
 * draft speculation (the draft proposes token IDs in the target's vocabulary).
 *
 * What it measures, from real {@link GenerationResult} metrics:
 *   1. Baseline: standard one-token-per-step decode (maxSpeculativeTokens = 0)
 *   2. Speculative: draft proposes K tokens per step ({@code maxSpeculativeTokens})
 *      — reports totalSpeculativeTokens / totalAcceptedTokens /
 *      averageAcceptanceRate / speculativeSteps / effectiveTokensPerSecond
 *   3. Output equivalence: greedy speculative decoding is LOSSLESS — accepted tokens
 *      are exactly what the target would have produced, so both outputs must match.
 *
 * STATUS: GenerationPipeline does not yet wire draft speculation into its decode
 * loops — it warns at create() and runs standard decode, so the speculation metrics
 * print as zeros (section 4 calls this out at runtime). This example is the intended
 * measurement harness for that feature: the config, the shared-vocabulary model
 * pairing, and the lossless-equivalence oracle are exactly how it will be validated
 * once wired.
 *
 * Speculation economics (why acceptance rate is the whole game): each speculative step
 * costs one draft forward per proposed token plus ONE target forward that verifies all
 * K proposals. At acceptance rate a, the target executes ~1/(1 + a*K) of its baseline
 * forwards. On CPU both models compete for the same cores, so the wall-clock win is
 * smaller than on GPU where the draft is nearly free — the acceptance metrics printed
 * here are hardware-independent, the tok/s comparison is not.
 *
 * Downloads on first run (cached under ~/.cache/dl4j-vlm-models): SmolDocling decoder
 * (~516MB) + tokenizer, SmolLM2-135M decoder (~515MB).
 *
 * System properties: -Dexample.gen.tokens=24  -Dexample.spec.k=4
 *                    -Dexample.prompt="..."
 */
public class DraftModelSpeculativeDecodingExample {

    public static void main(String[] args) throws Exception {
        int genTokens = Integer.getInteger("example.gen.tokens", 24);
        int specK = Integer.getInteger("example.spec.k", 4);
        String prompt = System.getProperty("example.prompt",
                "The three most important ideas in computer science are");

        // ============================================================
        // 1. LOAD TARGET + DRAFT (both SmolLM2 family, shared vocab)
        // ============================================================
        System.out.println("=== 1. Loading target (SmolLM2-1.7B) and draft (SmolLM2-135M) ===");
        File decoderFile = VLMModelDownloader.download(VLMModel.SMOLDOCLING_DECODER).getModelFile();
        File embedFile = VLMModelDownloader.download(VLMModel.SMOLDOCLING_EMBED_TOKENS).getModelFile();
        File tokenizerFile = VLMModelDownloader.download(VLMModel.SMOLDOCLING_TOKENIZER).getModelFile();
        File draftFile = VLMModelDownloader.download(VLMModel.SMOLLM2_135M_DECODER).getModelFile();

        long t0 = System.currentTimeMillis();
        SameDiff target = OnnxModelCache.importWithCache(decoderFile.getAbsolutePath());
        SameDiff embedTokens = OnnxModelCache.importWithCache(embedFile.getAbsolutePath());
        SameDiff draft = OnnxModelCache.importWithCache(draftFile.getAbsolutePath());
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile);
        System.out.println("  Loaded in " + (System.currentTimeMillis() - t0) + "ms"
                + " — target " + target.ops().length + " ops, draft " + draft.ops().length
                + " ops, vocab " + tokenizer.getVocabSize());

        // ============================================================
        // 2. BASELINE: standard decode, one target forward per token
        // ============================================================
        System.out.println("\n=== 2. Baseline decode (no speculation) ===");
        GenerationResult baseline;
        try (GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(target)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())   // greedy => speculation is lossless
                .maxNewTokens(genTokens)
                .build())) {
            baseline = pipeline.generate(prompt, genTokens);
        }
        System.out.println("  \"" + prompt + baseline.getText() + "\"");
        System.out.println(String.format(
                "  %d tokens | %.2f tok/s | first token %dms",
                baseline.getGeneratedTokenCount(), baseline.getTokensPerSecond(),
                baseline.getFirstTokenLatencyMs()));

        // ============================================================
        // 3. SPECULATIVE: draft proposes K, target verifies in one pass
        // ============================================================
        System.out.println("\n=== 3. Speculative decode (draft proposes K=" + specK + ") ===");
        GenerationResult speculative;
        try (GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(target)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(genTokens)
                .draftDecoder(draft)                        // wires a DraftModelSpeculator
                .maxSpeculativeTokens(specK)                // K proposals per verify step
                .build())) {
            speculative = pipeline.generate(prompt, genTokens);
        }
        System.out.println("  \"" + prompt + speculative.getText() + "\"");

        // ============================================================
        // 4. THE NUMBERS THAT MATTER
        // ============================================================
        System.out.println("\n=== 4. Speculation metrics ===");
        if (speculative.getSpeculativeSteps() == 0) {
            // Zero steps means the pipeline ran STANDARD decode: as of this writing,
            // speculative decoding is not yet wired into GenerationPipeline's decode
            // loops (the pipeline logs a warning at create()). This example is the
            // measurement harness for when that wiring lands — the config, model
            // pairing and lossless-equivalence oracle below are the correct usage.
            System.out.println("  !! Speculation DID NOT ENGAGE — the pipeline ran standard decode.");
            System.out.println("  !! GenerationPipeline does not yet wire draft speculation into its");
            System.out.println("  !! decode loops (see the create()-time warning). Metrics below are");
            System.out.println("  !! therefore zeros, and the tok/s difference is run-to-run noise.");
        }
        System.out.println("  Speculative steps executed:   " + speculative.getSpeculativeSteps());
        System.out.println("  Draft tokens proposed:        " + speculative.getTotalSpeculativeTokens());
        System.out.println("  Draft tokens accepted:        " + speculative.getTotalAcceptedTokens());
        System.out.println(String.format(
                "  Acceptance rate:              %.1f%%",
                speculative.getAverageAcceptanceRate() * 100));
        System.out.println(String.format(
                "  Baseline:     %.2f tok/s", baseline.getTokensPerSecond()));
        System.out.println(String.format(
                "  Speculative:  %.2f tok/s (effective %.2f tok/s)",
                speculative.getTokensPerSecond(), speculative.getEffectiveTokensPerSecond()));
        if (baseline.getTokensPerSecond() > 0) {
            System.out.println(String.format(
                    "  Wall-clock speedup:           %.2fx  (CPU note: draft and target share"
                            + " cores here; GPU drafts are nearly free)",
                    speculative.getTokensPerSecond() / baseline.getTokensPerSecond()));
        }

        // Greedy speculation is lossless: verified-accepted tokens are exactly the
        // target's greedy choices. Divergence here indicates a verification bug.
        boolean identical = baseline.getText().equals(speculative.getText());
        System.out.println("  Output identical to baseline: " + identical
                + (identical ? " (lossless verification holds)" : "  <-- BUG: report this"));

        tokenizer.close();
        System.out.println("\nDraft-model speculative decoding example completed.");
    }
}
