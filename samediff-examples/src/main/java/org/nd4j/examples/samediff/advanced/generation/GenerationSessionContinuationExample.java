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

import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.DownloadResult;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.LLMModel;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.QuantType;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline.GenerationSession;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;

import java.io.File;

/**
 * Resumable decoding with {@link GenerationSession}: generate a few tokens, keep the
 * KV cache alive, and continue the SAME generation later without re-processing the
 * prompt (no re-prefill).
 *
 * Why this matters:
 *
 *   - Chat/agent servers stream a response in chunks; each chunk continues from the
 *     retained KV state instead of paying prefill again.
 *   - Token budgets: generate 64 tokens, inspect them, decide whether to spend more.
 *   - Cooperative cancellation: stop a long generation at a step boundary.
 *
 * The key invariant (greedy decoding): splitting a generation into chunks produces
 * EXACTLY the same tokens as one uninterrupted call —
 *
 *     generate(N) + continueGeneration(M)  ==  generate(N + M)
 *
 * This example verifies that invariant against a one-shot reference.
 *
 * Session API tour ({@link GenerationPipeline#startSession(String, int)}):
 *   session.generate(n)             — decode the first n tokens (prefills the prompt)
 *   session.continueGeneration(n)   — decode n more from the retained KV state
 *   session.continueToCompletion(s) — loop in s-token steps until EOS/capacity
 *   session.getRemainingCapacity()  — new-token headroom left in the KV buffer
 *   session.getCachePosition()      — absolute position of the next-fed token
 *   session.isEosReached()          — true once a real stop token was produced
 *   session.getFullText()           — clean cumulative text across all calls
 *   session.close()                 — releases retained native buffers (try-with-resources)
 *
 * Notes: one session may be open per pipeline at a time; the session must be used from
 * the thread that created it; continuation is greedy-deterministic — with sampling
 * enabled the chunked and one-shot outputs may legitimately diverge.
 *
 * The Qwen3.5-0.8B GGUF (~508MB Q4_K_M) is downloaded once and cached under
 * ~/.cache/dl4j-llm-models. CPU decode of ~50 tokens takes a few minutes.
 */
public class GenerationSessionContinuationExample {

    public static void main(String[] args) throws Exception {

        // ============================================================
        // 1. LOAD MODEL + TOKENIZER (cached after first run)
        // ============================================================
        System.out.println("=== 1. Loading Qwen3.5-0.8B (Q4_K_M) ===");
        DownloadResult download = LLMModelDownloader.download(LLMModel.QWEN35_0_8B, QuantType.Q4_K_M);
        File ggufFile = download.getModelFile();
        SameDiff model = GGMLModelImport.importModel(ggufFile, ConversionOptions.forInference());
        Tokenizer tokenizer = HuggingFaceTokenizer.fromDirectory(ggufFile.getParentFile());
        System.out.println("  Model: " + ggufFile.getName() + ", ops: " + model.ops().length);

        GenerationPipeline pipeline = GenerationPipeline.create(
                GenerationPipelineConfig.builder()
                        .decoder(model)
                        .tokenizer(tokenizer)
                        .samplingConfig(SamplingConfig.greedy())   // parity check requires greedy
                        .maxNewTokens(24)
                        .build());

        String prompt = "The three primary colors are";
        int totalTokens = 24;

        try {
            // ============================================================
            // 2. ONE-SHOT REFERENCE
            // ============================================================
            System.out.println("\n=== 2. One-shot reference (" + totalTokens + " tokens) ===");
            GenerationResult oneShot = pipeline.generate(prompt, totalTokens);
            System.out.println("  \"" + oneShot.getText() + "\"");

            // ============================================================
            // 3. THE SAME GENERATION IN THREE CHUNKS
            // ============================================================
            System.out.println("\n=== 3. Chunked: 8 + 8 + 8 tokens via GenerationSession ===");
            String chunkedText;
            try (GenerationSession session = pipeline.startSession(prompt, totalTokens)) {
                GenerationResult first = session.generate(8);          // prefill + 8 tokens
                System.out.println("  chunk 1: +" + first.getGeneratedTokenCount()
                        + " tokens, cachePosition=" + session.getCachePosition()
                        + ", remaining=" + session.getRemainingCapacity());

                GenerationResult second = session.continueGeneration(8); // NO re-prefill
                System.out.println("  chunk 2: +" + second.getGeneratedTokenCount()
                        + " tokens, cachePosition=" + session.getCachePosition()
                        + ", remaining=" + session.getRemainingCapacity());

                GenerationResult third = session.continueGeneration(8);
                System.out.println("  chunk 3: +" + third.getGeneratedTokenCount()
                        + " tokens, cachePosition=" + session.getCachePosition()
                        + ", remaining=" + session.getRemainingCapacity());

                chunkedText = session.getFullText();
                System.out.println("  full session text: \"" + chunkedText + "\"");
                System.out.println("  EOS reached: " + session.isEosReached());
            }

            // ============================================================
            // 4. PARITY: chunked == one-shot (greedy invariant)
            // ============================================================
            System.out.println("\n=== 4. Parity check ===");
            boolean match = oneShot.getText().startsWith(chunkedText)
                    || chunkedText.startsWith(oneShot.getText())
                    || oneShot.getText().equals(chunkedText);
            System.out.println("  one-shot == chunked: " + match);
            if (!match) {
                System.out.println("  one-shot: \"" + oneShot.getText() + "\"");
                System.out.println("  chunked:  \"" + chunkedText + "\"");
                throw new IllegalStateException(
                        "FAILED: greedy continuation must reproduce the one-shot generation");
            }

            // ============================================================
            // 5. RUN-TO-COMPLETION IN STEPS
            // ============================================================
            System.out.println("\n=== 5. continueToCompletion in 8-token steps ===");
            try (GenerationSession session = pipeline.startSession("Water boils at", 16)) {
                // generate() runs the initial decode (prefill + first tokens); the
                // continueToCompletion() loop then drives fixed-size steps until EOS or
                // capacity — the pattern a streaming/chat server uses, with
                // session.cancel() available from another thread for cooperative stops.
                session.generate(8);
                GenerationResult r = session.continueToCompletion(8);
                System.out.println("  \"" + session.getFullText() + "\"");
                System.out.println("  finished with " + session.getRemainingCapacity()
                        + " tokens of capacity left, EOS=" + session.isEosReached()
                        + ", truncated=" + r.isTruncated());
            }

            System.out.println("\nSession continuation verified: chunked greedy decode matches one-shot.");
        } finally {
            pipeline.close();
        }
    }
}
