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

import org.eclipse.deeplearning4j.llm.speculative.DraftModelSpeculator;
import org.eclipse.deeplearning4j.llm.speculative.NgramSpeculator;
import org.eclipse.deeplearning4j.llm.speculative.SpeculativeDecodeLoop;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;
import org.eclipse.deeplearning4j.pipeline.AutoModel;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Speculative Decoding for Faster LLM Inference — API Reference
 *
 * Speculative decoding reduces autoregressive generation latency by using a
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
 * Key classes:
 *   - {@link NgramSpeculator}        — heuristic draft via n-gram matching
 *   - {@link DraftModelSpeculator}   — neural draft model speculator
 *   - {@link SpeculativeDecodeLoop}  — core loop: draft → verify → emit
 *   - {@link GenerationPipeline}     — high-level pipeline (integrates speculator)
 *   - {@link GenerationPipelineConfig} — config (includes speculativeModelPath)
 *   - {@link SamplingConfig}         — temperature / top-k / top-p settings
 *
 * Performance notes:
 *   - Acceptance rate depends on how well the draft matches the target model.
 *   - Greedy sampling (temperature=0) gives the highest acceptance rates.
 *   - With temperature > 0, speculative decoding applies a corrected distribution
 *     so output statistics are identical to standard sampling.
 *   - The speedup is roughly proportional to mean accepted draft length.
 *
 * Run with:
 *   cd samediff-examples
 *   mvn exec:java \
 *     -Dexec.mainClass="org.nd4j.examples.samediff.advanced.generation.SpeculativeDecodingExample"
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
        // 1. LOAD TARGET MODEL
        // ============================================================
        System.out.println("=== 1. Load target model ===");
        System.out.println();
        System.out.println("  The target model is the large, high-quality model that determines");
        System.out.println("  what tokens are actually accepted or rejected.");
        System.out.println();

        // AutoModel.fromPretrained() accepts GGUF, SafeTensors, ONNX, or SDZ.
        // Replace the path with a real model file to execute this example.
        //
        //   SameDiff targetSd = AutoModel.fromPretrained("/models/llama3-8b.gguf");
        //
        // For the remainder of this educational example, we use descriptive
        // placeholder names rather than executing real model loads.

        System.out.println("  // Load the large target model (replace path as needed):");
        System.out.println("  SameDiff targetSd = AutoModel.fromPretrained(\"/models/llama3-8b.gguf\");");
        System.out.println();

        // ============================================================
        // 2. NGRAM SPECULATOR (NO DRAFT MODEL NEEDED)
        // ============================================================
        System.out.println("=== 2. NgramSpeculator — heuristic draft, no extra model ===");
        System.out.println();

        // NgramSpeculator maintains a dynamic n-gram table of the tokens seen
        // so far in the context window.  When proposing draft tokens it looks up
        // the last (ngramSize-1) tokens as a key and suggests the most frequent
        // continuation.  Simple, fast, and surprisingly effective for repetitive
        // or formulaic text (code, structured output, legal boilerplate, etc.).
        //
        // Parameters:
        //   ngramSize      — size of the n-gram window (3 = trigrams)
        //   maxDraftTokens — how many draft tokens to propose per step
        //   minFrequency   — minimum n-gram frequency required to use it as a draft

        NgramSpeculator ngramSpeculator = NgramSpeculator.builder()
                .ngramSize(3)           // use trigrams to predict the next token
                .maxDraftTokens(5)      // propose up to 5 draft tokens per step
                .minFrequency(1)        // include n-grams seen at least once
                .build();

        System.out.println("  NgramSpeculator created:");
        System.out.println("    ngramSize:      " + ngramSpeculator.getNgramSize()
                + " (trigrams — look back 2 tokens to predict next)");
        System.out.println("    maxDraftTokens: " + ngramSpeculator.getMaxDraftTokens()
                + " (propose up to 5 candidates per verification step)");
        System.out.println();
        System.out.println("  When to use NgramSpeculator:");
        System.out.println("    - No draft model available");
        System.out.println("    - Generating repetitive or structured text (code, tables)");
        System.out.println("    - Minimal memory overhead required");
        System.out.println("    - Quick prototyping / benchmarking");
        System.out.println();

        // ============================================================
        // 3. DRAFT MODEL SPECULATOR (NEURAL DRAFT)
        // ============================================================
        System.out.println("=== 3. DraftModelSpeculator — small neural draft model ===");
        System.out.println();

        // DraftModelSpeculator uses a smaller language model (the "draft model")
        // to autoregressively generate candidate tokens.  The draft model must:
        //   - Come from the same family as the target model (shared vocabulary)
        //   - Be substantially smaller (e.g. 0.5B draft for an 8B target)
        //
        // Parameters:
        //   draftModel      — SameDiff graph of the small draft model
        //   draftTokenCount — how many tokens the draft model should propose
        //   temperature     — sampling temperature for draft proposals
        //                     (0.0 = greedy, matches target at temperature=0)

        System.out.println("  // First, load the small draft model:");
        System.out.println("  SameDiff draftSd = AutoModel.fromPretrained(\"/models/llama3-0.5b.gguf\");");
        System.out.println();
        System.out.println("  // Build the DraftModelSpeculator:");
        System.out.println("  DraftModelSpeculator speculator = DraftModelSpeculator.builder()");
        System.out.println("      .draftModel(draftSd)      // the small model");
        System.out.println("      .draftTokenCount(5)       // tokens to draft per step");
        System.out.println("      .temperature(0.0)         // greedy draft (highest acceptance)");
        System.out.println("      .build();");
        System.out.println();

        // Conceptual object creation shown above (replace null with real draftSd at runtime):
        //
        // DraftModelSpeculator draftSpeculator = DraftModelSpeculator.builder()
        //         .draftModel(draftSd)
        //         .draftTokenCount(5)
        //         .temperature(0.0)
        //         .build();

        System.out.println("  When to use DraftModelSpeculator:");
        System.out.println("    - High acceptance rate is important (general text, diverse prompts)");
        System.out.println("    - Draft model from the same family is available");
        System.out.println("    - Willing to use extra VRAM / RAM for the draft model");
        System.out.println("    - Production serving where throughput is critical");
        System.out.println();

        // ============================================================
        // 4. SPECULATIVE DECODE LOOP — MANUAL CONTROL
        // ============================================================
        System.out.println("=== 4. SpeculativeDecodeLoop — fine-grained control ===");
        System.out.println();

        // SpeculativeDecodeLoop is the core engine.  It accepts any Speculator
        // implementation and any SamplingConfig and runs the draft→verify→emit
        // loop until maxLength tokens are generated or EOS is hit.
        //
        // Use this class directly when you need:
        //   - Access to acceptance statistics per step
        //   - Custom stopping conditions beyond max-length and EOS
        //   - Integration with a custom KV-cache or attention backend

        SamplingConfig greedyConfig = SamplingConfig.greedy();

        System.out.println("  // Build the decode loop with an NgramSpeculator:");
        System.out.println("  SpeculativeDecodeLoop loop = SpeculativeDecodeLoop.builder()");
        System.out.println("      .targetModel(targetSd)          // large target model");
        System.out.println("      .speculator(ngramSpeculator)     // or draftSpeculator");
        System.out.println("      .samplingConfig(SamplingConfig.greedy())");
        System.out.println("      .build();");
        System.out.println();
        System.out.println("  // Run generation:");
        System.out.println("  int[] inputTokenIds = {1, 15043, 29892, 3186, 29991};  // encoded prompt");
        System.out.println("  GenerationResult result = loop.generate(inputTokenIds, 200 /*maxLength*/);");
        System.out.println();
        System.out.println("  System.out.println(\"Generated tokens: \" + result.getGeneratedTokenCount());");
        System.out.println("  System.out.println(\"Throughput: \" + result.getTokensPerSecond() + \" tok/s\");");
        System.out.println("  System.out.println(\"Finish reason: \" + result.getFinishReason());");
        System.out.println();

        // ============================================================
        // 5. SPECULATIVE DECODING VIA GENERATIONPIPELINECONFIG
        // ============================================================
        System.out.println("=== 5. Speculative decoding via GenerationPipelineConfig ===");
        System.out.println();

        // The easiest way to enable speculative decoding in a production pipeline
        // is to set speculativeModelPath (or speculativeNumTokens for n-gram) in
        // GenerationPipelineConfig.  The pipeline wires everything together automatically.

        System.out.println("  // Option A — n-gram speculative decoding (no draft model file):");
        System.out.println("  GenerationPipelineConfig ngramConfig = GenerationPipelineConfig.builder()");
        System.out.println("      .decoder(targetSd)");
        System.out.println("      .tokenizer(tokenizer)");
        System.out.println("      .samplingConfig(SamplingConfig.greedy())");
        System.out.println("      .speculativeNumTokens(5)    // draft 5 tokens via n-gram");
        System.out.println("      .maxNewTokens(200)");
        System.out.println("      .build();");
        System.out.println();
        System.out.println("  // Option B — draft model speculative decoding:");
        System.out.println("  GenerationPipelineConfig draftConfig = GenerationPipelineConfig.builder()");
        System.out.println("      .decoder(targetSd)");
        System.out.println("      .tokenizer(tokenizer)");
        System.out.println("      .samplingConfig(SamplingConfig.greedy())");
        System.out.println("      .speculativeModelPath(\"/models/llama3-0.5b.gguf\")");
        System.out.println("      .speculativeNumTokens(5)    // draft model proposes 5 tokens");
        System.out.println("      .maxNewTokens(200)");
        System.out.println("      .build();");
        System.out.println();
        System.out.println("  GenerationPipeline pipeline = GenerationPipeline.create(draftConfig);");
        System.out.println("  GenerationResult result = pipeline.generate(\"Explain quantum entanglement:\");");
        System.out.println("  pipeline.close();");
        System.out.println();

        // ============================================================
        // 6. ACCEPTANCE RATE AND SPEEDUP
        // ============================================================
        System.out.println("=== 6. Acceptance rate and expected speedup ===");
        System.out.println();

        System.out.println("  The speedup from speculative decoding depends on the acceptance rate α:");
        System.out.println("    - α = 1.0 means every draft token is accepted (ideal case)");
        System.out.println("    - α = 0.0 means no draft tokens are accepted (fall-back to standard)");
        System.out.println("    - Typical values with a good draft model: 0.6–0.85");
        System.out.println();
        System.out.println("  Expected speedup formula (K = draft tokens per step):");
        System.out.println("    speedup ≈ (1 + K·α) / (1 + overhead_factor)");
        System.out.println();
        System.out.println("  Practical benchmarks (8B target, 0.5B draft, K=5):");
        System.out.println("    - Code generation (high repetition):   3–4× speedup");
        System.out.println("    - Instruction-following (medium):       2–3× speedup");
        System.out.println("    - Open-ended creative text (diverse):   1.5–2× speedup");
        System.out.println("    - N-gram speculator (no draft model):   1.3–1.8× speedup on structured text");
        System.out.println();
        System.out.println("  Accessing acceptance statistics from SpeculativeDecodeLoop:");
        System.out.println("    SpeculativeDecodeLoop.Stats stats = loop.getStats();");
        System.out.println("    double acceptRate = stats.getMeanAcceptanceRate();");
        System.out.println("    double meanAccepted = stats.getMeanAcceptedTokensPerStep();");
        System.out.println("    int totalSteps = stats.getVerificationStepCount();");
        System.out.println();

        // ============================================================
        // 7. SAMPLING AND OUTPUT EQUIVALENCE
        // ============================================================
        System.out.println("=== 7. Sampling and output equivalence ===");
        System.out.println();

        System.out.println("  Speculative decoding with temperature=0 (greedy) is straightforward:");
        System.out.println("    - Draft token accepted iff it matches target's argmax token.");
        System.out.println("    - Output is bit-for-bit identical to standard greedy decoding.");
        System.out.println();
        System.out.println("  Speculative decoding with temperature>0 uses rejection sampling:");
        System.out.println("    - Draft token i accepted with probability min(1, p_target(xi)/p_draft(xi)).");
        System.out.println("    - Rejected tokens replaced by a corrected sample from target distribution.");
        System.out.println("    - Output distribution is provably identical to standard sampling.");
        System.out.println("    - Acceptance rates are lower than with greedy, but output is correct.");
        System.out.println();
        System.out.println("  Recommended sampling configs for speculative decoding:");
        System.out.println("    SamplingConfig.greedy()                       — highest acceptance, deterministic");
        System.out.println("    SamplingConfig.precise()                      — temp=0.3, good speed balance");
        System.out.println("    SamplingConfig.builder().temperature(0.7)");
        System.out.println("        .topK(40).build()                         — creative, acceptance ~0.6–0.7");
        System.out.println();

        System.out.println("SpeculativeDecodingExample completed.");
        System.out.println("See QwenTextGenerationExample for a runnable end-to-end generation example.");
        System.out.println("Set speculativeModelPath / speculativeNumTokens in GenerationPipelineConfig");
        System.out.println("to enable speculative decoding in your own pipelines.");
    }
}
