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

package org.nd4j.examples.samediff.quickstart.modeling;

import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.DownloadResult;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.LLMModel;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.QuantType;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;

import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 * Qwen LLM Text Generation — Complete Example
 *
 * Demonstrates the full pipeline for running a Qwen 3.5 large language model:
 *
 *   1. Download a quantized GGUF model from HuggingFace
 *   2. Import the GGUF into a SameDiff computation graph
 *   3. Load a HuggingFace tokenizer
 *   4. Build a GenerationPipeline for text generation
 *   5. Generate text with different sampling strategies
 *   6. Use chat templates for instruction-following
 *
 * Key classes:
 *   - {@link LLMModelDownloader} — Downloads and caches GGUF models from HuggingFace
 *   - {@link GGMLModelImport} — Imports GGUF files into SameDiff graphs
 *   - {@link HuggingFaceTokenizer} — Native Rust-backed tokenizer (BPE, SentencePiece, WordPiece)
 *   - {@link GenerationPipeline} — Autoregressive text generation with KV cache
 *   - {@link SamplingConfig} — Controls temperature, top-k, top-p, repetition penalty
 *   - {@link GenerationResult} — Output text, token IDs, throughput metrics, finish reason
 *
 * Model requirements:
 *   - The smallest Qwen3.5-0.8B Q4_K_M model is ~600MB
 *   - Models are cached in ~/.cache/dl4j-llm-models/ by default
 *   - Override cache directory with -Dllm.model.cache.dir=/path
 *
 * Run with:
 *   cd samediff-examples
 *   mvn exec:java -Dexec.mainClass="org.nd4j.examples.samediff.quickstart.modeling.QwenTextGenerationExample"
 *
 * For GPU acceleration, use nd4j-cuda backend:
 *   mvn exec:java -Dexec.mainClass="..." -Dnd4j.backend=nd4j-cuda-12.9-platform
 */
public class QwenTextGenerationExample {

    public static void main(String[] args) throws Exception {

        // ============================================================
        // 1. DOWNLOAD GGUF MODEL
        // ============================================================
        System.out.println("=== 1. Downloading Qwen3.5-0.8B GGUF Model ===");

        // LLMModelDownloader handles downloading and caching GGUF files from HuggingFace.
        // Available model sizes: 0.8B, 2B, 4B, 9B, 27B (dense), 35B-A3B, 122B-A10B, 397B-A17B (MoE)
        // Available quantizations: Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, F16
        DownloadResult downloadResult = LLMModelDownloader.download(LLMModel.QWEN35_0_8B, QuantType.Q4_K_M);
        File ggufFile = downloadResult.getModelFile();
        System.out.println("  Model file: " + ggufFile.getAbsolutePath());
        System.out.println("  File size: " + (ggufFile.length() / (1024 * 1024)) + " MB");
        System.out.println("  Downloaded now: " + downloadResult.isDownloadedNow());

        // ============================================================
        // 2. INSPECT GGUF METADATA (without full import)
        // ============================================================
        System.out.println("\n=== 2. Inspecting GGUF Metadata ===");

        // Quick metadata inspection without loading the full model
        boolean isGGUF = GGMLModelImport.isGGMLFile(ggufFile);
        System.out.println("  Is GGUF file: " + isGGUF);

        // ============================================================
        // 3. IMPORT GGUF INTO SAMEDIFF
        // ============================================================
        System.out.println("\n=== 3. Importing GGUF → SameDiff ===");

        // ConversionOptions control how quantized GGUF tensors are handled:
        //   forInference()        — Dequantize to FP32 (best accuracy, most memory)
        //   fp16()                — Dequantize to FP16 (half memory, good accuracy)
        //   preserveQuantization() — Keep quantized (smallest, requires quantized ops)
        ConversionOptions options = ConversionOptions.forInference();
        System.out.println("  Conversion mode: " + options.getQuantizationMode());
        System.out.println("  Target dtype: " + options.getTargetDataType());

        long importStart = System.currentTimeMillis();
        SameDiff model = GGMLModelImport.importModel(ggufFile, options);
        long importMs = System.currentTimeMillis() - importStart;

        System.out.println("  Import time: " + importMs + "ms");
        System.out.println("  Graph ops: " + model.ops().length);
        System.out.println("  Variables: " + model.variables().size());

        // ============================================================
        // 4. LOAD TOKENIZER
        // ============================================================
        System.out.println("\n=== 4. Loading HuggingFace Tokenizer ===");

        // The tokenizer can be loaded from:
        //   - A tokenizer.json file path
        //   - A directory containing tokenizer.json (+ optional tokenizer_config.json)
        //   - A raw JSON string
        // The GGUF file often embeds tokenizer info; LLMModelDownloader extracts it.
        File tokenizerFile = downloadResult.getModelFile().getParentFile();
        Tokenizer tokenizer = HuggingFaceTokenizer.fromDirectory(tokenizerFile);

        System.out.println("  Vocab size: " + tokenizer.getVocabSize());
        System.out.println("  BOS token: '" + tokenizer.getBosToken() + "' (id=" + tokenizer.getBosTokenId() + ")");
        System.out.println("  EOS token: '" + tokenizer.getEosToken() + "' (id=" + tokenizer.getEosTokenId() + ")");
        System.out.println("  Has chat template: " + (tokenizer.getChatTemplate() != null));

        // Demonstrate encoding/decoding
        String testText = "Hello, world!";
        Encoding encoding = tokenizer.encode(testText);
        System.out.println("  Encode '" + testText + "': " + Arrays.toString(encoding.getIds()));
        String decoded = tokenizer.decode(encoding.getIds());
        System.out.println("  Decode back: '" + decoded + "'");

        // ============================================================
        // 5. GREEDY TEXT GENERATION
        // ============================================================
        System.out.println("\n=== 5. Greedy Text Generation ===");

        // Build the generation pipeline.
        // For single-model GGUF imports, the decoder contains the embedding table
        // internally — no separate embedTokens model needed.
        GenerationPipelineConfig config = GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())  // Deterministic, always pick highest-probability token
                .maxNewTokens(50)                          // Maximum tokens to generate
                .graphOptimizerEnabled(true)               // Apply fusion/simplification passes
                .build();

        GenerationPipeline pipeline = GenerationPipeline.create(config);

        // Simple text completion
        String prompt = "The capital of France is";
        System.out.println("  Prompt: \"" + prompt + "\"");

        GenerationResult result = pipeline.generate(prompt);

        System.out.println("  Generated: \"" + result.getText() + "\"");
        System.out.println("  Tokens generated: " + result.getGeneratedTokenCount());
        System.out.println("  Prompt tokens: " + result.getPromptTokenCount());
        System.out.println("  Total tokens: " + result.getTotalTokenCount());
        System.out.println("  Generation time: " + result.getGenerationTimeMs() + "ms");
        System.out.println("  Throughput: " + String.format("%.1f", result.getTokensPerSecond()) + " tok/s");
        System.out.println("  Finish reason: " + result.getFinishReason());

        pipeline.close();

        // ============================================================
        // 6. SAMPLING STRATEGIES
        // ============================================================
        System.out.println("\n=== 6. Sampling Strategies ===");

        // Temperature sampling: higher temperature = more creative/random
        System.out.println("  Available presets:");
        System.out.println("    greedy()        — deterministic, temp=0, always best token");
        System.out.println("    precise()       — temp=0.3, topP=0.85, low randomness");
        System.out.println("    defaultConfig() — temp=0.7, topP=0.9, balanced");
        System.out.println("    creative()      — temp=0.9, topK=50, topP=0.95, high variety");

        // Custom sampling config
        SamplingConfig customSampling = SamplingConfig.builder()
                .temperature(0.8)           // Softmax temperature (higher = more random)
                .topK(40)                   // Only consider top 40 tokens
                .topP(0.9)                  // Nucleus sampling: top tokens covering 90% probability
                .repetitionPenalty(1.1)     // Penalize repeated tokens (>1.0 = discourage repeats)
                .doSample(true)             // Enable stochastic sampling
                .seed(42L)                  // Reproducible sampling
                .build();

        System.out.println("  Custom config: temp=" + customSampling.getTemperature()
                + ", topK=" + customSampling.getTopK()
                + ", topP=" + customSampling.getTopP()
                + ", repPenalty=" + customSampling.getRepetitionPenalty());

        GenerationPipelineConfig samplingPipelineConfig = GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(customSampling)
                .maxNewTokens(100)
                .build();

        GenerationPipeline samplingPipeline = GenerationPipeline.create(samplingPipelineConfig);

        result = samplingPipeline.generate("Once upon a time, in a distant galaxy,");
        System.out.println("  Creative output: \"" + result.getText().substring(0,
                Math.min(200, result.getText().length())) + "...\"");
        System.out.println("  Tokens: " + result.getGeneratedTokenCount()
                + ", Speed: " + String.format("%.1f", result.getTokensPerSecond()) + " tok/s");

        samplingPipeline.close();

        // ============================================================
        // 7. CHAT / INSTRUCTION FOLLOWING
        // ============================================================
        System.out.println("\n=== 7. Chat with Instruction Template ===");

        // Qwen uses the ChatML template format:
        //   <|im_start|>system\n{system message}<|im_end|>\n
        //   <|im_start|>user\n{user message}<|im_end|>\n
        //   <|im_start|>assistant\n
        //
        // The tokenizer's applyChatTemplate() handles this formatting.
        List<ChatTemplate.Message> messages = Arrays.asList(
                ChatTemplate.Message.system("You are a helpful AI assistant. Answer concisely."),
                ChatTemplate.Message.user("What are the three laws of thermodynamics? List them briefly.")
        );

        // Apply the chat template to format the conversation
        String chatPrompt = tokenizer.applyChatTemplate(messages, true);  // true = add generation prompt
        System.out.println("  Formatted prompt (first 200 chars):");
        System.out.println("    " + chatPrompt.substring(0, Math.min(200, chatPrompt.length())) + "...");

        GenerationPipelineConfig chatConfig = GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.precise())  // Low temperature for factual answers
                .maxNewTokens(200)
                .build();

        GenerationPipeline chatPipeline = GenerationPipeline.create(chatConfig);
        result = chatPipeline.generate(chatPrompt);

        System.out.println("  Assistant response:");
        System.out.println("    " + result.getText());
        System.out.println("  Finish reason: " + result.getFinishReason());
        System.out.println("  Latency to first token: " + result.getFirstTokenLatencyMs() + "ms");

        // Multi-turn conversation: append the assistant's response and continue
        System.out.println("\n  --- Multi-turn follow-up ---");
        List<ChatTemplate.Message> followUp = Arrays.asList(
                ChatTemplate.Message.system("You are a helpful AI assistant. Answer concisely."),
                ChatTemplate.Message.user("What are the three laws of thermodynamics? List them briefly."),
                ChatTemplate.Message.assistant(result.getText()),
                ChatTemplate.Message.user("Which one is most relevant to entropy?")
        );

        String followUpPrompt = tokenizer.applyChatTemplate(followUp, true);
        result = chatPipeline.generate(followUpPrompt);
        System.out.println("  Follow-up response:");
        System.out.println("    " + result.getText());

        chatPipeline.close();

        // ============================================================
        // 8. GENERATION RESULT METRICS
        // ============================================================
        System.out.println("\n=== 8. GenerationResult Metrics ===");
        System.out.println("  Key metrics available on GenerationResult:");
        System.out.println("    getText()                     — generated text");
        System.out.println("    getTokenIds()                 — int[] of generated token IDs");
        System.out.println("    getGeneratedTokenCount()      — number of new tokens");
        System.out.println("    getPromptTokenCount()         — prompt token count");
        System.out.println("    getFirstTokenLatencyMs()      — time to first token (prefill)");
        System.out.println("    getGenerationTimeMs()         — total wall-clock time");
        System.out.println("    getTokensPerSecond()          — overall throughput");
        System.out.println("    getDecodeTokensPerSecond()    — decode-only throughput");
        System.out.println("    getFinishReason()             — EOS, MAX_TOKENS, STOP_SEQUENCE, ERROR");
        System.out.println("    isComplete()                  — true if EOS or STOP_SEQUENCE");
        System.out.println("    isTruncated()                 — true if MAX_TOKENS hit");

        // ============================================================
        // 9. CONVERSION OPTIONS REFERENCE
        // ============================================================
        System.out.println("\n=== 9. ConversionOptions Reference ===");
        System.out.println("  ConversionOptions control GGUF → SameDiff import:");
        System.out.println("    forInference()        — FP32 dequant, best accuracy, most memory");
        System.out.println("    fp16()                — FP16 dequant, half memory, good accuracy");
        System.out.println("    forTraining()         — FP32 + gradient support");
        System.out.println("    preserveQuantization() — keep quantized format");
        System.out.println("  Custom builder:");
        System.out.println("    ConversionOptions.builder()");
        System.out.println("      .quantizationMode(QuantizationMode.DEQUANTIZE_TO_FLOAT16)");
        System.out.println("      .targetDataType(DataType.FLOAT16)");
        System.out.println("      .architectureOverride(\"llama\")  // skip auto-detection");
        System.out.println("      .useMemoryMapping(true)           // mmap for large files");
        System.out.println("      .build()");

        // Cleanup
        tokenizer.close();

        System.out.println("\nQwen text generation example completed successfully.");
    }
}
