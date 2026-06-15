/* *****************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.examples.samediff.advanced.generation;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * LLM Text Generation Pipeline Example.
 *
 * This example documents the SameDiff LLM generation pipeline (samediff-llm module),
 * which provides high-level APIs for autoregressive text generation with
 * decoder-only Transformer models loaded into SameDiff.
 *
 * <h3>Key Components:</h3>
 *
 * <b>GenerationPipeline</b> — Main generation engine.
 * <pre>
 * GenerationPipeline pipeline = GenerationPipeline.create(config);
 *
 * // Simple text generation
 * GenerationResult result = pipeline.generate("What is deep learning?");
 * GenerationResult result = pipeline.generate("prompt", maxNewTokens);
 *
 * // Streaming (token-by-token callback)
 * pipeline.generateStream("prompt", token -> System.out.print(token));
 * pipeline.generateStream("prompt", maxNewTokens, tokenCallback, stopCallback);
 *
 * // Vision-language (with image embeddings)
 * GenerationResult result = pipeline.generate(prefillEmbeddings, promptTokenIds);
 *
 * // Token embedding
 * INDArray embeddings = pipeline.embedTokens(tokenIds);
 *
 * pipeline.close(); // implements AutoCloseable
 * </pre>
 *
 * <b>GenerationPipelineConfig</b> — Configuration builder.
 * <pre>
 * GenerationPipelineConfig config = GenerationPipelineConfig.builder()
 *     .modelPath("/path/to/model.sdz")
 *     .tokenizerPath("/path/to/tokenizer.json")
 *     .maxSequenceLength(4096)
 *     .batchSize(1)
 *     .kvCacheStrategy(KVCacheStrategy.PAGED)
 *     .dspCompilationMode(DspCompilationMode.REDUCE_OVERHEAD)
 *     .tensorParallelConfig(TensorParallelConfig.create(2, 0))
 *     .build();
 * </pre>
 *
 * <b>SamplingConfig</b> — Token sampling strategy.
 * <pre>
 * // Greedy decoding (deterministic)
 * SamplingConfig greedy = SamplingConfig.greedy();
 *
 * // Top-k sampling
 * SamplingConfig topK = SamplingConfig.topK(50);
 *
 * // Top-p (nucleus) sampling
 * SamplingConfig topP = SamplingConfig.topP(0.9);
 *
 * // Temperature-scaled sampling with top-k and top-p
 * SamplingConfig custom = SamplingConfig.builder()
 *     .temperature(0.7)
 *     .topK(40)
 *     .topP(0.95)
 *     .repetitionPenalty(1.1)
 *     .build();
 * </pre>
 *
 * <b>KV Cache Strategies</b>:
 * <ul>
 *   <li>STATIC — Fixed-size cache, fastest but wastes memory for short sequences</li>
 *   <li>PAGED — PagedAttention (vLLM-style), efficient for variable lengths</li>
 *   <li>QUANTIZED — Quantized KV cache for reduced memory</li>
 *   <li>PREFIX — Prefix caching for shared prompt prefixes</li>
 * </ul>
 *
 * <h3>Speculative Decoding:</h3>
 * <pre>
 * // Use a small draft model to propose tokens, verified by the large model.
 * // Reduces latency by generating multiple tokens per forward pass.
 * GenerationPipelineConfig config = GenerationPipelineConfig.builder()
 *     .modelPath("large-model.sdz")
 *     .speculativeModelPath("draft-model.sdz")
 *     .speculativeNumTokens(5)
 *     .build();
 * </pre>
 *
 * <h3>Tensor Parallelism:</h3>
 * <pre>
 * // Split model across multiple GPUs for models that don't fit in one GPU.
 * TensorParallelConfig tp = TensorParallelConfig.create(numGpus, rank);
 * tp = tp.withDeviceIds(0, 1, 2, 3);
 * </pre>
 *
 * <h3>Vision-Language Models (VLM):</h3>
 * <pre>
 * // VisionLanguageModel supports image + text generation.
 * VisionLanguageModel vlm = VisionLanguageModel.fromDirectory(modelDir);
 * VisionLanguageModel vlm = VisionLanguageModel.loadSmolDocling(modelDir);
 *
 * // Generate text from image
 * String text = vlm.generate(image, "Describe this image");
 * String text = vlm.generate(image, "prompt", maxNewTokens);
 *
 * // Batch / multi-page generation
 * GenerationResult[] results = vlm.generateBatch(images, "prompt", maxNewTokens);
 * String doc = vlm.generateDocument(pageImages, "OCR this document", maxNewTokens);
 *
 * // Tiled image processing (for high-res images)
 * GenerationResult[] tiled = vlm.generatePagesTiled(pageSplitResults, ...);
 * </pre>
 *
 * <h3>Multi-Model Pipeline:</h3>
 * <pre>
 * // Chain multiple models in sequence (summarizer → classifier → etc.)
 * MultiModelPipeline pipeline = new MultiModelPipeline(config);
 * pipeline.registerModel("classifier", ModelType.CLASSIFIER, classifierModel);
 * PipelineResult result = pipeline.execute(stages, inputText);
 * </pre>
 *
 * <h3>Token Sampling Operations (sd.nn()):</h3>
 * At the SameDiff op level, autoregressive sampling is available via:
 * <pre>
 * // Single-op token sampling
 * SDVariable nextToken = sd.nn.tokenSample(logits);
 * SDVariable nextToken = sd.nn.tokenSample(logits, temperature, topK, topP);
 * </pre>
 *
 * NOTE: This example requires a model to be loaded. Use the OmniHub examples
 * to download models from HuggingFace first.
 */
public class LLMGenerationPipelineExample {
    private static final Logger log = LoggerFactory.getLogger(LLMGenerationPipelineExample.class);

    public static void main(String[] args) {

        log.info("=== LLM Generation Pipeline API Reference ===");
        log.info("");

        // =====================================================================
        // 1. GenerationPipeline usage
        // =====================================================================
        log.info("--- 1. GenerationPipeline ---");
        log.info("  GenerationPipeline pipeline = GenerationPipeline.create(config);");
        log.info("  GenerationResult result = pipeline.generate(\"prompt\");");
        log.info("  pipeline.generateStream(\"prompt\", token -> System.out.print(token));");
        log.info("");

        // =====================================================================
        // 2. SamplingConfig presets
        // =====================================================================
        log.info("--- 2. SamplingConfig presets ---");
        log.info("  SamplingConfig.greedy()          → deterministic, argmax");
        log.info("  SamplingConfig.topK(50)          → top-k sampling");
        log.info("  SamplingConfig.topP(0.9)         → nucleus sampling");
        log.info("  SamplingConfig.builder()");
        log.info("    .temperature(0.7)");
        log.info("    .topK(40)");
        log.info("    .topP(0.95)");
        log.info("    .repetitionPenalty(1.1)");
        log.info("    .build()");
        log.info("");

        // =====================================================================
        // 3. KV Cache strategies
        // =====================================================================
        log.info("--- 3. KV Cache strategies ---");
        log.info("  STATIC     — fixed-size, fastest for fixed-length");
        log.info("  PAGED      — PagedAttention (vLLM-style), memory efficient");
        log.info("  QUANTIZED  — reduced precision KV cache");
        log.info("  PREFIX     — share prefix across requests");
        log.info("");

        // =====================================================================
        // 4. Speculative decoding
        // =====================================================================
        log.info("--- 4. Speculative decoding ---");
        log.info("  Uses a small draft model to propose multiple tokens,");
        log.info("  verified in a single forward pass of the large model.");
        log.info("  Can reduce latency 2-3x with well-matched draft models.");
        log.info("");

        // =====================================================================
        // 5. Tensor & Pipeline parallelism
        // =====================================================================
        log.info("--- 5. Parallelism ---");
        log.info("  TensorParallelConfig.create(numGpus, rank)");
        log.info("    Splits layers across GPUs (column/row parallel linear)");
        log.info("  PipelineParallelRunner");
        log.info("    Splits model stages across devices");
        log.info("  DistributedDataParallelTrainer");
        log.info("    DDP training across multiple GPUs/nodes");
        log.info("");

        // =====================================================================
        // 6. VLM (Vision-Language) pipeline
        // =====================================================================
        log.info("--- 6. VisionLanguageModel ---");
        log.info("  VisionLanguageModel.fromDirectory(dir)");
        log.info("  vlm.generate(image, \"prompt\")");
        log.info("  vlm.generateBatch(images, \"prompt\", maxTokens)");
        log.info("  vlm.generateDocument(pageImages, \"prompt\", maxTokens)");
        log.info("");

        log.info("**************** LLM Generation Pipeline Example finished ********************");
    }
}
