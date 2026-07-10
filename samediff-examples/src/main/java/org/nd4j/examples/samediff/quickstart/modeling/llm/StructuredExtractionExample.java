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
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.DownloadResult;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.LLMModel;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.QuantType;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.node.ArrayNode;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Structured Information Extraction — LiquidAI LFM2-350M-Extract
 *
 * <p>This example demonstrates the "small specialized model" pattern for NLP: a 350M
 * parameter model purpose-built for structured extraction (JSON/XML/YAML from free text)
 * beats prompting a large general-purpose model in both cost and latency for this task.
 *
 * <p>The same {@link GenerationPipeline} API that drives Qwen, Gemma, and other models
 * works without modification for LFM2-350M-Extract.  The LFM2 family is a hybrid
 * conv/attention architecture (Liquid Foundation Models — gated short convolutions
 * interleaved with attention layers); the GGUF import layer handles the architecture
 * transparently.
 *
 * <h3>Workflow</h3>
 * <ol>
 *   <li>Download the LFM2-350M-Extract F16 GGUF from HuggingFace (cached after first run).</li>
 *   <li>Import the GGUF into a SameDiff computation graph.</li>
 *   <li>Load the matching HuggingFace tokenizer.</li>
 *   <li>For each input text: wrap in the ChatML extraction prompt, decode greedily,
 *       parse the JSON output, and print an entities/relationships table.</li>
 * </ol>
 *
 * <h3>Prompt format</h3>
 * LFM2-Extract uses a ChatML-style template with a terse JSON-extraction system prompt.
 * The format was taken from the benchmark runner in
 * {@code platform-tests/.../TestLLMBenchmarkSuite.java#formatExtractionPrompt}:
 * <pre>
 *   &lt;|im_start|&gt;system
 *   Return data as a JSON object. Extract all key entities, attributes, and values from the input text.
 *   &lt;|im_end|&gt;
 *   &lt;|im_start|&gt;user
 *   {input text}
 *   &lt;|im_end|&gt;
 *   &lt;|im_start|&gt;assistant
 * </pre>
 *
 * <h3>JSON handling</h3>
 * The model emits compact JSON, occasionally wrapped in markdown code fences ({@code ```json}).
 * This example strips fences before parsing and falls back to printing the raw output
 * if parsing fails — failure is surfaced explicitly, never hidden.
 * Jackson is available transitively through the {@code samediff-llm} dependency
 * (shaded as {@code org.nd4j.shade.jackson}).
 *
 * <h3>Model details</h3>
 * <ul>
 *   <li>Repo: {@code LiquidAI/LFM2-350M-Extract-GGUF}</li>
 *   <li>File: {@code LFM2-350M-Extract-F16.gguf} (~700 MB)</li>
 *   <li>Tokenizer: {@code LiquidAI/LFM2-350M/resolve/main/tokenizer.json}</li>
 *   <li>Quantization: F16 (the registry default for this model)</li>
 * </ul>
 *
 * <h3>Running</h3>
 * <pre>
 *   cd samediff-examples
 *   mvn exec:java \
 *     -Dexec.mainClass="org.nd4j.examples.samediff.quickstart.modeling.llm.StructuredExtractionExample"
 * </pre>
 * First run downloads the GGUF and tokenizer (~720 MB total) into
 * {@code ~/.cache/dl4j-llm-models/}. Override the cache directory with
 * {@code -Dllm.model.cache.dir=/path}. CPU runtime: ~60–90 s total
 * (import ~20 s, ~20–30 s per extraction on a modern CPU).
 */
public class StructuredExtractionExample {

    // ── Two representative input texts ────────────────────────────────────────────

    private static final String TEXT_ACQUISITION =
            "Anthropic, the AI safety company founded in 2021 by Dario Amodei and " +
            "Daniela Amodei, raised $7.3 billion in Series E funding led by Google " +
            "and Spark Capital in 2024. The company is headquartered in San Francisco " +
            "and employs approximately 800 people. Its flagship product, Claude, " +
            "competes directly with OpenAI's ChatGPT.";

    private static final String TEXT_HISTORY =
            "The Treaty of Versailles was signed on 28 June 1919 in the Palace of " +
            "Versailles, France, ending World War I. The principal signatories were " +
            "Georges Clemenceau of France, David Lloyd George of the United Kingdom, " +
            "Woodrow Wilson of the United States, and Vittorio Orlando of Italy. " +
            "Germany lost approximately 13% of its territory and 10% of its population " +
            "under the treaty's terms.";

    // ── ChatML extraction prompt (from TestLLMBenchmarkSuite#formatExtractionPrompt) ──

    // LFM2 REQUIRES the leading <|startoftext|> BOS token (model card: the chat template
    // begins with {{- bos_token -}}; llama.cpp adds it from GGUF add_bos_token metadata).
    // Without it the 350M model degenerates into repetition loops. The tokenizer maps the
    // literal special-token text to the BOS id even when encoding pre-formatted prompts.
    private static String buildExtractionPrompt(String inputText) {
        return "<|startoftext|><|im_start|>system\n" +
               "Return data as a JSON object. Extract all key entities, attributes, " +
               "and values from the input text.\n" +
               "<|im_end|>\n" +
               "<|im_start|>user\n" +
               inputText + "\n" +
               "<|im_end|>\n" +
               "<|im_start|>assistant\n";
    }

    // ── Entry point ───────────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {

        // ============================================================
        // 1. DOWNLOAD LFM2-350M-EXTRACT GGUF
        // ============================================================
        System.out.println("=== 1. Downloading LFM2-350M-Extract GGUF ===");
        System.out.println("  Model: LiquidAI/LFM2-350M-Extract-GGUF");
        System.out.println("  File:  LFM2-350M-Extract-F16.gguf (~700 MB)");
        System.out.println("  Quant: F16 (registry default for this model)");
        System.out.println("  Cache: ~/.cache/dl4j-llm-models/ (or -Dllm.model.cache.dir)");

        // LFM2_350M_EXTRACT uses F16 quantization (no Q4/Q8 variant in the public repo).
        DownloadResult downloadResult = LLMModelDownloader.download(
                LLMModel.LFM2_350M_EXTRACT, QuantType.F16);
        File ggufFile = downloadResult.getModelFile();

        System.out.println("  File:   " + ggufFile.getAbsolutePath());
        System.out.println("  Size:   " + (ggufFile.length() / (1024 * 1024)) + " MB");
        System.out.println("  Cached: " + !downloadResult.isDownloadedNow()
                + (downloadResult.isDownloadedNow() ? " (downloaded)" : " (already present)"));

        // ============================================================
        // 2. IMPORT GGUF → SAMEDIFF
        // ============================================================
        System.out.println("\n=== 2. Importing GGUF → SameDiff ===");
        System.out.println("  LFM2 is a hybrid conv/attention architecture (gated short");
        System.out.println("  convolutions + attention layers). GGUF import handles it");
        System.out.println("  transparently — no architecture-specific code needed here.");

        // forInference() dequantizes weights to FP32 — the accuracy-first choice and the
        // configuration the platform benchmark suite validates this model with. fp16()
        // halves memory but measurably degrades this 350M model's JSON output on CPU
        // (repeated keys, garbled entity values) — not worth it at ~1.4 GB fp32.
        ConversionOptions options = ConversionOptions.forInference();

        long importStart = System.currentTimeMillis();
        SameDiff model = GGMLModelImport.importModel(ggufFile, options);
        long importMs = System.currentTimeMillis() - importStart;

        System.out.println("  Import time: " + importMs + " ms");
        System.out.println("  Graph ops:   " + model.ops().length);
        System.out.println("  Variables:   " + model.variables().size());

        // ============================================================
        // 3. LOAD TOKENIZER
        // ============================================================
        System.out.println("\n=== 3. Loading Tokenizer ===");
        System.out.println("  Source: LiquidAI/LFM2-350M/resolve/main/tokenizer.json");

        // The Extract GGUF repo publishes no tokenizer.json; the base LFM2-350M repo does.
        // Download it under a model-specific cache name. Never scan the shared cache
        // directory for a tokenizer — another model's tokenizer.json may already sit
        // there, and mismatched token ids overflow the embedding table (gather errors).
        File tokenizerFile = LLMModelDownloader.downloadCustom(
                "https://huggingface.co/LiquidAI/LFM2-350M/resolve/main/tokenizer.json",
                "lfm2-350m-tokenizer.json");
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile);

        System.out.println("  Vocab size: " + tokenizer.getVocabSize());
        System.out.println("  BOS token:  '" + tokenizer.getBosToken()
                + "' (id=" + tokenizer.getBosTokenId() + ")");
        System.out.println("  EOS token:  '" + tokenizer.getEosToken()
                + "' (id=" + tokenizer.getEosTokenId() + ")");

        // ============================================================
        // 4. BUILD GENERATION PIPELINE
        // ============================================================
        System.out.println("\n=== 4. Building GenerationPipeline ===");
        System.out.println("  Sampling:  greedy (deterministic, best for structured output)");
        System.out.println("  MaxTokens: 384 (rich nested extractions can exceed 250 tokens; EOS ends earlier ones)");

        // Greedy decoding is the correct choice for structured extraction: the model
        // must emit well-formed JSON, and temperature > 0 introduces tokens that break
        // structure without improving content.
        GenerationPipelineConfig pipelineConfig = GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(384)
                .graphOptimizerEnabled(true)
                .build();

        GenerationPipeline pipeline = GenerationPipeline.create(pipelineConfig);

        // ============================================================
        // 5. EXTRACT INFORMATION FROM BOTH TEXTS
        // ============================================================
        System.out.println("\n=== 5. Structured Extraction ===");

        ObjectMapper mapper = new ObjectMapper();

        runExtraction(pipeline, mapper, "Company acquisition / funding round", TEXT_ACQUISITION);
        runExtraction(pipeline, mapper, "Historical treaty", TEXT_HISTORY);

        // ============================================================
        // 6. CLEANUP
        // ============================================================
        pipeline.close();
        tokenizer.close();

        System.out.println("\nStructured extraction example completed.");
        System.out.println("  Why a small specialist beats a big general model:");
        System.out.println("  - LFM2-350M-Extract was fine-tuned on extraction tasks: it emits");
        System.out.println("    well-formed JSON reliably without elaborate prompt engineering.");
        System.out.println("  - At 350M parameters it runs on CPU in tens of seconds and fits");
        System.out.println("    comfortably in 2 GB of RAM — orders of magnitude cheaper than");
        System.out.println("    a 7B+ general model for the same workload.");
        System.out.println("  - The GenerationPipeline API is identical for both: swap the model");
        System.out.println("    and you get a different capability at a different cost point.");
    }

    // ── Per-text extraction helper ────────────────────────────────────────────────

    /**
     * Runs one extraction, prints the raw model output, attempts JSON parsing,
     * and renders entities/relationships tables if parsing succeeds.
     *
     * <p>Parsing failures are surfaced explicitly — raw output is always printed
     * so the caller can diagnose model drift, truncation, or prompt issues.
     */
    private static void runExtraction(GenerationPipeline pipeline,
                                      ObjectMapper mapper,
                                      String label,
                                      String inputText) {
        System.out.println("\n─── " + label + " ───────────────────────────────────");
        System.out.println("Input:");
        System.out.println("  " + inputText);

        String prompt = buildExtractionPrompt(inputText);

        long t0 = System.currentTimeMillis();
        GenerationResult result = pipeline.generate(prompt);
        long elapsed = System.currentTimeMillis() - t0;

        String rawOutput = result.getText();

        System.out.println("\nRaw model output (" + result.getGeneratedTokenCount()
                + " tokens, " + elapsed + " ms, "
                + String.format("%.1f", result.getTokensPerSecond()) + " tok/s):");
        System.out.println("  " + rawOutput.replace("\n", "\n  "));

        // Strip markdown fences the model sometimes adds (```json ... ```)
        String cleaned = stripMarkdownFences(rawOutput.trim());

        // Attempt to parse as JSON
        JsonNode root;
        try {
            root = mapper.readTree(cleaned);
        } catch (JsonProcessingException e) {
            System.out.println("\n[PARSE FAILED] Output is not valid JSON after fence stripping.");
            System.out.println("  Parse error: " + e.getOriginalMessage());
            System.out.println("  Cleaned text: " + cleaned);
            System.out.println("  Tip: if truncated, increase maxNewTokens; if malformed, check prompt.");
            return;
        }

        System.out.println("\nExtracted fields:");
        renderJsonTable(root, mapper, "  ");
    }

    // ── JSON output rendering ─────────────────────────────────────────────────────

    /**
     * Renders a JSON object as a simple key-value (or nested) table.
     * Handles the two most common shapes the model produces:
     * <ul>
     *   <li>Flat object: {@code {"company": "Anthropic", "founded": "2021", ...}}</li>
     *   <li>Structured object with "entities" and/or "relationships" arrays</li>
     * </ul>
     */
    private static void renderJsonTable(JsonNode root, ObjectMapper mapper, String indent) {
        if (!root.isObject()) {
            // Scalar or array at root — just print
            System.out.println(indent + root.toString());
            return;
        }

        // Check for well-known structured keys: entities / relationships
        boolean hasEntities = root.has("entities");
        boolean hasRelationships = root.has("relationships");

        if (hasEntities || hasRelationships) {
            if (hasEntities) {
                System.out.println(indent + "Entities:");
                renderArray(root.get("entities"), mapper, indent + "  ");
            }
            if (hasRelationships) {
                System.out.println(indent + "Relationships:");
                renderArray(root.get("relationships"), mapper, indent + "  ");
            }
            // Print any remaining top-level keys
            for (Iterator<Map.Entry<String, JsonNode>> it = root.fields(); it.hasNext(); ) {
                Map.Entry<String, JsonNode> entry = it.next();
                if (!entry.getKey().equals("entities") && !entry.getKey().equals("relationships")) {
                    System.out.printf("%s%-20s %s%n", indent, entry.getKey() + ":",
                            flattenNode(entry.getValue()));
                }
            }
        } else {
            // Flat object — print each key/value on its own line
            for (Iterator<Map.Entry<String, JsonNode>> it = root.fields(); it.hasNext(); ) {
                Map.Entry<String, JsonNode> entry = it.next();
                JsonNode val = entry.getValue();
                if (val.isObject() || val.isArray()) {
                    System.out.printf("%s%-20s%n", indent, entry.getKey() + ":");
                    if (val.isArray()) {
                        renderArray(val, mapper, indent + "  ");
                    } else {
                        renderJsonTable(val, mapper, indent + "  ");
                    }
                } else {
                    System.out.printf("%s%-20s %s%n", indent, entry.getKey() + ":",
                            val.asText());
                }
            }
        }
    }

    /** Renders a JSON array — each element on its own line as a compact string. */
    private static void renderArray(JsonNode array, ObjectMapper mapper, String indent) {
        if (array == null || array.isNull()) {
            System.out.println(indent + "(none)");
            return;
        }
        if (!array.isArray()) {
            System.out.println(indent + array.asText());
            return;
        }
        ArrayNode arr = (ArrayNode) array;
        if (arr.size() == 0) {
            System.out.println(indent + "(empty)");
            return;
        }
        for (JsonNode item : arr) {
            if (item.isObject()) {
                // Render each field of the object inline, e.g. "name: X  type: Y  value: Z"
                List<String> parts = new ArrayList<>();
                for (Iterator<Map.Entry<String, JsonNode>> it = item.fields(); it.hasNext(); ) {
                    Map.Entry<String, JsonNode> e = it.next();
                    parts.add(e.getKey() + "=" + flattenNode(e.getValue()));
                }
                System.out.println(indent + "• " + String.join("  ", parts));
            } else {
                System.out.println(indent + "• " + item.asText());
            }
        }
    }

    /** Returns a compact single-line representation of any JsonNode. */
    private static String flattenNode(JsonNode node) {
        if (node.isTextual()) return node.asText();
        if (node.isNumber())  return node.asText();
        if (node.isBoolean()) return node.asText();
        if (node.isNull())    return "null";
        return node.toString(); // array or nested object: compact JSON
    }

    // ── Markdown fence stripping ──────────────────────────────────────────────────

    /**
     * Removes optional markdown code fences the model may emit around JSON:
     * <pre>
     *   ```json
     *   { ... }
     *   ```
     * </pre>
     * Returns the trimmed input unchanged if no fence is detected.
     */
    private static String stripMarkdownFences(String text) {
        if (text.startsWith("```")) {
            int firstNewline = text.indexOf('\n');
            if (firstNewline < 0) return text;           // degenerate: fence only, no body
            String body = text.substring(firstNewline + 1);
            if (body.endsWith("```")) {
                body = body.substring(0, body.length() - 3);
            }
            return body.trim();
        }
        return text;
    }
}
