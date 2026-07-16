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
package org.nd4j.examples.samediff.quickstart.pipeline;

import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.LLMModel;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.QuantType;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLMetadata;

import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 * The complete ingestion-to-output arc: HuggingFace download layer → tokenizer
 * plumbing → generated text. Every other LLM example assumes this glue works; this
 * one takes it apart on a real model (Qwen3.5-0.8B) and shows the seams that bite:
 *
 *   1. DOWNLOAD LAYER — the model catalog ({@link LLMModel} × {@link QuantType}),
 *      the cache directory contract ({@code llm.model.cache.dir}), cache checks
 *      without network ({@link LLMModelDownloader#isCached}), arbitrary-repo pulls
 *      ({@link LLMModelDownloader#downloadCustom}), and the HF_TOKEN requirement
 *      for gated repos (Gemma).
 *
 *   2. TWO TOKENIZER SOURCES, ONE TRUTH — a GGUF carries its own embedded
 *      tokenizer metadata ({@link GGMLMetadata.TokenizerInfo}: vocab, special
 *      token IDs, chat template), while HuggingFace repos ship tokenizer.json
 *      (loaded by the Rust-backed {@link HuggingFaceTokenizer}). Deployments mix
 *      them freely, so this example CROSS-CHECKS the two on the same model:
 *      special-token IDs, vocab sizes, chat templates. Includes the padded-vocab
 *      landmine: GGUF embedding matrices are often padded past the tokenizer
 *      vocabulary (248320 rows vs 248070 real tokens on Qwen3.5) — argmax over
 *      raw logits can produce IDs the tokenizer cannot decode.
 *
 *   3. ENCODING ANATOMY — ids/tokens/attention-mask triplet, special-token
 *      handling on encode AND decode, batch APIs, chat-template application for
 *      multi-turn conversations, round-trip fidelity.
 *
 *   4. TEXT OUT, INCREMENTALLY — generation via the production pipeline, then the
 *      streaming-UI detokenization recipe: BPE tokens do NOT decode independently
 *      (bytes split mid-character, markers like Ġ leak), so per-token
 *      {@code decode(new int[]{id})} produces garbage at boundaries. The correct
 *      recipe is prefix-decoding: decode ids[0..i] and emit the string DELTA.
 *      (There is currently no push/callback streaming API on the pipeline — the
 *      prefix-delta recipe over the returned ids is the supported approach.)
 *
 * First run downloads the Q4_K_M GGUF (~508MB) + tokenizer.json (~12MB) into
 * ~/.cache/dl4j-llm-models. System properties:
 *   -Dexample.gen.tokens=12      generation length (0 skips section 4's decode)
 *   -Dexample.download.custom=true   also demo downloadCustom (network required)
 */
public class HuggingFaceToTextExample {

    public static void main(String[] args) throws Exception {
        int genTokens = Integer.getInteger("example.gen.tokens", 12);

        // ============================================================
        // 1. THE DOWNLOAD LAYER
        // ============================================================
        System.out.println("=== 1. HuggingFace download layer ===");
        System.out.println("  Cache dir: " + LLMModelDownloader.getCacheDir()
                + "  (override: -D" + LLMModelDownloader.CACHE_DIR_PROPERTY + ")");

        // The catalog is LLMModel x QuantType; isCached() answers without network.
        System.out.println("  Catalog/cache check (no network):");
        LLMModel[] sample = {LLMModel.QWEN35_0_8B, LLMModel.GEMMA3_1B, LLMModel.MISTRAL_7B};
        QuantType[] quants = {QuantType.Q4_K_M, QuantType.Q8_0};
        for (LLMModel m : sample) {
            StringBuilder line = new StringBuilder("    " + String.format("%-14s", m.name()));
            for (QuantType q : quants) {
                line.append(q.name()).append("=")
                        .append(LLMModelDownloader.isCached(m, q) ? "cached " : "absent ");
            }
            System.out.println(line);
        }
        System.out.println("  Gated repos (e.g. Gemma) need -Dhf.token=... or the HF_TOKEN env var.");

        // download() = fetch-or-serve-from-cache; DownloadResult carries the file.
        File gguf = LLMModelDownloader.download(LLMModel.QWEN35_0_8B, QuantType.Q4_K_M).getModelFile();
        System.out.println("  Model file: " + gguf.getName() + " (" + gguf.length() / (1024 * 1024) + " MB)");

        // Arbitrary-repo escape hatch: any URL -> named file in the same cache.
        if (Boolean.getBoolean("example.download.custom")) {
            File custom = LLMModelDownloader.downloadCustom(
                    "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/merges.txt",
                    "qwen25-merges-demo.txt");
            System.out.println("  downloadCustom(): " + custom.getName()
                    + " (" + custom.length() / 1024 + " KB)");
        } else {
            System.out.println("  downloadCustom(url, fileName) pulls from ANY repo"
                    + " (skipped — enable with -Dexample.download.custom=true)");
        }

        // ============================================================
        // 2. TWO TOKENIZER SOURCES, CROSS-CHECKED
        // ============================================================
        System.out.println("\n=== 2. tokenizer.json vs GGUF-embedded metadata ===");
        Tokenizer tokenizer = HuggingFaceTokenizer.fromDirectory(gguf.getParentFile());
        GGMLMetadata metadata = GGMLModelImport.inspectModel(gguf);
        GGMLMetadata.TokenizerInfo embedded = metadata.getTokenizerInfo();

        System.out.println("  tokenizer.json (Rust-backed, native=" +
                HuggingFaceTokenizer.isNativeAvailable() + "):");
        System.out.println("    vocab=" + tokenizer.getVocabSize()
                + "  bos=" + tokenizer.getBosTokenId() + "  eos=" + tokenizer.getEosTokenId());
        if (embedded != null) {
            int embeddedVocab = embedded.getTokens() != null ? embedded.getTokens().size() : -1;
            System.out.println("  GGUF-embedded TokenizerInfo (model=" + embedded.getModel() + "):");
            System.out.println("    vocab=" + embeddedVocab
                    + "  bos=" + embedded.getBosTokenId() + "  eos=" + embedded.getEosTokenId()
                    + "  chatTemplate=" + (embedded.getChatTemplate() != null
                            ? embedded.getChatTemplate().length() + " chars" : "absent"));

            // The cross-check deployments should run before mixing sources:
            boolean eosMatch = embedded.getEosTokenId() == tokenizer.getEosTokenId();
            System.out.println("  Cross-check: eos IDs " + (eosMatch ? "MATCH" : "DIFFER — do not mix these sources!"));
        } else {
            System.out.println("  GGUF-embedded TokenizerInfo: not present in this file");
        }

        // The padded-vocab landmine: the embedding matrix has MORE rows than the
        // tokenizer has tokens. Any consumer of raw logits (argmax, top-k UI,
        // distillation targets) must clamp/slice to the tokenizer vocab.
        long embeddingRows = metadata.getVocabSize() > 0 ? metadata.getVocabSize() : -1;
        System.out.println("  Padded-vocab check: tokenizer=" + tokenizer.getVocabSize()
                + " vs GGUF header vocab=" + embeddingRows
                + " (Qwen3.5 embedding is padded to 248320 rows; IDs >= "
                + tokenizer.getVocabSize() + " are undecodable padding)");

        // ============================================================
        // 3. ENCODING ANATOMY
        // ============================================================
        System.out.println("\n=== 3. Encoding anatomy ===");
        String text = "Tokenizers map text to integers.";
        Encoding plain = tokenizer.encode(text, false);
        Encoding withSpecial = tokenizer.encode(text, true);
        System.out.println("  \"" + text + "\"");
        System.out.println("    ids (no specials, " + plain.getIds().length + "): "
                + Arrays.toString(plain.getIds()));
        System.out.println("    tokens: " + Arrays.toString(plain.getTokens()));
        System.out.println("    attention mask: " + Arrays.toString(plain.getAttentionMask()));
        System.out.println("    with specials: " + withSpecial.getIds().length
                + " ids (adds BOS/EOS per tokenizer config)");

        // Round trip + special-token visibility on decode
        String roundTrip = tokenizer.decode(plain.getIds(), true);
        System.out.println("  Round-trip exact: " + roundTrip.equals(text)
                + "  (\"" + roundTrip + "\")");
        System.out.println("  decode(skipSpecialTokens=false) keeps markers like <|im_end|>"
                + " — use true for user-facing text");

        // Chat template turns a conversation into the model's expected string.
        List<ChatTemplate.Message> chat = Arrays.asList(
                ChatTemplate.Message.system("You are terse."),
                ChatTemplate.Message.user("Name one prime number."));
        String rendered = tokenizer.applyChatTemplate(chat, true);
        System.out.println("  applyChatTemplate (addGenerationPrompt=true), first 90 chars:");
        System.out.println("    " + rendered.substring(0, Math.min(90, rendered.length()))
                .replace("\n", "\\n"));

        // ============================================================
        // 4. TEXT OUT — AND HOW TO STREAM IT CORRECTLY
        // ============================================================
        if (genTokens > 0) {
            System.out.println("\n=== 4. Generation + incremental detokenization ===");
            SameDiff model = GGMLModelImport.importModel(gguf.getAbsolutePath(),
                    ConversionOptions.forInference());
            GenerationResult result;
            try (GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                    .decoder(model)
                    .tokenizer(tokenizer)
                    .samplingConfig(SamplingConfig.greedy())
                    .maxNewTokens(genTokens)
                    .build())) {
                result = pipeline.generate("The capital of France is", genTokens);
            }
            System.out.println("  Full text: \"" + result.getText().replace("\n", "\\n") + "\"");
            System.out.println("  finishReason=" + result.getFinishReason()
                    + "  tokens=" + result.getGeneratedTokenCount()
                    + String.format("  %.2f tok/s", result.getTokensPerSecond()));

            int[] ids = result.getTokenIds();
            int show = Math.min(6, ids.length);

            // WRONG streaming recipe: decoding tokens independently. BPE merges span
            // token boundaries, so fragments/markers leak.
            StringBuilder naive = new StringBuilder();
            for (int i = 0; i < show; i++) {
                naive.append('[')
                        .append(tokenizer.decode(new int[]{ids[i]}, true).replace("\n", "\\n"))
                        .append(']');
            }
            System.out.println("  Naive per-token decode (WRONG for UIs): " + naive);

            // RIGHT streaming recipe: decode the growing prefix, emit the delta.
            System.out.println("  Prefix-delta decode (the streaming recipe):");
            String previous = "";
            StringBuilder stream = new StringBuilder();
            for (int i = 0; i < show; i++) {
                String current = tokenizer.decode(Arrays.copyOf(ids, i + 1), true);
                String delta = current.substring(Math.min(previous.length(), current.length()));
                stream.append('[').append(delta.replace("\n", "\\n")).append(']');
                previous = current;
            }
            System.out.println("    " + stream);
            System.out.println("  (No push/callback streaming API exists on the pipeline yet —"
                    + " prefix-delta over returned ids is the supported approach.)");

            model.close();
        }

        tokenizer.close();
        System.out.println("\nHuggingFace-to-text pipeline example completed.");
    }
}
