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

import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader.VLMModel;
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * SmolDocling Vision-Language Model (VLM) — Document Understanding Example
 *
 * SmolDocling is a 256M-parameter VLM designed for document understanding tasks:
 * converting scanned pages, PDFs, and images into structured text (markdown, DocTags).
 *
 * This example demonstrates the full VLM pipeline:
 *
 *   1. Download ONNX model components (vision encoder, decoder, embed tokens, tokenizer)
 *   2. Import ONNX models into SameDiff graphs
 *   3. Preprocess an input image (resize, normalize, tile)
 *   4. Encode image tiles through the vision encoder
 *   5. Merge vision embeddings with text prompt embeddings
 *   6. Run autoregressive text generation via GenerationPipeline
 *
 * Architecture overview:
 *   - Vision Encoder: SigLIP-so400m (27 layers, 1152 hidden, patch size 14)
 *   - Projector: Pixel shuffle (9x spatial compression) + linear
 *   - LLM Decoder: SmolLM2-1.7B (24 layers, 2048 hidden, 32 heads, 1 KV head)
 *   - Chat format: Idefics3 (<end_of_utterance> markers)
 *
 * Key classes:
 *   - {@link VLMModelDownloader} — Downloads and caches ONNX model components
 *   - {@link OnnxModelCache} — Imports ONNX to SameDiff with SDZ caching
 *   - {@link VLMImagePreprocessor} — Resize, normalize, pad images for the vision encoder
 *   - {@link ImageTiler} — Split large images into tiles for multi-frame encoding
 *   - {@link EmbeddingMerger} — Splice vision embeddings into text embedding sequences
 *   - {@link ImagePromptBuilder} — Build prompt strings with tile grid tokens
 *   - {@link GenerationPipeline} — Autoregressive text generation with KV cache
 *
 * Model requirements:
 *   - Total download: ~700MB (vision encoder ~355MB, decoder ~350MB, embed tokens ~18MB)
 *   - Models are cached alongside ONNX files as .sdz for faster subsequent loads
 *   - GPU recommended for reasonable throughput
 *
 * Run with:
 *   cd samediff-examples
 *   mvn exec:java -Dexec.mainClass="org.nd4j.examples.samediff.quickstart.modeling.SmolDoclingVLMExample"
 */
public class SmolDoclingVLMExample {

    public static void main(String[] args) throws Exception {

        // ============================================================
        // 1. DOWNLOAD MODEL COMPONENTS
        // ============================================================
        System.out.println("=== 1. Downloading SmolDocling Model Components ===");
        System.out.println("  SmolDocling consists of 4 components:");
        System.out.println("    - Vision encoder (SigLIP): encodes image tiles into embeddings");
        System.out.println("    - Embed tokens: converts token IDs to text embeddings");
        System.out.println("    - Decoder (SmolLM2): autoregressive text generation");
        System.out.println("    - Tokenizer: BPE tokenizer for text encoding/decoding");

        long t0 = System.currentTimeMillis();

        // VLMModelDownloader handles downloading ONNX components from HuggingFace
        File decoderFile = VLMModelDownloader.download(VLMModel.SMOLDOCLING_DECODER).getModelFile();
        File embedTokensFile = VLMModelDownloader.download(VLMModel.SMOLDOCLING_EMBED_TOKENS).getModelFile();
        File visionEncoderFile = VLMModelDownloader.download(VLMModel.SMOLDOCLING_VISION_ENCODER).getModelFile();
        File tokenizerDir = VLMModelDownloader.download(VLMModel.SMOLDOCLING_TOKENIZER).getModelFile();
        File preprocessorConfigFile = VLMModelDownloader.download(VLMModel.SMOLDOCLING_PREPROCESSOR_CONFIG).getModelFile();

        long downloadMs = System.currentTimeMillis() - t0;
        System.out.println("  Download time: " + downloadMs + "ms");

        // ============================================================
        // 2. IMPORT ONNX MODELS INTO SAMEDIFF
        // ============================================================
        System.out.println("\n=== 2. Importing ONNX → SameDiff ===");

        // OnnxModelCache imports ONNX files and caches the SameDiff result as .sdz
        // files alongside the originals. Subsequent loads skip ONNX parsing entirely.
        long t1 = System.currentTimeMillis();
        SameDiff decoder = OnnxModelCache.importWithCache(decoderFile.getAbsolutePath());
        SameDiff embedTokens = OnnxModelCache.importWithCache(embedTokensFile.getAbsolutePath());
        SameDiff visionEncoder = OnnxModelCache.importWithCache(visionEncoderFile.getAbsolutePath());
        long importMs = System.currentTimeMillis() - t1;

        System.out.println("  Import time: " + importMs + "ms");
        System.out.println("  Decoder:        " + decoder.ops().length + " ops");
        System.out.println("  Embed tokens:   " + embedTokens.ops().length + " ops");
        System.out.println("  Vision encoder: " + visionEncoder.ops().length + " ops");

        // Load the tokenizer
        Tokenizer tokenizer = HuggingFaceTokenizer.fromDirectory(tokenizerDir);
        System.out.println("  Tokenizer vocab: " + tokenizer.getVocabSize() + " tokens");

        // ============================================================
        // 3. PREPROCESS AN IMAGE
        // ============================================================
        System.out.println("\n=== 3. Image Preprocessing ===");

        // Create a synthetic test document image
        BufferedImage testImage = createTestDocumentImage();
        System.out.println("  Input image: " + testImage.getWidth() + "x" + testImage.getHeight() + " pixels");

        // Load the preprocessor config (normalization mean/std, target resolution)
        PreprocessorConfig ppConfig = PreprocessorConfig.fromFile(preprocessorConfigFile);
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);

        // Split the image into tiles for multi-frame encoding.
        // Large images are divided into a grid of tiles (e.g., 2x2), each processed
        // independently through the vision encoder then concatenated.
        ImageTiler.SplitImageResult tileResult = ImageTiler.splitImageForVLM(
                testImage, ppConfig.getTargetHeight());

        System.out.println("  Tile grid: " + tileResult.numRows + "x" + tileResult.numCols
                + " (" + tileResult.frames.size() + " tiles)");

        // Preprocess each tile: resize to target resolution, normalize pixel values
        List<INDArray> preprocessedFrames = new ArrayList<INDArray>();
        for (BufferedImage frame : tileResult.frames) {
            preprocessedFrames.add(preprocessor.preprocess(frame));
        }
        System.out.println("  Preprocessed frame shape: " + Arrays.toString(preprocessedFrames.get(0).shape()));

        // ============================================================
        // 4. VISION ENCODING
        // ============================================================
        System.out.println("\n=== 4. Vision Encoding ===");

        // Run each preprocessed tile through the SigLIP vision encoder.
        // Input:  [1, 3, 384, 384] per tile (RGB normalized)
        // Output: [1, numPatches, 1152] per tile (patch embeddings)
        String visionInput = visionEncoder.inputs().get(0);
        String visionOutput = visionEncoder.outputs().get(0);

        long t2 = System.currentTimeMillis();
        List<INDArray> encodedFrames = new ArrayList<INDArray>();
        for (INDArray frame : preprocessedFrames) {
            Map<String, INDArray> visionResult = visionEncoder.output(
                    java.util.Collections.singletonMap(visionInput, frame), visionOutput);
            encodedFrames.add(visionResult.get(visionOutput));
        }

        // Concatenate all tile embeddings along the sequence dimension
        INDArray visionEmbeddings = Nd4j.concat(1, encodedFrames.toArray(new INDArray[0]));
        long visionMs = System.currentTimeMillis() - t2;

        System.out.println("  Vision encoding time: " + visionMs + "ms");
        System.out.println("  Vision embeddings shape: " + Arrays.toString(visionEmbeddings.shape()));
        System.out.println("  Total vision tokens: " + visionEmbeddings.shape()[1]);

        // ============================================================
        // 5. BUILD MERGED EMBEDDINGS
        // ============================================================
        System.out.println("\n=== 5. Building Merged Embeddings ===");

        // Build the prompt string with image tile grid tokens.
        // ImagePromptBuilder creates the correct <row_X_col_Y> token pattern
        // that tells the model about the spatial layout of image tiles.
        String prompt = ImagePromptBuilder.buildImagePromptString(
                tileResult.numRows, tileResult.numCols, (int) visionEmbeddings.shape()[1]);

        // Encode the prompt text into token IDs
        int[] promptTokenIds = tokenizer.encode(prompt, false).getIds();
        System.out.println("  Prompt tokens: " + promptTokenIds.length);

        // Look up text embeddings for the prompt tokens
        INDArray inputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, promptTokenIds.length).castTo(DataType.INT64);
        String embedInput = embedTokens.inputs().get(0);
        String embedOutput = embedTokens.outputs().get(0);
        Map<String, INDArray> embedResult = embedTokens.output(
                java.util.Collections.singletonMap(embedInput, inputIds), embedOutput);
        INDArray textEmbeddings = embedResult.get(embedOutput);

        System.out.println("  Text embeddings shape: " + Arrays.toString(textEmbeddings.shape()));

        // Merge vision and text embeddings.
        // EmbeddingMerger replaces <image> token positions in the text embedding
        // sequence with the corresponding vision encoder outputs.
        Integer imageTokenId = tokenizer.getTokenId("<image>");
        System.out.println("  <image> token ID: " + imageTokenId);

        INDArray mergedEmbeddings = EmbeddingMerger.mergeEmbeddings(
                textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);
        System.out.println("  Merged embeddings shape: " + Arrays.toString(mergedEmbeddings.shape()));

        long hiddenSize = mergedEmbeddings.shape()[2];
        System.out.println("  Hidden size: " + hiddenSize);

        // ============================================================
        // 6. TEXT GENERATION WITH GENERATIONPIPELINE
        // ============================================================
        System.out.println("\n=== 6. Generating Text from Image ===");

        // Build the generation pipeline with the decoder and tokenizer.
        // For VLM, we provide pre-computed merged embeddings rather than a text prompt.
        GenerationPipelineConfig pipelineConfig = GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())    // Deterministic for document OCR
                .maxNewTokens(100)                          // Max output tokens
                .hiddenSize(hiddenSize)                     // Must match merged embedding dim
                .kvCacheStrategy(KvCacheStrategy.STATIC)    // Pre-allocated KV cache
                .build();

        GenerationResult result;
        try (GenerationPipeline pipeline = GenerationPipeline.create(pipelineConfig)) {
            System.out.println("  Pipeline created with auto-discovered I/O config");

            // Generate text from the merged vision+text embeddings.
            // The pipeline handles:
            //   - Prefill: process all input embeddings through the decoder
            //   - Decode: autoregressive generation one token at a time
            //   - KV cache: cache key/value tensors to avoid recomputation
            result = pipeline.generate(mergedEmbeddings, promptTokenIds);
        }

        // ============================================================
        // 7. INSPECT RESULTS
        // ============================================================
        System.out.println("\n=== 7. Generation Results ===");
        System.out.println("  Generated text:");
        System.out.println("    " + result.getText());
        System.out.println("  Tokens generated: " + result.getGeneratedTokenCount());
        System.out.println("  Prompt tokens: " + result.getPromptTokenCount());
        System.out.println("  Generation time: " + result.getGenerationTimeMs() + "ms");
        System.out.println("  Throughput: " + String.format("%.1f", result.getTokensPerSecond()) + " tok/s");
        System.out.println("  First token latency: " + result.getFirstTokenLatencyMs() + "ms");
        System.out.println("  Finish reason: " + result.getFinishReason());
        System.out.println("  Complete: " + result.isComplete());

        // ============================================================
        // 8. VLM PIPELINE SUMMARY
        // ============================================================
        System.out.println("\n=== 8. VLM Pipeline Summary ===");
        System.out.println("  Full VLM pipeline stages:");
        System.out.println("    1. Image → ImageTiler.splitImageForVLM()       → tiles");
        System.out.println("    2. Tiles → VLMImagePreprocessor.preprocess()    → normalized tensors");
        System.out.println("    3. Tensors → Vision encoder forward pass        → vision embeddings");
        System.out.println("    4. Prompt → tokenizer.encode()                  → token IDs");
        System.out.println("    5. Token IDs → embed_tokens forward pass        → text embeddings");
        System.out.println("    6. EmbeddingMerger.mergeEmbeddings()            → merged embeddings");
        System.out.println("    7. Merged → GenerationPipeline.generate()       → output text");
        System.out.println();
        System.out.println("  For simpler usage, VisionLanguageModel wraps all these steps:");
        System.out.println("    VisionLanguageModel vlm = VisionLanguageModel.fromOnnx(");
        System.out.println("        visionEncoder, decoder, embedTokens, tokenizer);");
        System.out.println("    String output = vlm.generate(image, \"Convert to markdown.\");");

        // Cleanup
        tokenizer.close();

        System.out.println("\nSmolDocling VLM example completed successfully.");
    }

    /**
     * Creates a synthetic test document image with text content,
     * simulating a scanned document page for OCR/document understanding.
     */
    private static BufferedImage createTestDocumentImage() {
        BufferedImage img = new BufferedImage(800, 600, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();

        // White background
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 800, 600);

        // Title
        g.setColor(Color.BLACK);
        g.setFont(new Font("Serif", Font.BOLD, 28));
        g.drawString("Sample Document", 50, 60);

        // Horizontal rule
        g.drawLine(50, 75, 750, 75);

        // Body text
        g.setFont(new Font("Serif", Font.PLAIN, 16));
        g.drawString("Section 1: Introduction", 50, 110);
        g.drawString("This is a test document for the SmolDocling vision-language model.", 50, 140);
        g.drawString("SmolDocling can convert document images into structured text formats.", 50, 165);

        g.drawString("Section 2: Features", 50, 210);
        g.drawString("- Supports scanned documents, PDFs, and photographs", 70, 240);
        g.drawString("- Outputs markdown, DocTags, or plain text", 70, 265);
        g.drawString("- Handles multi-page documents with tiled encoding", 70, 290);

        // Simple table
        g.drawString("Section 3: Data Table", 50, 335);
        g.drawRect(70, 350, 300, 25);
        g.drawRect(70, 375, 300, 25);
        g.drawRect(70, 400, 300, 25);
        g.drawLine(220, 350, 220, 425);
        g.setFont(new Font("SansSerif", Font.BOLD, 12));
        g.drawString("Item", 90, 368);
        g.drawString("Value", 240, 368);
        g.setFont(new Font("SansSerif", Font.PLAIN, 12));
        g.drawString("Model size", 90, 393);
        g.drawString("256M params", 240, 393);
        g.drawString("Input resolution", 90, 418);
        g.drawString("384x384 pixels", 240, 418);

        g.dispose();
        return img;
    }
}
