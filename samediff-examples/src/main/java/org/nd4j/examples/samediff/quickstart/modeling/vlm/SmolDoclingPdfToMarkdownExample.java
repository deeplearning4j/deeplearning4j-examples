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
package org.nd4j.examples.samediff.quickstart.modeling.vlm;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader.VLMModel;
import org.eclipse.deeplearning4j.vlm.model.encoder.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.encoder.VisionEncoder;
import org.eclipse.deeplearning4j.vlm.model.encoder.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.model.loading.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.output.DocTagsParser;
import org.eclipse.deeplearning4j.vlm.output.DocumentStructure;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * End-to-end document conversion with SmolDocling: PDF pages in, structured Markdown out.
 *
 * This is the "production-shaped" counterpart to {@link SmolDoclingVLMExample}. Where that
 * example walks the pipeline once on a synthetic AWT image, this one runs the complete
 * document-conversion loop the way the platform benchmarks
 * (platform-tests/TestSmolDoclingOptimizedPipeline, run-benchmark.sh) exercise it:
 *
 *   1. A real multi-page PDF is rendered to page images with PDFBox (150 DPI).
 *      Pass your own document via -Dexample.pdf.path=/path/to/file.pdf; without it the
 *      example authors a 3-page report PDF (title page, metrics table, conclusions) so the
 *      full loop runs out of the box.
 *   2. Each page is resized (longest edge 2048), split into 512x512 tiles plus a global
 *      thumbnail ({@link ImageTiler}), and batched into a single [1, frames, 3, 512, 512]
 *      tensor ({@link VisionEncoderUtils#preprocessFrames}).
 *   3. The {@link VisionEncoder} wrapper (the same class the benchmark uses) runs the
 *      SigLIP encoder over all tiles and returns [1, frames * seqPerFrame, hidden].
 *   4. The SmolDocling chat prompt is assembled with {@link ImagePromptBuilder} —
 *      including the tile-grid layout tokens — and merged with the vision embeddings
 *      ({@link EmbeddingMerger}).
 *   5. ONE {@link GenerationPipeline} instance decodes every page. The pipeline resets its
 *      frozen DSP executor between generations automatically, so page 2..N reuse the
 *      compiled plan/warm caches instead of paying page-1 warmup again. Per-page tok/s,
 *      steady-state tok/s and DSP plan state are reported so the effect is visible.
 *   6. Raw DocTags output is parsed ({@link DocTagsParser}) into a {@link DocumentStructure}
 *      and rendered to Markdown; per-page and whole-document .md files are written.
 *
 * System properties (all optional):
 *   -Dexample.pdf.path=/path/to/document.pdf   use a real PDF instead of the generated one
 *   -Dexample.pdf.dpi=150                      render DPI
 *   -Dexample.max.pages=2                      how many pages to convert
 *   -Dexample.max.tokens=160                   decode budget per page
 *   -Dexample.resize.edge=2048                 longest-edge resize before tiling
 *   -Dexample.max.tiles=-1                     tile cap (-1 = unlimited; each tile is a
 *                                              full vision-encoder pass — the main cost)
 *   -Dexample.output.dir=/tmp/...              where DocTags/Markdown files are written
 *
 * Backend: defaults to CPU (nd4j-native). Decode is dramatically faster on CUDA — switch
 * the {@code nd4j.backend} property in samediff-examples/pom.xml to
 * {@code nd4j-cuda-12.9-platform} for GPU (the platform benchmark sustains &gt;55 tok/s
 * with the OPTIMAL config on a single consumer GPU).
 *
 * Downloads on first run (~700MB total, cached under ~/.cache/dl4j-vlm-models): SigLIP
 * vision encoder, SmolLM2 decoder, embed_tokens, tokenizer and preprocessor config. ONNX
 * import results are cached as .sdz next to the models, so later runs skip import entirely.
 */
public class SmolDoclingPdfToMarkdownExample {

    public static void main(String[] args) throws Exception {
        int dpi = Integer.getInteger("example.pdf.dpi", 150);
        int maxPages = Integer.getInteger("example.max.pages", 2);
        int maxNewTokens = Integer.getInteger("example.max.tokens", 160);
        // Resource knobs: tile count is the dominant vision-encoder cost (each 512x512
        // frame is a full SigLIP forward pass). Full quality = resize 2048 / unlimited
        // tiles; constrained machines can run e.g. -Dexample.resize.edge=512
        // -Dexample.max.tiles=1 for a single-frame pass at reduced OCR fidelity.
        int resizeEdge = Integer.getInteger("example.resize.edge", 2048);
        int maxTiles = Integer.getInteger("example.max.tiles", -1);
        Path outputDir = Paths.get(System.getProperty("example.output.dir",
                System.getProperty("java.io.tmpdir") + File.separator + "smoldocling-pdf-example"));
        Files.createDirectories(outputDir);

        // ============================================================
        // 1. INPUT DOCUMENT
        // ============================================================
        System.out.println("=== 1. Input document ===");
        File pdfFile;
        String pdfProp = System.getProperty("example.pdf.path");
        if (pdfProp != null && new File(pdfProp).isFile()) {
            pdfFile = new File(pdfProp);
            System.out.println("  Using user-supplied PDF: " + pdfFile.getAbsolutePath());
        } else {
            pdfFile = outputDir.resolve("sample-report.pdf").toFile();
            createSampleReportPdf(pdfFile);
            System.out.println("  No -Dexample.pdf.path given; generated a 3-page sample report: "
                    + pdfFile.getAbsolutePath());
        }

        // ============================================================
        // 2. MODELS: DOWNLOAD, IMPORT (SDZ-CACHED), TOKENIZER
        // ============================================================
        System.out.println("\n=== 2. Downloading + importing SmolDocling components ===");
        long t0 = System.currentTimeMillis();
        File decoderFile = VLMModelDownloader.download(VLMModel.SMOLDOCLING_DECODER).getModelFile();
        File embedTokensFile = VLMModelDownloader.download(VLMModel.SMOLDOCLING_EMBED_TOKENS).getModelFile();
        File visionEncoderFile = VLMModelDownloader.download(VLMModel.SMOLDOCLING_VISION_ENCODER).getModelFile();
        File tokenizerFile = VLMModelDownloader.download(VLMModel.SMOLDOCLING_TOKENIZER).getModelFile();
        File preprocessorConfigFile = VLMModelDownloader.download(VLMModel.SMOLDOCLING_PREPROCESSOR_CONFIG).getModelFile();
        System.out.println("  Download/cache check: " + (System.currentTimeMillis() - t0) + "ms");

        // importAllWithCache loads vision encoder + decoder + embed_tokens in parallel,
        // runs the GraphOptimizer over each graph (fusing rms_norm / swish / xw_plus_b
        // patterns) and caches the result as .sdz. First run takes minutes; cached runs
        // load in seconds.
        t0 = System.currentTimeMillis();
        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionEncoderFile.getAbsolutePath(),
                decoderFile.getAbsolutePath(),
                embedTokensFile.getAbsolutePath());
        SameDiff visionEncoderSd = models[0];
        SameDiff decoder = models[1];
        SameDiff embedTokensSd = models[2];
        System.out.println("  Import time: " + (System.currentTimeMillis() - t0) + "ms");
        System.out.println("  Vision encoder ops: " + visionEncoderSd.ops().length
                + ", decoder ops: " + decoder.ops().length
                + ", embed_tokens ops: " + embedTokensSd.ops().length);

        // The downloader hands back the tokenizer.json FILE itself — load with fromFile
        // (fromDirectory expects a directory containing tokenizer.json).
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile);
        System.out.println("  Tokenizer vocab: " + tokenizer.getVocabSize());

        PreprocessorConfig ppConfig = PreprocessorConfig.fromFile(preprocessorConfigFile);
        int tileSize = ppConfig.getTargetHeight();   // 512 for SmolDocling
        System.out.println("  Tile size from preprocessor_config.json: " + tileSize);

        // ============================================================
        // 3. ONE GENERATION PIPELINE FOR THE WHOLE DOCUMENT
        // ============================================================
        System.out.println("\n=== 3. Creating the generation pipeline ===");
        // Document OCR must be deterministic -> greedy decoding. The pipeline is created
        // once and reused for every page: between generate() calls it resets the frozen
        // DSP executor but keeps compiled artifacts warm, which is exactly how the
        // multi-page benchmark drives it.
        GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokensSd)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxNewTokens)
                .build());
        System.out.println("  Pipeline ready (greedy, maxNewTokens=" + maxNewTokens + ")");

        // ============================================================
        // 4. PER-PAGE CONVERSION LOOP
        // ============================================================
        DocTagsParser parser = new DocTagsParser();
        StringBuilder documentMarkdown = new StringBuilder();
        List<double[]> pageMetrics = new ArrayList<>();   // {tokens, tok/s, steady, firstTokenMs}

        try (PDDocument pdf = PDDocument.load(pdfFile)) {
            PDFRenderer renderer = new PDFRenderer(pdf);
            int pages = Math.min(maxPages, pdf.getNumberOfPages());
            System.out.println("\n=== 4. Converting " + pages + "/" + pdf.getNumberOfPages()
                    + " page(s) at " + dpi + " DPI ===");

            for (int page = 0; page < pages; page++) {
                System.out.println("\n  --- Page " + (page + 1) + " ---");

                // 4a. Render + tile. resizeLongestEdge keeps the aspect ratio while
                // bounding work; splitImageForVLM produces the tile grid plus a global
                // downscaled frame (the model sees both detail and page layout).
                BufferedImage pageImage = renderer.renderImageWithDPI(page, dpi, ImageType.RGB);
                BufferedImage resized = ImageTiler.resizeLongestEdge(pageImage, resizeEdge);
                ImageTiler.SplitImageResult split = ImageTiler.splitImageForVLM(resized, tileSize, maxTiles);
                int frames = split.getTotalFrames();
                System.out.println("    Rendered " + pageImage.getWidth() + "x" + pageImage.getHeight()
                        + " -> grid " + split.numRows + "x" + split.numCols + " (" + frames + " frames)");

                // 4b. Normalize all frames into one [1, frames, 3, H, W] tensor.
                VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);
                INDArray imageInput = VisionEncoderUtils.preprocessFrames(split.frames, preprocessor, tileSize);
                preprocessor.shutdown();

                // 4c. Vision encoding through the reusable VisionEncoder wrapper.
                VisionEncoder visionEncoder = VisionEncoder.builder()
                        .model(visionEncoderSd)
                        .targetSize(tileSize)
                        .build();
                VisionEncoder.Result visionResult = visionEncoder.encode(imageInput, frames, split);
                INDArray visionEmbeddings = visionResult.getEmbeddings();
                imageInput.close();
                System.out.println("    Vision embeddings " + java.util.Arrays.toString(visionEmbeddings.shape())
                        + " in " + visionResult.getEncodingTimeMs() + "ms");

                // 4d. Prompt assembly. The grid tokens tell the decoder where each tile
                // sits on the page; the task instruction requests DocTags output.
                int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
                int seqPerFrame = (int) (visionEmbeddings.size(1) / frames);
                String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                        split.numRows, split.numCols, seqPerFrame);
                String chatPrompt = "<|im_start|>User:" + imagePrompt
                        + "Convert this page to docling.<end_of_utterance>\nAssistant:";
                int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();

                INDArray textEmbeddings = pipeline.embedTokens(promptTokenIds);
                INDArray inputsEmbeds = EmbeddingMerger.mergeEmbeddings(
                        textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);
                System.out.println("    Prompt tokens: " + promptTokenIds.length
                        + ", merged sequence: " + inputsEmbeds.size(1));

                // 4e. Decode.
                GenerationResult result = pipeline.generate(inputsEmbeds, promptTokenIds, maxNewTokens);
                System.out.println(String.format(
                        "    Generated %d tokens in %dms  |  %.2f tok/s overall, %.2f steady, %.2f late-steady",
                        result.getGeneratedTokenCount(), result.getGenerationTimeMs(),
                        result.getTokensPerSecond(), result.getSteadyStateTokensPerSecond(),
                        result.getLateSteadyStateTokensPerSecond()));
                System.out.println("    First token: " + result.getFirstTokenLatencyMs()
                        + "ms, finish reason: " + result.getFinishReason());
                pageMetrics.add(new double[]{result.getGeneratedTokenCount(), result.getTokensPerSecond(),
                        result.getSteadyStateTokensPerSecond(), result.getFirstTokenLatencyMs()});

                // 4f. DSP plan state — page 1 pays warmup/compile, later pages should
                // show the executor replaying instead of recompiling.
                reportDspState(decoder, page + 1);

                // 4g. Structure the output: DocTags -> DocumentStructure -> Markdown.
                String docTags = result.getText();
                DocumentStructure structure = parser.parse(docTags);
                String pageMarkdown = parser.toMarkdown(structure);
                System.out.println("    Parsed elements: " + structure.getElementCount()
                        + " (headers: " + structure.getHeaders().size()
                        + ", tables: " + structure.getTables().size() + ")");

                Files.write(outputDir.resolve("page-" + (page + 1) + ".doctags.txt"),
                        docTags.getBytes(StandardCharsets.UTF_8));
                Files.write(outputDir.resolve("page-" + (page + 1) + ".md"),
                        pageMarkdown.getBytes(StandardCharsets.UTF_8));

                documentMarkdown.append("\n\n<!-- page ").append(page + 1).append(" -->\n\n")
                        .append(pageMarkdown);

                String preview = docTags.substring(0, Math.min(220, docTags.length()))
                        .replace("\n", " ");
                System.out.println("    DocTags preview: " + preview + "...");
            }
        } finally {
            pipeline.close();
            tokenizer.close();
        }

        // ============================================================
        // 5. WHOLE-DOCUMENT OUTPUT + RUN SUMMARY
        // ============================================================
        System.out.println("\n=== 5. Document output ===");
        Path documentMd = outputDir.resolve("document.md");
        Files.write(documentMd, documentMarkdown.toString().getBytes(StandardCharsets.UTF_8));
        System.out.println("  Combined Markdown: " + documentMd.toAbsolutePath());
        System.out.println("  Per-page DocTags + Markdown files in: " + outputDir.toAbsolutePath());

        System.out.println("\n  Page | tokens | tok/s | steady tok/s | first-token ms");
        for (int i = 0; i < pageMetrics.size(); i++) {
            double[] m = pageMetrics.get(i);
            System.out.println(String.format("  %4d | %6.0f | %5.2f | %12.2f | %14.0f",
                    i + 1, m[0], m[1], m[2], m[3]));
        }

        System.out.println("\n  Tuning knobs used by the platform benchmark that also apply here:");
        System.out.println("    - GPU backend (pom nd4j.backend=nd4j-cuda-12.9-platform) for >10x decode speed");
        System.out.println("    - GenerationPipelineConfig.maxPrefillLength/maxKvCacheLength: fixed-size");
        System.out.println("      buffers so the frozen DSP plan is reused verbatim across pages");
        System.out.println("    - BenchmarkConfig.optimal() (the default) enables Triton section fusion +");
        System.out.println("      CUDA graph capture on GPU; BenchmarkConfig.cpuCascade() targets CPU");
        System.out.println("\nSmolDocling PDF -> Markdown example completed.");
    }

    /** Print a compact view of the decoder's DSP plan lifecycle after a page. */
    private static void reportDspState(SameDiff decoder, int page) {
        try {
            DspHandle dsp = decoder.dsp();
            if (dsp == null || !dsp.isCompiled()) {
                System.out.println("    DSP: no compiled plan (slot-by-slot execution)");
                return;
            }
            int phaseOrdinal = dsp.planPhase();
            PlanPhase[] phases = PlanPhase.values();
            String phase = phaseOrdinal >= 0 && phaseOrdinal < phases.length
                    ? phases[phaseOrdinal].name() : ("#" + phaseOrdinal);
            System.out.println("    DSP after page " + page + ": phase=" + phase
                    + ", executions=" + dsp.executeCount()
                    + ", segments=" + dsp.numSegments()
                    + ", graphReplays=" + dsp.totalGraphReplays());
        } catch (Throwable t) {
            // DSP introspection is diagnostic only — never fail the conversion over it.
            System.out.println("    DSP state unavailable: " + t.getMessage());
        }
    }

    // ================================================================
    // Sample-document authoring (PDFBox) — used when no PDF is supplied
    // ================================================================

    /**
     * Writes a 3-page report PDF with the element types SmolDocling is trained on:
     * title, section headers, paragraphs, a bulleted list, a data table with caption,
     * and page footers.
     */
    private static void createSampleReportPdf(File target) throws Exception {
        try (PDDocument doc = new PDDocument()) {
            // ---- Page 1: title, intro paragraphs, bullet list ----
            PDPage page1 = new PDPage(PDRectangle.LETTER);
            doc.addPage(page1);
            try (PDPageContentStream cs = new PDPageContentStream(doc, page1)) {
                text(cs, PDType1Font.HELVETICA_BOLD, 24, 72, 720, "Quarterly Engineering Report");
                text(cs, PDType1Font.HELVETICA_BOLD, 14, 72, 680, "1. Executive Summary");
                paragraph(cs, 72, 656,
                        "This report summarizes the engineering organization's delivery",
                        "performance for the quarter. Inference throughput improved across",
                        "all supported accelerators while model accuracy remained stable.",
                        "The document conversion pipeline now processes multi-page inputs",
                        "with a single reusable generation pipeline.");
                text(cs, PDType1Font.HELVETICA_BOLD, 14, 72, 540, "2. Highlights");
                paragraph(cs, 90, 516,
                        "- Decode throughput up 38 percent on consumer GPUs",
                        "- Graph optimizer fusions reduced op count by roughly a third",
                        "- KV cache reuse eliminated per-page warmup cost",
                        "- Distillation cut student model size by a factor of twenty");
                text(cs, PDType1Font.HELVETICA, 9, 72, 60, "Confidential - Page 1 of 3");
            }

            // ---- Page 2: section header, metrics table with caption ----
            PDPage page2 = new PDPage(PDRectangle.LETTER);
            doc.addPage(page2);
            try (PDPageContentStream cs = new PDPageContentStream(doc, page2)) {
                text(cs, PDType1Font.HELVETICA_BOLD, 14, 72, 720, "3. Throughput Metrics");
                paragraph(cs, 72, 696,
                        "The table below reports steady-state decode throughput in tokens",
                        "per second for each execution configuration.");

                String[][] rows = {
                        {"Configuration", "CPU tok/s", "GPU tok/s"},
                        {"SLOT_BY_SLOT", "2.1", "14.5"},
                        {"OPTIMAL", "3.4", "67.9"},
                        {"TRITON", "3.2", "66.1"},
                        {"CUDA_GRAPHS", "-", "63.8"},
                };
                float tableTop = 640, rowH = 22, colW = 150, left = 72;
                for (int r = 0; r < rows.length; r++) {
                    float y = tableTop - r * rowH;
                    for (int c = 0; c < 3; c++) {
                        text(cs, r == 0 ? PDType1Font.HELVETICA_BOLD : PDType1Font.HELVETICA,
                                11, left + c * colW + 4, y - 15, rows[r][c]);
                    }
                    cs.moveTo(left, y - rowH + 2);
                    cs.lineTo(left + 3 * colW, y - rowH + 2);
                    cs.stroke();
                }
                cs.moveTo(left, tableTop + 2);
                cs.lineTo(left + 3 * colW, tableTop + 2);
                cs.stroke();
                text(cs, PDType1Font.HELVETICA_OBLIQUE, 10, 72, tableTop - rows.length * rowH - 16,
                        "Table 1: Steady-state decode throughput by execution configuration.");
                text(cs, PDType1Font.HELVETICA, 9, 72, 60, "Confidential - Page 2 of 3");
            }

            // ---- Page 3: conclusions ----
            PDPage page3 = new PDPage(PDRectangle.LETTER);
            doc.addPage(page3);
            try (PDPageContentStream cs = new PDPageContentStream(doc, page3)) {
                text(cs, PDType1Font.HELVETICA_BOLD, 14, 72, 720, "4. Conclusions");
                paragraph(cs, 72, 696,
                        "Structured document understanding is now a first-class workload.",
                        "The vision encoder, embedding merger and decoder execute as one",
                        "pipeline, and DocTags output parses directly into a document tree",
                        "that renders to Markdown or HTML without manual cleanup.");
                text(cs, PDType1Font.HELVETICA_BOLD, 14, 72, 600, "5. Next Steps");
                paragraph(cs, 72, 576,
                        "Planned work includes paged KV cache strategies for very long",
                        "documents and speculative decoding with a distilled draft model.");
                text(cs, PDType1Font.HELVETICA, 9, 72, 60, "Confidential - Page 3 of 3");
            }

            doc.save(target);
        }
    }

    private static void text(PDPageContentStream cs, PDType1Font font, float size,
                             float x, float y, String line) throws Exception {
        cs.beginText();
        cs.setFont(font, size);
        cs.newLineAtOffset(x, y);
        cs.showText(line);
        cs.endText();
    }

    private static void paragraph(PDPageContentStream cs, float x, float topY,
                                  String... lines) throws Exception {
        float y = topY;
        for (String line : lines) {
            text(cs, PDType1Font.HELVETICA, 11, x, y, line);
            y -= 16;
        }
    }
}
