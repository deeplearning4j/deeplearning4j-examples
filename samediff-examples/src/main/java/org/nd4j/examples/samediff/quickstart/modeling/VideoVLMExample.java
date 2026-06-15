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

import org.eclipse.deeplearning4j.vlm.model.VideoVisionLanguageModel;
import org.eclipse.deeplearning4j.vlm.model.VisionLanguageModel;
import org.eclipse.deeplearning4j.vlm.preprocessing.VideoPreprocessor;
import org.eclipse.deeplearning4j.vlm.preprocessing.VideoFrameSampler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VideoFrameExtractor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Video VLM (Video Vision-Language Model) — Preprocessing Pipeline Example
 *
 * Demonstrates the video processing and Video Vision-Language Model preprocessing APIs
 * available in the samediff-vlm module. This example focuses on the full preprocessing
 * pipeline and API documentation. Generation methods are documented with accurate
 * signatures but not called, since a real model download is not required.
 *
 * The pipeline covered here consists of three composable layers:
 *
 *   1. VideoFrameExtractor — Low-level frame extraction from video files via JavaCV/FFmpeg
 *   2. VideoFrameSampler   — Frame selection strategy (uniform, fixed FPS, keyframe)
 *   3. VideoPreprocessor   — Resize, normalize, and stack frames into a 5D tensor
 *
 * These three are then consumed by VideoVisionLanguageModel, which:
 *   - Passes each frame through the vision encoder
 *   - Concatenates the resulting frame embeddings along the sequence dimension
 *   - Merges vision embeddings with text embeddings
 *   - Runs autoregressive decoding via the LLM backbone
 *
 * Supported video VLM architectures:
 *   - SmolVLM2:    Frames as image sequence, pixel shuffle spatial compression
 *   - Qwen3-VL:    Temporal patches with M-RoPE, DeepStack frame fusion
 *   - MiniCPM-V 4.5: 3D-Resampler grouping 6 frames into 64 vision tokens
 *
 * The synthetic frame preprocessing IS fully executed. The shape printed is
 * [1, numFrames, 3, 384, 384] — the 5D tensor consumed by VideoVisionLanguageModel.
 *
 * Run with:
 *   cd samediff-examples
 *   mvn exec:java -Dexec.mainClass="org.nd4j.examples.samediff.quickstart.modeling.VideoVLMExample"
 */
public class VideoVLMExample {

    public static void main(String[] args) throws Exception {

        // ============================================================
        // 1. VIDEO FRAME SAMPLER
        // ============================================================
        System.out.println("=== 1. VideoFrameSampler — Frame Selection Strategies ===");

        // VideoFrameSampler selects which frames to use from a video.
        // It wraps VideoFrameExtractor internally and applies the chosen strategy.
        // Three sampling strategies are available via the Strategy enum:
        //   UNIFORM   — evenly spaced frames up to maxFrames
        //   FIXED_FPS — extract at a target frame rate, optionally capped at maxFrames
        //   KEYFRAME  — only I-frames (no intermediate decoded frames)

        System.out.println("  Available strategies: " + Arrays.toString(VideoFrameSampler.Strategy.values()));

        // UNIFORM: evenly spaced frames, good default for most tasks
        VideoFrameSampler uniformSampler = VideoFrameSampler.builder()
                .strategy(VideoFrameSampler.Strategy.UNIFORM)
                .maxFrames(16)
                .build();

        System.out.println("  uniformSampler: strategy=" + uniformSampler.getStrategy()
                + ", maxFrames=" + uniformSampler.getMaxFrames());

        // FIXED_FPS: useful when temporal density matters (e.g., action recognition)
        // targetFPS controls how many frames per second are extracted
        VideoFrameSampler fpsSampler = VideoFrameSampler.builder()
                .strategy(VideoFrameSampler.Strategy.FIXED_FPS)
                .maxFrames(32)
                .targetFPS(1.0)
                .build();

        System.out.println("  fpsSampler: strategy=" + fpsSampler.getStrategy()
                + ", maxFrames=" + fpsSampler.getMaxFrames()
                + ", targetFPS=" + fpsSampler.getTargetFPS());

        // KEYFRAME: only I-frames — much faster since intermediate frames are skipped.
        // Good for scene change detection and coarse temporal understanding.
        VideoFrameSampler keyframeSampler = VideoFrameSampler.builder()
                .strategy(VideoFrameSampler.Strategy.KEYFRAME)
                .maxFrames(16)
                .build();

        System.out.println("  keyframeSampler: strategy=" + keyframeSampler.getStrategy()
                + ", maxFrames=" + keyframeSampler.getMaxFrames());

        // Optional: minFrameInterval applies post-sampling thinning.
        // Setting minFrameInterval=2 keeps every other frame after extraction.
        VideoFrameSampler thinnedSampler = VideoFrameSampler.builder()
                .strategy(VideoFrameSampler.Strategy.UNIFORM)
                .maxFrames(32)
                .minFrameInterval(2)
                .build();

        System.out.println("  thinnedSampler: strategy=" + thinnedSampler.getStrategy()
                + ", maxFrames=" + thinnedSampler.getMaxFrames()
                + ", minFrameInterval=" + thinnedSampler.getMinFrameInterval());

        // sampleFromFrames() re-samples from an already-loaded list of BufferedImages.
        // Useful when frames are obtained via another source (e.g., camera capture,
        // custom decoder) and you want to apply the same uniform sampling logic
        // without hitting the filesystem.
        System.out.println();
        System.out.println("  sampleFromFrames(List<BufferedImage>) — re-samples from pre-extracted frames");
        System.out.println("    Signature: List<BufferedImage> sampleFromFrames(List<BufferedImage> allFrames)");
        System.out.println("    Uniformly selects up to maxFrames from the supplied list.");
        System.out.println("    Returns the original list unchanged if allFrames.size() <= maxFrames.");

        // ============================================================
        // 2. VIDEO FRAME EXTRACTOR
        // ============================================================
        System.out.println("\n=== 2. VideoFrameExtractor — Low-Level FFmpeg Frame Access ===");

        // VideoFrameExtractor wraps JavaCV (FFmpeg bindings) for decoding video files.
        // JavaCV is an optional runtime dependency. Call isJavaCVAvailable() before
        // using any extraction method to avoid runtime exceptions when it is absent.
        boolean javaCVAvailable = VideoFrameExtractor.isJavaCVAvailable();
        System.out.println("  JavaCV available: " + javaCVAvailable);
        if (!javaCVAvailable) {
            System.out.println("  (Add org.bytedeco:javacv and org.bytedeco:ffmpeg to use video file extraction)");
        }

        // Build an extractor — all configuration is optional with sensible defaults
        VideoFrameExtractor extractor = VideoFrameExtractor.builder()
                .maxFrames(64)
                .build();

        System.out.println("  extractor: maxFrames=" + extractor.getMaxFrames());

        // Extraction methods (require JavaCV; will throw UnsupportedOperationException if absent):
        System.out.println();
        System.out.println("  Extraction methods (all require JavaCV on classpath):");
        System.out.println("    extractFrames(File videoFile, int maxFrames)");
        System.out.println("      -> Uniformly-spaced frames capped at maxFrames");
        System.out.println("    extractFrames(File videoFile)");
        System.out.println("      -> Uniformly-spaced frames using builder maxFrames");
        System.out.println("    extractAtFPS(File videoFile, double fps)");
        System.out.println("      -> Frames at target FPS, unbounded count");
        System.out.println("    extractAtFPS(File videoFile, double fps, int maxFrames)");
        System.out.println("      -> Frames at target FPS, capped at maxFrames");
        System.out.println("    extractKeyframes(File videoFile, int maxFrames)");
        System.out.println("      -> Only I-frames (keyframes), up to maxFrames");
        System.out.println("    getMetadata(File videoFile)");
        System.out.println("      -> VideoMetadata without decoding any frames");

        // VideoMetadata carries the basic properties of a video file
        System.out.println();
        System.out.println("  VideoFrameExtractor.VideoMetadata fields:");
        System.out.println("    int    width           — pixel width of each frame");
        System.out.println("    int    height          — pixel height of each frame");
        System.out.println("    double fps             — native frame rate of the video");
        System.out.println("    long   totalFrames     — total frame count");
        System.out.println("    double durationSeconds — duration in seconds");
        System.out.println("    String codecName       — video codec (e.g., h264, hevc)");
        System.out.println();
        System.out.println("  Access via getters: getWidth(), getHeight(), getFps(),");
        System.out.println("    getTotalFrames(), getDurationSeconds(), getCodecName()");

        // ============================================================
        // 3. VIDEO PREPROCESSOR
        // ============================================================
        System.out.println("\n=== 3. VideoPreprocessor — Frames to 5D Tensor ===");

        // VideoPreprocessor is the main entry point for turning video content
        // into the tensor format expected by video VLMs.
        //
        // Output tensor shape: [1, numFrames, 3, targetHeight, targetWidth]
        //   dimension 0: batch (always 1 for single video)
        //   dimension 1: frame index
        //   dimension 2: channel (RGB)
        //   dimensions 3-4: spatial resolution after resize
        //
        // The imagePreprocessor field handles per-frame normalization (mean/std subtraction,
        // channel normalization). If omitted from the builder, a default normalizer is used.
        // For model-specific normalization use VLMImagePreprocessor.fromConfig(PreprocessorConfig).

        VideoPreprocessor preprocessor = VideoPreprocessor.builder()
                .sampler(VideoFrameSampler.builder()
                        .strategy(VideoFrameSampler.Strategy.UNIFORM)
                        .maxFrames(16)
                        .build())
                .targetHeight(384)
                .targetWidth(384)
                .numPreprocessThreads(4)
                .temporalPatchSize(2)
                .build();

        System.out.println("  preprocessor config:");
        System.out.println("    sampler strategy: " + preprocessor.getSampler().getStrategy());
        System.out.println("    sampler maxFrames: " + preprocessor.getSampler().getMaxFrames());
        System.out.println("    targetHeight: " + preprocessor.getTargetHeight());
        System.out.println("    targetWidth:  " + preprocessor.getTargetWidth());
        System.out.println("    numPreprocessThreads: " + preprocessor.getNumPreprocessThreads());
        System.out.println("    temporalPatchSize: " + preprocessor.getTemporalPatchSize());

        System.out.println();
        System.out.println("  Core preprocessing methods:");
        System.out.println("    preprocessVideo(File videoFile)");
        System.out.println("      -> Extracts frames via sampler, returns [1, N, 3, H, W]");
        System.out.println("    preprocessFrames(List<BufferedImage> frames)");
        System.out.println("      -> Preprocesses a pre-extracted frame list, returns [1, N, 3, H, W]");
        System.out.println("    preprocessVideoTemporalAligned(File videoFile)");
        System.out.println("      -> Like preprocessVideo, but pads N to divisible by temporalPatchSize");
        System.out.println("    preprocessFramesTemporalAligned(List<BufferedImage> frames)");
        System.out.println("      -> Like preprocessFrames, but pads N to divisible by temporalPatchSize");
        System.out.println("      -> Padding duplicates the last frame as needed");

        // estimateVisionTokens computes how many vision tokens the model will see,
        // given the preprocessing configuration and vision encoder parameters.
        //   patchSize        — patch size of the vision encoder (e.g., 14 for SigLIP, 16 for CLIP)
        //   pixelShuffleFactor — spatial compression factor; 3x3 = 9x reduction (SmolVLM2)
        //                       1 means no pixel shuffle compression
        //   temporalPatchSize  — temporal compression; 2 means two frames merge into one group
        int numFrames = 8;
        int patchSize = 14;           // SigLIP patch size
        int pixelShuffleFactor = 3;   // SmolVLM2 pixel shuffle 3x3
        int estimatedTokens = preprocessor.estimateVisionTokens(numFrames, patchSize, pixelShuffleFactor);
        System.out.println();
        System.out.println("  estimateVisionTokens(numFrames, patchSize, pixelShuffleFactor):");
        System.out.println("    inputs:   numFrames=" + numFrames + ", patchSize=" + patchSize
                + ", pixelShuffleFactor=" + pixelShuffleFactor);
        System.out.println("    formula:  patchesPerFrame = (H/patchSize) * (W/patchSize)");
        System.out.println("              tokensPerFrame  = patchesPerFrame / (pixelShuffleFactor^2)");
        System.out.println("              effectiveFrames = numFrames / temporalPatchSize  (if > 1)");
        System.out.println("              totalTokens     = effectiveFrames * tokensPerFrame");
        System.out.println("    result:   estimated vision tokens = " + estimatedTokens);

        // ============================================================
        // 4. CREATE SYNTHETIC FRAMES AND PREPROCESS
        // ============================================================
        System.out.println("\n=== 4. Synthetic Frame Preprocessing (Live Demo) ===");

        // Create 8 synthetic BufferedImage frames with distinct colors
        // to simulate a real video input without requiring a video file.
        List<BufferedImage> syntheticFrames = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            BufferedImage frame = new BufferedImage(384, 384, BufferedImage.TYPE_3BYTE_BGR);
            Graphics2D g = frame.createGraphics();
            // Gradient color per frame to simulate temporal variation
            g.setColor(new Color(i * 30, 100, 200));
            g.fillRect(0, 0, 384, 384);
            g.setColor(Color.WHITE);
            g.drawString("Frame " + i, 150, 192);
            g.dispose();
            syntheticFrames.add(frame);
        }

        System.out.println("  Created " + syntheticFrames.size() + " synthetic frames ("
                + syntheticFrames.get(0).getWidth() + "x" + syntheticFrames.get(0).getHeight() + " px each)");

        // Build a preprocessor without temporal padding to keep the exact frame count
        VideoPreprocessor demoPreprocessor = VideoPreprocessor.builder()
                .sampler(VideoFrameSampler.builder()
                        .strategy(VideoFrameSampler.Strategy.UNIFORM)
                        .maxFrames(16)
                        .build())
                .targetHeight(384)
                .targetWidth(384)
                .numPreprocessThreads(1)
                .temporalPatchSize(1)
                .build();

        // preprocessFrames() runs per-frame resize + normalize and stacks into 5D tensor
        INDArray frameTensor = demoPreprocessor.preprocessFrames(syntheticFrames);

        System.out.println("  frameTensor shape: " + Arrays.toString(frameTensor.shape()));
        System.out.println("    [batch=1, frames=8, channels=3, height=384, width=384]");
        System.out.println("  frameTensor dtype: " + frameTensor.dataType());
        System.out.println("  Total elements: " + frameTensor.length());

        // Also demonstrate sampleFromFrames with the uniform sampler
        List<BufferedImage> resampledFrames = uniformSampler.sampleFromFrames(syntheticFrames);
        System.out.println("  sampleFromFrames(8 frames, maxFrames=16): returned "
                + resampledFrames.size() + " frames (unchanged since 8 <= 16)");

        // With a smaller maxFrames the sampler selects a uniform subset
        VideoFrameSampler smallSampler = VideoFrameSampler.builder()
                .strategy(VideoFrameSampler.Strategy.UNIFORM)
                .maxFrames(4)
                .build();
        List<BufferedImage> subsampledFrames = smallSampler.sampleFromFrames(syntheticFrames);
        System.out.println("  sampleFromFrames(8 frames, maxFrames=4): returned "
                + subsampledFrames.size() + " frames (downsampled)");

        // Demonstrate temporal alignment padding
        // temporalPatchSize=3 with 8 frames will pad to 9 (next multiple of 3)
        VideoPreprocessor temporalPreprocessor = VideoPreprocessor.builder()
                .sampler(VideoFrameSampler.builder()
                        .strategy(VideoFrameSampler.Strategy.UNIFORM)
                        .maxFrames(16)
                        .build())
                .targetHeight(384)
                .targetWidth(384)
                .numPreprocessThreads(1)
                .temporalPatchSize(3)
                .build();

        INDArray temporalAlignedTensor = temporalPreprocessor.preprocessFramesTemporalAligned(
                new ArrayList<>(syntheticFrames));  // pass a copy since it may be mutated
        System.out.println("  preprocessFramesTemporalAligned with temporalPatchSize=3:");
        System.out.println("    Input frames: 8  ->  output shape: "
                + Arrays.toString(temporalAlignedTensor.shape()));
        System.out.println("    (padded to 9 frames = ceil(8/3)*3, last frame duplicated)");

        // ============================================================
        // 5. VIDEO VLM ARCHITECTURE AND BUILDER API
        // ============================================================
        System.out.println("\n=== 5. VideoVisionLanguageModel — Builder and Factory Methods ===");

        // VideoVisionLanguageModel wraps a VisionLanguageModel (image VLM) with a
        // VideoPreprocessor and handles the per-frame encoding loop automatically.
        //
        // It accepts either a video File or a List<BufferedImage> as input, and
        // returns either a String (generated text) or a GenerationResult with metrics.

        System.out.println("  Builder pattern:");
        System.out.println("    VideoVisionLanguageModel.builder()");
        System.out.println("      .vlm(vlm)                    // VisionLanguageModel base model");
        System.out.println("      .videoPreprocessor(preprocessor) // VideoPreprocessor instance");
        System.out.println("      .maxFrames(32)               // Integer, overrides preprocessor sampler max");
        System.out.println("      .maxNewTokens(512)           // Integer, max tokens to generate");
        System.out.println("      .temperature(1.0)            // Double, sampling temperature");
        System.out.println("      .doSample(true)              // Boolean, false = greedy decoding");
        System.out.println("      .build()");

        System.out.println();
        System.out.println("  Factory methods (load from pre-exported SDZ model directories):");
        System.out.println("    VideoVisionLanguageModel.fromDirectory(File modelDir)");
        System.out.println("      -> Loads VisionLanguageModel.fromDirectory(modelDir)");
        System.out.println("         then wraps it with a default VideoPreprocessor");
        System.out.println("    VideoVisionLanguageModel.fromDirectory(File modelDir, VideoPreprocessor preprocessor)");
        System.out.println("      -> Same but uses the supplied preprocessor (null = default)");
        System.out.println("    VideoVisionLanguageModel.fromVLM(VisionLanguageModel vlm)");
        System.out.println("      -> Wraps an existing VLM with a default VideoPreprocessor");

        // ============================================================
        // 6. GENERATION API DOCUMENTATION
        // ============================================================
        System.out.println("\n=== 6. VideoVisionLanguageModel — Generation API ===");

        System.out.println("  Generation from video file (requires JavaCV for frame extraction):");
        System.out.println("    String generate(File videoFile, String prompt)");
        System.out.println("      -> Full pipeline: extract -> sample -> preprocess -> encode -> decode");
        System.out.println("      -> Returns generated text");
        System.out.println("    String generate(File videoFile, String prompt,");
        System.out.println("                    int maxNewTokens, double temperature, boolean doSample)");
        System.out.println("      -> Same with explicit generation parameters");
        System.out.println("    GenerationResult generateWithMetrics(File videoFile, String prompt)");
        System.out.println("      -> Returns GenerationResult with text + timing + token counts");
        System.out.println("    GenerationResult generateWithMetrics(File videoFile, String prompt,");
        System.out.println("                    int maxNewTokens, double temperature, boolean doSample)");
        System.out.println("      -> Same with explicit generation parameters");

        System.out.println();
        System.out.println("  Generation from pre-extracted frames (no JavaCV needed):");
        System.out.println("    String generate(List<BufferedImage> frames, String prompt)");
        System.out.println("      -> Preprocesses frames then encodes; no filesystem I/O");
        System.out.println("    String generate(List<BufferedImage> frames, String prompt,");
        System.out.println("                    int maxNewTokens, double temperature, boolean doSample)");
        System.out.println("    GenerationResult generateWithMetrics(List<BufferedImage> frames,");
        System.out.println("                    String prompt, int maxNewTokens,");
        System.out.println("                    double temperature, boolean doSample)");

        System.out.println();
        System.out.println("  Low-level generation from a pre-built frame tensor:");
        System.out.println("    GenerationResult generateFromFrameTensor(INDArray frameTensor, String prompt,");
        System.out.println("                    int maxNewTokens, double temperature, boolean doSample)");
        System.out.println("      -> frameTensor must be [1, numFrames, 3, H, W]");
        System.out.println("      -> Each frame is encoded through the vision encoder separately");
        System.out.println("      -> Frame embeddings are concatenated along the sequence dimension");
        System.out.println("      -> VisionLanguageModel.generateFromEmbeddings() handles decoding");

        System.out.println();
        System.out.println("  Video metadata (without full frame extraction):");
        System.out.println("    VideoFrameExtractor.VideoMetadata getVideoMetadata(File videoFile)");
        System.out.println("      -> Returns width, height, fps, totalFrames, durationSeconds, codecName");

        // ============================================================
        // 7. SUPPORTED VIDEO ARCHITECTURES
        // ============================================================
        System.out.println("\n=== 7. Supported Video VLM Architectures ===");

        System.out.println("  SmolVLM2 (Idefics3 + SmolLM2):");
        System.out.println("    - Frames processed as an independent image sequence");
        System.out.println("    - Vision encoder: SigLIP-so400m, patch size 14, 384x384 resolution");
        System.out.println("    - Pixel shuffle projector: 3x3 spatial compression (9x token reduction)");
        System.out.println("    - Each frame produces (384/14)^2 / 9 = ~73 vision tokens");
        System.out.println("    - Frames concatenated along sequence dim before decoding");
        System.out.println("    - Recommended: UNIFORM sampler, 16 frames, temporalPatchSize=1");

        System.out.println();
        System.out.println("  Qwen3-VL (QwenVL + Qwen3 LLM backbone):");
        System.out.println("    - Temporal patches: adjacent frames merged at the patch embed level");
        System.out.println("    - M-RoPE: 3D rotary positional encoding (time, height, width)");
        System.out.println("    - DeepStack fusion: combines information across temporal patch groups");
        System.out.println("    - Vision encoder: native resolution + dynamic tiling");
        System.out.println("    - Recommended: FIXED_FPS sampler, targetFPS=1.0, temporalPatchSize=2");

        System.out.println();
        System.out.println("  MiniCPM-V 4.5 (InternViT + MiniCPM3 LLM backbone):");
        System.out.println("    - 3D-Resampler: groups N consecutive frames (default N=6)");
        System.out.println("    - Each group produces a fixed 64 vision tokens regardless of frame count");
        System.out.println("    - Temporal compression: 6 frames -> 64 tokens (dense temporal fusion)");
        System.out.println("    - Recommended: UNIFORM sampler, maxFrames divisible by 6, temporalPatchSize=6");

        // ============================================================
        // 8. PRACTICAL PREPROCESSING CONFIGURATION GUIDE
        // ============================================================
        System.out.println("\n=== 8. Preprocessing Configuration Guide ===");

        System.out.println("  SmolVLM2 configuration:");
        System.out.println("    VideoPreprocessor.builder()");
        System.out.println("      .sampler(VideoFrameSampler.builder()");
        System.out.println("          .strategy(VideoFrameSampler.Strategy.UNIFORM).maxFrames(16).build())");
        System.out.println("      .targetHeight(384).targetWidth(384)");
        System.out.println("      .numPreprocessThreads(4)");
        System.out.println("      .temporalPatchSize(1)  // no temporal grouping");
        System.out.println("      .build()");

        System.out.println();
        System.out.println("  Qwen3-VL configuration:");
        System.out.println("    VideoPreprocessor.builder()");
        System.out.println("      .sampler(VideoFrameSampler.builder()");
        System.out.println("          .strategy(VideoFrameSampler.Strategy.FIXED_FPS).targetFPS(1.0).maxFrames(32).build())");
        System.out.println("      .targetHeight(420).targetWidth(420)  // or model-specific resolution");
        System.out.println("      .numPreprocessThreads(4)");
        System.out.println("      .temporalPatchSize(2)   // pair adjacent frames");
        System.out.println("      .build()");

        System.out.println();
        System.out.println("  MiniCPM-V 4.5 configuration:");
        System.out.println("    VideoPreprocessor.builder()");
        System.out.println("      .sampler(VideoFrameSampler.builder()");
        System.out.println("          .strategy(VideoFrameSampler.Strategy.UNIFORM).maxFrames(24).build()) // 24 = 4 groups of 6");
        System.out.println("      .targetHeight(448).targetWidth(448)");
        System.out.println("      .numPreprocessThreads(4)");
        System.out.println("      .temporalPatchSize(6)   // 3D-Resampler groups 6 frames");
        System.out.println("      .build()");

        System.out.println();
        System.out.println("  End-to-end usage with a loaded model:");
        System.out.println("    // Load model from directory of exported SDZ files");
        System.out.println("    VideoVisionLanguageModel videoVLM =");
        System.out.println("        VideoVisionLanguageModel.fromDirectory(new File(\"SmolVLM2-256M-Video-Instruct-sdz\"));");
        System.out.println();
        System.out.println("    // Option A: generate from a video file (requires JavaCV)");
        System.out.println("    String description = videoVLM.generate(");
        System.out.println("        new File(\"clip.mp4\"), \"Describe what is happening in this video.\");");
        System.out.println();
        System.out.println("    // Option B: generate from pre-extracted frames");
        System.out.println("    String description = videoVLM.generate(");
        System.out.println("        syntheticFrames, \"What objects are visible in these frames?\");");
        System.out.println();
        System.out.println("    // Option C: generate with metrics");
        System.out.println("    GenerationResult result = videoVLM.generateWithMetrics(");
        System.out.println("        new File(\"clip.mp4\"), \"Describe this video.\");");
        System.out.println("    System.out.println(result.getText());");
        System.out.println("    System.out.println(\"Tokens generated: \" + result.getGeneratedTokenCount());");
        System.out.println("    System.out.println(\"Speed: \" + result.getTokensPerSecond() + \" tok/s\");");
        System.out.println();
        System.out.println("    videoVLM.close();");

        // ============================================================
        // 9. SUMMARY
        // ============================================================
        System.out.println("\n=== 9. Pipeline Summary ===");
        System.out.println("  Full video VLM pipeline (high-level):");
        System.out.println("    1. Video file -> VideoFrameExtractor.extractFrames()  -> List<BufferedImage>");
        System.out.println("    2. List<BufferedImage> -> VideoFrameSampler.sampleFromFrames() -> subset");
        System.out.println("    3. subset -> VideoPreprocessor.preprocessFrames()    -> [1, N, 3, H, W]");
        System.out.println("    4. [1, N, 3, H, W] -> VideoVisionLanguageModel.generateFromFrameTensor()");
        System.out.println("         a. For each frame f: VisionEncoder.encode([1, 3, H, W]) -> [1, T, D]");
        System.out.println("         b. Nd4j.concat(1, embeddings) -> [1, N*T, D] combined vision");
        System.out.println("         c. VLM.generateFromEmbeddings(combined, prompt) -> String");
        System.out.println();
        System.out.println("  Simplified one-call API:");
        System.out.println("    VideoVisionLanguageModel.generate(File, String) -> String");
        System.out.println("    VideoVisionLanguageModel.generate(List<BufferedImage>, String) -> String");

        // Print the live tensor shape one more time as a summary
        System.out.println();
        System.out.println("  Live demo result — synthetic frame tensor shape: "
                + Arrays.toString(frameTensor.shape()));
        System.out.println("  Temporal-aligned tensor shape (temporalPatchSize=3): "
                + Arrays.toString(temporalAlignedTensor.shape()));

        // Cleanup tensors
        frameTensor.close();
        temporalAlignedTensor.close();

        System.out.println("\nVideoVLMExample completed successfully.");
    }
}
