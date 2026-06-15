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

import org.eclipse.deeplearning4j.pipeline.AutoModel;
import org.eclipse.deeplearning4j.pipeline.ModelFormat;
import org.eclipse.deeplearning4j.pipeline.ModelManifest;
import org.eclipse.deeplearning4j.pipeline.Pipeline;
import org.eclipse.deeplearning4j.pipeline.PipelineLoader;
import org.eclipse.deeplearning4j.pipeline.PipelineLoader.LoadConfig;
import org.eclipse.deeplearning4j.omnihub.OmniHubUtils;
import org.nd4j.autodiff.samediff.SameDiff;

import java.io.File;
import java.nio.file.Path;

/**
 * AutoModel Format-Agnostic Model Loading — API Reference
 *
 * AutoModel is the recommended entry point for loading any supported model
 * format into SameDiff. It detects the format automatically from file
 * extensions and embedded model manifests, eliminating the need to call
 * format-specific importers (GGMLModelImport, SafeTensors, OnnxImport, etc.)
 * directly.
 *
 * Supported formats detected automatically:
 *   - GGUF    (.gguf)       — quantized LLM format (LLaMA, Mistral, Qwen, etc.)
 *   - SafeTensors (.safetensors) — HuggingFace serialization format
 *   - ONNX    (.onnx)       — Open Neural Network Exchange
 *   - SDZ     (.sdz)        — Native SameDiff serialization (zip-compressed)
 *   - FlatBuffers (.fb)     — SameDiff FlatBuffers format
 *
 * Key classes:
 *   - {@link AutoModel}       — Static factory methods for format-agnostic loading
 *   - {@link Pipeline}        — High-level model pipeline with metadata and lifecycle
 *   - {@link PipelineLoader}  — Configurable loader (cache, quantization, device placement)
 *   - {@link LoadConfig}      — Builder for load-time options
 *   - {@link ModelManifest}   — Parsed model metadata (architecture, format, tensor names)
 *   - {@link ModelFormat}     — Enum of supported formats (GGUF, SAFE_TENSORS, ONNX, SDZ)
 *   - OmniHubUtils            — Downloads pretrained models from HuggingFace / model hubs
 *
 * Run with:
 *   cd samediff-examples
 *   mvn exec:java -Dexec.mainClass="org.nd4j.examples.samediff.quickstart.pipeline.AutoModelExample"
 */
public class AutoModelExample {

    public static void main(String[] args) throws Exception {

        // ============================================================
        // 1. BASIC AUTOMODEL LOADING
        // ============================================================
        System.out.println("=== 1. AutoModel.fromPretrained(path) ===");
        System.out.println();
        System.out.println("  AutoModel detects the model format from the file extension");
        System.out.println("  and embedded metadata, then delegates to the appropriate");
        System.out.println("  format-specific importer automatically.");
        System.out.println();

        // AutoModel.fromPretrained() accepts any supported format.
        // The return type is SameDiff — the unified runtime representation.
        //
        // For a real model, replace this path with an actual file:
        //   SameDiff sd = AutoModel.fromPretrained("/path/to/model.gguf");
        //   SameDiff sd = AutoModel.fromPretrained("/path/to/model.onnx");
        //   SameDiff sd = AutoModel.fromPretrained("/path/to/model.safetensors");
        //   SameDiff sd = AutoModel.fromPretrained("/path/to/model.sdz");
        //
        // Format detection order:
        //   1. File extension (.gguf, .onnx, .safetensors, .sdz, .fb)
        //   2. Magic bytes in the file header (GGUF=0x46554747, ONNX protobuf, etc.)
        //   3. model_manifest.json in the same directory
        //   4. Directory scan for known layout patterns (HuggingFace model dir)

        System.out.println("  Code pattern:");
        System.out.println("    String path = \"/path/to/model.gguf\";  // or .onnx, .safetensors, .sdz");
        System.out.println("    SameDiff sd = AutoModel.fromPretrained(path);");
        System.out.println("    // sd is ready for inference — SameDiff.exec(), GenerationPipeline, etc.");
        System.out.println();

        // ============================================================
        // 2. LOADING WITH CUSTOM LOADCONFIG
        // ============================================================
        System.out.println("=== 2. AutoModel.fromPretrained(path, LoadConfig) ===");
        System.out.println();
        System.out.println("  LoadConfig lets you control caching, quantization, and device placement.");
        System.out.println();

        // LoadConfig is built via a fluent builder.
        // cacheConvertedModel(true) — after importing a GGUF/ONNX to SameDiff, save
        //   the SameDiff graph to disk as .sdz so future loads skip re-import (much faster).
        // quantizationHint()        — request a specific precision on load
        //   (only honored if the format supports it at import time)
        // deviceId()                — which GPU/CPU device to place tensors on
        // memoryMapped()            — mmap large model files instead of copying to heap

        LoadConfig config = LoadConfig.builder()
                .cacheConvertedModel(true)          // Save .sdz cache next to source file
                .deviceId(0)                        // GPU 0 (use -1 for CPU)
                .memoryMapped(true)                 // mmap the source file
                .build();

        System.out.println("  LoadConfig options demonstrated:");
        System.out.println("    cacheConvertedModel(true) — convert once, load fast thereafter");
        System.out.println("    deviceId(0)               — target device (0=first GPU, -1=CPU)");
        System.out.println("    memoryMapped(true)         — mmap source file (less heap pressure)");
        System.out.println();
        System.out.println("  Code pattern:");
        System.out.println("    LoadConfig config = LoadConfig.builder()");
        System.out.println("        .cacheConvertedModel(true)");
        System.out.println("        .deviceId(0)");
        System.out.println("        .memoryMapped(true)");
        System.out.println("        .build();");
        System.out.println("    SameDiff sd = AutoModel.fromPretrained(\"/path/to/model.gguf\", config);");
        System.out.println();

        // ============================================================
        // 3. LOADING AS A PIPELINE (WITH METADATA)
        // ============================================================
        System.out.println("=== 3. AutoModel.pipelineFromPretrained(path) ===");
        System.out.println();
        System.out.println("  pipelineFromPretrained() returns a Pipeline object which wraps");
        System.out.println("  SameDiff and also exposes the model's ModelManifest — architecture");
        System.out.println("  metadata, tokenizer paths, and format information.");
        System.out.println();

        // Pipeline provides lifecycle management (AutoCloseable) and exposes
        // the ModelManifest for introspection without fully loading the model.
        //
        //   Pipeline pipeline = AutoModel.pipelineFromPretrained("/path/to/model.gguf");
        //   ModelManifest manifest = pipeline.getManifest();
        //   String arch = manifest.getArchitecture();       // "llama", "mistral", "qwen", etc.
        //   ModelFormat format = manifest.getFormat();      // GGUF, SAFE_TENSORS, ONNX, SDZ
        //   String tokenizerPath = manifest.getTokenizerPath();
        //   SameDiff sd = pipeline.getModel();
        //   pipeline.close(); // releases native resources

        System.out.println("  Code pattern:");
        System.out.println("    Pipeline pipeline = AutoModel.pipelineFromPretrained(\"/path/to/model.gguf\");");
        System.out.println("    try (pipeline) {");
        System.out.println("        ModelManifest manifest = pipeline.getManifest();");
        System.out.println("        System.out.println(\"Architecture: \" + manifest.getArchitecture());");
        System.out.println("        System.out.println(\"Format: \" + manifest.getFormat());");
        System.out.println("        SameDiff sd = pipeline.getModel();");
        System.out.println("        // use sd for inference ...");
        System.out.println("    }");
        System.out.println();

        // ModelManifest fields:
        System.out.println("  ModelManifest provides:");
        System.out.println("    manifest.getArchitecture()     — detected arch (\"llama\", \"mistral\", ...)");
        System.out.println("    manifest.getFormat()           — ModelFormat enum value");
        System.out.println("    manifest.getModelName()        — human-readable name from metadata");
        System.out.println("    manifest.getNumParameters()    — total parameter count");
        System.out.println("    manifest.getTokenizerPath()    — path to associated tokenizer.json");
        System.out.println("    manifest.getTensorNames()      — List<String> of all tensor names");
        System.out.println("    manifest.getContextLength()    — maximum sequence length");
        System.out.println("    manifest.getVocabSize()        — vocabulary size");
        System.out.println();

        // ============================================================
        // 4. PIPELINELOADER WITH FULL CONFIG
        // ============================================================
        System.out.println("=== 4. PipelineLoader — Fine-grained control ===");
        System.out.println();
        System.out.println("  PipelineLoader is the lower-level API that AutoModel delegates to.");
        System.out.println("  Use it directly when you need more control over the loading process.");
        System.out.println();

        // PipelineLoader.LoadConfig has more options than AutoModel.fromPretrained()
        LoadConfig advancedConfig = LoadConfig.builder()
                .cacheConvertedModel(true)          // Save .sdz next to original on first load
                .deviceId(0)                        // Place model on GPU 0
                .memoryMapped(true)                 // mmap source file (avoids large heap allocation)
                .validateOnLoad(true)               // Run graph validation after loading
                .build();

        System.out.println("  Advanced LoadConfig:");
        System.out.println("    LoadConfig config = LoadConfig.builder()");
        System.out.println("        .cacheConvertedModel(true)");
        System.out.println("        .deviceId(0)");
        System.out.println("        .memoryMapped(true)");
        System.out.println("        .validateOnLoad(true)");
        System.out.println("        .build();");
        System.out.println();
        System.out.println("  Loading via PipelineLoader:");
        System.out.println("    PipelineLoader loader = new PipelineLoader(config);");
        System.out.println("    Pipeline pipeline = loader.load(\"/path/to/model.gguf\");");
        System.out.println();

        // ============================================================
        // 5. MODELFORMAT ENUM — SUPPORTED FORMATS
        // ============================================================
        System.out.println("=== 5. ModelFormat — Supported formats ===");
        System.out.println();

        // ModelFormat is used by AutoModel and ModelManifest to indicate which
        // format was detected / will be used.
        System.out.println("  ModelFormat values:");
        System.out.println("    ModelFormat.GGUF          — GGML/GGUF quantized LLM format");
        System.out.println("    ModelFormat.SAFE_TENSORS  — HuggingFace SafeTensors");
        System.out.println("    ModelFormat.ONNX          — Open Neural Network Exchange");
        System.out.println("    ModelFormat.SDZ           — SameDiff native (zip-compressed)");
        System.out.println("    ModelFormat.FLAT_BUFFERS  — SameDiff FlatBuffers");
        System.out.println("    ModelFormat.UNKNOWN       — could not detect format");
        System.out.println();
        System.out.println("  Checking format programmatically:");
        System.out.println("    ModelFormat fmt = AutoModel.detectFormat(\"/path/to/model.gguf\");");
        System.out.println("    if (fmt == ModelFormat.GGUF) { ... }");
        System.out.println();

        // ============================================================
        // 6. OMNIHUB + AUTOMODEL — DOWNLOAD AND LOAD
        // ============================================================
        System.out.println("=== 6. OmniHub + AutoModel — download and load in one step ===");
        System.out.println();
        System.out.println("  OmniHubUtils.downloadModel() returns a local path after downloading");
        System.out.println("  and caching the model from HuggingFace (or other supported hubs).");
        System.out.println("  That path is then passed directly to AutoModel.fromPretrained().");
        System.out.println();

        // OmniHubUtils handles:
        //   - Resolving the model ID to a download URL
        //   - Caching in ~/.cache/dl4j-models/ (or DL4J_CACHE env var)
        //   - Verifying checksums on re-use
        //   - Extracting model archives if needed
        //
        // AutoModel handles:
        //   - Detecting the downloaded file format
        //   - Importing to SameDiff
        //   - Optionally caching the converted .sdz for faster future loads

        System.out.println("  Full pattern:");
        System.out.println("    // Step 1: download (or use cached copy)");
        System.out.println("    Path modelPath = OmniHubUtils.downloadModel(\"KompileAI/some-model\");");
        System.out.println("    System.out.println(\"Model at: \" + modelPath);");
        System.out.println();
        System.out.println("    // Step 2: load — format detected automatically");
        System.out.println("    SameDiff sd = AutoModel.fromPretrained(modelPath.toString());");
        System.out.println();
        System.out.println("    // Or, combine download + caching with LoadConfig:");
        System.out.println("    LoadConfig cfg = LoadConfig.builder()");
        System.out.println("        .cacheConvertedModel(true)   // skip re-import on next run");
        System.out.println("        .build();");
        System.out.println("    Path modelPath2 = OmniHubUtils.downloadModel(\"KompileAI/some-model\");");
        System.out.println("    SameDiff sd2 = AutoModel.fromPretrained(modelPath2.toString(), cfg);");
        System.out.println();

        // ============================================================
        // 7. FORMAT DETECTION EXAMPLES
        // ============================================================
        System.out.println("=== 7. How AutoModel detects formats ===");
        System.out.println();
        System.out.println("  Detection is attempted in this order:");
        System.out.println("  1. File extension:");
        System.out.println("       .gguf             -> ModelFormat.GGUF");
        System.out.println("       .safetensors      -> ModelFormat.SAFE_TENSORS");
        System.out.println("       .onnx             -> ModelFormat.ONNX");
        System.out.println("       .sdz              -> ModelFormat.SDZ");
        System.out.println("       .fb               -> ModelFormat.FLAT_BUFFERS");
        System.out.println("  2. Magic bytes (file header):");
        System.out.println("       0x46554747        -> GGUF");
        System.out.println("       protobuf tag 0x0a -> ONNX");
        System.out.println("       {'__': ...} JSON  -> SafeTensors");
        System.out.println("  3. model_manifest.json in the same directory");
        System.out.println("  4. Directory layout matching known patterns");
        System.out.println("       (e.g., config.json + pytorch_model.bin = HuggingFace)");
        System.out.println();
        System.out.println("  If detection fails, AutoModel throws ModelFormatException.");
        System.out.println("  You can override with: LoadConfig.builder().forceFormat(ModelFormat.GGUF)");
        System.out.println();

        // ============================================================
        // 8. COMMON USAGE PATTERNS SUMMARY
        // ============================================================
        System.out.println("=== 8. Quick-reference patterns ===");
        System.out.println();
        System.out.println("  // Simplest usage — infer format, load to CPU:");
        System.out.println("  SameDiff sd = AutoModel.fromPretrained(\"/models/llama3.gguf\");");
        System.out.println();
        System.out.println("  // Load to GPU with converted-model cache:");
        System.out.println("  SameDiff sd = AutoModel.fromPretrained(\"/models/llama3.gguf\",");
        System.out.println("      LoadConfig.builder().cacheConvertedModel(true).deviceId(0).build());");
        System.out.println();
        System.out.println("  // Load as Pipeline for metadata access:");
        System.out.println("  try (Pipeline p = AutoModel.pipelineFromPretrained(\"/models/llama3.gguf\")) {");
        System.out.println("      System.out.println(p.getManifest().getArchitecture());");
        System.out.println("      System.out.println(p.getManifest().getNumParameters() / 1_000_000 + \"M params\");");
        System.out.println("  }");
        System.out.println();
        System.out.println("  // OmniHub download + AutoModel load:");
        System.out.println("  Path p = OmniHubUtils.downloadModel(\"KompileAI/some-model\");");
        System.out.println("  SameDiff sd = AutoModel.fromPretrained(p.toString());");
        System.out.println();

        System.out.println("AutoModel example completed. Replace placeholder paths with real model");
        System.out.println("files or use OmniHubUtils to download models before running inference.");
    }
}
