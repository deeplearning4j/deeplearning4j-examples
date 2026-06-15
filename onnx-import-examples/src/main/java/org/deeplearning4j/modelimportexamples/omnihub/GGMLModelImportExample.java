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

package org.deeplearning4j.modelimportexamples.omnihub;

import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.GGMLModelExport;
import org.nd4j.ggml.GGMLMetadata;
import org.nd4j.ggml.ConversionOptions;
import org.nd4j.autodiff.samediff.SameDiff;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Low-Level GGUF/GGML Model Import and Export Example.
 *
 * This example demonstrates the direct GGML import API (nd4j-ggml module),
 * as opposed to the higher-level AutoModel pipeline shown in HuggingFaceGGUFImport.
 *
 * GGUF (GGML Universal Format) is the standard format for quantized LLMs,
 * widely used by llama.cpp, ollama, and other inference engines.
 *
 * The nd4j-ggml module provides:
 * - GGMLModelImport: Import GGUF models → SameDiff computation graphs
 * - GGMLModelExport: Export SameDiff models → GGUF format
 * - GGMLMetadata: Inspect model metadata without loading weights
 * - ConversionOptions: Control dequantization, precision, and architecture
 *
 * Supported architectures (auto-detected from GGUF metadata):
 * - LLaMA, LLaMA4, Mistral, Gemma, Phi, GPT, Granite, OLMo, OpenELM
 * - MiniCPM-V, Qwen3VL, SmolVLM2 (multimodal)
 * - Whisper (audio)
 * - GenericArchitecture (fallback for unknown architectures)
 *
 * Supported quantization types: Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1,
 * Q5_K, Q6_K, Q8_0, IQ1_M, IQ1_S, IQ2_S, IQ2_XS
 *
 * NOTE: This example requires a GGUF model file. Download one from HuggingFace:
 *   e.g., TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF (tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)
 */
public class GGMLModelImportExample {
    private static final Logger log = LoggerFactory.getLogger(GGMLModelImportExample.class);

    public static void main(String[] args) throws Exception {

        // =====================================================================
        // 1. Check if a file is GGUF format
        // =====================================================================
        log.info("=== Step 1: Format detection ===");

        // For this example, we use a placeholder path.
        // Replace with an actual GGUF model path to run.
        String ggufPath = "path/to/model.gguf";
        File ggufFile = new File(ggufPath);

        if (!ggufFile.exists()) {
            log.info("No GGUF file found at {}. Showing API usage only.", ggufPath);
            showAPIUsage();
            return;
        }

        // Detect whether the file is a valid GGUF/GGML file
        boolean isGGML = GGMLModelImport.isGGMLFile(ggufFile);
        log.info("Is GGML file: {}", isGGML);

        // Detect the specific GGML format variant
        log.info("Format: {}", GGMLModelImport.detectFormat(ggufFile));

        // =====================================================================
        // 2. Inspect model metadata without loading weights
        // =====================================================================
        log.info("=== Step 2: Inspect model metadata ===");

        GGMLMetadata metadata = GGMLModelImport.inspectModel(ggufPath);
        log.info("Model metadata: {}", metadata);

        // =====================================================================
        // 3. Import with ConversionOptions for inference (FP16)
        // =====================================================================
        log.info("=== Step 3: Import model for inference ===");

        // ConversionOptions.forInference() dequantizes to float16 — good balance
        // of memory usage and precision for inference.
        SameDiff model = GGMLModelImport.importModel(ggufFile, ConversionOptions.forInference());
        log.info("Model loaded for inference. Variables: {}", model.variables().size());

        // =====================================================================
        // 4. Import with different precision options
        // =====================================================================
        log.info("=== Step 4: Precision options ===");

        // For training (full float32 precision):
        // SameDiff trainModel = GGMLModelImport.importModel(ggufFile, ConversionOptions.forTraining());

        // For minimal memory (keep quantized weights):
        // SameDiff quantModel = GGMLModelImport.importModel(ggufFile, ConversionOptions.preserveQuantization());

        // For explicit float16:
        // SameDiff fp16Model = GGMLModelImport.importModel(ggufFile, ConversionOptions.fp16());

        // Custom options via builder:
        // ConversionOptions customOpts = ConversionOptions.builder()
        //     .quantizationMode(QuantizationMode.DEQUANTIZE_FP16)
        //     .build();

        // =====================================================================
        // 5. Convert GGUF to SDZ (native SameDiff format)
        // =====================================================================
        log.info("=== Step 5: Format conversion ===");

        // Convert GGUF → SDZ for fast subsequent loading
        // SDZ is DL4J's native format — loading SDZ is much faster than re-importing GGUF
        // GGMLModelImport.convertToSDZ("model.gguf", "model.sdz");

        // =====================================================================
        // 6. Export SameDiff model back to GGUF
        // =====================================================================
        log.info("=== Step 6: GGUF export ===");

        // Export a SameDiff model to GGUF format (for use with llama.cpp, etc.)
        // GGMLModelExport.exportModel(model, new File("output.gguf"));

        // Re-quantize an existing GGUF file to a different quantization level
        // GGMLModelExport.requantize(inputFile, outputFile, "Q4_0");

        // List supported quantization types
        log.info("Supported quantization types: {}", GGMLModelExport.getSupportedQuantizationTypes());
        log.info("Supported architectures: {}", GGMLModelExport.getSupportedArchitectures());

        log.info("**************** GGML Import Example finished ********************");
    }

    /**
     * Shows the GGML API usage without requiring an actual model file.
     */
    private static void showAPIUsage() {
        log.info("");
        log.info("=== GGML Import API Reference ===");
        log.info("");
        log.info("--- Import (GGUF → SameDiff) ---");
        log.info("  GGMLModelImport.importModel(\"model.gguf\")");
        log.info("  GGMLModelImport.importModel(file, ConversionOptions.forInference())");
        log.info("  GGMLModelImport.importModel(file, ConversionOptions.forTraining())");
        log.info("  GGMLModelImport.importModel(file, ConversionOptions.fp16())");
        log.info("  GGMLModelImport.importModel(file, ConversionOptions.preserveQuantization())");
        log.info("");
        log.info("--- Inspection ---");
        log.info("  GGMLModelImport.isGGMLFile(file)          → boolean");
        log.info("  GGMLModelImport.detectFormat(file)        → GGMLFormat");
        log.info("  GGMLModelImport.inspectModel(\"model.gguf\") → GGMLMetadata");
        log.info("");
        log.info("--- Conversion ---");
        log.info("  GGMLModelImport.convertToSDZ(\"in.gguf\", \"out.sdz\")");
        log.info("");
        log.info("--- Export (SameDiff → GGUF) ---");
        log.info("  GGMLModelExport.exportModel(sd, new File(\"out.gguf\"))");
        log.info("  GGMLModelExport.convertSDZToGGUF(sdzFile, ggufFile, options)");
        log.info("  GGMLModelExport.requantize(inputGguf, outputGguf, \"Q4_0\")");
        log.info("  GGMLModelExport.validateForExport(sd)");
        log.info("");
        log.info("--- Supported architectures ---");
        log.info("  LLaMA, Mistral, Gemma, Phi, GPT, Granite, OLMo, OpenELM");
        log.info("  MiniCPM-V, Qwen3VL, SmolVLM2, Whisper, GenericArchitecture");
        log.info("");
        log.info("--- ConversionOptions presets ---");
        log.info("  forInference()          → dequantize to float16");
        log.info("  forTraining()           → full float32");
        log.info("  fp16()                  → force float16");
        log.info("  preserveQuantization()  → keep original quantization");

        log.info("**************** GGML Import Example finished ********************");
    }
}
