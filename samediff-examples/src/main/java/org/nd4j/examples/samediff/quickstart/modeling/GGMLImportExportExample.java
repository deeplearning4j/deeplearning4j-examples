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

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.GGMLModelExport;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.export.ExportOptions;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.ggml.format.GGMLFormatDetector;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.ggml.format.GGMLTensorInfo;
import org.nd4j.ggml.format.GGUFReader;
import org.nd4j.ggml.format.GGUFWriter;
import org.nd4j.ggml.quantization.Dequantizer;
import org.nd4j.ggml.quantization.DequantizerFactory;
import org.nd4j.ggml.quantization.Quantizer;
import org.nd4j.ggml.quantization.QuantizerFactory;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.nio.file.Files;
import java.util.List;
import java.util.Set;

/**
 * GGML / GGUF Model Import & Export - Complete API Reference
 *
 * This example covers the GGML ecosystem integration:
 *
 *   1. Format detection - Identify GGML/GGUF files
 *   2. Model import - Load GGUF models into SameDiff
 *   3. Model export - Save SameDiff models as GGUF
 *   4. Low-level GGUF I/O - Read/write GGUF files directly
 *   5. Quantization - Quantize/dequantize tensors
 *   6. Architecture support - LLaMA, Mistral, generic
 *
 * Key classes:
 *   - GGMLModelImport / GGMLModelExport
 *   - ConversionOptions / ExportOptions
 *   - GGUFReader / GGUFWriter
 *   - QuantizerFactory / DequantizerFactory
 *   - GGMLMetadata / GGMLTensorInfo / GGMLDataType
 */
public class GGMLImportExportExample {

    public static void main(String[] args) throws Exception {

        File tempDir = Files.createTempDirectory("ggml_example").toFile();

        // ============================================================
        // 1. GGUF FILE FORMAT DETECTION
        // ============================================================
        System.out.println("=== GGUF Format Detection ===");
        {
            // GGMLModelImport can detect GGUF files by magic bytes
            // Magic bytes: GGUF=0x46554747, ggml=0x67676D6C, ggmf=0x67676D66, ggjt=0x67676A74
            System.out.println("  Supported detection methods:");
            System.out.println("    GGMLModelImport.isGGMLFile(File) -> boolean");
            System.out.println("    GGMLModelImport.detectFormat(File) -> GGMLFormat (GGUF or GGML)");
        }

        // ============================================================
        // 2. CONVERSION OPTIONS (Import Configuration)
        // ============================================================
        System.out.println("\n=== ConversionOptions (Import) ===");
        {
            // Factory presets for common configurations
            ConversionOptions inferenceOpts = ConversionOptions.forInference();
            ConversionOptions trainingOpts = ConversionOptions.forTraining();
            ConversionOptions fp16Opts = ConversionOptions.fp16();
            ConversionOptions preserveOpts = ConversionOptions.preserveQuantization();

            // Custom builder
            ConversionOptions custom = ConversionOptions.builder()
                    .targetDataType(DataType.FLOAT)         // output tensor precision
                    .preserveTokenizerInfo(true)             // keep vocab metadata
                    .forTraining(false)                       // inference mode
                    .useMemoryMapping(true)                   // mmap for large files
                    .tensorBatchSize(8)                       // process N tensors at a time
                    .maxFileSize(10L * 1024 * 1024 * 1024)   // 10 GB max
                    .build();

            System.out.println("  Preset options available:");
            System.out.println("    forInference() - FLOAT32, read-only");
            System.out.println("    forTraining()  - FLOAT32, with gradient support");
            System.out.println("    fp16()         - FLOAT16, memory-efficient inference");
            System.out.println("    preserveQuantization() - Keep GGML quantized types");

            // Import example (requires a real .gguf file):
            // SameDiff sd = GGMLModelImport.importModel(new File("model.gguf"), inferenceOpts);
        }

        // ============================================================
        // 3. EXPORT OPTIONS
        // ============================================================
        System.out.println("\n=== ExportOptions (Export) ===");
        {
            // Factory presets
            ExportOptions f16Export = ExportOptions.f16();
            ExportOptions f32Export = ExportOptions.f32();
            ExportOptions q4kExport = ExportOptions.q4k();
            ExportOptions q8Export = ExportOptions.q8_0();
            ExportOptions q5kExport = ExportOptions.q5k();
            ExportOptions q6kExport = ExportOptions.q6k();

            // Custom with per-layer quantization overrides
            ExportOptions custom = ExportOptions.builder()
                    .modelName("my-model")
                    .modelAuthor("DL4J")
                    .modelDescription("Example model exported from SameDiff")
                    .validateBeforeExport(true)
                    .includeTokenizer(true)
                    .includeStatistics(true)
                    .build();

            // Supported quantization types
            System.out.println("  Quantization types:");
            System.out.println("    F32, F16, BF16");
            System.out.println("    Q4_0, Q4_1, Q5_0, Q5_1, Q8_0");
            System.out.println("    Q4_K, Q5_K, Q6_K (k-quants, best quality)");

            // Architecture support
            Set<String> archs = GGMLModelExport.getSupportedArchitectures();
            System.out.println("  Supported architectures: " + archs);
        }

        // ============================================================
        // 4. MODEL EXPORT (SameDiff -> GGUF)
        // ============================================================
        System.out.println("\n=== Model Export (SameDiff -> GGUF) ===");
        {
            // Build a small model with LLaMA-style tensor naming
            SameDiff sd = SameDiff.create();

            int vocabSize = 1000;
            int hiddenSize = 128;
            int numHeads = 4;
            int headDim = hiddenSize / numHeads;
            int intermediateSize = 256;

            // Embedding
            sd.var("model.embed_tokens.weight",
                    Nd4j.randn(DataType.FLOAT, vocabSize, hiddenSize).muli(0.02));

            // Layer 0 attention
            sd.var("model.layers.0.self_attn.q_proj.weight",
                    Nd4j.randn(DataType.FLOAT, hiddenSize, hiddenSize).muli(0.02));
            sd.var("model.layers.0.self_attn.k_proj.weight",
                    Nd4j.randn(DataType.FLOAT, hiddenSize, hiddenSize).muli(0.02));
            sd.var("model.layers.0.self_attn.v_proj.weight",
                    Nd4j.randn(DataType.FLOAT, hiddenSize, hiddenSize).muli(0.02));
            sd.var("model.layers.0.self_attn.o_proj.weight",
                    Nd4j.randn(DataType.FLOAT, hiddenSize, hiddenSize).muli(0.02));

            // Layer 0 MLP
            sd.var("model.layers.0.mlp.gate_proj.weight",
                    Nd4j.randn(DataType.FLOAT, hiddenSize, intermediateSize).muli(0.02));
            sd.var("model.layers.0.mlp.up_proj.weight",
                    Nd4j.randn(DataType.FLOAT, hiddenSize, intermediateSize).muli(0.02));
            sd.var("model.layers.0.mlp.down_proj.weight",
                    Nd4j.randn(DataType.FLOAT, intermediateSize, hiddenSize).muli(0.02));

            // Norms
            sd.var("model.layers.0.input_layernorm.weight",
                    Nd4j.ones(DataType.FLOAT, hiddenSize));
            sd.var("model.layers.0.post_attention_layernorm.weight",
                    Nd4j.ones(DataType.FLOAT, hiddenSize));
            sd.var("model.norm.weight",
                    Nd4j.ones(DataType.FLOAT, hiddenSize));

            // LM head
            sd.var("lm_head.weight",
                    Nd4j.randn(DataType.FLOAT, hiddenSize, vocabSize).muli(0.02));

            // Check if exportable
            boolean canExport = GGMLModelExport.canExport(sd);
            System.out.println("  Model exportable: " + canExport);

            // Detect architecture
            String detectedArch = GGMLModelExport.detectArchitecture(sd);
            System.out.println("  Detected architecture: " + detectedArch);

            // Validate before export
            List<String> issues = GGMLModelExport.validateForExport(sd);
            System.out.println("  Validation issues: " + (issues.isEmpty() ? "none" : issues));

            // Export as Q4_K quantized GGUF
            File outputFile = new File(tempDir, "model-q4k.gguf");
            GGMLModelExport.exportModel(sd, outputFile, ExportOptions.q4k());
            System.out.println("  Exported to: " + outputFile.getName());
            System.out.println("  File size: " + (outputFile.length() / 1024) + " KB");

            // Also export as F16
            File f16File = new File(tempDir, "model-f16.gguf");
            GGMLModelExport.exportModel(sd, f16File, ExportOptions.f16());
            System.out.println("  F16 size:  " + (f16File.length() / 1024) + " KB");
            System.out.println("  Q4_K size: " + (outputFile.length() / 1024) + " KB");
            System.out.println("  Compression ratio: " + String.format("%.1fx",
                    (double) f16File.length() / outputFile.length()));
        }

        // ============================================================
        // 5. LOW-LEVEL GGUF WRITER / READER
        // ============================================================
        System.out.println("\n=== Low-level GGUF Writer / Reader ===");
        {
            File ggufFile = new File(tempDir, "custom.gguf");

            // Write a custom GGUF file
            try (GGUFWriter writer = new GGUFWriter(ggufFile, 3)) {  // version 3
                // Add metadata
                writer.addMetadata("general.architecture", "llama");
                writer.addMetadata("general.name", "example-model");
                writer.addMetadataInt("llama.block_count", 1);
                writer.addMetadataInt("llama.embedding_length", 64);
                writer.addMetadataInt("llama.attention.head_count", 4);
                writer.addMetadataInt("llama.attention.head_count_kv", 4);
                writer.addMetadataInt("llama.context_length", 2048);

                // Register tensors (declares shape and type before writing data)
                writer.registerTensor("token_embd.weight",
                        new long[]{64, 1000}, GGMLDataType.GGML_TYPE_F16);
                writer.registerTensor("blk.0.attn_q.weight",
                        new long[]{64, 64}, GGMLDataType.GGML_TYPE_F16);

                // Write header (locks metadata, must come before tensor data)
                writer.writeHeader();

                // Write tensor data
                byte[] embdData = new byte[64 * 1000 * 2];  // F16 = 2 bytes per element
                writer.writeTensorData("token_embd.weight", embdData);

                byte[] attnData = new byte[64 * 64 * 2];
                writer.writeTensorData("blk.0.attn_q.weight", attnData);

                // Finalize (validates all registered tensors were written)
                writer.finalizeFile();
            }

            System.out.println("  Created GGUF file: " + ggufFile.getName());
            System.out.println("  File size: " + ggufFile.length() + " bytes");

            // Read it back
            try (GGUFReader reader = new GGUFReader(ggufFile)) {
                GGMLMetadata metadata = reader.getMetadata();
                System.out.println("  Architecture: " + metadata.getArchitecture());
                System.out.println("  Model name:   " + metadata.getModelName());
                System.out.println("  Tensors:");
                for (GGMLTensorInfo tensor : metadata.getTensors()) {
                    System.out.println("    " + tensor.getName() +
                            " shape=" + tensor.getShapeString() +
                            " type=" + tensor.getDataType() +
                            " elements=" + tensor.getNumElements());
                }
                System.out.println("  Total parameters: " + metadata.getTotalParameters());
            }
        }

        // ============================================================
        // 6. QUANTIZATION API
        // ============================================================
        System.out.println("\n=== Quantization API ===");
        {
            // Create test data
            float[] data = new float[256]; // must be multiple of block size
            for (int i = 0; i < data.length; i++) {
                data[i] = (float) (Math.random() * 2 - 1); // uniform [-1, 1]
            }

            // Quantization block sizes
            System.out.println("  Quantization type specifications:");
            System.out.println("    Q4_0:  block=32, 18 bytes/block (4-bit, no min)");
            System.out.println("    Q4_1:  block=32, 20 bytes/block (4-bit, with min)");
            System.out.println("    Q5_0:  block=32, 22 bytes/block (5-bit, no min)");
            System.out.println("    Q5_1:  block=32, 24 bytes/block (5-bit, with min)");
            System.out.println("    Q8_0:  block=32, 34 bytes/block (8-bit)");
            System.out.println("    Q4_K:  block=256, 144 bytes/block (k-quant 4-bit)");
            System.out.println("    Q5_K:  block=256, 176 bytes/block (k-quant 5-bit)");
            System.out.println("    Q6_K:  block=256, 210 bytes/block (k-quant 6-bit)");

            // Check available quantizers
            GGMLDataType[] typesToTest = {
                    GGMLDataType.GGML_TYPE_Q4_0,
                    GGMLDataType.GGML_TYPE_Q8_0,
                    GGMLDataType.GGML_TYPE_Q4_K
            };

            for (GGMLDataType type : typesToTest) {
                if (QuantizerFactory.hasQuantizer(type)) {
                    Quantizer quantizer = QuantizerFactory.getQuantizer(type);

                    // Quantize
                    byte[] quantized = quantizer.quantize(data);

                    // Get quality statistics
                    Quantizer.QuantizationStats stats = quantizer.getQuantizationStats(data);

                    // Dequantize
                    Dequantizer dequantizer = DequantizerFactory.getDequantizer(type);
                    float[] recovered = dequantizer.dequantize(quantized, data.length);

                    // Compute error
                    double maxError = 0;
                    for (int i = 0; i < data.length; i++) {
                        maxError = Math.max(maxError, Math.abs(data[i] - recovered[i]));
                    }

                    System.out.println("\n  " + type + ":");
                    System.out.println("    Block size:     " + quantizer.getBlockSize());
                    System.out.println("    Bytes/block:    " + quantizer.getBytesPerBlock());
                    System.out.println("    Compressed:     " + quantized.length + " bytes (from " + (data.length * 4) + ")");
                    System.out.println("    Compression:    " + String.format("%.1fx", (data.length * 4.0) / quantized.length));
                    System.out.println("    Max abs error:  " + String.format("%.6f", maxError));
                    System.out.println("    Stats - min: " + String.format("%.4f", stats.getMinValue()) +
                            ", max: " + String.format("%.4f", stats.getMaxValue()));
                }
            }

            // INDArray-based quantization
            INDArray tensor = Nd4j.rand(DataType.FLOAT, 256);
            Quantizer q4k = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_K);
            byte[] quantizedTensor = q4k.quantize(tensor);
            System.out.println("\n  INDArray quantization: " + tensor.length() + " floats -> " +
                    quantizedTensor.length + " bytes");
        }

        // ============================================================
        // 7. ROUND-TRIP: Export -> Import
        // ============================================================
        System.out.println("\n=== Round-Trip: Export -> Import ===");
        {
            // Create a simple model
            SameDiff original = SameDiff.create();
            original.var("model.embed_tokens.weight",
                    Nd4j.randn(DataType.FLOAT, 500, 64).muli(0.02));
            original.var("model.layers.0.self_attn.q_proj.weight",
                    Nd4j.randn(DataType.FLOAT, 64, 64).muli(0.02));
            original.var("model.layers.0.self_attn.k_proj.weight",
                    Nd4j.randn(DataType.FLOAT, 64, 64).muli(0.02));
            original.var("model.layers.0.self_attn.v_proj.weight",
                    Nd4j.randn(DataType.FLOAT, 64, 64).muli(0.02));
            original.var("model.layers.0.self_attn.o_proj.weight",
                    Nd4j.randn(DataType.FLOAT, 64, 64).muli(0.02));
            original.var("model.layers.0.mlp.gate_proj.weight",
                    Nd4j.randn(DataType.FLOAT, 64, 128).muli(0.02));
            original.var("model.layers.0.mlp.up_proj.weight",
                    Nd4j.randn(DataType.FLOAT, 64, 128).muli(0.02));
            original.var("model.layers.0.mlp.down_proj.weight",
                    Nd4j.randn(DataType.FLOAT, 128, 64).muli(0.02));
            original.var("model.layers.0.input_layernorm.weight",
                    Nd4j.ones(DataType.FLOAT, 64));
            original.var("model.layers.0.post_attention_layernorm.weight",
                    Nd4j.ones(DataType.FLOAT, 64));
            original.var("model.norm.weight",
                    Nd4j.ones(DataType.FLOAT, 64));
            original.var("lm_head.weight",
                    Nd4j.randn(DataType.FLOAT, 64, 500).muli(0.02));

            int originalParams = 0;
            for (String name : original.variableNames()) {
                originalParams += (int) original.getVariable(name).getArr().length();
            }
            System.out.println("  Original model: " + originalParams + " parameters");

            // Export to GGUF (F32 for lossless round-trip)
            File rtFile = new File(tempDir, "roundtrip.gguf");
            GGMLModelExport.exportModel(original, rtFile, ExportOptions.f32());
            System.out.println("  Exported to GGUF: " + (rtFile.length() / 1024) + " KB");

            // Import back
            SameDiff imported = GGMLModelImport.importModel(rtFile);
            int importedParams = 0;
            for (String name : imported.variableNames()) {
                importedParams += (int) imported.getVariable(name).getArr().length();
            }
            System.out.println("  Imported model: " + importedParams + " parameters");
            System.out.println("  Round-trip preserved: " + (originalParams == importedParams));
        }

        // Cleanup
        for (File f : tempDir.listFiles()) {
            f.delete();
        }
        tempDir.delete();

        System.out.println("\nAll GGML/GGUF operations demonstrated successfully.");
    }
}
