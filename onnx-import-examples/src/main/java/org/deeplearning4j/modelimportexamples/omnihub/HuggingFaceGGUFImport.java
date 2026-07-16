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

import org.eclipse.deeplearning4j.omnihub.HuggingFaceHubDownloader;
import org.eclipse.deeplearning4j.omnihub.OmniHubUtils;
import org.eclipse.deeplearning4j.pipeline.AutoModel;
import org.eclipse.deeplearning4j.pipeline.ModelManifest;
import org.eclipse.deeplearning4j.pipeline.PipelineLoader;
import org.nd4j.autodiff.samediff.SameDiff;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * GGUF Model Import from HuggingFace Hub.
 *
 * This example demonstrates how to download and load GGUF (GGML Universal Format)
 * models from HuggingFace Hub into the DL4J/SameDiff ecosystem.
 *
 * GGUF is the successor to GGML format, widely used for quantized LLMs (e.g., via llama.cpp).
 * DL4J can import GGUF models and convert them to SameDiff computation graphs.
 *
 * The pipeline:
 * 1. HuggingFaceHubDownloader downloads the model repository (with glob filtering)
 * 2. AutoModel.fromPretrained() auto-detects the format and loads via the appropriate PipelineLoader
 * 3. For GGUF, weights are dequantized and the model is built as a SameDiff graph
 * 4. The converted model is optionally cached in SDZ format for fast subsequent loads
 *
 * Supported model formats:
 * - GGUF (.gguf) - GGML Universal Format (quantized LLMs)
 * - SafeTensors (.safetensors) - HuggingFace SafeTensors format
 * - ONNX (.onnx) - Open Neural Network Exchange
 * - PyTorch (.bin, .pt, .pth) - PyTorch pickle format
 * - SDZ (.sdz) - Native SameDiff ZIP format
 *
 * Authentication: Set the HF_TOKEN environment variable for private/gated models.
 *
 * NOTE: This example downloads large model files. Ensure you have sufficient disk
 * space and a stable internet connection.
 */
public class HuggingFaceGGUFImport {
    private static final Logger log = LoggerFactory.getLogger(HuggingFaceGGUFImport.class);

    public static void main(String[] args) throws Exception {
        // =====================================================================
        // Example 1: Download and inspect a GGUF model repository
        // =====================================================================
        log.info("=== Step 1: Download GGUF model from HuggingFace ===");

        String modelId = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";
        String filePattern = "*.Q4_K_M.gguf"; // Only download the Q4_K_M quantized variant

        // Download the repository - files are cached to $OMNIHUB_HOME/huggingface/
        File modelDir = HuggingFaceHubDownloader.downloadRepo(
                modelId,
                filePattern,  // glob pattern to filter files
                false,        // don't force re-download if cached
                "main"        // git revision (branch/tag)
        );
        log.info("Model downloaded to: {}", modelDir.getAbsolutePath());

        // =====================================================================
        // Example 2: Inspect model format using AutoModel
        // =====================================================================
        log.info("=== Step 2: Inspect model manifest ===");

        ModelManifest manifest = AutoModel.inspect(modelDir);
        log.info("Model format: {}", manifest.getFormat());
        log.info("Model type: {}", manifest.getType());
        log.info("Model directory: {}", manifest.getModelDirectory());

        // =====================================================================
        // Example 3: Load the model as a SameDiff graph
        // =====================================================================
        log.info("=== Step 3: Load model into SameDiff ===");

        // Configure loading options
        PipelineLoader.LoadConfig config = PipelineLoader.LoadConfig.builder()
                .dataType("float32")          // Convert weights to float32
                .device("cpu")                // Target device
                .useMmap(true)                // Memory-map large files
                .cacheConvertedModel(true)     // Cache the SDZ conversion
                .dequantize(true)             // Dequantize quantized weights
                .build();

        SameDiff model = AutoModel.fromPretrained(modelDir, config);
        log.info("Model loaded successfully!");
        log.info("Number of variables: {}", model.variables().size());

        // =====================================================================
        // Example 4: Shortcut - load directly via OmniHubUtils
        // =====================================================================
        // The simplest API - combines download + auto-detection + loading:
        //
        // SameDiff model = OmniHubUtils.loadFromHuggingFace(modelId, filePattern, false);

        log.info("**************** GGUF Import Example finished ********************");
    }
}
