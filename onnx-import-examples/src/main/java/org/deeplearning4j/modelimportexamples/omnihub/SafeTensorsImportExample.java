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

import org.eclipse.deeplearning4j.pipeline.AutoModel;
import org.eclipse.deeplearning4j.pipeline.ModelManifest;
import org.eclipse.deeplearning4j.pipeline.PipelineLoader;
import org.eclipse.deeplearning4j.safetensors.SafeTensorsPipelineLoader;
import org.eclipse.deeplearning4j.safetensors.SafeTensorsReader;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Map;

/**
 * SafeTensors Model Import Example.
 *
 * SafeTensors is HuggingFace's recommended format for storing model weights.
 * It is a simple, safe format that supports zero-copy deserialization and
 * memory-mapped I/O, making it fast to load.
 *
 * This example demonstrates three levels of SafeTensors loading:
 *
 * 1. AutoModel.fromPretrained() — Highest-level API. Auto-detects format,
 *    builds the SameDiff graph from config.json, and loads weights.
 *
 * 2. SafeTensorsPipelineLoader — Mid-level API. Direct loader that handles
 *    multi-shard files and caching. Use when you need more control.
 *
 * 3. SafeTensorsReader — Low-level API. Reads raw tensors from a .safetensors
 *    file as INDArray maps. Use when you need direct tensor access.
 *
 * SafeTensors features:
 * - Zero-copy memory mapping for fast loading
 * - Multi-file shard support (model-00001-of-00003.safetensors, etc.)
 * - Automatic float32 conversion option
 * - SDZ caching for fast subsequent loads
 *
 * NOTE: This example requires SafeTensors model files downloaded from HuggingFace.
 */
public class SafeTensorsImportExample {
    private static final Logger log = LoggerFactory.getLogger(SafeTensorsImportExample.class);

    public static void main(String[] args) throws Exception {

        // =====================================================================
        // 1. High-level: AutoModel (recommended)
        // =====================================================================
        log.info("=== 1. AutoModel — auto-detect and load ===");

        // AutoModel auto-detects the model format from files in the directory.
        // It looks for config.json, *.safetensors, *.gguf, etc.
        // The simplest way to load any HuggingFace model:
        //
        // File modelDir = new File("/path/to/hf-model-dir");
        // SameDiff model = AutoModel.fromPretrained(modelDir);
        //
        // With explicit config:
        // PipelineLoader.LoadConfig config = PipelineLoader.LoadConfig.builder()
        //     .dataType("float32")
        //     .device("cpu")
        //     .cacheConvertedModel(true)
        //     .build();
        // SameDiff model = AutoModel.fromPretrained(modelDir, config);

        log.info("  AutoModel.fromPretrained(dir) — auto-detects SafeTensors, GGUF, SDZ");
        log.info("  AutoModel.inspect(dir) — returns ModelManifest with format info");

        // =====================================================================
        // 2. Mid-level: SafeTensorsPipelineLoader
        // =====================================================================
        log.info("=== 2. SafeTensorsPipelineLoader — direct loader ===");

        // The SafeTensorsPipelineLoader implements PipelineLoader for SafeTensors.
        // It handles multi-shard files and optional SDZ caching.
        //
        // SafeTensorsPipelineLoader loader = new SafeTensorsPipelineLoader();
        //
        // Load a model:
        // SameDiff sd = loader.loadModel(file, config);
        //
        // Load with manifest:
        // ModelManifest manifest = AutoModel.inspect(modelDir);
        // SameDiff sd = loader.loadModel(manifest, config);
        //
        // Load a multi-component pipeline:
        // Map<String, SameDiff> components = loader.loadPipeline(manifest, config);

        log.info("  SafeTensorsPipelineLoader handles multi-file shards");
        log.info("  Supports cacheConvertedModel for SDZ caching");

        // =====================================================================
        // 3. Low-level: SafeTensorsReader — raw tensor access
        // =====================================================================
        log.info("=== 3. SafeTensorsReader — raw tensor access ===");

        // SafeTensorsReader reads individual tensors from .safetensors files.
        // Use this when you need direct access to weight tensors.
        //
        // Single file:
        // Map<String, INDArray> weights = SafeTensorsReader.loadFile(file);
        //
        // Multiple shards:
        // Map<String, INDArray> allWeights = SafeTensorsReader.loadFiles(shardFiles);
        //
        // With reader instance (for header inspection):
        // SafeTensorsReader reader = SafeTensorsReader.open(file);
        // Map<String, INDArray> tensors = reader.readAllTensors();
        //
        // Inspect file header without loading weights:
        // SafeTensorsHeader header = SafeTensorsPipelineLoader.inspectFile(file);

        log.info("  SafeTensorsReader.loadFile(file) → Map<String, INDArray>");
        log.info("  SafeTensorsReader.loadFiles(files) → merged Map<String, INDArray>");
        log.info("  SafeTensorsPipelineLoader.inspectFile(file) → SafeTensorsHeader");

        // =====================================================================
        // 4. Loading weights into existing SameDiff graph
        // =====================================================================
        log.info("=== 4. Weight loading workflow ===");

        // A common workflow: define your own SameDiff graph and load pretrained weights.
        //
        // SameDiff sd = buildMyGraph();
        // Map<String, INDArray> weights = SafeTensorsReader.loadFile(new File("model.safetensors"));
        // for (Map.Entry<String, INDArray> entry : weights.entrySet()) {
        //     String name = entry.getKey();
        //     INDArray weight = entry.getValue();
        //     if (sd.hasVariable(name)) {
        //         sd.getVariable(name).setArray(weight);
        //     }
        // }

        log.info("  Load weights from SafeTensors into custom SameDiff graphs");
        log.info("  Useful for transfer learning or custom architectures");

        log.info("**************** SafeTensors Import Example finished ********************");
    }
}
