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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * TorchScript Model Import Example.
 *
 * DL4J can import TorchScript models (.pt files created by torch.jit.script
 * or torch.jit.trace) and convert them to SameDiff computation graphs.
 *
 * <h3>TorchScript Import API:</h3>
 * <pre>
 * // Basic import
 * SameDiff sd = TorchScriptModelImport.importModel("model.pt");
 *
 * // Import with options
 * SameDiff sd = TorchScriptModelImport.importModel(file, options);
 *
 * // Convert to SDZ format (for fast loading)
 * TorchScriptModelImport.convertToSDZ("model.pt", "model.sdz");
 *
 * // Inspect without loading weights
 * TorchScriptMetadata meta = TorchScriptModelImport.inspectModel("model.pt");
 *
 * // Format detection
 * TorchScriptFormat fmt = TorchScriptModelImport.detectFormat(file);
 * boolean ok = TorchScriptModelImport.isSupportedFile(file);
 * </pre>
 *
 * <h3>Supported Operations:</h3>
 * The importer handles standard PyTorch operations including:
 * - Linear layers (nn.Linear)
 * - Convolutions (nn.Conv1d/2d/3d)
 * - Normalization (nn.BatchNorm, nn.LayerNorm, nn.GroupNorm)
 * - Activations (ReLU, GELU, SiLU, Softmax, etc.)
 * - Pooling (MaxPool, AvgPool, AdaptiveAvgPool)
 * - Attention (nn.MultiheadAttention)
 * - Recurrent (nn.LSTM, nn.GRU)
 * - Elementwise ops, reshape, transpose, etc.
 *
 * <h3>Model Preparation (Python side):</h3>
 * <pre>
 * # Option 1: torch.jit.trace (for models without control flow)
 * traced_model = torch.jit.trace(model, example_inputs)
 * traced_model.save("model.pt")
 *
 * # Option 2: torch.jit.script (for models with control flow)
 * scripted_model = torch.jit.script(model)
 * scripted_model.save("model.pt")
 * </pre>
 *
 * <h3>Import Pipeline Comparison:</h3>
 * <pre>
 * Format        | Import Path                                  | Best For
 * --------------|----------------------------------------------|------------------
 * GGUF          | GGMLModelImport.importModel(file)            | Quantized LLMs
 * SafeTensors   | AutoModel.fromPretrained(dir)                | HuggingFace models
 * TorchScript   | TorchScriptModelImport.importModel(file)     | PyTorch models
 * ONNX          | SameDiff via samediff-import-onnx             | Cross-framework
 * Keras HDF5    | KerasModelImport.importKerasModelAndWeights() | Keras/TF models
 * SDZ           | SameDiff.load(file)                          | Native DL4J
 * </pre>
 *
 * NOTE: This example requires a TorchScript model file (.pt).
 * Create one in Python using torch.jit.trace or torch.jit.script.
 */
public class TorchScriptImportExample {
    private static final Logger log = LoggerFactory.getLogger(TorchScriptImportExample.class);

    public static void main(String[] args) {
        log.info("=== TorchScript Model Import API Reference ===");
        log.info("");

        // =====================================================================
        // 1. Basic import
        // =====================================================================
        log.info("--- 1. Basic Import ---");
        log.info("  SameDiff sd = TorchScriptModelImport.importModel(\"model.pt\");");
        log.info("  SameDiff sd = TorchScriptModelImport.importModel(file, options);");
        log.info("");

        // =====================================================================
        // 2. Format conversion
        // =====================================================================
        log.info("--- 2. Format Conversion ---");
        log.info("  TorchScriptModelImport.convertToSDZ(\"model.pt\", \"model.sdz\");");
        log.info("  → SDZ loads 10-100x faster than re-importing TorchScript");
        log.info("");

        // =====================================================================
        // 3. Inspection (no weight loading)
        // =====================================================================
        log.info("--- 3. Inspection ---");
        log.info("  TorchScriptMetadata meta = TorchScriptModelImport.inspectModel(\"model.pt\");");
        log.info("  TorchScriptFormat fmt = TorchScriptModelImport.detectFormat(file);");
        log.info("  boolean ok = TorchScriptModelImport.isSupportedFile(file);");
        log.info("");

        // =====================================================================
        // 4. Complete import workflow
        // =====================================================================
        log.info("--- 4. Typical Workflow ---");
        log.info("  1. Export from Python: torch.jit.trace(model, input).save(\"model.pt\")");
        log.info("  2. Import in Java: SameDiff sd = TorchScriptModelImport.importModel(\"model.pt\")");
        log.info("  3. Convert for fast loading: TorchScriptModelImport.convertToSDZ(...)");
        log.info("  4. Run inference: INDArray out = sd.outputSingle(inputs, \"output\")");
        log.info("  5. Optionally apply LoRA: attach LoraConfig for fine-tuning");
        log.info("");

        // =====================================================================
        // 5. All supported import formats summary
        // =====================================================================
        log.info("--- 5. All Import Formats ---");
        log.info("  GGUF         → GGMLModelImport (quantized LLMs, llama.cpp compatible)");
        log.info("  SafeTensors  → AutoModel.fromPretrained (HuggingFace standard)");
        log.info("  TorchScript  → TorchScriptModelImport (PyTorch .pt files)");
        log.info("  ONNX         → samediff-import-onnx (cross-framework standard)");
        log.info("  Keras HDF5   → KerasModelImport (Keras 2 .h5 files)");
        log.info("  SDZ          → SameDiff.load (native format, fastest loading)");

        log.info("**************** TorchScript Import Example finished ********************");
    }
}
