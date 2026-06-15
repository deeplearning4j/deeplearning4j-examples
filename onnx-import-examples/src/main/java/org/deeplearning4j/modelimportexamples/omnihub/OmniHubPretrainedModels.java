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

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.eclipse.deeplearning4j.omnihub.OmniHubUtils;
import org.eclipse.deeplearning4j.omnihub.models.Pretrained;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * OmniHub Pretrained Model Loading Examples.
 *
 * OmniHub is the DL4J model zoo system that provides easy access to pretrained models.
 * It supports loading models from:
 * 1. The DL4J model zoo (hosted pretrained models in DL4J and SameDiff formats)
 * 2. HuggingFace Hub (GGUF, SafeTensors formats auto-detected)
 *
 * Models are cached locally in the omnihub home directory (configurable via OMNIHUB_HOME
 * environment variable, defaults to ~/.omnihub/).
 *
 * The Pretrained class provides typed access to zoo models:
 * - Pretrained.dl4j()     -> DL4J models (ComputationGraph)
 * - Pretrained.samediff() -> SameDiff models
 *
 * For HuggingFace models, use OmniHubUtils.loadFromHuggingFace() which:
 * - Downloads the model repository (with optional file pattern filtering)
 * - Auto-detects format (GGUF, SafeTensors, etc.)
 * - Converts to SameDiff graph via the AutoModel pipeline
 * - Caches the converted model for fast subsequent loads
 * - Supports private/gated models via HF_TOKEN environment variable
 */
public class OmniHubPretrainedModels {
    private static final Logger log = LoggerFactory.getLogger(OmniHubPretrainedModels.class);

    public static void main(String[] args) throws Exception {

        // =====================================================================
        // Example 1: Load a DL4J ComputationGraph from the model zoo
        // =====================================================================
        log.info("=== Loading VGG19 (no top) from DL4J model zoo ===");

        // Using the typed Pretrained API - downloads on first use, cached afterwards
        ComputationGraph vgg19 = Pretrained.dl4j().vgg19noTop(false);
        log.info("VGG19 loaded. Layers: {}", vgg19.getNumLayers());
        log.info("VGG19 parameters: {}", vgg19.numParams());

        // =====================================================================
        // Example 2: Load a SameDiff model from the model zoo
        // =====================================================================
        log.info("=== Loading ResNet18 from SameDiff model zoo ===");

        // ResNet18 converted from PyTorch via ONNX
        SameDiff resnet18 = Pretrained.samediff().resnet18(false);
        log.info("ResNet18 loaded. Variables: {}", resnet18.variables().size());

        // Run inference with a dummy input (batch=1, channels=3, height=224, width=224)
        INDArray dummyInput = Nd4j.randn(1, 3, 224, 224);
        resnet18.getVariable("input").setArray(dummyInput);
        INDArray output = resnet18.outputSingle(null, "output");
        log.info("ResNet18 output shape: {}", java.util.Arrays.toString(output.shape()));

        // =====================================================================
        // Example 3: Load directly using OmniHubUtils (lower-level API)
        // =====================================================================
        log.info("=== Loading SameDiff model using OmniHubUtils directly ===");

        // Age prediction model (converted from ONNX model zoo)
        SameDiff ageModel = OmniHubUtils.loadSameDiffModel("age_googlenet.fb");
        log.info("Age GoogLeNet loaded. Variables: {}", ageModel.variables().size());

        // =====================================================================
        // Example 4: Load from HuggingFace Hub
        // =====================================================================
        // Note: This requires internet access and may download large files.
        // For gated models, set HF_TOKEN environment variable.
        //
        // Uncomment to run:
        //
        // log.info("=== Loading from HuggingFace ===");
        // SameDiff hfModel = OmniHubUtils.loadFromHuggingFace(
        //     "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",   // HuggingFace model ID
        //     "*.gguf",                                      // File pattern filter
        //     false                                          // Force re-download
        // );
        // log.info("HF model loaded. Variables: {}", hfModel.variables().size());

        log.info("**************** OmniHub Example finished ********************");
    }
}
