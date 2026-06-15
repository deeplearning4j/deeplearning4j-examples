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

package org.deeplearning4j.modelimportexamples.keras.advanced.layertypes;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.autodiff.samediff.SameDiff;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Keras Advanced Layer Import Example.
 *
 * This example documents the comprehensive set of Keras layer types supported
 * by DL4J's Keras model import, including many new layer types added for
 * Transformer and modern architecture support.
 *
 * <h3>Import Methods:</h3>
 * <ul>
 *   <li>{@code KerasModelImport.importKerasModelAndWeights("model.h5")} → ComputationGraph</li>
 *   <li>{@code KerasModelImport.importKerasSequentialModelAndWeights("model.h5")} → MultiLayerNetwork</li>
 *   <li>{@code KerasModelImport.importKerasModelToSameDiff("model.h5")} → SameDiff (NEW)</li>
 *   <li>{@code KerasModelImport.importKerasSequentialModelToSameDiff("model.h5")} → SameDiff (NEW)</li>
 * </ul>
 *
 * <h3>Newly Supported Layer Types:</h3>
 *
 * <b>Attention Layers:</b>
 * <ul>
 *   <li>MultiHeadAttention — Transformer multi-head attention (BERT/GPT style)</li>
 *   <li>Attention — Luong-style scaled dot-product attention</li>
 *   <li>AdditiveAttention — Bahdanau-style additive attention</li>
 * </ul>
 *
 * <b>Normalization Layers:</b>
 * <ul>
 *   <li>LayerNormalization — Per-sample normalization (Transformer standard)</li>
 *   <li>GroupNormalization — Group-based normalization (groups=32 default)</li>
 *   <li>UnitNormalization — L2 unit norm (no trainable parameters)</li>
 *   <li>BatchNormalization — Standard batch normalization</li>
 * </ul>
 *
 * <b>Core Layers:</b>
 * <ul>
 *   <li>EinsumDense — Einstein summation-based dense layer (flexible tensor ops)</li>
 *   <li>Identity — Pass-through layer</li>
 *   <li>RepeatVector — Repeats input N times along new axis</li>
 *   <li>SpatialDropout1D/2D/3D — Structured dropout for spatial data</li>
 * </ul>
 *
 * <b>Convolutional Layers:</b>
 * <ul>
 *   <li>Conv1DTranspose (Deconvolution1D) — 1D transposed convolution</li>
 *   <li>Conv2DTranspose (Deconvolution2D) — 2D transposed convolution</li>
 *   <li>Conv3DTranspose (Deconvolution3D) — 3D transposed convolution</li>
 *   <li>SeparableConv1D — 1D depthwise separable convolution</li>
 *   <li>SeparableConv2D — 2D depthwise separable convolution</li>
 *   <li>DepthwiseConv2D — 2D depthwise convolution</li>
 *   <li>AtrousConv1D/2D — Dilated (atrous) convolution</li>
 * </ul>
 *
 * <b>Embedding Layers:</b>
 * <ul>
 *   <li>Embedding — Standard word embedding lookup</li>
 *   <li>Embedding2D — 2D embedding layer</li>
 * </ul>
 *
 * <b>Spatial Manipulation:</b>
 * <ul>
 *   <li>Upsampling1D/2D/3D — Spatial upsampling</li>
 *   <li>ZeroPadding1D/2D/3D — Zero padding</li>
 *   <li>Cropping1D/2D/3D — Spatial cropping</li>
 *   <li>SpaceToDepth — Rearrange spatial dims to depth</li>
 * </ul>
 *
 * <b>Advanced Activations:</b>
 * <ul>
 *   <li>PReLU — Parametric ReLU (learned slope)</li>
 *   <li>ELU — Exponential Linear Unit</li>
 *   <li>LeakyReLU — Leaky ReLU with fixed slope</li>
 *   <li>ThresholdedReLU — Thresholded ReLU</li>
 * </ul>
 *
 * <b>Noise/Regularization:</b>
 * <ul>
 *   <li>GaussianNoise — Additive Gaussian noise</li>
 *   <li>GaussianDropout — Multiplicative Gaussian noise</li>
 *   <li>AlphaDropout — Dropout for SELU networks</li>
 * </ul>
 *
 * <b>Wrappers:</b>
 * <ul>
 *   <li>Bidirectional — Bidirectional wrapper for RNN layers</li>
 *   <li>TFOpLayer — Raw TF op passthrough (for tf.keras models with custom ops)</li>
 * </ul>
 *
 * <h3>SameDiff Conversion (New Feature):</h3>
 * The importKerasModelToSameDiff() methods convert Keras models first to
 * ComputationGraph, then to SameDiff via ComputationGraphSameDiffConverter.
 * This enables using SameDiff's DSP execution, mixed precision, and LoRA
 * with imported Keras models.
 */
public class KerasAdvancedLayerImportExample {
    private static final Logger log = LoggerFactory.getLogger(KerasAdvancedLayerImportExample.class);

    public static void main(String[] args) throws Exception {

        // =====================================================================
        // 1. Standard Keras import (HDF5 format)
        // =====================================================================
        log.info("=== 1. Keras Model Import Methods ===");

        // Import a Keras Functional model as ComputationGraph:
        // ComputationGraph model = KerasModelImport.importKerasModelAndWeights("model.h5");
        //
        // Import with separate config and weights files:
        // ComputationGraph model = KerasModelImport.importKerasModelAndWeights("model.json", "weights.h5");
        //
        // Import with enforcement of training config (optimizer, loss):
        // ComputationGraph model = KerasModelImport.importKerasModelAndWeights("model.h5", true);
        //
        // Import from InputStream:
        // ComputationGraph model = KerasModelImport.importKerasModelAndWeights(inputStream);

        log.info("  Functional API → ComputationGraph");
        log.info("  Sequential API → MultiLayerNetwork");
        log.info("  Both → SameDiff (new conversion path)");

        // =====================================================================
        // 2. NEW: Import Keras to SameDiff
        // =====================================================================
        log.info("=== 2. Keras → SameDiff Import (NEW) ===");

        // Convert Keras model directly to SameDiff graph.
        // This enables DSP execution, mixed precision, and LoRA fine-tuning
        // on imported Keras models.
        //
        // SameDiff sd = KerasModelImport.importKerasModelToSameDiff("model.h5");
        //
        // Sequential model to SameDiff:
        // SameDiff sd = KerasModelImport.importKerasSequentialModelToSameDiff("model.h5");
        //
        // With separate config/weights:
        // SameDiff sd = KerasModelImport.importKerasModelToSameDiff("model.json", "weights.h5");

        log.info("  importKerasModelToSameDiff(\"model.h5\") → SameDiff");
        log.info("  Uses ComputationGraphSameDiffConverter internally");

        // =====================================================================
        // 3. Attention layer import details
        // =====================================================================
        log.info("=== 3. Attention Layer Import ===");

        // Keras MultiHeadAttention layer:
        //   keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)
        // Maps to: DotProductAttentionVertex in DL4J
        //
        // Keras Attention layer:
        //   keras.layers.Attention(use_scale=True)
        // Maps to: DotProductAttentionLayer
        //
        // Keras AdditiveAttention layer:
        //   keras.layers.AdditiveAttention()
        // Maps to: Additive attention implementation

        log.info("  MultiHeadAttention → DotProductAttentionVertex");
        log.info("  Attention → DotProductAttentionLayer");
        log.info("  AdditiveAttention → Additive attention impl");

        // =====================================================================
        // 4. Normalization layer import details
        // =====================================================================
        log.info("=== 4. Normalization Layer Import ===");

        // Keras LayerNormalization:
        //   keras.layers.LayerNormalization(epsilon=1e-6, axis=-1)
        // Maps to: DL4J LayerNormalization (per-sample, feature-axis)
        //
        // Keras GroupNormalization:
        //   keras.layers.GroupNormalization(groups=32, epsilon=1e-5)
        // Maps to: DL4J GroupNormalization
        //
        // Keras UnitNormalization:
        //   keras.layers.UnitNormalization(axis=-1)
        // Maps to: DL4J UnitNormalization (L2 normalize, no trainable params)

        log.info("  LayerNormalization (per-sample, Transformer standard)");
        log.info("  GroupNormalization (groups=32 default)");
        log.info("  UnitNormalization (L2 norm, no learnable params)");

        // =====================================================================
        // 5. EinsumDense layer import
        // =====================================================================
        log.info("=== 5. EinsumDense Layer Import ===");

        // Keras EinsumDense:
        //   keras.layers.EinsumDense("abc,cd->abd", output_shape=(None, 64))
        // Maps to: DL4J EinsumDense
        //
        // EinsumDense uses Einstein summation notation to define arbitrary
        // linear transformations. This is heavily used in modern Transformer
        // implementations (e.g., T5, PaLM).

        log.info("  EinsumDense supports arbitrary einsum equations");
        log.info("  Used extensively in Transformer architectures");

        // =====================================================================
        // 6. Convolutional layer import details
        // =====================================================================
        log.info("=== 6. Transposed & Separable Convolution Import ===");

        // Keras Conv1DTranspose:
        //   keras.layers.Conv1DTranspose(64, 3, strides=2, padding='same')
        // Maps to: DL4J Deconvolution1D
        //
        // Keras SeparableConv1D:
        //   keras.layers.SeparableConv1D(64, 3, depth_multiplier=1)
        // Maps to: DL4J SeparableConvolution1D (currently approximated)
        //
        // Keras DepthwiseConv2D:
        //   keras.layers.DepthwiseConv2D(3, depth_multiplier=1)
        // Maps to: DL4J DepthwiseConvolution2D

        log.info("  Conv1DTranspose → Deconvolution1D");
        log.info("  SeparableConv1D → SeparableConvolution1D");
        log.info("  DepthwiseConv2D → DepthwiseConvolution2D");

        // =====================================================================
        // 7. TF Op Layer passthrough
        // =====================================================================
        log.info("=== 7. TF Op Layer Import ===");

        // When Keras models contain raw TF ops (e.g., from tf.keras functional API),
        // DL4J uses KerasTFOpLayer to wrap them. This enables import of models
        // that mix Keras layers with raw TensorFlow operations.
        //
        // Example in Keras/Python:
        //   x = tf.keras.layers.Dense(64)(input)
        //   x = tf.nn.gelu(x)  # Raw TF op embedded in Keras model
        //
        // DL4J maps this via TFOpLayerImpl which wraps the raw op.

        log.info("  TFOpLayer handles raw TF ops in tf.keras models");
        log.info("  Enables import of hybrid Keras + TF models");

        // =====================================================================
        // 8. Supported loss functions (TF.Keras format)
        // =====================================================================
        log.info("=== 8. TF.Keras Loss Functions ===");

        // DL4J supports TF.Keras loss name format (no underscores):
        //   "binarycrossentropy" (tf.keras) = "binary_crossentropy" (Keras 2)
        //   "sparsecategoricalcrossentropy"  = "sparse_categorical_crossentropy"
        //   "categoricalcrossentropy"        = "categorical_crossentropy"
        //   "meansquarederror"               = "mean_squared_error"
        //   "meanabsoluteerror"              = "mean_absolute_error"
        //   "huber"                          = "huber_loss"

        log.info("  Supports both Keras 2 and TF.Keras loss name formats");

        log.info("**************** Keras Advanced Layer Import Example finished ********************");
    }
}
