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

package org.deeplearning4j.examples.quickstart.modeling.convolution;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Deconvolution and Upsampling Layer Examples.
 *
 * This example demonstrates spatial upsampling methods available in DL4J,
 * commonly used in autoencoders, image generation (GANs), and semantic
 * segmentation (U-Net style architectures).
 *
 * <h3>Deconvolution (Transposed Convolution):</h3>
 * Also known as "Conv2DTranspose". Learnable upsampling that increases spatial
 * dimensions while applying a learned kernel. The transpose of standard convolution.
 * - Deconvolution2D: 2D transposed convolution (standard for images)
 * - Deconvolution3D: 3D transposed convolution (for volumetric data)
 * - Deconvolution1D: 1D transposed convolution (for sequences)
 *
 * <h3>Upsampling:</h3>
 * Non-learnable (parameter-free) spatial expansion via repetition.
 * Simpler and faster than deconvolution but less expressive.
 * - Upsampling1D: Repeat along time/sequence dimension
 * - Upsampling2D: Repeat along height and width (nearest-neighbor)
 * - Upsampling3D: Repeat along depth, height, and width
 *
 * <h3>SpaceToBatch / SpaceToDepth:</h3>
 * - SpaceToBatchLayer: Rearranges spatial blocks into the batch dimension
 * - SpaceToDepthLayer: Rearranges spatial blocks into the depth/channel dimension
 *   (inverse of PixelShuffle / depth-to-space)
 *
 * This example builds a simple autoencoder that encodes MNIST to a latent
 * space and decodes using Deconvolution2D + Upsampling2D.
 */
public class DeconvolutionUpsamplingExample {
    private static final Logger log = LoggerFactory.getLogger(DeconvolutionUpsamplingExample.class);

    public static void main(String[] args) throws Exception {
        int seed = 123;

        // =====================================================================
        // Example 1: Encoder-Decoder with Deconvolution2D
        // =====================================================================
        log.info("=== Example 1: Autoencoder with Deconvolution2D ===");

        // This autoencoder encodes 28x28 MNIST images to a latent space,
        // then decodes back to 28x28 using transposed convolutions.
        MultiLayerConfiguration deconvConf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                // Encoder: 28x28x1 → 14x14x16 → 7x7x32
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nIn(1)
                        .nOut(16)
                        .stride(2, 2)
                        .activation(Activation.RELU)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(32)
                        .stride(2, 2)
                        .activation(Activation.RELU)
                        .build())
                // Decoder: 7x7x32 → 14x14x16 using Deconvolution2D
                // Deconvolution2D is the learned inverse of convolution.
                // It expands spatial dimensions while applying a learned filter.
                .layer(new Deconvolution2D.Builder(3, 3)
                        .nOut(16)
                        .stride(2, 2)
                        .activation(Activation.RELU)
                        .build())
                // 14x14x16 → 28x28x1
                .layer(new Deconvolution2D.Builder(3, 3)
                        .nOut(1)
                        .stride(2, 2)
                        .activation(Activation.SIGMOID)
                        .build())
                // Use CNN loss to compare reconstructed image to input
                .layer(new CnnLossLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .setInputType(InputType.convolutional(28, 28, 1))
                .build();

        MultiLayerNetwork deconvModel = new MultiLayerNetwork(deconvConf);
        deconvModel.init();
        log.info("Deconvolution autoencoder parameters: {}", deconvModel.numParams());

        // Test forward pass
        INDArray testInput = Nd4j.randn(4, 1, 28, 28);
        INDArray reconstructed = deconvModel.output(testInput);
        log.info("Input shape:  {}", java.util.Arrays.toString(testInput.shape()));
        log.info("Output shape: {}", java.util.Arrays.toString(reconstructed.shape()));

        // =====================================================================
        // Example 2: Upsampling layers (parameter-free)
        // =====================================================================
        log.info("=== Example 2: Upsampling2D (parameter-free) ===");

        // Upsampling2D repeats each spatial element. No learned parameters.
        // Useful for simple nearest-neighbor upscaling, often combined with
        // a convolution layer (Upsampling + Conv instead of Deconvolution).
        MultiLayerConfiguration upConf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                // Encoder
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nIn(1)
                        .nOut(16)
                        .stride(2, 2)
                        .activation(Activation.RELU)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(32)
                        .stride(2, 2)
                        .activation(Activation.RELU)
                        .build())
                // Decoder: Upsampling2D + Conv (avoids checkerboard artifacts)
                .layer(new Upsampling2D.Builder(2).build())  // 2x spatial expansion
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(16)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new Upsampling2D.Builder(2).build())  // 2x spatial expansion
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(1)
                        .stride(1, 1)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(new CnnLossLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .setInputType(InputType.convolutional(28, 28, 1))
                .build();

        MultiLayerNetwork upModel = new MultiLayerNetwork(upConf);
        upModel.init();
        log.info("Upsampling autoencoder parameters: {}", upModel.numParams());

        // =====================================================================
        // Example 3: SpaceToDepthLayer
        // =====================================================================
        log.info("=== Example 3: SpaceToDepthLayer ===");

        // SpaceToDepthLayer rearranges spatial blocks into channel dimension.
        // Input [N, C, H, W] → Output [N, C*blockSize^2, H/blockSize, W/blockSize]
        // This is the inverse of PixelShuffle/depth-to-space.
        // Used in YOLO-style object detection for feature map manipulation.
        MultiLayerConfiguration s2dConf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(1e-3))
                .list()
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nIn(1)
                        .nOut(16)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                // blockSize=2: 28x28x16 → 14x14x64
                .layer(new SpaceToDepthLayer.Builder(2,
                        SpaceToDepthLayer.DataFormat.NCHW).build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(32)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.AVG)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(28, 28, 1))
                .build();

        MultiLayerNetwork s2dModel = new MultiLayerNetwork(s2dConf);
        s2dModel.init();
        log.info("SpaceToDepth model parameters: {}", s2dModel.numParams());

        log.info("**************** Deconvolution & Upsampling Example finished ********************");
    }
}
