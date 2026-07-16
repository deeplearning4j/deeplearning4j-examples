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

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * MobileNet-style Depthwise Separable Convolution for MNIST classification.
 *
 * This example demonstrates the use of DepthwiseConvolution2D and SeparableConvolution2D layers,
 * which are the building blocks of efficient architectures like MobileNet (Howard et al., 2017).
 *
 * Standard convolution applies a filter across ALL input channels simultaneously.
 * Depthwise separable convolution factorizes this into two steps:
 *
 * 1. DepthwiseConvolution2D: applies a separate spatial filter to each input channel independently.
 *    With depthMultiplier=1, the output has the same number of channels as the input.
 *    This captures spatial patterns within each feature map.
 *
 * 2. A standard 1x1 convolution (pointwise): linearly combines the depthwise outputs
 *    across channels to produce the final output feature maps.
 *
 * SeparableConvolution2D combines both steps into one layer (depthwise + pointwise internally).
 *
 * This approach dramatically reduces parameters and computation:
 * Standard conv: nIn * nOut * kH * kW parameters
 * Separable conv: nIn * kH * kW + nIn * nOut parameters (much fewer when kH*kW is large)
 */
public class DepthwiseSeparableConvMNIST {
    private static final Logger log = LoggerFactory.getLogger(DepthwiseSeparableConvMNIST.class);

    public static void main(String[] args) throws Exception {
        int batchSize = 64;
        int nEpochs = 1;
        int seed = 123;

        log.info("Load data...");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);

        log.info("Build model with depthwise separable convolutions...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                // Initial standard convolution to expand from 1 channel to 32
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nIn(1)
                        .nOut(32)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                // SeparableConvolution2D: performs depthwise conv + pointwise conv in one layer.
                // Internally: first applies a separate kxk filter per input channel (depthwise),
                // then a 1x1 conv to mix channels (pointwise) producing nOut feature maps.
                .layer(new SeparableConvolution2D.Builder(3, 3)
                        .nOut(64)
                        .stride(1, 1)
                        .depthMultiplier(1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Another separable conv block
                .layer(new SeparableConvolution2D.Builder(3, 3)
                        .nOut(128)
                        .stride(1, 1)
                        .depthMultiplier(1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Global average pooling reduces spatial dimensions to 1x1
                .layer(new GlobalPoolingLayer.Builder(PoolingType.AVG).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Number of parameters: {}", model.numParams());

        log.info("Train model...");
        model.setListeners(new ScoreIterationListener(50),
                new EvaluativeListener(mnistTest, 1, InvocationType.EPOCH_END));
        model.fit(mnistTrain, nEpochs);

        log.info("**************** Depthwise Separable Conv Example finished ********************");
    }
}
