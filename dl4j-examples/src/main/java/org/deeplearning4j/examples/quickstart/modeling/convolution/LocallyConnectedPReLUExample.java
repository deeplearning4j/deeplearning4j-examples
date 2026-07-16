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
 * LocallyConnected2D and PReLU Layer Example.
 *
 * <h3>LocallyConnected2D:</h3>
 * Like Conv2D but with separate (unshared) weights at each spatial position.
 * In standard convolution, the same filter is applied everywhere (weight sharing).
 * LocallyConnected uses different weights at each spatial location.
 *
 * Use cases:
 * - When spatial invariance is NOT desired (e.g., face recognition where
 *   different regions contain semantically different features)
 * - DeepFace-style architectures
 * - More parameters than Conv2D but can capture position-specific patterns
 *
 * <h3>PReLU (Parametric ReLU):</h3>
 * Generalization of LeakyReLU where the negative slope is learned during training.
 * f(x) = max(0, x) + alpha * min(0, x)
 * where alpha is a learnable parameter (initialized to 0.25 by default).
 *
 * PReLU can share a single alpha across all channels or learn per-channel alphas,
 * controlled by the sharedAxes parameter.
 *
 * Reference: "Delving Deep into Rectifiers" by He et al. (2015)
 */
public class LocallyConnectedPReLUExample {
    private static final Logger log = LoggerFactory.getLogger(LocallyConnectedPReLUExample.class);

    public static void main(String[] args) throws Exception {
        int batchSize = 64;
        int nEpochs = 1;
        int seed = 123;

        log.info("Load data...");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);

        // =====================================================================
        // Build model with LocallyConnected2D and PReLU
        // =====================================================================
        log.info("Build model with LocallyConnected2D and PReLU...");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.RELU)
                .updater(new Adam(1e-3))
                .list()
                // Standard convolution for initial feature extraction
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(1)
                        .nOut(16)
                        .stride(2, 2)
                        .activation(Activation.IDENTITY)
                        .build())
                // PReLU: learned activation slope for negative values
                // sharedAxes controls parameter sharing:
                //   [1] = per-channel alpha (most common)
                //   [] (empty) = per-element alpha (most parameters)
                .layer(new PReLULayer.Builder()
                        .sharedAxes(1)    // Share alpha across spatial dims, separate per channel
                        .build())
                // LocallyConnected2D: separate weights at each spatial position.
                // Same builder API as ConvolutionLayer but unshared weights.
                // kernelSize, stride, padding, nIn, nOut all work the same way.
                .layer(new LocallyConnected2D.Builder()
                        .nIn(16)
                        .nOut(32)
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .setInputSize(12, 12)  // Must specify spatial input size
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new PReLULayer.Builder()
                        .sharedAxes(1)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
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

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Number of parameters: {}", model.numParams());
        log.info("(Note: LocallyConnected has more parameters than Conv2D due to unshared weights)");

        log.info("Train model...");
        model.setListeners(new ScoreIterationListener(50),
                new EvaluativeListener(mnistTest, 1, InvocationType.EPOCH_END));
        model.fit(mnistTrain, nEpochs);

        log.info("**************** LocallyConnected & PReLU Example finished ********************");
    }
}
