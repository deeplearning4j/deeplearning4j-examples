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

package org.deeplearning4j.examples.advanced.modeling.capsulenet;

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
 * Capsule Network (CapsNet) for MNIST digit classification.
 *
 * Architecture: Conv2D -> PrimaryCapsules -> CapsuleLayer -> CapsuleStrength -> Output
 *
 * This demonstrates the DL4J capsule layer types:
 * - PrimaryCapsules: converts convolutional feature maps into capsule vectors
 * - CapsuleLayer: performs dynamic routing between capsules (Sabour et al., 2017)
 * - CapsuleStrengthLayer: extracts the length (norm) of each capsule vector as class probability
 *
 * PrimaryCapsules takes CNN feature maps and reshapes them into groups of vectors (capsules).
 * Each capsule encodes both the probability of a feature and its instantiation parameters
 * (pose, orientation, etc.). CapsuleLayer then routes lower-level capsules to higher-level
 * ones using an iterative dynamic routing-by-agreement algorithm.
 *
 * Reference: "Dynamic Routing Between Capsules" (Sabour, Frosst, Hinton, 2017)
 */
public class CapsNetMNIST {
    private static final Logger log = LoggerFactory.getLogger(CapsNetMNIST.class);

    public static void main(String[] args) throws Exception {
        int batchSize = 64;
        int nEpochs = 1;
        int seed = 123;

        log.info("Load data...");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);

        log.info("Build CapsNet model...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                // Initial convolution layer to extract low-level features
                .layer(new ConvolutionLayer.Builder(9, 9)
                        .nIn(1)
                        .nOut(256)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                // PrimaryCapsules: convert CNN features into capsule vectors.
                // channels=32 groups of capsules, each with capsuleDimensions=8.
                // A 9x9 conv with stride 2 is applied internally to produce the capsule feature maps.
                .layer(new PrimaryCapsules.Builder(8, 32)
                        .kernelSize(9, 9)
                        .stride(2, 2)
                        .build())
                // CapsuleLayer: 10 output capsules (one per digit class), each 16-dimensional.
                // Dynamic routing with 3 iterations refines the coupling coefficients between
                // primary capsules and digit capsules.
                .layer(new CapsuleLayer.Builder(10, 16, 3).build())
                // CapsuleStrengthLayer: compute the L2 norm of each capsule vector.
                // The resulting scalar per capsule represents the probability that the
                // corresponding entity (digit class) is present.
                .layer(new CapsuleStrengthLayer.Builder().build())
                // Standard output layer on top of capsule strengths
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

        log.info("**************** CapsNet Example finished ********************");
    }
}
