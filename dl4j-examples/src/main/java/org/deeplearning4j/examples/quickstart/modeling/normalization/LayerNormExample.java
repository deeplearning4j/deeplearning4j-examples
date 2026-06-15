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

package org.deeplearning4j.examples.quickstart.modeling.normalization;

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
 * Layer Normalization example on MNIST.
 *
 * This example demonstrates the use of LayerNormalization as an alternative
 * to BatchNormalization. While BatchNorm normalizes across the batch dimension
 * (computing mean/variance over the batch for each feature), LayerNorm normalizes
 * across the feature dimension per individual sample.
 *
 * Key differences from BatchNormalization:
 * - LayerNorm normalizes per sample, not across the batch
 * - No running statistics are needed (no separate train/eval behavior)
 * - Works well with small batch sizes and variable-length sequences
 * - Widely used in Transformer architectures (BERT, GPT, etc.)
 *
 * Formula: y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
 * where mean and variance are computed across the feature dimension for each sample.
 *
 * Parameters:
 * - gamma (scale): learned gain, initialized to 1.0
 * - beta (center): learned bias, initialized to 0.0
 * - epsilon: small constant for numerical stability (default 1e-5)
 *
 * Reference: "Layer Normalization" by Ba, Kiros, and Hinton (2016)
 */
public class LayerNormExample {
    private static final Logger log = LoggerFactory.getLogger(LayerNormExample.class);

    public static void main(String[] args) throws Exception {
        int batchSize = 64;
        int nEpochs = 1;
        int seed = 123;

        log.info("Load data...");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);

        log.info("Build model with LayerNormalization...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(784)
                        .nOut(256)
                        .activation(Activation.IDENTITY) // Apply activation after normalization
                        .build())
                // LayerNormalization normalizes across the 256 features for each sample independently.
                // normalizedShape is inferred from the previous layer output when not set explicitly.
                .layer(new LayerNormalization.Builder()
                        .epsilon(1e-5)
                        .build())
                .layer(new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new LayerNormalization.Builder()
                        .epsilon(1e-5)
                        .build())
                .layer(new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.feedForward(784))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Number of parameters: {}", model.numParams());

        log.info("Train model...");
        model.setListeners(new ScoreIterationListener(50),
                new EvaluativeListener(mnistTest, 1, InvocationType.EPOCH_END));
        model.fit(mnistTrain, nEpochs);

        log.info("**************** Layer Normalization Example finished ********************");
    }
}
