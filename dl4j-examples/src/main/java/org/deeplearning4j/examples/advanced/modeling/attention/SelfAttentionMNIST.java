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

package org.deeplearning4j.examples.advanced.modeling.attention;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
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
 * Self-Attention layer example for MNIST classification.
 *
 * This example treats each MNIST image as a sequence of 28 rows (time steps),
 * each with 28 features (pixels per row), and applies multi-head self-attention
 * to capture dependencies between rows.
 *
 * Architecture: LSTM -> SelfAttention (multi-head) -> LastTimeStep -> Dense -> Output
 *
 * The SelfAttentionLayer implements scaled dot-product multi-head attention
 * (Vaswani et al., "Attention Is All You Need", 2017). When projectInput=true,
 * learned projection matrices Wq, Wk, Wv, Wo are used to project the input into
 * query, key, and value spaces across multiple heads, enabling the model to attend
 * to different representation subspaces at different positions.
 *
 * Key parameters:
 * - nHeads: number of parallel attention heads
 * - nIn/nOut: input/output feature dimensions
 * - projectInput: whether to use learned Q/K/V projections (required for multi-head)
 * - scaled: whether to scale attention scores by 1/sqrt(headSize) for stable gradients
 */
public class SelfAttentionMNIST {
    private static final Logger log = LoggerFactory.getLogger(SelfAttentionMNIST.class);

    public static void main(String[] args) throws Exception {
        int batchSize = 64;
        int nEpochs = 1;
        int seed = 123;
        int hiddenSize = 64;
        int nHeads = 4;

        log.info("Load data...");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);

        log.info("Build self-attention model...");
        // We treat each MNIST image (28x28) as a sequence: 28 time steps of 28 features.
        // InputType.recurrent(28, 28) is set via the feedforward->recurrent preprocessor
        // triggered by setInputType below.
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                // LSTM to encode sequential features
                .layer(new LSTM.Builder()
                        .nOut(hiddenSize)
                        .activation(Activation.TANH)
                        .build())
                // Multi-head self-attention: each input position attends to all other positions.
                // projectInput=true enables learned Q/K/V/O projection matrices.
                // nHeads=4 with nOut=64 gives headSize=16 per head.
                .layer(new SelfAttentionLayer.Builder()
                        .nOut(hiddenSize)
                        .nHeads(nHeads)
                        .projectInput(true)
                        .scale(true)
                        .build())
                // Take the last time step output as a fixed-length representation
                .layer(new LastTimeStep(new LSTM.Builder()
                        .nOut(hiddenSize)
                        .activation(Activation.TANH)
                        .build()))
                .layer(new DenseLayer.Builder()
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build())
                // Treat 784 flat pixels as a sequence of 28 time steps x 28 features
                .setInputType(InputType.recurrent(28, 28))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Number of parameters: {}", model.numParams());

        log.info("Train model...");
        model.setListeners(new ScoreIterationListener(50),
                new EvaluativeListener(mnistTest, 1, InvocationType.EPOCH_END));
        model.fit(mnistTrain, nEpochs);

        log.info("**************** Self-Attention Example finished ********************");
    }
}
