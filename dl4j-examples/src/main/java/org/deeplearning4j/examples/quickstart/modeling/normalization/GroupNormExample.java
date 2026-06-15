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
 * Group Normalization example on MNIST.
 *
 * Group Normalization divides channels into groups and normalizes within each
 * group. Unlike BatchNormalization, it doesn't depend on batch statistics,
 * making it effective with:
 * - Small batch sizes (where BatchNorm statistics are unreliable)
 * - Online learning (single-sample updates)
 * - Object detection / segmentation tasks
 *
 * GroupNorm is a generalization:
 * - groups = nChannels → equivalent to InstanceNormalization
 * - groups = 1       → equivalent to LayerNormalization
 * - groups = 32      → standard GroupNorm (default, as used in ResNeXt/EfficientNet)
 *
 * For input [batch, channels, ...], each group normalizes channels/groups
 * features: y = gamma * (x - mean_g) / sqrt(var_g + eps) + beta
 *
 * Reference: "Group Normalization" by Wu and He (2018)
 */
public class GroupNormExample {
    private static final Logger log = LoggerFactory.getLogger(GroupNormExample.class);

    public static void main(String[] args) throws Exception {
        int batchSize = 64;
        int nEpochs = 1;
        int seed = 123;

        log.info("Load data...");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);

        log.info("Build model with GroupNormalization...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                // Conv block 1 with GroupNorm
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(1)
                        .nOut(32)
                        .stride(1, 1)
                        .activation(Activation.IDENTITY)
                        .build())
                // GroupNorm with 8 groups over 32 channels = 4 channels per group
                .layer(new GroupNormalization.Builder()
                        .nChannels(32)
                        .groups(8)
                        .epsilon(1e-5)
                        .build())
                .layer(new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Conv block 2 with GroupNorm
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nOut(64)
                        .stride(1, 1)
                        .activation(Activation.IDENTITY)
                        .build())
                // GroupNorm with 16 groups over 64 channels = 4 channels per group
                .layer(new GroupNormalization.Builder()
                        .nChannels(64)
                        .groups(16)
                        .epsilon(1e-5)
                        .build())
                .layer(new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Dense + output
                .layer(new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
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

        log.info("**************** Group Normalization Example finished ********************");
    }
}
