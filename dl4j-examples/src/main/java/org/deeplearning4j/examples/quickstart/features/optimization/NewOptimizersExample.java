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

package org.deeplearning4j.examples.quickstart.features.optimization;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * New Optimizers: AdaBelief and Adam8bit.
 *
 * DL4J now includes two notable new optimizers beyond the standard set:
 *
 * <h3>AdaBelief (arXiv:2010.07468):</h3>
 * Modifies Adam by using the "belief" in the gradient — the deviation of the
 * actual gradient from the predicted gradient. This leads to:
 * - Faster convergence than Adam on many tasks
 * - Better generalization (comparable to SGD+momentum)
 * - Good default: beta1=0.9, beta2=0.999, epsilon=1e-14
 *
 * <h3>Adam8bit (bitsandbytes-style):</h3>
 * Quantizes optimizer state (momentum + variance) to INT8 using block-wise
 * quantization (per-block absmax scaling). Benefits:
 * - ~4x reduction in optimizer state memory
 * - Enables training larger models with limited GPU memory
 * - Minimal quality loss due to block-wise dequantization during updates
 * - Configurable: blockSize (default 2048), pagedOptimizer (for CPU offload)
 *
 * <h3>Full Optimizer List:</h3>
 * <pre>
 * Optimizer     | Description
 * --------------|----------------------------------------------------
 * Adam          | Standard Adam (Kingma & Ba, 2014)
 * AdaBelief     | NEW — Gradient-belief adaptive learning rate
 * Adam8bit      | NEW — 8-bit quantized Adam (bitsandbytes-style)
 * AMSGrad       | Adam variant with long-term memory of squared gradients
 * AdaDelta      | Adaptive learning rate (Zeiler, 2012)
 * AdaGrad       | Adaptive gradient accumulation (Duchi et al., 2011)
 * AdaMax        | Adam variant using infinity norm
 * Nadam         | Adam + Nesterov momentum
 * Nesterovs     | SGD + Nesterov momentum
 * RmsProp       | Root mean square propagation (Hinton)
 * Sgd           | Stochastic gradient descent
 * NoOp          | No-op updater (external optimization)
 * </pre>
 */
public class NewOptimizersExample {
    private static final Logger log = LoggerFactory.getLogger(NewOptimizersExample.class);

    public static void main(String[] args) throws Exception {
        int batchSize = 64;
        int nEpochs = 1;
        int seed = 123;

        log.info("Load data...");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);

        // =====================================================================
        // 1. AdaBelief Optimizer
        // =====================================================================
        log.info("=== 1. Training with AdaBelief ===");

        // AdaBelief adapts the learning rate based on the "belief" in the gradient.
        // When the observed gradient matches the predicted gradient (EMA), the
        // update is larger. When there's high uncertainty, the update is conservative.
        AdaBelief adaBelief = new AdaBelief(1e-3);  // Default: beta1=0.9, beta2=0.999, eps=1e-14

        MultiLayerConfiguration adaBeliefConf = buildMnistConfig(seed, adaBelief);
        MultiLayerNetwork adaBeliefModel = new MultiLayerNetwork(adaBeliefConf);
        adaBeliefModel.init();
        adaBeliefModel.setListeners(new ScoreIterationListener(100));

        log.info("Training with AdaBelief (lr=1e-3)...");
        adaBeliefModel.fit(mnistTrain, nEpochs);
        log.info("AdaBelief final score: {}", adaBeliefModel.score());

        // =====================================================================
        // 2. Adam8bit Optimizer (quantized optimizer state)
        // =====================================================================
        log.info("=== 2. Training with Adam8bit ===");

        // Adam8bit quantizes optimizer state to INT8 using block-wise quantization.
        // This reduces optimizer memory by ~4x, enabling training of larger models.
        //
        // The quantization uses per-block absmax scaling:
        //   - Divide state into blocks of 'blockSize' elements
        //   - For each block, compute absmax and scale to INT8 range
        //   - During update, dequantize per-block before applying
        Adam8bit adam8bit = new Adam8bit(1e-3);     // Default: blockSize=2048

        MultiLayerConfiguration adam8bitConf = buildMnistConfig(seed, adam8bit);
        MultiLayerNetwork adam8bitModel = new MultiLayerNetwork(adam8bitConf);
        adam8bitModel.init();
        adam8bitModel.setListeners(new ScoreIterationListener(100));

        // Reset iterator
        mnistTrain.reset();

        log.info("Training with Adam8bit (lr=1e-3, blockSize=2048)...");
        adam8bitModel.fit(mnistTrain, nEpochs);
        log.info("Adam8bit final score: {}", adam8bitModel.score());

        // =====================================================================
        // 3. Comparison summary
        // =====================================================================
        log.info("=== Optimizer Memory Comparison ===");
        long numParams = adaBeliefModel.numParams();
        log.info("Model parameters: {}", numParams);
        log.info("  Adam     state: {} bytes (2x FP32 = 8 bytes/param)", numParams * 8);
        log.info("  Adam8bit state: {} bytes (2x INT8 = 2 bytes/param)", numParams * 2);
        log.info("  Memory savings: ~4x reduction in optimizer state");

        log.info("**************** New Optimizers Example finished ********************");
    }

    /**
     * Builds a standard MNIST classification config with the specified updater.
     */
    private static MultiLayerConfiguration buildMnistConfig(int seed, IUpdater updater) {
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(updater)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(784)
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();
    }
}
