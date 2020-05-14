/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.multigpu.advanced.w2vsentiment;

import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;

import static org.deeplearning4j.examples.multigpu.advanced.w2vsentiment.DataSetsBuilder.TEST_PATH;
import static org.deeplearning4j.examples.multigpu.advanced.w2vsentiment.DataSetsBuilder.TRAIN_PATH;

/**
 * Example: Given a movie review (raw text), classify that movie review as either positive or negative based on the words it contains.
 * This example is the multi-gpu version of the dl4j-example example of the same name.
 *
 * Here the dataset is presaved to save time on multiple epochs.
 * @author Alex Black
 */
public class ImdbReviewClassificationRNN {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(ImdbReviewClassificationRNN.class);

    public static void main(String[] args) throws Exception {

        int vectorSize = 300;   //Size of the word vectors. 300 in the Google News model
        int nEpochs = 1;        //Number of epochs (full passes of training data) to train on

//        Nd4j.setDataType(DataBuffer.Type.DOUBLE);

        CudaEnvironment.getInstance().getConfiguration()
            // key option enabled
            .allowMultiGPU(true)

            // we're allowing larger memory caches
            .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)

            // cross-device access is used for faster model averaging over pcie
            .allowCrossDeviceAccess(true);

        //Set up network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .updater(new Adam.Builder().learningRate(2e-2).build())
            .l2(1e-5)
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
            .list()
            .layer(0, new LSTM.Builder().nIn(vectorSize).nOut(256)
                .activation(Activation.TANH).build())
            .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new PerformanceListener(10, true));

        if (!new File(TRAIN_PATH).exists() || !new File(TEST_PATH).exists()) {
            new DataSetsBuilder().run(args);
        }
        //DataSetIterators for training and testing respectively
        DataSetIterator train = new ExistingMiniBatchDataSetIterator(new File(TRAIN_PATH));
        DataSetIterator test = new ExistingMiniBatchDataSetIterator(new File(TEST_PATH));

        ParallelWrapper pw = new ParallelWrapper.Builder<>(net)
            .prefetchBuffer(16 * Nd4j.getAffinityManager().getNumberOfDevices())
            .reportScoreAfterAveraging(true)
            .averagingFrequency(10)
            .workers(Nd4j.getAffinityManager().getNumberOfDevices())
            .build();

        log.info("Starting training...");
        for (int i = 0; i < nEpochs; i++) {
            pw.fit(train);
            train.reset();
        }

        log.info("Starting evaluation...");

        //Run evaluation. This is on 25k reviews, so can take some time
        Evaluation evaluation = net.evaluate(test);
        System.out.println(evaluation.stats());
    }

}
