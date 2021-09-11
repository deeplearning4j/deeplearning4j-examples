/*******************************************************************************
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

package org.deeplearning4j.examples.quickstart.features.presavingdatasets;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

import static org.deeplearning4j.examples.quickstart.features.presavingdatasets.PreSaveFirst.TEST_FOLDER;
import static org.deeplearning4j.examples.quickstart.features.presavingdatasets.PreSaveFirst.TRAIN_FOLDER;


/**
 *
 * YOU NEED TO RUN PreSave first
 * before using this class.
 *
 * This class demonstrates how to  use a pre saved
 * dataset to minimize time spent loading data.
 * This is critical if you want to have ANY speed
 * with deeplearning4j.
 *
 * Deeplearning4j does not force you to use a particular data format.
 * Unfortunately this flexibility means that many people get training wrong.
 *
 * With more flexibility comes more complexity. This class demonstrates how
 * to minimize time spent training while using an existing iterator and an existing dataset.
 *
 * We use an {@link AsyncDataSetIterator}  to load data in the background
 * and {@link PreSaveFirst} to pre save the data to 2 specified directories,
 * trainData and testData
 *
 *
 * Created by agibsonccc on 9/16/15.
 * Modified by dmichelin on 12/10/2016 to add documentation
 */
public class LoadPreSavedLenetMnistExample {
    private static final Logger log = LoggerFactory.getLogger(LoadPreSavedLenetMnistExample.class);

    public static void main(String[] args) throws Exception {
        int nChannels = 1; // Number of input channels
        int outputNum = 10; // The number of possible outcomes
        int nEpochs = 1; // Number of training epochs
        int seed = 123; //

        /*
            Load the pre saved data. NOTE: YOU NEED TO RUN PreSave first.

         */
        if (!new File(TRAIN_FOLDER).exists() || !new File(TEST_FOLDER).exists()) {
            throw new RuntimeException("You have not presaved the datasets. Run the main class in PreSaveFirst.java!!");
        }
        else {
            log.info("Load data....");
        }
        /**
         * Note the {@link ExistingMiniBatchDataSetIterator}
         * takes in a pattern of "mnist-train-%d.bin"
         * and "mnist-test-%d.bin"
         *
         * The %d is an integer. You need a %d
         * as part of the template in order to have
         * the iterator work.
         * It uses this %d integer to
         * index what number it is in the current dataset.
         * This is how pre save will save the data.
         *
         * If you still don't understand what this is, please see an example with printf in c:
         * http://www.sitesbay.com/program/c-program-print-number-pattern
         * and in java:
         * https://docs.oracle.com/javase/tutorial/java/data/numberformat.html
         */
        DataSetIterator existingTrainingData = new ExistingMiniBatchDataSetIterator(new File(TRAIN_FOLDER),"mnist-train-%d.bin");
       //note here that we use as ASYNC iterator which loads data in the background, this is crucial to avoid disk as a bottleneck
        //when loading data
        DataSetIterator mnistTrain = new AsyncDataSetIterator(existingTrainingData);
        DataSetIterator existingTestData = new ExistingMiniBatchDataSetIterator(new File(TEST_FOLDER),"mnist-test-%d.bin");
        DataSetIterator mnistTest = new AsyncDataSetIterator(existingTestData);

        /*
            Construct the neural network
         */
        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .l2(0.0005)
            .updater(new Nesterovs(0.01, 0.9))
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(new ConvolutionLayer.Builder(5, 5)
                //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                .nIn(nChannels)
                .stride(1, 1)
                .nOut(20)
                .activation(Activation.IDENTITY)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)
                .build())
            .layer(new ConvolutionLayer.Builder(5, 5)
                //Note that nIn need not be specified in later layers
                .stride(1, 1)
                .nOut(50)
                .activation(Activation.IDENTITY)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)
                .build())
            .layer(new DenseLayer.Builder().activation(Activation.RELU)
                .nOut(500).build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
            .build();

        /*
        Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
        (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
            and the dense layer
        (b) Does some additional configuration validation
        (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
            layer based on the size of the previous layer (but it won't override values manually set by the user)

        InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
        For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
        MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
        row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
        */

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(1), new EvaluativeListener(mnistTest, 1, InvocationType.EPOCH_END));
        model.fit(mnistTrain, nEpochs);

        log.info("Evaluate model....");
        Evaluation eval = model.evaluate(mnistTest);
        System.out.println(eval.stats());

        log.info("****************Example finished********************");
    }
}
