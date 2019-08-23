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

/**-
 * This program trains a RNN to predict category of a news headlines. It uses word vector generated from PrepareWordVector.java.
 * - Labeled News are stored in \dl4j-examples\src\main\resources\NewsData\LabelledNews folder in train and test folders.
 * - categories.txt file in \dl4j-examples\src\main\resources\NewsData\LabelledNews folder contains category code and description.
 * - This categories are used along with actual news for training.
 * - news word vector is contained  in \dl4j-examples\src\main\resources\NewsData\NewsWordVector.txt file.
 * - Trained model is stored in \dl4j-examples\src\main\resources\NewsData\NewsModel.net file
 * - News Data contains only 3 categories currently.
 * - Data set structure is as given below
 * - categories.txt - this file contains various categories in category id,category description format. Sample categories are as below
 * 0,crime
 * 1,politics
 * 2,bollywood
 * 3,Business&Development
 * - For each category id above, there is a file containig actual news headlines, e.g.
 * 0.txt - contains news for crime headlines
 * 1.txt - contains news for politics headlines
 * 2.txt - contains news for bollywood
 * 3.txt - contains news for Business&Development
 * - You can add any new category by adding one line in categories.txt and respective news file in folder mentioned above.
 * - Below are training results with the news data given with this example.
 * ==========================Scores========================================
 * Accuracy:        0.9343
 * Precision:       0.9249
 * Recall:          0.9327
 * F1 Score:        0.9288
 * ========================================================================
 * <p>
 * Note :
 * - This code is a modification of original example named Word2VecSentimentRNN.java
 * - Results may vary with the data you use to train this network
 * <p>
 * <b>KIT Solutions Pvt. Ltd. (www.kitsol.com)</b>
 */

package org.deeplearning4j.examples.recurrent.processnews;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.download.DownloaderUtility;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;


public class TrainNews {
    public static String DATA_PATH = "";
    public static WordVectors wordVectors;
    private static TokenizerFactory tokenizerFactory;

    public static void main(String[] args) throws Exception {
        String dataLocalPath = DownloaderUtility.NEWSDATA.Download();
        DATA_PATH = new File(dataLocalPath,"LabelledNews").getAbsolutePath();

        int batchSize = 50;     //Number of examples in each minibatch
        int nEpochs = 10;        //Number of epochs (full passes of training data) to train on
        int truncateReviewsToLength = 300;  //Truncate reviews with length (# words) greater than this

        //DataSetIterators for training and testing respectively
        //Using AsyncDataSetIterator to do data loading in a separate thread; this may improve performance vs. waiting for data to load
        wordVectors = WordVectorSerializer.readWord2VecModel(new File(dataLocalPath,"NewsWordVector.txt"));

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        NewsIterator iTrain = new NewsIterator.Builder()
            .dataDirectory(DATA_PATH)
            .wordVectors(wordVectors)
            .batchSize(batchSize)
            .truncateLength(truncateReviewsToLength)
            .tokenizerFactory(tokenizerFactory)
            .train(true)
            .build();

        NewsIterator iTest = new NewsIterator.Builder()
            .dataDirectory(DATA_PATH)
            .wordVectors(wordVectors)
            .batchSize(batchSize)
            .tokenizerFactory(tokenizerFactory)
            .truncateLength(truncateReviewsToLength)
            .train(false)
            .build();

        //DataSetIterator train = new AsyncDataSetIterator(iTrain,1);
        //DataSetIterator test = new AsyncDataSetIterator(iTest,1);

        int inputNeurons = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length; // 100 in our case
        int outputs = iTrain.getLabels().size();

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        //Set up network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .updater(new RmsProp(0.0018))
            .l2(1e-5)
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
            .list()
            .layer( new LSTM.Builder().nIn(inputNeurons).nOut(200)
                .activation(Activation.TANH).build())
            .layer(new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(outputs).build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        System.out.println("Starting training...");
        net.setListeners(new ScoreIterationListener(1), new EvaluativeListener(iTest, 1, InvocationType.EPOCH_END));
        net.fit(iTrain, nEpochs);

        System.out.println("Evaluating...");
        Evaluation eval = net.evaluate(iTest);
        System.out.println(eval.stats());

        net.save(new File(dataLocalPath,"NewsModel.net"), true);
        System.out.println("----- Example complete -----");
    }

}
