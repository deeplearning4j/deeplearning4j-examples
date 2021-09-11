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

package org.deeplearning4j.examples.advanced.modelling.textclassification.pretrainedword2vec;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
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
import org.deeplearning4j.examples.utils.DataUtilities;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.common.resources.Downloader;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Scanner;

/**
 * Example: Given a movie review (raw text), classify that movie review as either positive or negative based on the words it contains.
 * This is done by combining Word2Vec vectors and a recurrent neural network model. Each word in a review is vectorized
 * (using the Word2Vec model) and fed into a recurrent neural network.
 * Training data is the "Large Movie Review Dataset" from http://ai.stanford.edu/~amaas/data/sentiment/
 * This data set contains 25,000 training reviews + 25,000 testing reviews
 * <p>
 * Process:
 * 0. If path to the wordvectors is not set and a download not found previously in the default location you will be prompted if you want to download it.
 * 1. Automatic on first run of example: Download data (movie reviews) + extract
 * 2. Load existing Word2Vec model (for example: Google News word vectors.)
 * 3. Load each each review. Convert words to vectors + reviews to sequences of vectors
 * 4. Train network
 * <p>
 * With the current configuration, gives approx. 83% accuracy after 1 epoch. Better performance may be possible with
 * additional tuning.
 *
 * @author Alex Black
 */
public class ImdbReviewClassificationRNN {

    /**
     * Data URL for downloading
     */
    public static final String DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
    /**
     * Location to save and extract the training/testing data
     */
    public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/");
    /**
     * Location (local file system) for the Google News vectors. Set this manually.
     */
    public static String wordVectorsPath = "/PATH/TO/YOUR/VECTORS/GoogleNews-vectors-negative300.bin.gz";

    public static void main(String[] args) throws Exception {
        if (wordVectorsPath.startsWith("/PATH/TO/YOUR/VECTORS/")) {
            System.out.println("wordVectorsPath has not been set. Checking default location in ~/dl4j-examples-data for download...");
            checkDownloadW2VECModel();
        }
        //Download and extract data
        downloadData();

        int batchSize = 64;     //Number of examples in each minibatch
        int vectorSize = 300;   //Size of the word vectors. 300 in the Google News model
        int nEpochs = 1;        //Number of epochs (full passes of training data) to train on
        int truncateReviewsToLength = 256;  //Truncate reviews with length (# words) greater than this
        final int seed = 0;     //Seed for reproducibility

        Nd4j.getMemoryManager().setAutoGcWindow(10000);  //https://deeplearning4j.konduit.ai/config/config-memory/config-workspaces#garbage-collector

        //Set up network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .updater(new Adam(5e-3))
            .l2(1e-5)
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
            .list()
            .layer(new LSTM.Builder().nIn(vectorSize).nOut(256)
                .activation(Activation.TANH).build())
            .layer(new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        //DataSetIterators for training and testing respectively
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(wordVectorsPath));
        SentimentExampleIterator train = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, true);
        SentimentExampleIterator test = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false);

        System.out.println("Starting training");
        net.setListeners(new ScoreIterationListener(1), new EvaluativeListener(test, 1, InvocationType.EPOCH_END));
        net.fit(train, nEpochs);

        //After training: load a single example and generate predictions
        File shortNegativeReviewFile = new File(FilenameUtils.concat(DATA_PATH, "aclImdb/test/neg/12100_1.txt"));
        String shortNegativeReview = FileUtils.readFileToString(shortNegativeReviewFile, (Charset) null);

        INDArray features = test.loadFeaturesFromString(shortNegativeReview, truncateReviewsToLength);
        INDArray networkOutput = net.output(features);
        long timeSeriesLength = networkOutput.size(2);
        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

        System.out.println("\n\n-------------------------------");
        System.out.println("Short negative review: \n" + shortNegativeReview);
        System.out.println("\n\nProbabilities at last time step:");
        System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
        System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));

        System.out.println("----- Example complete -----");
    }

    public static void downloadData() throws Exception {
        //Create directory if required
        File directory = new File(DATA_PATH);
        if (!directory.exists()) directory.mkdir();

        //Download file:
        String archizePath = DATA_PATH + "aclImdb_v1.tar.gz";
        File archiveFile = new File(archizePath);
        String extractedPath = DATA_PATH + "aclImdb";
        File extractedFile = new File(extractedPath);

        if (!archiveFile.exists()) {
            System.out.println("Starting data download (80MB)...");
            FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile);
            System.out.println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath());
            //Extract tar.gz file to output directory
            DataUtilities.extractTarGz(archizePath, DATA_PATH);
        } else {
            //Assume if archive (.tar.gz) exists, then data has already been extracted
            System.out.println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath());
            if (!extractedFile.exists()) {
                //Extract tar.gz file to output directory
                DataUtilities.extractTarGz(archizePath, DATA_PATH);
            } else {
                System.out.println("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
            }
        }
    }

    public static void checkDownloadW2VECModel() throws IOException {
        String defaultwordVectorsPath = FilenameUtils.concat(System.getProperty("user.home"), "dl4j-examples-data/w2vec300");
        String md5w2vec = "1c892c4707a8a1a508b01a01735c0339";
        wordVectorsPath = new File(defaultwordVectorsPath, "GoogleNews-vectors-negative300.bin.gz").getAbsolutePath();
        if (new File(wordVectorsPath).exists()) {
            System.out.println("\n\tGoogleNews-vectors-negative300.bin.gz file found at path: " + defaultwordVectorsPath);
            System.out.println("\tChecking md5 of existing file..");
            if (Downloader.checkMD5OfFile(md5w2vec, new File(wordVectorsPath))) {
                System.out.println("\tExisting file hash matches.");
                return;
            } else {
                System.out.println("\tExisting file hash doesn't match. Retrying download...");
            }
        } else {
            System.out.println("\n\tNo previous download of GoogleNews-vectors-negative300.bin.gz found at path: " + defaultwordVectorsPath);
        }
        System.out.println("\tWARNING: GoogleNews-vectors-negative300.bin.gz is a 1.5GB file.");
        System.out.println("\tPress \"ENTER\" to start a download of GoogleNews-vectors-negative300.bin.gz to " + defaultwordVectorsPath);
        Scanner scanner = new Scanner(System.in);
        scanner.nextLine();
        System.out.println("Starting model download (1.5GB!)...");
        Downloader.download("Word2Vec", new URL("https://dl4jdata.blob.core.windows.net/resources/wordvectors/GoogleNews-vectors-negative300.bin.gz"), new File(wordVectorsPath), md5w2vec, 5);
        System.out.println("Successfully downloaded word2vec model to " + wordVectorsPath);
    }
}


