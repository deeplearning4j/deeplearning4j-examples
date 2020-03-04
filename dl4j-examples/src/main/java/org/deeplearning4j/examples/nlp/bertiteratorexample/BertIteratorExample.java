/*******************************************************************************
 * Copyright (c) 2015-2020 Skymind, Inc.
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
package org.deeplearning4j.examples.nlp.bertiteratorexample;


import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.iterator.BertIterator;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.examples.utilities.DataUtilities;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.*;


/**
 * @author andrewtuzhykov@gmail.com
 */

public class BertIteratorExample {


    /**
     * Data URL for downloading
     */
    public static final String DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";

    // Bert Base Uncased Vocabulary
    public static final String VOCAB_URL = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt";

    /**
     * Location to save and extract the training/testing data
     */
    public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/");


    /**
     * Get BertIterator instance.
     *
     * @param isTraining specifies which dataset iterator we want to get: train or test.
     * @param t          BertWordPieceTokenizerFactory initialized with provided vocab.
     * @return BertIterator with specified parameters.
     */

    public static BertIterator getBertDataSetIterator(boolean isTraining, BertWordPieceTokenizerFactory t) {

        String path = FilenameUtils.concat(DATA_PATH, (isTraining ? "aclImdb/train/" : "aclImdb/test/"));
        String positiveBaseDir = FilenameUtils.concat(path, "pos");
        String negativeBaseDir = FilenameUtils.concat(path, "neg");
        Random rng = new Random(42);

        File filePositive = new File(positiveBaseDir);
        File fileNegative = new File(negativeBaseDir);

        Map<String, List<File>> reviewFilesMap = new HashMap<>();
        reviewFilesMap.put("Positive", Arrays.asList(Objects.requireNonNull(filePositive.listFiles())));
        reviewFilesMap.put("Negative", Arrays.asList(Objects.requireNonNull(fileNegative.listFiles())));


        BertIterator b = BertIterator.builder()
            .tokenizer(t)
            .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 256)
            .minibatchSize(32)
            .sentenceProvider(new FileLabeledSentenceProvider(reviewFilesMap, rng))
            .featureArrays(BertIterator.FeatureArrays.INDICES_MASK)
            .vocabMap(t.getVocab())
            .task(BertIterator.Task.SEQ_CLASSIFICATION)
            .build();


        return b;
    }


    public static void main(String[] args) throws Exception {


        //Download and extract data
        downloadData();


        final int seed = 0;
        //Seed for reproducibility
        String pathToVocab = DATA_PATH + "vocab.txt";
        // Path to vocab

        // BertWordPieceTokenizerFactory initialized with given vocab
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(new File(pathToVocab), true, true, StandardCharsets.UTF_8);


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .updater(new Adam(1e-3))
            .l2(1e-5)
            .weightInit(WeightInit.XAVIER)
            .list()
            // matching EmbeddingSequenceLayer outputs with Bidirectional LSTM inputs
            .setInputType(InputType.recurrent(1))
//          // initialized weights with normal distribution, amount of inputs according to vocab size and off L2 for this layer
            .layer(0, new EmbeddingSequenceLayer.Builder().weightInit(new NormalDistribution(0, 1)).l2(0)
                .hasBias(true).nIn(t.getVocab().size()).nOut(128).build())
//           // two Bidirectional LSTM layers in a row with dropout and tanh as activation function
            .layer(new Bidirectional(new LSTM.Builder().nOut(256)
                .dropOut(0.8).activation(Activation.TANH).build()))
            .layer(new Bidirectional(new LSTM.Builder().nOut(256)
                .dropOut(0.8).activation(Activation.TANH).build()))
            .layer(new GlobalPoolingLayer(PoolingType.MAX))
            // defining last layer with 2 outputs (2 classes - positive and negative),
            // small dropout to avoid overfitting and MCXENT loss function
            .layer(new OutputLayer.Builder().nOut(2)
                .dropOut(0.97).activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
            .build();


        // Getting train and test BertIterators for both: test and train,
        // changing argument isTraining: true to get train and false to get test respectively
        BertIterator train = getBertDataSetIterator(true, t);
        BertIterator test = getBertDataSetIterator(false, t);


        // preprocessor for DataType matching
        MultiDataSetPreProcessor mdsPreprocessor = new MultiDataSetPreProcessor() {
            @Override
            public void preProcess(MultiDataSet multiDataSet) {
                multiDataSet.setFeaturesMaskArray(0, multiDataSet.getFeaturesMaskArray(0).castTo(DataType.FLOAT));
            }
        };

        // Applying preprocessor for both: train and test datasets
        train.setPreProcessor(mdsPreprocessor);
        test.setPreProcessor(mdsPreprocessor);

        // initialize MultiLayerNetwork instance with described above configuration
        MultiLayerNetwork net = new MultiLayerNetwork(conf);


        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats-" + System.currentTimeMillis() + ".dl4j"));
        int listenerFrequency = 20;
        net.setListeners(new StatsListener(statsStorage, listenerFrequency), new ScoreIterationListener(50));
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);


        // Setting to train net for 19 epochs (note: previous net state persist after each iteration)
        for (int i = 1; i <= 19; i++) {

            net.fit(train);


            // Get and print accuracy, precision, recall & F1 and confusion matrix
            Evaluation eval = net.doEvaluation(test, new Evaluation[]{new Evaluation()})[0];
            System.out.println(eval.stats());
        }

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


        // Download Bert Base Uncased Vocab
        String vocabPath = DATA_PATH + "vocab.txt";
        File vocabFile = new File(vocabPath);

        if (!vocabFile.exists()) {
            try (BufferedInputStream inputStream = new BufferedInputStream(new URL(VOCAB_URL).openStream());
                 FileOutputStream file = new FileOutputStream(DATA_PATH + "vocab.txt")) {
                byte data[] = new byte[1024];
                int byteContent;
                while ((byteContent = inputStream.read(data, 0, 1024)) != -1) {
                    file.write(data, 0, byteContent);
                }
            } catch (IOException e) {
                System.out.println("Something went wrong getting Bert Base Vocabulary");
            }

        } else {
            System.out.println("Vocab file already exists at " + vocabFile.getAbsolutePath());
        }

    }


}

