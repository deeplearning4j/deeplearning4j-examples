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

package org.deeplearning4j.examples.dataexamples;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.impl.misc.SVMLightRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.resources.Downloader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;

public class SVMLightExample {
    private static Logger log = LoggerFactory.getLogger(SVMLightExample.class);

    public static final String DATA_LOCAL_PATH;

    static {
        final String DATA_URL = "https://deeplearning4jblob.blob.core.windows.net/dl4j-examples/dl4j-examples/DataExamples.zip";
        final String MD5 = "25de3941b3491af7234321f35b93ec25";
        final int DOWNLOAD_RETRIES = 10;
        final String DOWNLOAD_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "DataExamples.zip");
        final String EXTRACT_DIR = FilenameUtils.concat(System.getProperty("user.home"), "dl4j-examples-data/dl4j-examples");
        DATA_LOCAL_PATH = FilenameUtils.concat(EXTRACT_DIR, "DataExamples");
        if (!new File(DATA_LOCAL_PATH).exists()) {
            try {
                System.out.println("_______________________________________________________________________");
                System.out.println("Downloading data (2MB) and extracting to \n\t" + DATA_LOCAL_PATH);
                System.out.println("_______________________________________________________________________");
                Downloader.downloadAndExtract("files",
                    new URL(DATA_URL),
                    new File(DOWNLOAD_PATH),
                    new File(EXTRACT_DIR),
                    MD5,
                    DOWNLOAD_RETRIES);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        else {
            System.out.println("_______________________________________________________________________");
            System.out.println("Example data present in \n\t" + DATA_LOCAL_PATH);
            System.out.println("_______________________________________________________________________");
        }
    }

    public static void main(String[] args) throws Exception {

        int numOfFeatures = 784;     // For MNIST data set, each row is a 1D expansion of a handwritten digits picture of size 28x28 pixels = 784
        int numOfClasses = 10;       // 10 classes (types of senders) in the data set. Zero indexing. Classes have integer values 0, 1 or 2 ... 9
        int batchSize = 10;          // 1000 examples, with batchSize is 10, around 100 iterations per epoch
        int printIterationsNum = 20; // print score every 20 iterations

        int hiddenLayer1Num = 200;
        long seed = 42;
        int nEpochs = 4;

        Configuration config = new Configuration();
        config.setBoolean(SVMLightRecordReader.ZERO_BASED_INDEXING, true);
        config.setInt(SVMLightRecordReader.NUM_FEATURES, numOfFeatures);

        SVMLightRecordReader trainRecordReader = new SVMLightRecordReader();
        trainRecordReader.initialize(config, new FileSplit(new File(DATA_LOCAL_PATH,"MnistSVMLightExample/mnist_svmlight_train_1000.txt")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, numOfFeatures, numOfClasses);

        SVMLightRecordReader testRecordReader = new SVMLightRecordReader();
        testRecordReader.initialize(config, new FileSplit(new File(DATA_LOCAL_PATH,"MnistSVMLightExample/mnist_svmlight_test_100.txt")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, numOfFeatures, numOfClasses);


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .updater(Adam.builder().learningRate(0.02).beta1(0.9).beta2(0.999).build())
            .l2(1e-4)
            .list()
            .layer(new DenseLayer.Builder().nIn(numOfFeatures).nOut(hiddenLayer1Num)
                .build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(hiddenLayer1Num).nOut(numOfClasses).build())
            .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(printIterationsNum));

        for ( int n = 0; n < nEpochs; n++) {

            model.fit(trainIter);

            log.info(String.format("Epoch %d finished training", n + 1));

            // evaluate the model on test data, once every second epoch
            if ((n + 1) % 2 == 0) {
                //evaluate the model on the test set
                Evaluation eval = new Evaluation(numOfClasses);
                testIter.reset();
                while(testIter.hasNext()) {
                    DataSet t = testIter.next();
                    INDArray features = t.getFeatures();
                    INDArray labels = t.getLabels();
                    INDArray predicted = model.output(features, false);
                    eval.eval(labels, predicted);
                }
                log.info(String.format("Evaluation on test data - [Epoch %d] [Accuracy: %.3f, P: %.3f, R: %.3f, F1: %.3f] ",
                    n + 1, eval.accuracy(), eval.precision(), eval.recall(), eval.f1()));
                log.info(eval.stats());
            }
        }
        System.out.println("Finished...");
    }
}

