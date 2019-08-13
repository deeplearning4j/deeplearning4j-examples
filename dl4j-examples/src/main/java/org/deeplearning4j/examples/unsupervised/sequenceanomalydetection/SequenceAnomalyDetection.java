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

package org.deeplearning4j.examples.unsupervised.sequenceanomalydetection;


import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.resources.Downloader;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.*;

/**Detection anomaly data from these sequence data which comes from the sensor
 * The normal data should have low reconfiguration error,whereas those anomaly data which the autoencoder has not encountered  have high reconstruction error
 * @author wangfeng
 */
public class SequenceAnomalyDetection {

    private static int trainBatchSize = 64;
    private static int testBatchSize = 1;
    private static int numEpochs = 38;

    public static final String DATA_LOCAL_PATH;

    static {
        final String DATA_URL = "https://deeplearning4jblob.blob.core.windows.net/dl4j-examples/dl4j-examples/anomalysequencedata.zip";
        final String MD5 = "51bb7c50e265edec3a241a2d7cce0e73";
        final int DOWNLOAD_RETRIES = 10;
        final String DOWNLOAD_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "anomalysequencedata.zip");
        final String EXTRACT_DIR = FilenameUtils.concat(System.getProperty("user.home"), "dl4j-examples-data/dl4j-examples");
        DATA_LOCAL_PATH = FilenameUtils.concat(EXTRACT_DIR, "anomalysequencedata");
        if (!new File(DATA_LOCAL_PATH).exists()) {
            try {
                System.out.println("_______________________________________________________________________");
                System.out.println("Downloading data (3MB) and extracting to \n\t" + DATA_LOCAL_PATH);
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
        } else {
            System.out.println("_______________________________________________________________________");
            System.out.println("Example data present in \n\t" + DATA_LOCAL_PATH);
            System.out.println("_______________________________________________________________________");
        }
    }

    public static void main(String[] args) throws Exception {

        File modelFile = new File(DATA_LOCAL_PATH, "anomalyDetectionModel.gz");
        DataSetIterator trainIterator = new AnomalyDataSetIterator(new File(DATA_LOCAL_PATH, "ads.csv").getAbsolutePath(), trainBatchSize);
        DataSetIterator testIterator = new AnomalyDataSetIterator(new File(DATA_LOCAL_PATH,"test.csv").getAbsolutePath(), testBatchSize);

        MultiLayerNetwork net = true ? createModel(trainIterator.inputColumns(), trainIterator.totalOutcomes()) : MultiLayerNetwork.load(modelFile, true);
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        uiServer.attach(statsStorage);

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainIterator);              //Collect training data statistics
        trainIterator.reset();
        trainIterator.setPreProcessor(normalizer);
        testIterator.setPreProcessor(normalizer);	//Note: using training normalization statistics
        NormalizerSerializer.getDefault().write(normalizer, new File(DATA_LOCAL_PATH, "anomalyDetectionNormlizer.ty").getAbsolutePath());

        // training
        net.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
        net.fit(trainIterator, numEpochs);

        // save model to disk
        ModelSerializer.writeModel(net, modelFile,true);

        List<Pair<Double,String>> evalList = new ArrayList<>();
        Queue<String> queue = ((AnomalyDataSetIterator)testIterator).getCurrentLines();
        double totalScore = 0;
          while (testIterator.hasNext()) {
            DataSet ds = testIterator.next();
            double score = net.score(ds);
            String currentLine = queue.poll();
            totalScore += score;
            evalList.add(new ImmutablePair<>(score, currentLine));
        }

        Collections.sort(evalList, Comparator.comparing(Pair::getLeft));
        Stack<String> anomalyData = new Stack<>();
        double threshold = totalScore / evalList.size();
        for (Pair<Double, String> pair: evalList) {
            double s = pair.getLeft();
            if (s >  threshold) {
                anomalyData.push(pair.getRight());
            }
        }

        //output anomaly data
        System.out.println("based on the score, all anomaly data is following with descending order:\n");
        for (int i = anomalyData.size(); i > 0; i--) {
            System.out.println(anomalyData.pop());
        }

    }

    public static MultiLayerNetwork createModel(int inputNum, int outputNum) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED).inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .seed(123456)
                .optimizationAlgo( OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new RmsProp.Builder().learningRate(0.05).rmsDecay(0.002).build())
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.TANH)
                .list()
                .layer(new LSTM.Builder().name("encoder0").nIn(inputNum).nOut(100).build())
                .layer(new LSTM.Builder().name("encoder1").nOut(80).build())
                .layer(new LSTM.Builder().name("encoder2").nOut(5).build())
                .layer(new LSTM.Builder().name("decoder1").nOut(80).build())
                .layer(new LSTM.Builder().name("decoder2").nOut(100).build())
                .layer(new RnnOutputLayer.Builder().name("output").nOut(outputNum)
                        .activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        return net;
    }

}
