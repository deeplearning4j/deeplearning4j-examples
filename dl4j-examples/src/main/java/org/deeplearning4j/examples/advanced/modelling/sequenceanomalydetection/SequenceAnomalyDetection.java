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

package org.deeplearning4j.examples.advanced.modelling.sequenceanomalydetection;


import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.examples.utils.DownloaderUtility;
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
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.*;

/**Detection anomaly data from these sequence data which comes from the sensor
 * The normal data should have low reconfiguration error,whereas those anomaly data which the autoencoder has not encountered  have high reconstruction error
 * @author wangfeng
 */
public class SequenceAnomalyDetection {

    private static int trainBatchSize = 64;
    private static int testBatchSize = 1;
    private static int numEpochs = 38;

    public static String dataLocalPath;


    public static void main(String[] args) throws Exception {

        dataLocalPath = DownloaderUtility.ANOMALYSEQUENCEDATA.Download();
        File modelFile = new File(dataLocalPath, "anomalyDetectionModel.gz");
        DataSetIterator trainIterator = new AnomalyDataSetIterator(new File(dataLocalPath, "ads.csv").getAbsolutePath(), trainBatchSize);
        DataSetIterator testIterator = new AnomalyDataSetIterator(new File(dataLocalPath,"test.csv").getAbsolutePath(), testBatchSize);

        MultiLayerNetwork net = true ? createModel(trainIterator.inputColumns(), trainIterator.totalOutcomes()) : MultiLayerNetwork.load(modelFile, true);
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        uiServer.attach(statsStorage);

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainIterator);              //Collect training data statistics
        trainIterator.reset();
        trainIterator.setPreProcessor(normalizer);
        testIterator.setPreProcessor(normalizer);	//Note: using training normalization statistics
        NormalizerSerializer.getDefault().write(normalizer, new File(dataLocalPath, "anomalyDetectionNormlizer.ty").getAbsolutePath());

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
