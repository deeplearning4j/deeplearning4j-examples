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

package org.deeplearning4j.examples.advanced.modelling.sequenceprediction;


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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;


/**
 * lottery rule:every day has 120 term, during 10:00-22:00(72),every 10 minutes generate new lottery number, during 22:00-02:00(48) every 5 minutes generate new lottery number,
 * lottery number rule: its length is 5 bit, every bit choose one from 0-9 digit
 * this example only try to get the lottery algorithm by an unidentified structure underlying the data,maybe the inputting features still is less,
 * although the model don't seem to get the original lottery algorithm, but this example can still be used as a reference for processing sequence data.
 *
 * @author wangfeng.
 */
public class TrainLotteryModelSeqPrediction {
    private static Logger log = LoggerFactory.getLogger(TrainLotteryModelSeqPrediction.class);

    private static int batchSize = 64;
    private static long seed = 123;
    private static int numEpochs = 3;
    private static boolean modelType = true;

    public static String dataLocalPath;


    public static void main(String[] args) throws Exception {

        dataLocalPath = DownloaderUtility.LOTTERYDATA.Download();
        File modelFile = new File(dataLocalPath, "lotteryPredictModel.json");
        DataSetIterator trainIterator = new LotteryDataSetIterator(new File(dataLocalPath,"cqssc_train.csv").getAbsolutePath(), batchSize, modelType);
        DataSetIterator testIterator = new LotteryDataSetIterator(new File(dataLocalPath, "cqssc_test.csv").getAbsolutePath(), 2, modelType);
        DataSetIterator validateIterator = new LotteryDataSetIterator(new File(dataLocalPath, "cqssc_validate.csv").getAbsolutePath(), 2, modelType);

        MultiLayerNetwork model = getNetModel(trainIterator.inputColumns(), trainIterator.totalOutcomes());
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        uiServer.attach(statsStorage);

        // print layers and parameters
        System.out.println(model.summary());

        // training
        model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));

        long startTime = System.currentTimeMillis();
        model.fit(trainIterator, numEpochs);
        long endTime = System.currentTimeMillis();
        System.out.println("=============run time=====================" + (endTime - startTime));

        // save model to disk
        model.save(modelFile, true);

        int luckySize = 5;
        if (modelType) {
            while (testIterator.hasNext()) {
                DataSet ds = testIterator.next();
                //predictions all at once
                INDArray output = model.output(ds.getFeatures());
                INDArray label = ds.getLabels();
                INDArray preOutput = Nd4j.argMax(output, 2);
                INDArray realLabel = Nd4j.argMax(label, 2);
                StringBuilder peLabel = new StringBuilder();
                StringBuilder reLabel = new StringBuilder();
                for (int dataIndex = 0; dataIndex < 5; dataIndex++) {
                    peLabel.append(preOutput.getInt(dataIndex));
                    reLabel.append(realLabel.getInt(dataIndex));
                }
                log.info("test-->real lottery {}  prediction {} status {}", reLabel.toString(), peLabel.toString(), peLabel.toString().equals(reLabel.toString()));
            }
            while (validateIterator.hasNext()) {
                DataSet ds = validateIterator.next();
                //predictions all at once
                INDArray output = model.feedForward(ds.getFeatures()).get(0);
                INDArray label = ds.getLabels();
                INDArray preOutput = Nd4j.argMax(output, 2);
                INDArray realLabel = Nd4j.argMax(label, 2);
                StringBuilder peLabel = new StringBuilder();
                StringBuilder reLabel = new StringBuilder();
                for (int dataIndex = 0; dataIndex < 5; dataIndex++) {
                    peLabel.append(preOutput.getInt(dataIndex));
                    reLabel.append(realLabel.getInt(dataIndex));
                }
                log.info("validate-->real lottery {}  prediction {} status {}", reLabel.toString(), peLabel.toString(), peLabel.toString().equals(reLabel.toString()));
            }

            String currentLottery = "27578";
            INDArray initCondition = Nd4j.zeros(1, 5, 10);
            String[] featureAry = currentLottery.split("");
            for (int j = 0; j < featureAry.length; j++) {
                int p = Integer.parseInt(featureAry[j]);
                initCondition.putScalar(new int[]{0, j, p}, 1);
            }
            INDArray output = model.output(initCondition);
            INDArray preOutput = Nd4j.argMax(output, 2);
            StringBuilder latestLottery = new StringBuilder();
            for (int dataIndex = 0; dataIndex < 5; dataIndex++) {
                latestLottery.append(preOutput.getInt(dataIndex));
            }
            System.out.println("current lottery numbers==" + currentLottery + "==prediction===next lottery numbers==" + latestLottery.toString());

        } else {
            int predictCount = 2;
            String predictDateNum = "20180716100";//20180716,100
            //Create input for initialization
            //For single time step:input has shape [miniBatchSize,inputSize] or [miniBatchSize,inputSize,1]. miniBatchSize=1 for single example.<br>
            //For multiple time steps:input has shape  [miniBatchSize,inputSize,inputTimeSeriesLength]
            INDArray initCondition = Nd4j.zeros(predictCount, 10, predictDateNum.length());

            String[] featureAry = predictDateNum.split("");
            for (int j = 0; j < featureAry.length; j++) {
                int p = Integer.parseInt(featureAry[j]);
                for (int i = 0; i < predictCount; i++) {
                    initCondition.putScalar(new int[]{i, p, j}, 1);
                }
            }
            StringBuilder[] sb = new StringBuilder[predictCount];
            for (int i = 0; i < predictCount; i++) {
                sb[i] = new StringBuilder(predictDateNum);
            }
            //Clear the previous state of the RNN layers (if any)
            model.rnnClearPreviousState();
            INDArray output = model.rnnTimeStep(initCondition);
            //output.size(x) will get the size along a specified dimension,
            //output.tensorAlongDimension(...) will get the vector along a particular dimension
            output = output.tensorAlongDimension((int) output.size(2) - 1, 1, 0);    //Gets the last time step output
            Random random = new Random(12345);
            for (int i = 0; i < luckySize; i++) {
                //Set up next input (single time step) by sampling from previous output
                INDArray nextInput = Nd4j.zeros(predictCount, 10);
                //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
                for (int s = 0; s < predictCount; s++) {
                    double[] outputProbDistribution = new double[10];
                    for (int j = 0; j < outputProbDistribution.length; j++) {
                        outputProbDistribution[j] = output.getDouble(s, j);
                    }
                    double sum = 0.0;
                    int luckyNum = 0;
                    double d = random.nextDouble();
                    for (int j = 0; j < outputProbDistribution.length; j++) {
                        sum += outputProbDistribution[j];
                        if (d <= sum) luckyNum = i;
                    }
                    //Prepare next time step input
                    nextInput.putScalar(new int[]{s, luckyNum}, 1.0f);
                    sb[s].append(luckyNum);
                }
                //Do one time step of forward pass
                output = model.rnnTimeStep(nextInput);
            }
            for (int i = 0; i < predictCount; i++) {
                System.out.println("==prediction====result===" + predictCount + "======" + sb[i].toString());
            }
        }
    }

    //create the neural network
    private static MultiLayerNetwork getNetModel(int inputNum, int outputNum) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .trainingWorkspaceMode(WorkspaceMode.ENABLED).inferenceWorkspaceMode(WorkspaceMode.ENABLED)
            .seed(seed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .weightInit(WeightInit.XAVIER)
            .updater(new RmsProp.Builder().rmsDecay(0.95).learningRate(1e-2).build())
            .list()
            .layer(new LSTM.Builder().name("lstm1")
                .activation(Activation.TANH).nIn(inputNum).nOut(100).build())
            .layer(new LSTM.Builder().name("lstm2")
                .activation(Activation.TANH).nOut(80).build())
            .layer(new RnnOutputLayer.Builder().name("output")
                .activation(Activation.SOFTMAX).nOut(outputNum).lossFunction(LossFunctions.LossFunction.MSE)
                .build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        return net;
    }


}
