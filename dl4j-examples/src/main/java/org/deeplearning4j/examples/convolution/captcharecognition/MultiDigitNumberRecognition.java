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

package org.deeplearning4j.examples.convolution.captcharecognition;


import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.resources.Downloader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;

/**
 * @Description This is a demo that multi-digit number recognition. The maximum length is 6 digits.
 * If it is less than 6 digits, then zero is added to last
 * Training set: There were 14108 images, and they were used to train a model.
 * Testing set: in total 108 images,they copied from the training set,mainly to determine whether it's good that the model fited training data
 * Verification set: The total quantity of the images has 248 that's the unknown image,the main judgment that the model is good or bad
 * Other: Using the current architecture and hyperparameters, the accuracy of the best model prediction validation set is (215-227) / 248 with different epochs
 * of course, if you're interested, you can continue to optimize.
 * @author WangFeng
 */
public class MultiDigitNumberRecognition {

    private static final Logger log = LoggerFactory.getLogger(MultiDigitNumberRecognition.class);

    private static long seed = 123;
    private static int epochs = 4;
    private static int batchSize = 15;
    private static String rootPath = System.getProperty("user.dir");

    private static String modelDirPath = rootPath.substring(0, rootPath.lastIndexOf(File.separatorChar)) + File.separatorChar + "out" + File.separatorChar + "models";
    private static String modelPath = modelDirPath + File.separatorChar + "validateCodeCheckModel.json";

    public static final String DATA_LOCAL_PATH;

    static {
        final String DATA_URL = "https://deeplearning4jblob.blob.core.windows.net/dl4j-examples/dl4j-examples/captchaImage.zip";
        final String MD5 = "1d159c9587fdbb1cbfd66f0d62380e61";
        final int DOWNLOAD_RETRIES = 10;
        final String DOWNLOAD_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "captchaImage.zip");
        final String EXTRACT_DIR = FilenameUtils.concat(System.getProperty("user.home"), "dl4j-examples-data/dl4j-examples");
        DATA_LOCAL_PATH = FilenameUtils.concat(EXTRACT_DIR,"captchaImage");
        if (!new File(DATA_LOCAL_PATH).exists()) {
            try {

                System.out.println("_______________________________________________________________________");
                System.out.println("Downloading data (42MB) and extracting to \n\t" + DATA_LOCAL_PATH);
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
        long startTime = System.currentTimeMillis();
        System.out.println(startTime);

        File modelDir = new File(modelDirPath);

        // create directory
        boolean hasDir = modelDir.exists() || modelDir.mkdirs();
        log.info( modelPath );
        //create model
        ComputationGraph model =  createModel();
        //monitor the model score
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        uiServer.attach(statsStorage);

        //construct the iterator
        MultiDataSetIterator trainMulIterator = new MultiRecordDataSetIterator(batchSize, "train");
        MultiDataSetIterator testMulIterator = new MultiRecordDataSetIterator(batchSize,"test");
        MultiDataSetIterator validateMulIterator = new MultiRecordDataSetIterator(batchSize,"validate");

        //fit
        model.setListeners(new ScoreIterationListener(10), new StatsListener( statsStorage), new EvaluativeListener(testMulIterator, 1, InvocationType.EPOCH_END));
        model.fit(trainMulIterator, epochs);

        //save
        model.save(new File(modelPath), true);
        long endTime = System.currentTimeMillis();

        System.out.println("=============run time=====================" + (endTime - startTime));

        System.out.println("=====eval model=====test==================");
        modelPredict(model, testMulIterator);

        System.out.println("=====eval model=====validate==================");
        modelPredict(model, validateMulIterator);

    }

    public static ComputationGraph createModel() {

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
            .l2(1e-3)
            .updater(new Adam(1e-3))
            .weightInit( WeightInit.XAVIER_UNIFORM)
            .graphBuilder()
            .addInputs("trainFeatures")
            .setInputTypes(InputType.convolutional(60, 160, 1))
            .setOutputs("out1", "out2", "out3", "out4", "out5", "out6")
            .addLayer("cnn1",  new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0})
                .nIn(1).nOut(48).activation( Activation.RELU).build(), "trainFeatures")
            .addLayer("maxpool1",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}, new int[]{2, 2}, new int[]{0, 0})
                .build(), "cnn1")
            .addLayer("cnn2",  new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0})
                .nOut(64).activation( Activation.RELU).build(), "maxpool1")
            .addLayer("maxpool2",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,1}, new int[]{2, 1}, new int[]{0, 0})
                .build(), "cnn2")
            .addLayer("cnn3",  new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0})
                .nOut(128).activation( Activation.RELU).build(), "maxpool2")
            .addLayer("maxpool3",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}, new int[]{2, 2}, new int[]{0, 0})
                .build(), "cnn3")
            .addLayer("cnn4",  new ConvolutionLayer.Builder(new int[]{4, 4}, new int[]{1, 1}, new int[]{0, 0})
                .nOut(256).activation( Activation.RELU).build(), "maxpool3")
            .addLayer("maxpool4",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}, new int[]{2, 2}, new int[]{0, 0})
                .build(), "cnn4")
            .addLayer("ffn0",  new DenseLayer.Builder().nOut(3072)
                .build(), "maxpool4")
            .addLayer("ffn1",  new DenseLayer.Builder().nOut(3072)
                .build(), "ffn0")
            .addLayer("out1", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")
            .addLayer("out2", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")
            .addLayer("out3", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")
            .addLayer("out4", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")
            .addLayer("out5", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")
            .addLayer("out6", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")
            .build();

        // Construct and initialize model
        ComputationGraph model = new ComputationGraph(config);
        model.init();

        return model;
    }

    public static void modelPredict(ComputationGraph model, MultiDataSetIterator iterator) {
        int sumCount = 0;
        int correctCount = 0;

        while (iterator.hasNext()) {
            MultiDataSet mds = iterator.next();
            INDArray[]  output = model.output(mds.getFeatures());
            INDArray[] labels = mds.getLabels();
            int dataNum = batchSize > output[0].rows() ? output[0].rows() : batchSize;
            for (int dataIndex = 0;  dataIndex < dataNum; dataIndex ++) {
                String reLabel = "";
                String peLabel = "";
                INDArray preOutput = null;
                INDArray realLabel = null;
                for (int digit = 0; digit < 6; digit ++) {
                    preOutput = output[digit].getRow(dataIndex);
                    peLabel += Nd4j.argMax(preOutput, 1).getInt(0);
                    realLabel = labels[digit].getRow(dataIndex);
                    reLabel += Nd4j.argMax(realLabel, 1).getInt(0);
                }
                if (peLabel.equals(reLabel)) {
                    correctCount ++;
                }
                sumCount ++;
                log.info("real image {}  prediction {} status {}",  reLabel,peLabel, peLabel.equals(reLabel));
            }
        }
        iterator.reset();
        System.out.println("validate result : sum count =" + sumCount + " correct count=" + correctCount );
    }
}
