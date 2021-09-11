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

package org.deeplearning4j.examples.quickstart.features.earlystopping;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**Early stopping example on a subset of MNIST
 * Idea: given a small subset of MNIST (1000 examples + 500 test set), conduct training and get the parameters that
 * have the minimum test set loss
 * This is an over-simplified example, but the principles used here should apply in more realistic cases.
 *
 * For further details on early stopping, see https://deeplearning4j.konduit.ai/getting-started/tutorials/early-stopping
 *
 * @author Alex Black
 */
public class EarlyStoppingMNIST {

    public static void main(String[] args) throws Exception {

        //Configure network:
        int nChannels = 1;
        int outputNum = 10;
        int batchSize = 25;
        int seed = 123;
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .l2(0.0005)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.RELU)
            .updater(new Nesterovs(0.02, 0.9))
            .list()
            .layer(new ConvolutionLayer.Builder(5, 5)
                .nIn(nChannels)
                .stride(1, 1)
                .nOut(20).dropOut(0.5)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(new DenseLayer.Builder()
                .nOut(500).build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(InputType.convolutionalFlat(28, 28, 1)) //See note in LenetMnistExample
            .build();

        //Get data:
        DataSetIterator mnistTrain1024 = new MnistDataSetIterator(batchSize,1024,false,true,true,12345);
        DataSetIterator mnistTest512 = new MnistDataSetIterator(batchSize,512,false,false,true,12345);

        String tempDir = System.getProperty("java.io.tmpdir");
        String exampleDirectory = FilenameUtils.concat(tempDir, "DL4JEarlyStoppingExample/");
        File dirFile = new File(exampleDirectory); //We have to create the temp directory or the sample will fail.
        dirFile.mkdir(); // If mkdir fails, it is probably because the directory already exists. Which is fine.
        EarlyStoppingModelSaver saver = new LocalFileModelSaver(exampleDirectory);
        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(100), new ScoreImprovementEpochTerminationCondition(8)) //Max of 50 epochs
                .evaluateEveryNEpochs(1)
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES)) //Max of 20 minutes
                .scoreCalculator(new DataSetLossCalculator(mnistTest512, true))     //Calculate test set score
                .modelSaver(saver)
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,configuration,mnistTrain1024);

        //Conduct early stopping training:
        EarlyStoppingResult result = trainer.fit();
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        //Print score vs. epoch
        Map<Integer,Double> scoreVsEpoch = result.getScoreVsEpoch();
        List<Integer> list = new ArrayList<>(scoreVsEpoch.keySet());
        Collections.sort(list);
        System.out.println("Score vs. Epoch:");
        for( Integer i : list){
            System.out.println(i + "\t" + scoreVsEpoch.get(i));
        }
    }
}
