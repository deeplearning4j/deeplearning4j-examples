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

package org.deeplearning4j.examples.quickstart.modeling.feedforward.classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.utils.DownloaderUtility;
import org.deeplearning4j.examples.utils.PlotUtil;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;
import java.util.concurrent.TimeUnit;

/**
 * "Moon" Data Classification Example
 *
 * Based on the data from Jason Baldridge:
 * 	https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots)
 *
 */
@SuppressWarnings("DuplicatedCode")
public class MoonClassifier {

    public static boolean visualize = true;
    public static String dataLocalPath;

    public static void main(String[] args) throws Exception {
        int seed = 123;
        double learningRate = 0.005;
        int batchSize = 50;
        int nEpochs = 100;

        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 50;

        dataLocalPath = DownloaderUtility.CLASSIFICATIONDATA.Download();

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(dataLocalPath,"moon_data_train.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,2);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(dataLocalPath,"moon_data_eval.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,2);

        //log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));    //Print score every 100 parameter updates

        model.fit( trainIter, nEpochs );

        System.out.println("Evaluate model....");
        Evaluation eval = model.evaluate(testIter);

        //Print the evaluation statistics
        System.out.println(eval.stats());
        System.out.println("\n****************Example finished********************");

        //Training is complete. Code that follows is for plotting the data & predictions only
        generateVisuals(model, trainIter, testIter);
    }

    public static void generateVisuals(MultiLayerNetwork model, DataSetIterator trainIter, DataSetIterator testIter) throws Exception {
        if (visualize) {
            double xMin = -1.5;
            double xMax = 2.5;
            double yMin = -1;
            double yMax = 1.5;

            //Let's evaluate the predictions at every point in the x/y input space, and plot this in the background
            int nPointsPerAxis = 100;

            //Generate x,y points that span the whole range of features
            INDArray allXYPoints = PlotUtil.generatePointsOnGraph(xMin, xMax, yMin, yMax, nPointsPerAxis);
            //Get train data and plot with predictions
            PlotUtil.plotTrainingData(model, trainIter, allXYPoints, nPointsPerAxis);
            TimeUnit.SECONDS.sleep(3);
            //Get test data, run the test data through the network to generate predictions, and plot those predictions:
            PlotUtil.plotTestData(model, testIter, allXYPoints, nPointsPerAxis);
        }
    }

}
