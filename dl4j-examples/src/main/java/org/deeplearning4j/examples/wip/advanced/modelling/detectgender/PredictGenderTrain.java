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

package org.deeplearning4j.examples.wip.advanced.modelling.detectgender;

/*
 * Created by KIT Solutions (www.kitsol.com) on 9/28/2016.
 */

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.utils.DownloaderUtility;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;

@SuppressWarnings("DuplicatedCode")
public class PredictGenderTrain
{
    public String filePath;
    public static String dataLocalPath;


    public static void main(String[] args) throws Exception {

        dataLocalPath = DownloaderUtility.PREDICTGENDERDATA.Download();
        PredictGenderTrain dg = new PredictGenderTrain();
        dg.filePath =  new File(dataLocalPath,"Data").getAbsolutePath();
        System.out.println(dg.filePath);
        dg.train();
    }

    /**
     * This function uses GenderRecordReader and passes it to RecordReaderDataSetIterator for further training.
     */
    public void train()
    {
        int seed = 123456;
        double learningRate = 0.005;// was .01 but often got errors: "o.d.optimize.solvers.BaseOptimizer - Hit termination condition on iteration 0"
        int batchSize = 100;
        int nEpochs = 10;
        int numInputs;
        int numOutputs;
        int numHiddenNodes;

        try(GenderRecordReader rr = new GenderRecordReader(new ArrayList<String>() {{add("M");add("F");}}))
        {
            long st = System.currentTimeMillis();
            System.out.println("Preprocessing start time : " + st);

            rr.initialize(new FileSplit(new File(this.filePath)));

            long et = System.currentTimeMillis();
            System.out.println("Preprocessing end time : " + et);
            System.out.println("time taken to process data : " + (et-st) + " ms");

            numInputs = rr.maxLengthName * 5;  // multiplied by 5 as for each letter we use five binary digits like 00000
            numOutputs = 2;
            numHiddenNodes = 2 * numInputs + numOutputs;


            GenderRecordReader rr1 = new GenderRecordReader(new ArrayList<String>() {{add("M");add("F");}});
            rr1.initialize(new FileSplit(new File(this.filePath)));

            DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, numInputs, 2);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .biasInit(1)
                .l2(1e-4)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.RELU)
                    .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.RELU)
                    .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.SOFTMAX)
                    .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();

            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
            uiServer.attach(statsStorage);
            model.setListeners(new StatsListener(statsStorage));

            for ( int n = 0; n < nEpochs; n++)
            {
                while(trainIter.hasNext())
                {
                    model.fit(trainIter.next());
                }
                trainIter.reset();
            }

            model.save(new File(this.filePath + "PredictGender.net"),true);

            System.out.println("Evaluate model....");
            Evaluation eval = new Evaluation(numOutputs);
            while(trainIter.hasNext()){
                DataSet t = trainIter.next();
                INDArray features = t.getFeatures();
                INDArray lables = t.getLabels();
                INDArray predicted = model.output(features,false);
                eval.eval(lables, predicted);
            }

            //Print the evaluation statistics
            System.out.println(eval.stats());
        }
        catch(Exception e)
        {
            System.out.println("Exception111 : " + e.getMessage());
        }
    }
}
