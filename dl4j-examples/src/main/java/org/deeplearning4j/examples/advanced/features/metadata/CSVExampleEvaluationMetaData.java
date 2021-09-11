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

package org.deeplearning4j.examples.advanced.features.metadata;

import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.utils.DownloaderUtility;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.meta.Prediction;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * This example is a version of the basic CSV example, but adds the following:
 * (a) Meta data tracking - i.e., where data for each example comes from
 * (b) Additional evaluation information - getting metadata for prediction errors
 *
 * @author Alex Black
 */
public class CSVExampleEvaluationMetaData {

    @SuppressWarnings("unused")
    public static void main(String[] args) throws  Exception {
        //First: get the dataset using the record reader. This is as per CSV example - see that example for details
        RecordReader recordReader = new CSVRecordReader(0, ',');
        recordReader.initialize(new FileSplit(new File(DownloaderUtility.IRISDATA.Download(),"iris.txt")));
        int labelIndex = 4;
        int numClasses = 3;
        int batchSize = 150;

        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        iterator.setCollectMetaData(true);  //Instruct the iterator to collect metadata, and store it in the DataSet objects
        DataSet allData = iterator.next();
        allData.shuffle(123);
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);  //Use 65% of data for training

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        //Let's view the example metadata in the training and test sets:
        List<RecordMetaData> trainMetaData = trainingData.getExampleMetaData(RecordMetaData.class);
        List<RecordMetaData> testMetaData = testData.getExampleMetaData(RecordMetaData.class);

        //Let's show specifically which examples are in the training and test sets, using the collected metadata
        System.out.println("  +++++ Training Set Examples MetaData +++++");
        String format = "%-20s\t%s";
        for(RecordMetaData recordMetaData : trainMetaData){
            System.out.println(String.format(format, recordMetaData.getLocation(), recordMetaData.getURI()));
            //Also available: recordMetaData.getReaderClass()
        }
        System.out.println("\n\n  +++++ Test Set Examples MetaData +++++");
        for(RecordMetaData recordMetaData : testMetaData){
            System.out.println(recordMetaData.getLocation());
        }


        //Normalize data as per basic CSV example
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set


        //Configure a simple model. We're not using an optimal configuration here, in order to show evaluation/errors, later
        final int numInputs = 4;
        int outputNum = 3;
        long seed = 6;

        System.out.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .updater(new Sgd(0.1))
            .l2(1e-4)
            .list()
            .layer(new DenseLayer.Builder().nIn(numInputs).nOut(3).build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX).nIn(3).nOut(outputNum).build())
            .build();

        //Fit the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        for( int i=0; i<50; i++ ) {
            model.fit(trainingData);
        }

        //Evaluate the model on the test set
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output, testMetaData);          //Note we are passing in the test set metadata here
        System.out.println(eval.stats());

        //Get a list of prediction errors, from the Evaluation object
        //Prediction errors like this are only available after calling iterator.setCollectMetaData(true)
        List<Prediction> predictionErrors = eval.getPredictionErrors();
        System.out.println("\n\n+++++ Prediction Errors +++++");
        for(Prediction p : predictionErrors){
            System.out.println("Predicted class: " + p.getPredictedClass() + ", Actual class: " + p.getActualClass()
                + "\t" + p.getRecordMetaData(RecordMetaData.class).getLocation());
        }

        //We can also load a subset of the data, to a DataSet object:
        List<RecordMetaData> predictionErrorMetaData = new ArrayList<>();
        for( Prediction p : predictionErrors ) predictionErrorMetaData.add(p.getRecordMetaData(RecordMetaData.class));
        DataSet predictionErrorExamples = iterator.loadFromMetaData(predictionErrorMetaData);
        normalizer.transform(predictionErrorExamples);  //Apply normalization to this subset

        //We can also load the raw data:
        List<Record> predictionErrorRawData = recordReader.loadFromMetaData(predictionErrorMetaData);

        //Print out the prediction errors, along with the raw data, normalized data, labels and network predictions:
        for(int i=0; i<predictionErrors.size(); i++ ){
            Prediction p = predictionErrors.get(i);
            RecordMetaData meta = p.getRecordMetaData(RecordMetaData.class);
            INDArray features = predictionErrorExamples.getFeatures().getRow(i,true);
            INDArray labels = predictionErrorExamples.getLabels().getRow(i,true);
            List<Writable> rawData = predictionErrorRawData.get(i).getRecord();

            INDArray networkPrediction = model.output(features);

            System.out.println(meta.getLocation() + ": "
                + "\tRaw Data: " + rawData
                + "\tNormalized: " + features
                + "\tLabels: " + labels
                + "\tPredictions: " + networkPrediction);
        }


        //Some other useful evaluation methods:
        List<Prediction> list1 = eval.getPredictions(1,2);                  //Predictions: actual class 1, predicted class 2
        List<Prediction> list2 = eval.getPredictionByPredictedClass(2);     //All predictions for predicted class 2
        List<Prediction> list3 = eval.getPredictionsByActualClass(2);       //All predictions for actual class 2
    }
}
