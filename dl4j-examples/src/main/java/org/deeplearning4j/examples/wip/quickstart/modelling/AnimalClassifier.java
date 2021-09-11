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

package org.deeplearning4j.examples.wip.quickstart.modelling;


import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.examples.utils.DownloaderUtility;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * Issue Summary: The labels in the test data are all the same. So even though net trains properly it gives terrible eval results. And this is confusing to users!
 *
 * Github Issue: https://github.com/eclipse/deeplearning4j-examples/issues/937
 *
 * This example is intended to be a simple CSV classifier that separates the training data
 * from the test data for the classification of animals. It would be suitable as a beginner's
 * example because not only does it load CSV data into the network, it also shows how to extract the
 * data and display the results of the classification, as well as a simple method to map the labels
 * from the testing data into the results.
 *
 * @author Clay Graham
 */
public class AnimalClassifier {

    private static Logger log = LoggerFactory.getLogger(AnimalClassifier.class);

    private static Map<Integer, String> eats;
    private static Map<Integer, String> sounds;
    private static Map<Integer, String> classifiers;

    public static String dataLocalPath;

    @SuppressWarnings("DuplicatedCode")
    public static void main(String[] args) throws Exception {
        dataLocalPath = DownloaderUtility.DATAEXAMPLES.Download();
        eats = readEnumCSV("animals/eats.csv");
        sounds = readEnumCSV("animals/sounds.csv");
        classifiers = readEnumCSV("animals/classifiers.csv");

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 4;     //5 values in each row of the animals.csv CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 3;     //3 classes (types of animals) in the animals data set. Classes have integer values 0, 1 or 2

        int batchSizeTraining = 30;    //Animals training data set: 30 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
        DataSet trainingData = readCSVDataset(
            "animals/animals_train.csv",
            batchSizeTraining, labelIndex, numClasses);

        // this is the data we want to classify
        int batchSizeTest = 44;
        DataSet testData = readCSVDataset("animals/animals.csv",
            batchSizeTest, labelIndex, numClasses);

        // make the data model for records prior to normalization, because it
        // changes the data.
        Map<Integer, Map<String, Object>> animals = makeAnimalsForTesting(testData);


        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

        //Configure neural network
        final int numInputs = 4;
        int outputNum = 3;
        int epochs = 1000;
        long seed = 6;

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .updater(new Sgd(0.1))
            .l2(1e-4)
            .list()
            .layer(new DenseLayer.Builder().nIn(numInputs).nOut(3).build())
            .layer(new DenseLayer.Builder().nIn(3).nOut(3).build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX).nIn(3).nOut(outputNum).build())
            .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        for (int i = 0; i < epochs; i++) {
            model.fit(trainingData);
        }

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(testData.getFeatures());

        eval.eval(testData.getLabels(), output);
        log.info(eval.stats());

        setFittedClassifiers(output, animals);
        logAnimals(animals);
    }

    private static void logAnimals(Map<Integer, Map<String, Object>> animals) {
        for (Map<String, Object> a : animals.values())
            log.info(a.toString());
    }

    private static void setFittedClassifiers(INDArray output, Map<Integer, Map<String, Object>> animals) {
        for (int i = 0; i < output.rows(); i++) {

            // set the classification from the fitted results
            animals.get(i).put("classifier",
                classifiers.get(maxIndex(getFloatArrayFromSlice(output.slice(i)))));
        }
    }

    /**
     * This method is to show how to convert the INDArray to a float array. This is to
     * provide some more examples on how to convert INDArray to types that are more java
     * centric.
     */
    private static float[] getFloatArrayFromSlice(INDArray rowSlice) {
        float[] result = new float[rowSlice.columns()];
        for (int i = 0; i < rowSlice.columns(); i++) {
            result[i] = rowSlice.getFloat(i);
        }
        return result;
    }

    /**
     * find the maximum item index. This is used when the data is fitted and we
     * want to determine which class to assign the test row to
     */
    private static int maxIndex(float[] vals) {
        int maxIndex = 0;
        for (int i = 1; i < vals.length; i++) {
            float newnumber = vals[i];
            if ((newnumber > vals[maxIndex])) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * take the dataset loaded for the matric and make the record model out of it so
     * we can correlate the fitted classifier to the record.
     */
    private static Map<Integer, Map<String, Object>> makeAnimalsForTesting(DataSet testData) {
        Map<Integer, Map<String, Object>> animals = new HashMap<>();

        INDArray features = testData.getFeatures();
        for (int i = 0; i < features.rows(); i++) {
            INDArray slice = features.slice(i);
            Map<String, Object> animal = new HashMap<>();

            //set the attributes
            animal.put("yearsLived", slice.getInt(0));
            animal.put("eats", eats.get(slice.getInt(1)));
            animal.put("sounds", sounds.get(slice.getInt(2)));
            animal.put("weight", slice.getFloat(3));

            animals.put(i, animal);
        }
        return animals;
    }


    private static Map<Integer, String> readEnumCSV(String csvFileClasspath) {
        try {
            List<String> lines = IOUtils.readLines(new FileInputStream(new File(dataLocalPath, csvFileClasspath)), StandardCharsets.UTF_8);
            Map<Integer, String> enums = new HashMap<>();
            for (String line : lines) {
                String[] parts = line.split(",");
                enums.put(Integer.parseInt(parts[0]), parts[1]);
            }
            return enums;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * used for testing and training*
     */
    private static DataSet readCSVDataset(
        String csvFileClasspath, int batchSize, int labelIndex, int numClasses)
        throws IOException, InterruptedException {

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(dataLocalPath, csvFileClasspath)));
        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
        return iterator.next();
    }
}
