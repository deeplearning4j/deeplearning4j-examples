package org.deeplearning4j.examples.recurrent.seqclassification;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.primitives.Pair;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Sequence Classification Example Using a LSTM Recurrent Neural Network
 *
 * This example learns how to classify univariate time series as belonging to one of six categories.
 * Categories are: Normal, Cyclic, Increasing trend, Decreasing trend, Upward shift, Downward shift
 *
 * Data is the UCI Synthetic Control Chart Time Series Data Set
 * Details:     https://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series
 * Data:        https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data
 * Image:       https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/data.jpeg
 *
 * This example proceeds as follows:
 * 1. Download and prepare the data (in downloadUCIData() method)
 *    (a) Split the 600 sequences into train set of size 450, and test set of size 150
 *    (b) Write the data into a format suitable for loading using the CSVSequenceRecordReader for sequence classification
 *        This format: one time series per file, and a separate file for the labels.
 *        For example, train/features/0.csv is the features using with the labels file train/labels/0.csv
 *        Because the data is a univariate time series, we only have one column in the CSV files. Normally, each column
 *        would contain multiple values - one time step per row.
 *        Furthermore, because we have only one label for each time series, the labels CSV files contain only a single value
 *
 * 2. Load the training data using CSVSequenceRecordReader (to load/parse the CSV files) and SequenceRecordReaderDataSetIterator
 *    (to convert it to DataSet objects, ready to train)
 *    For more details on this step, see: http://deeplearning4j.org/usingrnns#data
 *
 * 3. Normalize the data. The raw data contain values that are too large for effective training, and need to be normalized.
 *    Normalization is conducted using NormalizerStandardize, based on statistics (mean, st.dev) collected on the training
 *    data only. Note that both the training data and test data are normalized in the same way.
 *
 * 4. Configure the network
 *    The data set here is very small, so we can't afford to use a large network with many parameters.
 *    We are using one small LSTM layer and one RNN output layer
 *
 * 5. Train the network for 40 epochs
 *    At each epoch, evaluate and print the accuracy and f1 on the test set
 *
 * @author Alex Black
 */
public class UCISequenceClassificationExample {
    private static final Logger log = LoggerFactory.getLogger(UCISequenceClassificationExample.class);

    //'baseDir': Base directory for the data. Change this if you want to save the data somewhere else
    private static File baseDir = new File("src/main/resources/uci/");
    private static File baseTrainDir = new File(baseDir, "train");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");
    private static File baseTestDir = new File(baseDir, "test");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");

    public static void main(String[] args) throws Exception {
        downloadUCIData();

        // ----- Load the training data -----
        //Note that we have 450 training files for features: train/features/0.csv through train/features/449.csv
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));

        int miniBatchSize = 10;
        int numLabelClasses = 6;
        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        //Normalize the training data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);              //Collect training data statistics
        trainData.reset();

        //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
        trainData.setPreProcessor(normalizer);


        // ----- Load the test data -----
        //Same process as for the training data.
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, 149));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, 149));

        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        testData.setPreProcessor(normalizer);   //Note that we are using the exact same normalization process as the training data


        // ----- Configure the network -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)    //Random number generator seed for improved repeatability. Optional.
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS)
                .learningRate(0.005)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(10).nOut(numLabelClasses).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new ScoreIterationListener(20));   //Print the score (loss function value) every 20 iterations


        // ----- Train the network, evaluating the test set performance at each epoch -----
        int nEpochs = 40;
        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainData);

            //Evaluate on the test set:
            Evaluation evaluation = net.evaluate(testData);
            log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));

            testData.reset();
            trainData.reset();
        }

        log.info("----- Example Complete -----");
    }


    //This method downloads the data, and converts the "one time series per line" format into a suitable
    //CSV sequence format that DataVec (CsvSequenceRecordReader) and DL4J can read.
    private static void downloadUCIData() throws Exception {
        if (baseDir.exists()) return;    //Data already exists, don't download it again

        String url = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data";
        String data = IOUtils.toString(new URL(url));

        String[] lines = data.split("\n");

        //Create directories
        baseDir.mkdir();
        baseTrainDir.mkdir();
        featuresDirTrain.mkdir();
        labelsDirTrain.mkdir();
        baseTestDir.mkdir();
        featuresDirTest.mkdir();
        labelsDirTest.mkdir();

        int lineCount = 0;
        List<Pair<String, Integer>> contentAndLabels = new ArrayList<>();
        for (String line : lines) {
            String transposed = line.replaceAll(" +", "\n");

            //Labels: first 100 examples (lines) are label 0, second 100 examples are label 1, and so on
            contentAndLabels.add(new Pair<>(transposed, lineCount++ / 100));
        }

        //Randomize and do a train/test split:
        Collections.shuffle(contentAndLabels, new Random(12345));

        int nTrain = 450;   //75% train, 25% test
        int trainCount = 0;
        int testCount = 0;
        for (Pair<String, Integer> p : contentAndLabels) {
            //Write output in a format we can read, in the appropriate locations
            File outPathFeatures;
            File outPathLabels;
            if (trainCount < nTrain) {
                outPathFeatures = new File(featuresDirTrain, trainCount + ".csv");
                outPathLabels = new File(labelsDirTrain, trainCount + ".csv");
                trainCount++;
            } else {
                outPathFeatures = new File(featuresDirTest, testCount + ".csv");
                outPathLabels = new File(labelsDirTest, testCount + ".csv");
                testCount++;
            }

            FileUtils.writeStringToFile(outPathFeatures, p.getFirst());
            FileUtils.writeStringToFile(outPathLabels, p.getSecond().toString());
        }
    }
}
