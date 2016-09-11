package org.deeplearning4j.examples.recurrent.seqClassification;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.collections.map.HashedMap;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.datavec.api.berkeley.Pair;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.util.CSVUtils;
import org.deeplearning4j.examples.util.NDArrayUtils;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.*;

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

    protected static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    protected static Map<Integer,String> classifiers = CSVUtils.readEnumCSV("/recurrent/seqClassification/UCISequence/uci/classifiers.csv");
    protected static File baseDir;
    protected static File baseTrainDir;
    protected static File featuresDirTrain;
    protected static File labelsDirTrain;
    protected static File baseTestDir;
    protected static File featuresDirTest;
    protected static File labelsDirTest;

    public static void main(String[] args) throws Exception {

        //'baseDir': Base directory for the data. Change this if you want to save the data somewhere else
        baseDir = new ClassPathResource("/recurrent/seqClassification/UCISequence/uci/").getFile();
        baseTrainDir = new File(baseDir, "train");
        featuresDirTrain = new File(baseTrainDir, "features");
        labelsDirTrain = new File(baseTrainDir, "labels");
        baseTestDir = new File(baseDir, "test");
        featuresDirTest = new File(baseTestDir, "features");
        labelsDirTest = new File(baseTestDir, "labels");

        trainNetworkAndMapTestClassifications(args);

        log.info("----- Example Complete -----");
    }

    public static Map<Integer,Map<String,Object>> trainNetworkAndMapTestClassifications(String[] args)
            throws IOException, InterruptedException {

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

        //make a copy of the test data
        DataSet ds = testData.next().copy();

        //build a model we can use to correlate classifications
        Map<Integer,Map<String,Object>> sequences = makeFeatureModelForTesting(ds);

        //reset it because we dont know if using next altered it prior to normalization
        testData.reset();

        // normailze
        testData.setPreProcessor(normalizer);   //Note that we are using the exact same normalization process as the training data

        // ----- Configure the network -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)    //Random number generator seed for improved repeatability. Optional.
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .learningRate(0.005)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(0, new GravesLSTM.Builder().activation("tanh").nIn(1).nOut(10).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax").nIn(10).nOut(numLabelClasses).build())
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

        INDArray output = net.output(ds.getFeatureMatrix());

        // the output is a list of rows, indexed to the original test
        // model with the scoring for the row by each classifier. What we need
        // to do is find the best score and then map it to the class.
        List scoringOutput = NDArrayUtils.makeRowsFromNDArray(output,6);

        //for each row model expect the index to be the same
        for (int i = 0; i <scoringOutput.size() ; i++) {
            // get the orginating model row
            Map<String,Object> sequenceToClassify = sequences.get(i);
            List<List<Double>> scoringForEachClassifierOnRow = (List<List<Double>>)scoringOutput.get(i);
            sequenceToClassify.put("classification",
                    getWinningClassificationForRow(scoringForEachClassifierOnRow,classifiers));

            log.info(String.format("row=%d class=%s",sequenceToClassify.get("rowNumber"), sequenceToClassify.get("classification")));

        }

        return sequences;

    }

    /**
     * find the highest score for each classifiaction row from the output of the network
     * and map the highest to the classification
     *
     * @param scoringForEachClassifierOnRow
     * @param classifiers
     * @return
     */
    public static String getWinningClassificationForRow(
            List<List<Double>> scoringForEachClassifierOnRow, Map<Integer,String> classifiers){

        TreeMap<Double,Integer> scoringByClassifier = new TreeMap<>(Collections.reverseOrder());
        for (int i = 0; i <scoringForEachClassifierOnRow.size() ; i++) {
            scoringByClassifier.put(
                    getSumForDataRow(scoringForEachClassifierOnRow.get(i)),
                    i);
        }

        //already sorted so get highest score
        return classifiers.get(scoringByClassifier.firstEntry().getValue());
    }


    /**
     * sum the row, if we were using java8:
     * Double sum = scoringForEachClassifierOnRow.get(i).stream().mapToDouble(Double::doubleValue).sum();
     *
     * @param row
     * @return
     */
    private static Double getSumForDataRow(List<Double> row){
        Double sum = 0.0;
        for (Double d: row ) {
            sum+=d;
        }
        return sum;
    }

    /**
     * take the testing dataset and create a model that can later be used to map the
     * resultant classification to.
     *
     * @param ds
     * @return
     */
    private static Map<Integer,Map<String,Object>> makeFeatureModelForTesting(DataSet ds) {

        Map<Integer,Map<String,Object>> items = new HashMap<>();

        INDArray features = ds.getFeatureMatrix();
        try {
            List<List<Double>> rows = NDArrayUtils.makeRowsFromNDArray(features,6);
            for (int i = 0; i < rows.size(); i++) {
                List<Double> row = rows.get(i);
                Map<String,Object> itemModel = new HashedMap();
                itemModel.put("rowNumber",i);
                itemModel.put("rowData",row);
                items.put(i,itemModel);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return items;
    }


}
