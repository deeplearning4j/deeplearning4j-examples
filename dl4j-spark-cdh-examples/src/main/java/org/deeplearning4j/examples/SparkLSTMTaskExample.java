package org.deeplearning4j.examples;

import org.deeplearning4j.eval.ROC;
import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.api.TrainingMaster;

/**
 * EXERCISE 4: train a LSTM using spark to predict mortality using the Physionet
 * Challenge 2012 data publicly available at https://physionet.org/challenge/2012/
 *
 *
 */

public class SparkLSTMTaskExample {
    private static File baseDir = new File("resources/physionet2012/");

    private static File featuresDir = new File(baseDir, "sequence");

    /* Task-specific configuration */
    private static File labelsDir = new File(baseDir, "mortality");

    // For logging with SL4J
    private static final Logger log = LoggerFactory.getLogger(SparkLSTMTaskExample.class);


    // Number of training, validation, test examples
    public static final int NB_TRAIN_EXAMPLES = 3200;
    public static final int NB_VALID_EXAMPLES = 400;
    public static final int NB_TEST_EXAMPLES = 4000 - NB_TRAIN_EXAMPLES - NB_VALID_EXAMPLES;

    // Number of features (inputs), heart rate, blood pressure
    public static final int NB_INPUTS = 86;

    public static final int NB_EPOCHS = 5;
    public static final int RANDOM_SEED = 1234;
    public static final double LEARNING_RATE = 0.032;
    public static final int BATCH_SIZE = 40;
    public static final int lstmLayerSize = 200;    //Number of units in each GravesLSTM layer

    public static void main(String[] args) throws IOException, InterruptedException {

        // Step 1: Set up Spark Conf
        boolean useSparkLocal = false;

        SparkConf sparkConf = new SparkConf(); 

        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }

        sparkConf.setAppName("Time Series LSTM Physionet 2012");

        // Spark application, connection to Spark environment
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        // STEP 2: ETL/vectorization

        // Load training data

        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1, ",");
        trainFeatures.initialize( new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));

        int numLabelClasses = 2;

        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels,
                BATCH_SIZE, numLabelClasses, false,SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        // Load validation data
        SequenceRecordReader validFeatures = new CSVSequenceRecordReader(1, ",");
        validFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES , NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES  - 1));
        SequenceRecordReader validLabels = new CSVSequenceRecordReader();
        validLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES , NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES  - 1));
        DataSetIterator validData = new SequenceRecordReaderDataSetIterator(validFeatures, validLabels,
                BATCH_SIZE, numLabelClasses, false,SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        // Load test data
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader(1, ",");
        testFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES+ NB_VALID_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES + NB_TEST_EXAMPLES - 1));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES+ NB_VALID_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES + NB_TEST_EXAMPLES - 1));
        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels,
                BATCH_SIZE, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        // STEP 3: Parallelize data

        List<DataSet> trainDataList = new ArrayList<>();
        List<DataSet> validDataList = new ArrayList<>();
        List<DataSet> testDataList = new ArrayList<>();

        while (trainData.hasNext()) {
            trainDataList.add(trainData.next());
        }
        while(validData.hasNext()){
            validDataList.add(validData.next());
        }
        while (testData.hasNext()) {
            testDataList.add(testData.next());
        }

        JavaRDD<DataSet> JtrainData = sc.parallelize(trainDataList);
        JavaRDD<DataSet> JvalidData = sc.parallelize(validDataList);
        JavaRDD<DataSet> JtestData = sc.parallelize(testDataList);

        // STEP 4: Model configuration

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(RANDOM_SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(LEARNING_RATE)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAM)
                .dropOut(0.25)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(NB_INPUTS).nOut(lstmLayerSize).activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.XENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
                        .nIn(lstmLayerSize).nOut(numLabelClasses).build())
                .pretrain(false).backprop(true)
                .build();


        // Step 5: Spark Distributed Training Configuration

        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(BATCH_SIZE)    //Each DataSet object: contains (by default) 32 examples
                .averagingFrequency(5)
                .workerPrefetchNumBatches(2)
                .batchSizePerWorker(BATCH_SIZE)
                .build();

        // Step 6: Create and train Spark network

        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);

        for (int i = 0; i < NB_EPOCHS; i++) {
            sparkNet.fit(JtrainData);
            log.info("Completed Epoch {}", i);

            ROC roc = sparkNet.evaluateROC(JtrainData);
            log.info("***** Train Evaluation *****");
            log.info("{}", roc.calculateAUC());

            roc = sparkNet.evaluateROC(JvalidData);
            log.info("***** Valid Evaluation *****");
            log.info("{}", roc.calculateAUC());
        }

        ROC roc = sparkNet.evaluateROC(JtestData);
        log.info("***** Test Evaluation *****");
        log.info("{}", roc.calculateAUC());
    }
}
