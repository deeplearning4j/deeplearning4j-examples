package org.deeplearning4j.keras;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Generic Keras model import runner. Takes a CSV file in a format that CSVRecordReader understands
 * and a Keras model file in HDF5.
 *
 * Example usage:
 *  $SPARK_HOME/bin/spark-submit \
 *    --class org.deeplearning4j.keras.KerasImportCSVSparkRunner \
 *      -indexLabel 5 \
 *      -numClasses 4 \
 *      -modelFileName irisModel.h5 \
 *      -dataFileName iris.txt \
 *    --master $MASTER \
 *    --files iris.txt irisModel.h5
 *    dl4j.jar
 *
 *
 * @author Max Pumperla
 */
public class KerasImportCSVSparkRunner {

    private static final Logger log = LoggerFactory.getLogger(KerasImportCSVSparkRunner.class);

    @Parameter(names = "-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private boolean useSparkLocal = false;

    @Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
    private int batchSizePerWorker = 16;

    @Parameter(names = "-indexLabel", description = "Index of column that has labels")
    private int indexLabel = -1;

    @Parameter(names = "-modelFileName", description = "Name of the keras model file")
    private String modelFileName = "model.h5";

    @Parameter(names = "-dataFileName", description = "Name of the CSV file")
    private String dataFileName = "data.csv";

    @Parameter(names = "-numClasses", description = "Number of output classes")
    private int numClasses = -1;

    public static void main(String[] args) throws Exception {
        new KerasImportCSVSparkRunner().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {

        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            jcmdr.usage();
            try { Thread.sleep(500); } catch (Exception e2) { }
            throw e;
        }

        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("DL4J Keras model import runner");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        // Assume no header and comma separation
        int numLinesToSkip = 0;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);

        recordReader.initialize(new FileSplit(new File(dataFileName)));

        DataSetIterator iterator = new RecordReaderDataSetIterator(
            recordReader, batchSizePerWorker, indexLabel, numClasses);

        // Load all the data into a DataSet
        DataSet data = iterator.next();

        List<DataSet> dataList = new ArrayList<>();
        while (iterator.hasNext()) {
            dataList.add(iterator.next());
        }
        JavaRDD<DataSet> testData = sc.parallelize(dataList);


        MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights(modelFileName);

        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(40)
            .averagingFrequency(5)
            .workerPrefetchNumBatches(2)
            .batchSizePerWorker(40)
            .build();

        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, network, tm);

        // Distributed evaluation
        Evaluation evaluation = sparkNet.doEvaluation(testData, 64, new Evaluation(10))[0]; //Work-around for 0.9.1 bug: see https://deeplearning4j.org/releasenotes
        log.info(evaluation.stats());
        log.info("***** Example Complete *****");
    }
}
