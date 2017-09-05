package org.deeplearning4j.examples;

import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.input.PortableDataStream;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.writable.Writable;
import org.datavec.spark.functions.pairdata.BytesPairWritable;
import org.datavec.spark.functions.pairdata.PairSequenceRecordReaderBytesFunction;
import org.datavec.spark.functions.pairdata.PathToKeyConverter;
import org.datavec.spark.functions.pairdata.PathToKeyConverterFilename;
import org.datavec.spark.util.DataVecSparkUtil;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.datavec.DataVecSequencePairDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.util.SparkUtils;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.util.List;

public class SparkLSTMClinicalTimeSeriesClassificationExampleHDFS{

    // For logging with SL4J
    private static final Logger log = LoggerFactory.getLogger(SparkLSTMClinicalTimeSeriesClassificationExampleHDFS.class);

    // Number of training, validation, test examples
    public static final int NB_TRAIN_EXAMPLES = 3200;
    public static final int NB_VALID_EXAMPLES = 400;
    public static final int NB_TEST_EXAMPLES = 4000 - NB_TRAIN_EXAMPLES - NB_VALID_EXAMPLES;

    // Number of features (inputs), heart rate, blood pressure
    public static final int NB_INPUTS = 86;

    public static final int NB_EPOCHS = 25;
    public static final int RANDOM_SEED = 1234;
    public static final double LEARNING_RATE = 0.042;
    public static final int BATCH_SIZE = 40;
    public static final int lstmLayerSize = 200;    //Number of units in each GravesLSTM layer


    public static void main(String[] args) throws IOException, InterruptedException {
        // Set up Spark Context
        boolean useSparkLocal = false;

        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName("Spark Lstm example");
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        // Set Spark Configuration

        Configuration config = sc.hadoopConfiguration();
        config.set("fs.azure", "org.apache.hadoop.fs.azure.NativeAzureFileSystem");

        // Path to sequence and mortality directories via Azure blob container
        String featuresPath = "wasb:///resources/physionet2012/sequence";
        String labelsPath = "wasb:///resources/physionet2012/mortality";

        JavaPairRDD<Integer, String> featureFilePaths = SparkUtils.listPaths(sc, featuresPath).filter(s -> s.endsWith("csv")).mapToPair(new MapFn());
        JavaPairRDD<Integer, String> labelFilePaths = SparkUtils.listPaths(sc, labelsPath).filter(s -> s.endsWith("csv")).mapToPair(new MapFn());
        JavaPairRDD<Text, BytesPairWritable> rdd = featureFilePaths.join(labelFilePaths).mapToPair(new MapFn2());

        SequenceRecordReader srr1 = new CSVSequenceRecordReader(1, ",");;
        SequenceRecordReader srr2 = new CSVSequenceRecordReader();
        PairSequenceRecordReaderBytesFunction fn = new PairSequenceRecordReaderBytesFunction(srr1, srr2);
        JavaRDD<Tuple2<List<List<Writable>>, List<List<Writable>>>> writables = rdd.map(fn);

        int nClasses = 2;
        boolean regression = false;
        JavaRDD<DataSet> dataSets = writables.map(new DataVecSequencePairDataSetFunction(nClasses, regression, DataVecSequencePairDataSetFunction.AlignmentMode.ALIGN_END));

        // Split into train, validation, and test sets
        JavaRDD<DataSet>[] splits = dataSets.randomSplit(new double[] { 0.80, 0.10, 0.10 } ,1);

        JavaRDD<DataSet> JtrainData = splits[0];
        JavaRDD<DataSet> JvalidData = splits[1];
        JavaRDD<DataSet> JtestData = splits[2];

        // Neural Network Configuration

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(RANDOM_SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(LEARNING_RATE)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAM)
                .dropOut(0.25)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(NB_INPUTS).nOut(lstmLayerSize).activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
                        .nIn(lstmLayerSize).nOut(nClasses).build())
                .pretrain(false).backprop(true)
                .build();

        // Set Training Master

        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1)
                .averagingFrequency(3)
                .workerPrefetchNumBatches(2)
                .batchSizePerWorker(BATCH_SIZE)
                .build();

        // Initialize SparkMultiLayer

        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);

        // Train and evaluate data

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

    private static class MapFn implements PairFunction<String, Integer, String>{
        @Override
        public Tuple2<Integer, String> call(String s) throws Exception {
            int idx = s.lastIndexOf("/");
            idx = Math.max(idx, s.lastIndexOf("\\"));

            String sub = s.substring(idx+1, s.length()-4 );
            return new Tuple2<>(Integer.parseInt(sub), s);
        }
    }

    private static class MapFn2 implements PairFunction<Tuple2<Integer, Tuple2<String, String>>, Text, BytesPairWritable>{



        @Override
        public Tuple2<Text, BytesPairWritable> call(Tuple2<Integer, Tuple2<String, String>> t2) throws Exception {
            Configuration hc = new Configuration();
            Text t = new Text(String.valueOf(t2._1()));
            byte[] first;
            byte[] second;
            FileSystem fileSystem = FileSystem.get(hc);
            try (BufferedInputStream bis = new BufferedInputStream(fileSystem.open(new Path(t2._2()._1())))) {
                first = IOUtils.toByteArray(bis);
            }
            try (BufferedInputStream bis = new BufferedInputStream(fileSystem.open(new Path(t2._2()._2())))) {
                second = IOUtils.toByteArray(bis);
            }

            return new Tuple2<>(t, new BytesPairWritable(first, second, t2._2()._1(), t2._2()._1()));
        }
    }
}
