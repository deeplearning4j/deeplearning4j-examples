package org.deeplearning4j.rnn.seq2seq;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by susaneraly on 7/23/17.
 */
public class AdditionRNNSparkLocal {

    public static final int NUM_DIGITS =2;
    //Random number generator seed, for reproducability
    public static final int seed = 1234;

    //Tweak these to tune the dataset size = batchSize * totalBatches
    public static int batchSize = 16;
    public static int totalBatches = 100;
    public static int nEpochs = 10;
    public static int nIterations = 1;
    public static int testSize = 100;

    //Tweak the number of hidden nodes
    public static final int numHiddenNodes = 128;

    //This is the size of the one hot vector
    public static final int FEATURE_VEC_SIZE = 14;
    private static final Logger log = LoggerFactory.getLogger(AdditionRNNSparkLocal.class);

    @Parameter(names = "-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private boolean useSparkLocal = true;

    @Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
    private int batchSizePerWorker = batchSize;

    @Parameter(names = "-numEpochs", description = "Number of epochs for training")
    private int numEpochs = 5;

    public static void main(String[] args) throws Exception {
        new AdditionRNNSparkLocal().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        //Handle command line arguments
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            throw e;
        }

        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("DL4J Spark MLP Example");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        //Load the data into memory then parallelize
        //This isn't a good approach in general - but is simple to use for this example
        CustomSequenceIterator iterator = new CustomSequenceIterator(seed, batchSize, totalBatches);
        List<MultiDataSet> trainDataList = new ArrayList<>();
        List<MultiDataSet> testDataList = new ArrayList<>();
        while (iterator.hasNext()) {
            trainDataList.add(iterator.next());
            testDataList.add(iterator.generateTest(testSize));
        }

        JavaRDD<MultiDataSet> trainData = sc.parallelize(trainDataList);
        JavaRDD<MultiDataSet> testData = sc.parallelize(testDataList);

        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
            .learningRate(0.25)
            .updater(Updater.ADAM)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(nIterations)
            .seed(seed)
            .graphBuilder()
            .addInputs("additionIn", "sumOut")
            .setInputTypes(InputType.recurrent(FEATURE_VEC_SIZE), InputType.recurrent(FEATURE_VEC_SIZE))
            .addLayer("encoder", new GravesLSTM.Builder().nIn(FEATURE_VEC_SIZE).nOut(numHiddenNodes).activation(Activation.SOFTSIGN).build(),"additionIn")
            .addVertex("lastTimeStep", new LastTimeStepVertex("additionIn"), "encoder")
            .addVertex("duplicateTimeStep", new DuplicateToTimeSeriesVertex("sumOut"), "lastTimeStep")
            .addLayer("decoder", new GravesLSTM.Builder().nIn(FEATURE_VEC_SIZE+numHiddenNodes).nOut(numHiddenNodes).activation(Activation.SOFTSIGN).build(), "sumOut","duplicateTimeStep")
            .addLayer("output", new RnnOutputLayer.Builder().nIn(numHiddenNodes).nOut(FEATURE_VEC_SIZE).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build(), "decoder")
            .setOutputs("output")
            .pretrain(false).backprop(true)
            .build();

        //Configuration for Spark training: see http://deeplearning4j.org/spark for explanation of these configuration options
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)    //Each DataSet object: contains (by default) 32 examples
            .averagingFrequency(5)
            .workerPrefetchNumBatches(2)            //Async prefetching: 2 examples per worker
            .batchSizePerWorker(batchSizePerWorker)
            .build();

        SparkComputationGraph sparkNet = new SparkComputationGraph(sc, configuration, tm);

        //Execute training:
        for (int i = 0; i < numEpochs; i++) {
            sparkNet.fitMultiDataSet(trainData);
            log.info("Completed Epoch {}", i);
        }

        //Perform evaluation (distributed)
        Evaluation evaluation = sparkNet.evaluateMDS(testData);
        log.info("***** Evaluation *****");
        log.info(evaluation.stats());

        //Delete the temp training files, now that we are done with them
        tm.deleteTempFiles(sc);

        log.info("***** Example Complete *****");
    }
}
