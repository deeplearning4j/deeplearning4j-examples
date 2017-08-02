package org.deeplearning4j.stats;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.rnn.SparkLSTMCharacterExample;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.stats.EventStats;
import org.deeplearning4j.spark.stats.StatsUtils;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * This example is designed to show how to use DL4J's Spark training benchmarking/debugging/timing functionality.
 * For details: See https://deeplearning4j.org/spark#sparkstats
 *
 * The idea with this tool is to capture statistics on various aspects of Spark training, in order to identify
 * and debug performance issues.
 *
 * For the sake of the example, we will be using a network configuration and data as per the SparkLSTMCharacterExample.
 *
 *
 * To run the example locally: Run the example as-is. The example is set up to use Spark local.
 *
 * To run the example using Spark submit (for example on a cluster): pass "-useSparkLocal false" as the application argument,
 *   OR first modify the example by setting the field "useSparkLocal = false"
 *
 * NOTE: On some clusters without internet access, this example may fail with "Error querying NTP server"
 * See: https://deeplearning4j.org/spark#sparkstatsntp
 *
 * @author Alex Black
 */
public class TrainingStatsExample {
    private static final Logger log = LoggerFactory.getLogger(TrainingStatsExample.class);

    @Parameter(names="-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private boolean useSparkLocal = true;

    public static void main(String[] args) throws Exception {
        new TrainingStatsExample().entryPoint(args);
    }

    private void entryPoint(String[] args) throws Exception {
        //Handle command line arguments
        JCommander jcmdr = new JCommander(this);
        try{
            jcmdr.parse(args);
        } catch(ParameterException e){
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try{ Thread.sleep(500); } catch(Exception e2){ }
            throw e;
        }


        //Set up network configuration:
        MultiLayerConfiguration config = getConfiguration();

        //Set up the Spark-specific configuration
        int examplesPerWorker = 8;      //i.e., minibatch size that each worker gets
        int averagingFrequency = 3;     //Frequency with which parameters are averaged

        //Set up Spark configuration and context
        SparkConf sparkConf = new SparkConf();
        if(useSparkLocal){
            sparkConf.setMaster("local[*]");
            log.info("Using Spark Local");
        }
        sparkConf.setAppName("DL4J Spark Stats Example");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        //Get data. See SparkLSTMCharacterExample for details
        JavaRDD<DataSet> trainingData = SparkLSTMCharacterExample.getTrainingData(sc);

        //Set up the TrainingMaster. The TrainingMaster controls how learning is actually executed on Spark
        //Here, we are using standard parameter averaging
        int examplesPerDataSetObject = 1;   //We haven't pre-batched our data: therefore each DataSet object contains 1 example
        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
                .workerPrefetchNumBatches(2)    //Async prefetch 2 batches for each worker
                .averagingFrequency(averagingFrequency)
                .batchSizePerWorker(examplesPerWorker)
                .build();

        //Create the Spark network
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, config, tm);

        //*** Tell the network to collect training statistics. These will NOT be collected by default ***
        sparkNetwork.setCollectTrainingStats(true);

        //Fit for 1 epoch:
        sparkNetwork.fit(trainingData);

        //Delete the temp training files, now that we are done with them (if fitting for multiple epochs: would be re-used)
        tm.deleteTempFiles(sc);

        //Get the statistics:
        SparkTrainingStats stats = sparkNetwork.getSparkTrainingStats();
        Set<String> statsKeySet = stats.getKeySet();    //Keys for the types of statistics
        log.info("--- Collected Statistics ---");
        for(String s : statsKeySet){
            log.info(s);
        }

        //Demo purposes: get one statistic and print it
        String first = statsKeySet.iterator().next();
        List<EventStats> firstStatEvents = stats.getValue(first);
        EventStats es = firstStatEvents.get(0);
        log.info("Training stats example:");
        log.info("Machine ID:     " + es.getMachineID());
        log.info("JVM ID:         " + es.getJvmID());
        log.info("Thread ID:      " + es.getThreadID());
        log.info("Start time ms:  " + es.getStartTime());
        log.info("Duration ms:    " + es.getDurationMs());

        //Export a HTML file containing charts of the various stats calculated during training
        StatsUtils.exportStatsAsHtml(stats, "SparkStats.html",sc);
        log.info("Training stats exported to {}", new File("SparkStats.html").getAbsolutePath());

        log.info("****************Example finished********************");
    }


    //Configuration for the network we will be training
    private static MultiLayerConfiguration getConfiguration(){
        int lstmLayerSize = 200;					//Number of units in each GravesLSTM layer
        int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters

        Map<Character, Integer> CHAR_TO_INT = SparkLSTMCharacterExample.getCharToInt();
        int nIn = CHAR_TO_INT.size();
        int nOut = CHAR_TO_INT.size();

        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .learningRate(0.1)
            .updater(Updater.RMSPROP)   //To configure: .updater(new RmsProp(0.95))
            .seed(12345)
            .regularization(true).l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(lstmLayerSize).activation(Activation.TANH).build())
            .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize).activation(Activation.TANH).build())
            .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
                .nIn(lstmLayerSize).nOut(nOut).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
            .pretrain(false).backprop(true)
            .build();

        return conf;
    }
}
