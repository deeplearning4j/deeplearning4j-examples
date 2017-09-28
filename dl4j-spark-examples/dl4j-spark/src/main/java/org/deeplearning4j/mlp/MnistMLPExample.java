package org.deeplearning4j.mlp;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Train a simple/small MLP on MNIST data using Spark, then evaluate it on the test set in a distributed manner
 *
 * Note that the network being trained here is too small to make proper use of Spark - but it shows the configuration
 * and evaluation used for Spark training.
 *
 *
 * To run the example locally: Run the example as-is. The example is set up to use Spark local by default.
 * NOTE: Spark local should only be used for development/testing. For data parallel training on a single machine
 * (for example, multi-GPU systems) instead use ParallelWrapper (which is faster than using Spark for training on a single machine).
 * See for example MultiGpuLenetMnistExample in dl4j-cuda-specific-examples
 *
 * To run the example using Spark submit (for example on a cluster): pass "-useSparkLocal false" as the application argument,
 *   OR first modify the example by setting the field "useSparkLocal = false"
 *
 * @author Alex Black
 */
public class MnistMLPExample {
    private static final Logger log = LoggerFactory.getLogger(MnistMLPExample.class);

    @Parameter(names = "-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private boolean useSparkLocal = true;

    @Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
    private int batchSizePerWorker = 16;

    @Parameter(names = "-numEpochs", description = "Number of epochs for training")
    private int numEpochs = 2;

    public static void main(String[] args) throws Exception {
        new MnistMLPExample().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        //Handle command line arguments
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try { Thread.sleep(500); } catch (Exception e2) { }
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
        DataSetIterator iterTrain = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
        DataSetIterator iterTest = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
        List<DataSet> trainDataList = new ArrayList<>();
        List<DataSet> testDataList = new ArrayList<>();
        while (iterTrain.hasNext()) {
            trainDataList.add(iterTrain.next());
        }
        while (iterTest.hasNext()) {
            testDataList.add(iterTest.next());
        }

        JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);
        JavaRDD<DataSet> testData = sc.parallelize(testDataList);


        //----------------------------------
        //Create network configuration and conduct network training
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .activation(Activation.LEAKYRELU)
            .weightInit(WeightInit.XAVIER)
            .learningRate(0.1)
            .updater(Updater.NESTEROVS)// To configure: .updater(Nesterovs.builder().momentum(0.9).build())
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(28 * 28).nOut(500).build())
            .layer(1, new DenseLayer.Builder().nIn(500).nOut(100).build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX).nIn(100).nOut(10).build())
            .pretrain(false).backprop(true)
            .build();

        //Configuration for Spark training: see http://deeplearning4j.org/spark for explanation of these configuration options
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)    //Each DataSet object: contains (by default) 32 examples
            .averagingFrequency(5)
            .workerPrefetchNumBatches(2)            //Async prefetching: 2 examples per worker
            .batchSizePerWorker(batchSizePerWorker)
            .build();

        //Create the Spark network
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);

        //Execute training:
        for (int i = 0; i < numEpochs; i++) {
            sparkNet.fit(trainData);
            log.info("Completed Epoch {}", i);
        }

        //Perform evaluation (distributed)
//        Evaluation evaluation = sparkNet.evaluate(testData);
        Evaluation evaluation = sparkNet.doEvaluation(testData, 64, new Evaluation(10))[0]; //Work-around for 0.9.1 bug: see https://deeplearning4j.org/releasenotes
        log.info("***** Evaluation *****");
        log.info(evaluation.stats());

        //Delete the temp training files, now that we are done with them
        tm.deleteTempFiles(sc);

        log.info("***** Example Complete *****");
    }
}
