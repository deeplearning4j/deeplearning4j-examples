package org.deeplearning4j.transferlearning.vgg16;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.input.PortableDataStream;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.transferlearning.vgg16.dataHelpers.FeaturizedPreSave;
import org.deeplearning4j.transferlearning.vgg16.dataHelpers.FlowerDataSetIteratorFeaturized;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import scala.Tuple2;

import java.io.IOException;
import java.io.OutputStream;

/**
 * @author susaneraly on 3/10/17.
 *
 * Important:
 * Run the class "FeaturizePreSave" before attempting to run this. The outputs at the boundary of the frozen and unfrozen
 * vertices of a model are saved. These are referred to as "featurized" datasets in this description.
 * On a dataset of about 3000 images which is what is downloaded this can take "a while"
 *
 * Here we see how the transfer learning helper can be used to fit from a featurized datasets.
 * We attempt to train the same model architecture as the one in "EditLastLayerOthersFrozen".
 * Since the helper avoids the forward pass through the frozen layers we save on computation time when running multiple epochs.
 * In this manner, users can iterate quickly tweaking learning rates, weight initialization etc` to settle on a model that gives good results.
 */
public class FitFromFeaturized {

    public static final String featureExtractionLayer = FeaturizedPreSave.featurizeExtractionLayer;
    protected static final long seed = 12345;
    protected static final int numClasses = 5;
    protected static final int nEpochs = 3;
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FitFromFeaturized.class);
    @Parameter(names = "-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private boolean useSparkLocal = true;

    @Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
    private int batchSizePerWorker = 16;

    @Parameter(names = "-numEpochs", description = "Number of epochs for training")
    private int numEpochs = 15;
    @Parameter(names = "-hdfsRoot", description = "The root directory for hdfs for training")
    private String hdfsRoot = "/tmp";


    public static void main(String...args) throws Exception {
        new FitFromFeaturized().runMain(args);
    }

    public  void runMain(String [] args) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
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
            System.exit(1);
        }


        //Import vgg
        //Note that the model imported does not have an output layer (check printed summary)
        //  nor any training related configs (model from keras was imported with only weights and json)
        log.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
        ZooModel zooModel = new VGG16();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        log.info(vgg16.summary());
        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .learningRate(3e-5)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .seed(seed)
            .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor(featureExtractionLayer) //the specified layer and below are "frozen"
            .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
            .addLayer("predictions",
                new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(4096).nOut(numClasses)
                    .weightInit(WeightInit.DISTRIBUTION)
                    .dist(new NormalDistribution(0,0.2 * (2.0/(4096 + numClasses)))) //This weight init dist gave better results than Xavier
                    .activation(Activation.SOFTMAX).build(),
                "fc2")
            .build();

        //Instantiate the transfer learning helper to fit and output from the featurized dataset
        //The .unfrozenGraph() is the unfrozen subset of the computation graph passed in.
        //If using with a UI or a listener attach them directly to the unfrozenGraph instance
        //With each iteration updated params from unfrozenGraph are copied over to the original model
        TransferLearningHelper transferLearningHelper = new TransferLearningHelper(vgg16Transfer);
        log.info(transferLearningHelper.unfrozenGraph().summary());

        //Configuration for Spark training: see http://deeplearning4j.org/spark for explanation of these configuration options
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)    //Each DataSet object: contains (by default) 32 examples
            .averagingFrequency(5)
            .workerPrefetchNumBatches(2)            //Async prefetching: 2 examples per worker
            .batchSizePerWorker(batchSizePerWorker)
            .build();

        log.info(vgg16Transfer.summary());
        SparkConf sparkConf = new SparkConf();
        if(useSparkLocal)
            sparkConf.setMaster("local[*]");
        sparkConf.setAppName("vgg16");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        FileSystem fs = FileSystem.get(sc.hadoopConfiguration());

        SparkComputationGraph sparkComputationGraph = new SparkComputationGraph(sc,transferLearningHelper.unfrozenGraph(),tm);

        DataSetIterator trainIter = FlowerDataSetIteratorFeaturized.trainIterator();
        DataSetIterator testIter = FlowerDataSetIteratorFeaturized.testIterator();
        System.out.println("Writing train to hdfs");
        int trainCountWrote = 0;
        while(trainIter.hasNext()) {
            OutputStream os = fs.create(new Path(hdfsRoot + "/" + "train","dataset" + trainCountWrote++));
            trainIter.next().save(os);
            os.close();
        }

        System.out.println("Writing test to hdfs");
        String testDir = hdfsRoot + "/" + "test";
        int testCountWrote = 0;
        while(testIter.hasNext()) {
            OutputStream os = fs.create(new Path(testDir,"dataset" + testCountWrote++));
            testIter.next().save(os);
            os.close();
        }


        for (int epoch = 0; epoch < nEpochs; epoch++) {
            sparkComputationGraph.fit(hdfsRoot + "/train");
            log.info("Epoch #" + epoch + " complete");
        }

        JavaRDD<DataSet> data = sc.binaryFiles(testDir + "/*").map(new LoadDataFunction());


        Evaluation eval = sparkComputationGraph.evaluate(data);
        log.info("Eval stats BEFORE fit.....");
        log.info(eval.stats()+"\n");
        testIter.reset();


        log.info("Model build complete");
    }

    private static class LoadDataFunction implements Function<Tuple2<String, PortableDataStream>, DataSet> {
        @Override
        public DataSet call(Tuple2<String, PortableDataStream> v1) throws Exception {
            DataSet d = new DataSet();
            d.load(v1._2().open());
            return d;
        }
    }
}
