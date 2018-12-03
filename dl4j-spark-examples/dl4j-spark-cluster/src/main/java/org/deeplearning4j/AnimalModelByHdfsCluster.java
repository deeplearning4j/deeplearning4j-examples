package org.deeplearning4j;


import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.DatasetIteratorFromHdfs;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;


/**
 * @Description: Training Model based on distributed File system here's hdfs,the dataset is image
 * The hadoop clusters've two nodes, one is master node that its domain name is "cluster1" , the domain name of the slave node is "cluster2"
 * @author wangfeng
 */
public class AnimalModelByHdfsCluster {
    private static final Logger log = LoggerFactory.getLogger(AnimalModelByHdfsCluster.class);
    private static int height = 100;
    private static int width = 100;
    private static int channels = 3;
    private static long seed = 42;
    protected static int batchSize = 10;
    protected static int epochs = 50;

    private static String rootPath = System.getProperty("user.dir");
    private static String modelPath = "/home/out/AnimalModelByHdfsClusterModel.json";
    ;
    public static void main(String[] args) throws  Exception{

        File modelFile = new File(modelPath);
        boolean hasFile = modelFile.exists()?true:modelFile.createNewFile();
        log.info( modelFile.getPath() );


        long startTime = System.currentTimeMillis();
        System.out.println(startTime);
        MultiLayerNetwork network = lenetModel();
        network.init();

        File statsFile = new File("/home/AnimalModelByHdfsTrainingStats.dl4j");
        StatsStorage statsStorage = new FileStatsStorage(statsFile);
        network.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));

        DataSetIterator trainIterator = new DatasetIteratorFromHdfs(batchSize,true);

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        trainIterator.setPreProcessor(scaler);

        for ( int i = 0; i < epochs; i ++ ) {
            System.out.println("Epoch=====================" + i);
            network.fit(trainIterator);
        }
        ModelSerializer.writeModel(network, modelFile,true);
        long endTime = System.currentTimeMillis();
        System.out.println("=============run time=====================" + (endTime - startTime));

        log.info("Evaluate model....");
        DataSetIterator validateIterator = new DatasetIteratorFromHdfs(batchSize,false);
        //scaler.fit(trainIterator);
        validateIterator.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(validateIterator);
        log.info(eval.stats(true));

        // Example on how to get predict results with trained model. Result for first example in minibatch is printed
        validateIterator.reset();
        org.nd4j.linalg.dataset.DataSet validateDataSet = validateIterator.next();
        List<String> allClassLabels = validateIterator.getLabels();
        int labelIndex = validateDataSet.getLabels().argMax(1).getInt(0);
        int[] predictedClasses = network.predict(validateDataSet.getFeatures());
        String expectedResult = allClassLabels.get(labelIndex);
        String modelPrediction = allClassLabels.get(predictedClasses[0]);
        System.out.print("\n\n======================================================\n\n");
        System.out.print("\n\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelPrediction + "\n\n");


    }
    public static MultiLayerNetwork lenetModel() {
        /**
         * Revisde Lenet Model approach developed by ramgo2 achieves slightly above random
         * Reference: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
         **/
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.005) // tried 0.0001, 0.0005
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.0001,0.9))
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}).name("cnn1")
                        .nIn(channels).nOut(50).biasInit(0).build())
                .layer(1, new SubsamplingLayer.Builder(new int[]{2,2}, new int[]{2,2}).name("maxpool1").build())
                .layer(2, new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{5, 5}, new int[]{1, 1}).name("cnn2")
                        .nOut(100).biasInit(0).build())
                .layer(3, new SubsamplingLayer.Builder(new int[]{2,2}, new int[]{2,2}).name("maxpool2").build())
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(4)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }

}
