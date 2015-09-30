package org.deeplearning4j.examples.test;
import java.util.Collections;
import java.util.Random;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 9/16/15.
 */
public class Test {
    private static Logger LOGGER = LoggerFactory.getLogger(Test.class);

    public static void main(String[] args) {
        final int numRows = 4;
        final int numColumns = 1;
        int outputNum = 3;
        int numSamples = 150;
        int batchSize = 150;
        int iterations = 10;
        int splitTrainNum = (int) (batchSize * .8);
        int seed = 123;
        int listenerFreq = iterations/5;
        Nd4j.getRandom().setSeed(seed);
        LOGGER.info("Load data....");
        DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);
        DataSet iris = iter.next();
        iris.normalizeZeroMeanZeroUnitVariance();

        LOGGER.info("Split data....");
        SplitTestAndTrain testAndTrain = iris.splitTestAndTrain(splitTrainNum, new Random(seed));
        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        MultiLayerNetwork model = null;
        LOGGER.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) // Seed to lock in weight initialization for tuning
                .iterations(iterations) // # training iterations predict/classify & backprop
                .learningRate(1e-6f) // Optimization step size
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT) // Backprop method (calculate the gradients)
                .l1(1e-1).regularization(true).l2(2e-4)
                .useDropConnect(true)
                .list(2) // # NN layers (does not count input layer)
                .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                                .nIn(numRows * numColumns) // # input nodes
                                .nOut(3) // # fully connected hidden layer nodes. Add list if multiple layers.
                                .weightInit(WeightInit.XAVIER) // Weight initialization method
                                .k(1) // # contrastive divergence iterations
                                .activation("relu") // Activation function type
                                .lossFunction(LossFunctions.LossFunction.RMSE_XENT) // Loss function type
                                .updater(Updater.ADAGRAD)
                                .dropOut(0.5)
                                .build()
                ) // NN layer type
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .nIn(3) // # input nodes
                                .nOut(outputNum) // # output nodes
                                .activation("softmax")
                                .build()
                ) // NN layer type
                .build();
        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

        LOGGER.info("Train model....");
        model.fit(train);

//        for(int i=0; i<10; i++){
//            LOGGER.info("Output: {}", model.output(test.getFeatureMatrix(), Layer.TrainingMode.TEST).getRow(0));
//        }
    }

}
