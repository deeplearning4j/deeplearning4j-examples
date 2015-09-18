package org.deeplearning4j.examples.regression;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.core.io.ClassPathResource;

/**
 * Created by agibsonccc on 9/16/15.
 */
public class RegressionExample {

    public static void main(String[] args) throws Exception {
        int seed = 123;
        int iterations = 100;
        RecordReader reader = new CSVRecordReader();
        reader.initialize(new FileSplit(new ClassPathResource("regression-example.txt").getFile()));
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) // Seed to lock in weight initialization for tuning
                .iterations(iterations) // # training iterations predict/classify & backprop
                .learningRate(1e-3f) // Optimization step size
                .optimizationAlgo(OptimizationAlgorithm.LBFGS) // Backprop method (calculate the gradients)
                .constrainGradientToUnitNorm(true).l2(2e-4).regularization(true)
                .list(1) // # NN layers (does not count input layer)
                .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                .nIn(12) // # input nodes
                                .nOut(1) // # output nodes
                                .activation("identity")
                                .weightInit(WeightInit.XAVIER)
                                .build()
                ) // NN layer type
                .build();
        DataSetIterator iter = new RecordReaderDataSetIterator(reader,null,2029,12,1,true);
        DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(0.9);
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(new ScoreIterationListener(1));
        network.fit(testAndTrain.getTrain());



    }

}
