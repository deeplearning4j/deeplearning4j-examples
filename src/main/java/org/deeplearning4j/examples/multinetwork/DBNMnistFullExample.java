package org.deeplearning4j.examples.multinetwork;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collections;

/**
 * Created by agibsonccc on 9/11/14.
 */
public class DBNMnistFullExample {

    private static final Logger LOG = LoggerFactory.getLogger(DBNMnistFullExample.class);

    public static void main(String[] args) throws Exception {
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int numSamples = 60000;
        int batchSize = 100;
        int iterations = 10;
        int seed = 123;
        int listenerFreq = batchSize / 5;

        LOG.info("Load data....");
        DataSetIterator iter = new MnistDataSetIterator(batchSize,numSamples,true);
        DataSet trainingSet = iter.next();
        LOG.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .iterations(iterations)
                .momentum(0.5)
                .momentumAfter(Collections.singletonMap(3, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(3)
                .layer(0, new RBM.Builder().nIn(numRows*numColumns).nOut(500)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                        .visibleUnit(RBM.VisibleUnit.BINARY)
                        .hiddenUnit(RBM.HiddenUnit.BINARY)
                        .build())
                .layer(1, new RBM.Builder().nIn(500).nOut(250)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                        .visibleUnit(RBM.VisibleUnit.BINARY)
                        .hiddenUnit(RBM.HiddenUnit.BINARY)
                        .build())
                .layer(2, new RBM.Builder().nIn(250).nOut(200)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                        .visibleUnit(RBM.VisibleUnit.BINARY)
                        .hiddenUnit(RBM.HiddenUnit.BINARY)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(listenerFreq));

        LOG.info("Train model....");
        model.fit(trainingSet); // achieves end to end pre-training


        //logistic regression
        MultiLayerConfiguration confRegression = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .iterations(iterations)
                .momentum(0.5)
                .momentumAfter(Collections.singletonMap(3, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(1)
                .layer(0, new OutputLayer.Builder().nIn(200).nOut(10)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.MCXENT)
                        .build())
                .build();

        MultiLayerNetwork modelRegression = new MultiLayerNetwork(confRegression);
        modelRegression.init();
        modelRegression.setListeners(new ScoreIterationListener(listenerFreq));
        //train on the feature matrix based on the pretraining separately
        trainingSet.setFeatures(model.output(trainingSet.getFeatureMatrix()));


        modelRegression.fit(trainingSet);

 
        LOG.info("****************Example finished********************");

    }

}
