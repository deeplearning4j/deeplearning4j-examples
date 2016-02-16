package org.deeplearning4j.examples.convolution;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.Random;

/**
 * @author sonali
 */
public class CNNIrisExample {

    private static final Logger LOG = LoggerFactory.getLogger(CNNIrisExample.class);

    public static void main(String[] args) {

        final int numRows = 2;
        final int numColumns = 2;
        int nChannels = 1;
        int outputNum = 3;
        int iterations = 10;
        int splitTrainNum = 100;
        int seed = 123;
        int listenerFreq = 1;


        /**
         *Set a neural network configuration with multiple layers
         */
        LOG.info("Load data....");
        DataSetIterator irisIter = new IrisDataSetIterator(150, 150);
        DataSet iris = irisIter.next();
        iris.normalizeZeroMeanZeroUnitVariance();
        System.out.println("Loaded " + iris.labelCounts());
        Nd4j.shuffle(iris.getFeatureMatrix(), new Random(seed), 1);
        Nd4j.shuffle(iris.getLabels(),new Random(seed),1);
        SplitTestAndTrain trainTest = iris.splitTestAndTrain(splitTrainNum, new Random(seed));

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(2)
                .layer(0, new ConvolutionLayer.Builder(1, 1)
                        .nIn(nChannels)
                        .nOut(1000)
                        .activation("relu")
                        .weightInit(WeightInit.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);

        new ConvolutionLayerSetup(builder,numRows,numColumns,nChannels);

        MultiLayerConfiguration conf = builder.build();

        LOG.info("Build model....");
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

        LOG.info("Train model....");
        System.out.println("Training on " + trainTest.getTrain().labelCounts());
        model.fit(trainTest.getTrain());

        LOG.info("Evaluate weights....");
        for(org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
            LOG.info("Weights: " + w);
        }

        LOG.info("Evaluate model....");
        System.out.println("Training on " + trainTest.getTest().labelCounts());

        Evaluation eval = new Evaluation(outputNum);
        INDArray output = model.output(trainTest.getTest().getFeatureMatrix());
        eval.eval(trainTest.getTest().getLabels(), output);
        LOG.info(eval.stats());

        LOG.info("****************Example finished********************");
    }
}
