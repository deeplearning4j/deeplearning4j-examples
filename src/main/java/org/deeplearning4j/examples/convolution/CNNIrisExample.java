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
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Random;

/**
 * @author sonali
 */
public class CNNIrisExample {

    private static Logger log = LoggerFactory.getLogger(CNNIrisExample.class);

    public static void main(String[] args) {

        final int numRows = 2;
        final int numColumns = 2;
        int nChannels = 1;
        int outputNum = 3;
        int numSamples = 150;
        int batchSize = 110;
        int iterations = 10;
        int splitTrainNum = 100;
        int seed = 123;
        int listenerFreq = 1;


        /**
         *Set a neural network configuration with multiple layers
         */
        log.info("Load data....");
        DataSetIterator irisIter = new IrisDataSetIterator(batchSize, numSamples);
        DataSet iris = irisIter.next();
        iris.normalizeZeroMeanZeroUnitVariance();

        SplitTestAndTrain trainTest = iris.splitTestAndTrain(splitTrainNum, new Random(seed));

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .batchSize(batchSize)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .constrainGradientToUnitNorm(true)
                .l2(2e-4)
                .regularization(true)
                .useDropConnect(true)
                .list(2)
                .layer(0, new ConvolutionLayer.Builder(new int[]{1, 1})
                        .nIn(nChannels)
                        .nOut(6).dropOut(0.5)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(6)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())

                .backprop(true).pretrain(false);

        new ConvolutionLayerSetup(builder,numRows,numColumns,nChannels);

        MultiLayerConfiguration conf = builder.build();

        log.info("Build model....");
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Train model....");
        model.fit(trainTest.getTrain());

        log.info("Evaluate weights....");
        for(org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
            log.info("Weights: " + w);
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        INDArray output = model.output(trainTest.getTest().getFeatureMatrix());
        eval.eval(trainTest.getTest().getLabels(), output);
        log.info(eval.stats());

        log.info("****************Example finished********************");
    }
}
