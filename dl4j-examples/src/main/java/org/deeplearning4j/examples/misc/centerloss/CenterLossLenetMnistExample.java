package org.deeplearning4j.examples.misc.centerloss;

import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.examples.unsupervised.variational.plot.PlotUtil;
import org.deeplearning4j.examples.userInterface.util.GradientsAndParamsListener;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Example: training an embedding using the center loss model, on MNIST
 * The motivation is to use the class labels to learn embeddings that have the following properties:
 * (a) Intra-class similarity (i.e., similar vectors for same numbers)
 * (b) Inter-class dissimilarity (i.e., different vectors for different numbers)
 *
 * Refer to the paper "A Discriminative Feature Learning Approach for Deep Face Recognition", Wen et al. (2016)
 * http://ydwen.github.io/papers/WenECCV16.pdf
 *
 * This
 *
 * @author Alex Black
 */
public class CenterLossLenetMnistExample {
    private static final Logger log = LoggerFactory.getLogger(CenterLossLenetMnistExample.class);

    public static void main(String[] args) throws Exception {
        int outputNum = 10; // The number of possible outcomes
        int batchSize = 64; // Test batch size
        int nEpochs = 10;   // Number of training epochs
        int seed = 123;

        //Lambda defines the relative strength of the center loss component.
        //lambda = 0.0 is equivalent to training with standard softmax only
        double lambda = 1.0;

        //Alpha can be thought of as the learning rate for the centers for each class
        double alpha = 0.1;

        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(10000, false, 12345);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .regularization(true).l2(0.0005)
            .learningRate(0.01)
            .activation(Activation.LEAKYRELU)
            .weightInit(WeightInit.RELU)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.ADAM)
            .list()
            .layer(0, new ConvolutionLayer.Builder(5, 5).stride(1, 1).nOut(32).activation(Activation.LEAKYRELU).build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())
            .layer(2, new ConvolutionLayer.Builder(5, 5).stride(1, 1).nOut(64).build())
            .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())
            .layer(4, new DenseLayer.Builder().nOut(256).build())
            //Layer 5 is our embedding layer: 2 dimensions, just so we can plot it on X/Y grid. Usually use more in practice
            .layer(5, new DenseLayer.Builder().activation(Activation.IDENTITY).weightInit(WeightInit.XAVIER).nOut(2)
                //Larger L2 value on the embedding layer: can help to stop the embedding layer weights
                // (and hence activations) from getting too large. This is especially problematic with small values of
                // lambda such as 0.0
                .l2(0.1).build())
            .layer(6, new CenterLossOutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(2).nOut(outputNum)
                .weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX)
                //Alpha and lambda hyperparameters are specific to center loss model: see comments above and paper
                .alpha(alpha).lambda(lambda)
                .build())
            .setInputType(InputType.convolutionalFlat(28, 28, 1))
            .backprop(true).pretrain(false).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        log.info("Train model....");
        model.setListeners(new GradientsAndParamsListener(model,100),new ScoreIterationListener(100));

        List<Pair<INDArray, INDArray>> embeddingByEpoch = new ArrayList<>();
        List<Integer> epochNum = new ArrayList<>();

        DataSet testData = mnistTest.next();
        for (int i = 0; i < nEpochs; i++) {
            model.fit(mnistTrain);

            log.info("*** Completed epoch {} ***", i);

            //Feed forward to the embedding layer (layer 5) to get the 2d embedding to plot later
            INDArray embedding = model.feedForwardToLayer(5, testData.getFeatures()).get(6);

            embeddingByEpoch.add(new Pair<>(embedding, testData.getLabels()));
            epochNum.add(i);
        }

        //Create a scatterplot: slider allows embeddings to be view at the end of each epoch
        PlotUtil.scatterPlot(embeddingByEpoch, epochNum, "MNIST Center Loss Embedding: l = " + lambda + ", a = " + alpha);
    }
}
