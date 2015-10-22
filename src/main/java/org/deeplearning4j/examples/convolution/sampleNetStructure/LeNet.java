package org.deeplearning4j.examples.convolution.sampleNetStructure;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

/**
 * Reference: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
 * Created by nyghtowl on 9/11/15.
 */
public class LeNet {

    private int height;
    private int width;
    private int channels = 3;
    private int outputNum = 1000;
    private long seed = 123;
    private int iterations = 90;

    public LeNet(int height, int width, int channels, int outputNum, long seed, int iterations) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.outputNum = outputNum;
        this.seed = seed;
        this.iterations = iterations;
    }

    public MultiLayerNetwork init() {
        Updater updater = Updater.NESTEROVS;
        String activation = "sigmoid";
        WeightInit weightStrategy = WeightInit.DISTRIBUTION;
        GaussianDistribution distribution = new GaussianDistribution(0, 0.1); // TODO confirm std

        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .learningRate(7*10e-5)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(7)
                .layer(0, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1})
                        .nIn(channels)
                        .nOut(6)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2}, new int[]{2, 2})
                        .build())
                .layer(2, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1})
                        .nOut(16)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .biasInit(1)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2}, new int[]{2, 2})
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nOut(120)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .layer(5, new DenseLayer.Builder()
                        .nOut(84)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(outputNum)
                        .activation("softmax") // radial basis function required
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .backprop(true)
                .pretrain(false);


        new ConvolutionLayerSetup(conf,height,width,channels);
        MultiLayerNetwork model = new MultiLayerNetwork(conf.build());
        model.init();

        return model;
    }

}