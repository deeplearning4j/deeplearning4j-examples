package org.deeplearning4j.examples.convolution.sampleNetStructure;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.HashMap;
import java.util.Map;

/**
 * Reference: http://arxiv.org/pdf/1312.6229v4.pdf
 * Created by nyghtowl on 11/17/15.
 */
public class OverFeatAccurate {
    private int height; // TODO paper expects 256 pixels
    private int width;
    private int channels = 3;
    private int outputNum = 1000;
    private long seed = 123;
    private int iterations = 90;
    // TODO extract 5 random crops (and their horizontal flips) of size 221x221 pixels and present these to the network in mini-batches of size 128

    public OverFeatAccurate(int height, int width, int channels, int outputNum, long seed, int iterations) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.outputNum = outputNum;
        this.seed = seed;
        this.iterations = iterations;
    }

    public MultiLayerNetwork init() {
        double nonZeroBias = 1;
        double dropOut = 0.5;
        SubsamplingLayer.PoolingType poolingType = SubsamplingLayer.PoolingType.MAX;

        Map<Integer, Double> learningRateSchedule = new HashMap<>();
        //TODO supposed to change lr at each epoch vs iteration...
        learningRateSchedule.put(30, 0.5);
        learningRateSchedule.put(50, 0.5);
        learningRateSchedule.put(60, 0.5);
        learningRateSchedule.put(70, 0.5);
        learningRateSchedule.put(80, 0.5);

        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation("relu")
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new GaussianDistribution(0.0, 1 * 10e-2))
                .learningRate(5 * 10e-2)
                .learningRateAfter(learningRateSchedule)
                .momentum(.6)
                .regularization(true)
                .l2(1 * 10e-5)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(12)
                .layer(0, new ConvolutionLayer.Builder(new int[]{7, 7}, new int[]{2, 2})
                        .name("cnn1")
                        .nIn(channels)
                        .nOut(96)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{3, 3})
                        .name("maxpool1")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(new int[]{7, 7}, new int[]{1, 1})
                        .name("cnn2")
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(poolingType, new int[]{2, 2}, new int[]{2, 2})
                        .name("maxpool2")
                        .build())
                .layer(4, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn3")
                        .nOut(512)
                        .build())
                .layer(5, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn4")
                        .nOut(512)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(6, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn5")
                        .nOut(1024)
                        .biasInit(nonZeroBias)
                        .build())
                        // TODO changes need to how the convolutions function on specific pixels in layer 5 in order to match paper
                .layer(7, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn6")
                        .nOut(1024)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(8, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{3, 3})
                        .name("maxpool3")
                        .build())
                .layer(9, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(4096)
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(10, new DenseLayer.Builder()
                        .name("ffn2")
                        .nOut(4096)
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true)
                .pretrain(false);


        new ConvolutionLayerSetup(conf,height,width,channels);
        MultiLayerNetwork model = new MultiLayerNetwork(conf.build());
        model.init();

        return model;
    }

}

