package org.deeplearning4j.examples.convolution.sampleNetStructure;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Reference: http://arxiv.org/pdf/1409.1556.pdf
 * http://www.robots.ox.ac.uk/~vgg/research/very_deep/
 * https://gist.github.com/ksimonyan/211839e770f7b538e2d8
 *
 * On ImageNet error proven to decrease with depth but plateaued on the 16 weight layer example
 * Following is based on 16 layer
 *
 * Created by nyghtowl on 9/11/15.
 */

public class VGGNet {
    private int height;
    private int width;
    private int channels = 3;
    private int outputNum = 1300;
    private long seed = 123;
    private int iterations = 370; // 74 epochs - this based on batch of 256

    public VGGNet(int height, int width, int channels, int outputNum, long seed, int iterations) {
        this.height = height; // TODO configure inputs to be 224 (Based on paper) but this can and should vary
        this.width = width; // TODO configure inputs to be 224 (Based on paper)
        this.channels = channels; // TODO prepare input to subtract mean RGB value from each pixel
        this.outputNum = outputNum;
        this.seed = seed;
        this.iterations = iterations;
    }

    // TODO pretrain with smaller net for first couple CNN layer weights or http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    public MultiLayerNetwork init() {
        Updater updater = Updater.NESTEROVS;
        String activation = "relu";
        WeightInit weightStrategy = WeightInit.DISTRIBUTION;
        GaussianDistribution distribution = new GaussianDistribution(0, 0.01);

        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-4) // TODO start at 10^-2 and decrease by factor of 10 when validation set accuracy stops improving
                .regularization(true)
                .l2(5 * 1e-4)
                .momentum(0.9)
                .list(21)
                .layer(0, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nIn(channels)
                        .nOut(64)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .layer(1, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(64)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(3, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(128)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .layer(4, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(128)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(6, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(256)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .layer(7, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(256)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .layer(8, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(256)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .layer(9, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(10, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(512)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .layer(11, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(512)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .layer(12, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(512)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .layer(13, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(14, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(512)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .layer(15, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(512)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .layer(16, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(512)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .layer(17, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(18, new DenseLayer.Builder()
                        .nOut(4096)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .dropOut(0.5)
                        .build())
                .layer(19, new DenseLayer.Builder()
                        .nOut(4096)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .dropOut(0.5)
                        .build())
                .layer(20, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
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

