package org.deeplearning4j.examples.convolution.sampleNetStructure;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;


/**
 * References:
 * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
 * https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt
 *
 * Dl4j's AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
 * and the example code referenced.
 *
 * Model is built in dl4j based on available functionality and notes indicate where there are gaps waiting for enhancements.
 * Created by nyghtowl on 9/11/15.
 *
 * Bias initialization in the paper is 1 in certain layers but 0.1 in the example code
 * Weight distribution uses 0.1 std for all layers in the paper but 0.005 in the dense layers in the example code
 *
 */
public class AlexNet {

    private int height;
    private int width;
    private int channels = 3;
    private int outputNum = 1000;
    private long seed = 123;
    private int iterations = 90;

    public AlexNet(int height, int width, int channels, int outputNum, long seed, int iterations) {
        // TODO consider ways to make this adaptable to other problems not just imagenet
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.outputNum = outputNum;
        this.seed = seed;
        this.iterations = iterations;
        // TODO batch size set to 128 for ImageNet based on paper - base it on memory bandwidth
    }

    public MultiLayerNetwork init() {
        double nonZeroBias = 1;
        double dropOut = 0.5;
        SubsamplingLayer.PoolingType poolingType = SubsamplingLayer.PoolingType.MAX;

        // TODO split and link kernel maps on GPUs - 2nd, 4th, 5th convolution should only connect maps on the same gpu, 3rd connects to all in 2nd
        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new GaussianDistribution(0.0, 0.01))
                .activation("relu")
                .updater(Updater.NESTEROVS

                )
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // TODO confirm this is required
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        // TODO add lr_mult & decay_mult for weights and biases separately apply 1 & 1 to weights and 2 & 0 to bias
                .learningRate(1e-3)
                .learningRateScoreBasedDecayRate(1e-1)
                .regularization(true)
                .l2(5 * 1e-4)
                .momentum(0.9)
                .miniBatch(false)
                .list(13)
                        //conv1
                .layer(0, new ConvolutionLayer.Builder(new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3})
                        .nIn(channels)
                        .nOut(96)
                        .build())
                .layer(1, new LocalResponseNormalization.Builder()
                        .build())
                .layer(2, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .build())
                        //conv2
                .layer(3, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{2, 2}) // TODO verrify stride
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(4, new LocalResponseNormalization.Builder()
                        .k(2).n(5).alpha(1e-4).beta(0.75)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .build())
                        //conv3
                .layer(6, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(384)
                        .build())
                        //conv4
                .layer(7, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(384)
                        .biasInit(nonZeroBias)
                        .build())
                        //conv5
                .layer(8, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(9, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .build())
                .layer(10, new DenseLayer.Builder()
                        .nOut(4096)
                        .dist(new GaussianDistribution(0, 0.005))
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(11, new DenseLayer.Builder()
                        .nOut(4096)
                        .dist(new GaussianDistribution(0, 0.005))
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
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

