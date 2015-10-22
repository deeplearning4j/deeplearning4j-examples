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
        Updater updater = Updater.NESTEROVS;
        String activation = "relu";
        WeightInit weightStrategy = WeightInit.DISTRIBUTION;
        GaussianDistribution distribution = new GaussianDistribution(0, 0.01);
        double nonZeroBias = 1;
        double dropOut = 0.5;
        SubsamplingLayer.PoolingType poolingType = SubsamplingLayer.PoolingType.MAX;

        // TODO split and link kernel maps on GPUs - 2nd, 4th, 5th convolution should only connect maps on the same gpu, 3rd connects to all in 2nd
        // TODO add local response normalization after 1st, 2nd convolution and before max-pooling
        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-2) // TODO setup to decrease by 10 when validation error rate stops improving
                        // TODO add lr_mult & decay_mult for weights and biases separately apply 1 & 1 to weights and 2 & 0 to bias
                .l2(5 * 1e-4)
                .momentum(0.9)
                .list(11)
                        //conv1
                .layer(0, new ConvolutionLayer.Builder(new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3})
                        .nIn(channels)
                        .nOut(96)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .build())
                        //conv2
                .layer(2, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{2, 2}) // TODO verrify stride
                        .nOut(256)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .build())
                        //conv3
                .layer(4, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(384)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .build())
                        //conv4
                .layer(5, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(384)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .biasInit(nonZeroBias)
                        .build())
                        //conv5
                .layer(6, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(256)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(distribution)
                        .updater(updater)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(7, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .build())
                .layer(8, new DenseLayer.Builder()
                        .nOut(4096)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(new GaussianDistribution(0, 0.005))
                        .updater(updater)
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(9, new DenseLayer.Builder()
                        .nOut(4096)
                        .activation(activation)
                        .weightInit(weightStrategy)
                        .dist(new GaussianDistribution(0, 0.005))
                        .updater(updater)
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
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

