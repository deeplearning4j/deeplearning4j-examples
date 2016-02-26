package org.deeplearning4j.examples.convolution;

import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.Arrays;

/**
 * CIFAR-10 is an image dataset created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. The dataset inculdes 60K
 * tiny RGB images sized 32 x 32 pixels covering 10 classes. There are 50K training images and 10K test images.
 *
 * Use this example to run cifar-10.
 *
 * Reference: https://www.cs.toronto.edu/~kriz/cifar.html
 * Dataset url: https://s3.amazonaws.com/dl4j-distribution/cifar-small.bin
 */
public class Cifar {

    //This version of the example requires DL4J 0.4-rc3.9 or later (or, DL4J snapshots) and is disabled here
    public static void main(String[] args) throws IOException {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;

        int height = 32;
        int width = 32;
        int channels = 3;
        int numTrainSamples = 100;
        int numTestSamples = 100;
        int batchSize = 30;

        int outputNum = 10;
        int iterations = 5;
        int epochs = 5;
        int seed = 123;
        int listenerFreq = 5;

        System.out.println("Load data...");
        MultipleEpochsIterator cifar = new MultipleEpochsIterator(epochs, new CifarDataSetIterator(batchSize, numTrainSamples, "TRAIN"));

        //setup the network
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation("relu")
                .weightInit(WeightInit.XAVIER) // consider standard distribution with std .05
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.01)
                .momentum(0.9)
                .regularization(true)
                .l2(0.04)
                .updater(Updater.NESTEROVS)
                .useDropConnect(true)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn1")
                        .nIn(channels)
                        .stride(1, 1)
                        .padding(2, 2)
                        .nOut(32)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool1")
                        .build())
                .layer(2, new LocalResponseNormalization.Builder(3, 5e-05, 0.75).build())
                .layer(3, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn2")
                        .stride(1, 1)
                        .padding(2, 2)
                        .nOut(32)
                        .build())
                .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool2")
                        .build())
                .layer(5, new LocalResponseNormalization.Builder(3, 5e-05, 0.75).build())
                .layer(6, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn3")
                        .stride(1, 1)
                        .padding(2, 2)
                        .nOut(64)
                        .build())
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool3")
                        .build())
                .layer(8, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(250)
                        .dropOut(0.5)
                    .build())
                .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false)
                .cnnInputSize(height, width, channels);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        System.out.println("Train model...");
        network.fit(cifar);

        CifarDataSetIterator cifarTest = new CifarDataSetIterator(batchSize, numTestSamples, "TEST");
        Evaluation eval = new Evaluation(cifarTest.getLabels());
        while(cifarTest.hasNext()) {
            DataSet testDS = cifarTest.next(batchSize);
            INDArray output = network.output(testDS.getFeatureMatrix());
            eval.eval(testDS.getLabels(), output);
        }
        System.out.println(eval.stats());

    }


}
