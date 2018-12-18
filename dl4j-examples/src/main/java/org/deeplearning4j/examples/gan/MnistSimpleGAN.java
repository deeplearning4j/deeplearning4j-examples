package org.deeplearning4j.examples.gan;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;

public class MnistSimpleGAN {

    private static final int latentDim = 100;

    private static final double LEARNING_RATE = 0.0002;
    private static final IUpdater UPDATER_ZERO = Sgd.builder().learningRate(0.0).build();
    private static final IUpdater UPDATER = Adam.builder().learningRate(LEARNING_RATE).beta1(0.5).build();


    public static MultiLayerNetwork getGenerator() {
        MultiLayerConfiguration genConf = new NeuralNetConfiguration.Builder().list()
            .layer(new DenseLayer.Builder().nIn(100).nOut(256).weightInit(WeightInit.NORMAL).build())
            .layer(new ActivationLayer.Builder(new ActivationLReLU(0.2)).build())
            .layer(new DenseLayer.Builder().nIn(256).nOut(512).build())
            .layer(new ActivationLayer.Builder(new ActivationLReLU(0.2)).build())
            .layer(new DenseLayer.Builder().nIn(512).nOut(1024).build())
            .layer(new ActivationLayer.Builder(new ActivationLReLU(0.2)).build())
            .layer(new DenseLayer.Builder().nIn(1024).nOut(784).activation(Activation.TANH).build())
            .build();
        return new MultiLayerNetwork(genConf);
    }


    public static MultiLayerNetwork getDiscriminator() {
        MultiLayerConfiguration discConf = new NeuralNetConfiguration.Builder()
            .seed(42)
            .updater(UPDATER)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.IDENTITY)
            .list()
            .layer(new DenseLayer.Builder().nIn(784).nOut(1024).updater(UPDATER_ZERO).build())
            .layer(new ActivationLayer.Builder(new ActivationLReLU(0.2)).build())
            .layer(new DropoutLayer.Builder(1 - 0.5).build())
            .layer(new DenseLayer.Builder().nIn(1024).nOut(512).updater(UPDATER_ZERO).build())
            .layer(new ActivationLayer.Builder(new ActivationLReLU(0.2)).build())
            .layer(new DropoutLayer.Builder(1 - 0.5).build())
            .layer(new DenseLayer.Builder().nIn(512).nOut(256).updater(UPDATER_ZERO).build())
            .layer(new ActivationLayer.Builder(new ActivationLReLU(0.2)).build())
            .layer(new DropoutLayer.Builder(1 - 0.5).build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT).nIn(256).nOut(1)
                .activation(Activation.SIGMOID).updater(UPDATER_ZERO).build())
            .build();

        return new MultiLayerNetwork(discConf);
    }

    public static void main(String[] args) throws Exception {

        MultiLayerNetwork generator = getGenerator();
        generator.init();

        MultiLayerNetwork discriminator = getDiscriminator();
        discriminator.init();


        GAN gan = new GAN.Builder()
            .generator(generator)
            .discriminator(discriminator)
            .latentDimension(latentDim)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            .gradientNormalizationThreshold(1.0)
            .updater(new RmsProp.Builder().learningRate(0.0008).rmsDecay(1e-8).build())
            .build();

        Nd4j.getMemoryManager().setAutoGcWindow(15 * 1000);

        int batchSize = 20;
        MnistDataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 42);


        // Sample from latent space once to visualize progress on image generation.
        int numSamples = 9;
        INDArray fakeIn = Nd4j.rand(new int[]{batchSize,  latentDim});
        JFrame frame = GANVisualizationUtils.initFrame();
        JPanel panel = GANVisualizationUtils.initPanel(frame, numSamples);

        for (int i = 0; i< 10; i++) {
            gan.fit(trainData, 1);
            System.out.println("Iteration " + i + " Visualizing...");
            INDArray[] samples = new INDArray[numSamples];
            for (int k = 0; k < numSamples; k++) {
                INDArray input = fakeIn.getRow(k);
                samples[k] = gan.getGenerator().output(input, false);
            }
            GANVisualizationUtils.visualize(samples, frame, panel);
        }
    }
}
