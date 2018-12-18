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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;


/**
 * Training and visualizing a deep convolutional generative adversarial network (DCGAN) on handwritten digits.
 *
 * @author Max Pumperla, wmeddie
 */
public class MnistDCGANExample {

    private static final int LATENT_DIM = 32;
    private static final int HEIGHT = 28;
    private static final int WIDTH = 28;
    private static final int CHANNELS = 1;
    private static final int MEDIUM_HIDDEN = 128;
    private static final int LARGE_HIDDEN = 256;



    public static MultiLayerNetwork getGenerator () {
        MultiLayerConfiguration genConf = new NeuralNetConfiguration.Builder().list()
            .layer(0, new DenseLayer.Builder().nIn(LATENT_DIM).nOut(WIDTH / 2 * HEIGHT / 2 * MEDIUM_HIDDEN)
                .activation(Activation.LEAKYRELU).build())
            .layer(1, new Convolution2D.Builder().nIn(MEDIUM_HIDDEN).nOut(LARGE_HIDDEN).kernelSize(5, 5)
                .convolutionMode(ConvolutionMode.Same).activation(Activation.LEAKYRELU).build())
            // Up-sampling to 28x28xlargeHidden
            .layer(2, new Deconvolution2D.Builder().nIn(LARGE_HIDDEN).nOut(LARGE_HIDDEN).stride(2, 2)
                .kernelSize(5, 5).convolutionMode(ConvolutionMode.Same)
                .activation(Activation.LEAKYRELU).build())
            .layer(3, new Convolution2D.Builder().nIn(LARGE_HIDDEN).nOut(LARGE_HIDDEN).kernelSize(5, 5)
                .convolutionMode(ConvolutionMode.Same).activation(Activation.LEAKYRELU).build())
            .layer(4, new Convolution2D.Builder().nIn(LARGE_HIDDEN).nOut(LARGE_HIDDEN).kernelSize(5, 5)
                .convolutionMode(ConvolutionMode.Same).activation(Activation.LEAKYRELU).build())
            .layer(5, new Convolution2D.Builder().nIn(LARGE_HIDDEN).nOut(CHANNELS).kernelSize(7, 7)
                .convolutionMode(ConvolutionMode.Same).activation(Activation.TANH).build())
            .layer(6, new ActivationLayer.Builder().activation(Activation.IDENTITY).build())
            .inputPreProcessor(1,
                new FeedForwardToCnnPreProcessor(HEIGHT / 2, WIDTH / 2, MEDIUM_HIDDEN))
            .inputPreProcessor(6, new CnnToFeedForwardPreProcessor(HEIGHT, WIDTH, CHANNELS))
            .setInputType(InputType.feedForward(LATENT_DIM))
            .build();
        return new MultiLayerNetwork(genConf);
    }


    public static MultiLayerNetwork getDiscriminator () {
        MultiLayerConfiguration discConf = new NeuralNetConfiguration.Builder()
            .updater(new RmsProp.Builder().learningRate(0.0008).rmsDecay(1e-8).build())
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            .gradientNormalizationThreshold(1.0)
            .list()
            .layer(0, new Convolution2D.Builder().nIn(CHANNELS).nOut(MEDIUM_HIDDEN).kernelSize(3, 3)
                .activation(Activation.LEAKYRELU).build())
            .layer(1, new Convolution2D.Builder().nIn(MEDIUM_HIDDEN).nOut(MEDIUM_HIDDEN).kernelSize(3, 3).stride(2, 2)
                .activation(Activation.LEAKYRELU).build())
            .layer(2, new Convolution2D.Builder().nIn(MEDIUM_HIDDEN).nOut(MEDIUM_HIDDEN).kernelSize(3, 3).stride(2, 2)
                .activation(Activation.LEAKYRELU).build())
            .layer(3, new Convolution2D.Builder().nIn(MEDIUM_HIDDEN).nOut(MEDIUM_HIDDEN).kernelSize(3, 3).stride(2, 2)
                .activation(Activation.LEAKYRELU).build())
            .layer(4, new DropoutLayer.Builder().dropOut(0.6).build())
            .layer(5, new DenseLayer.Builder().nIn(MEDIUM_HIDDEN * 2 * 2).nOut(1).activation(Activation.SIGMOID).build())
            .layer(6, new LossLayer.Builder().lossFunction(LossFunctions.LossFunction.XENT).build())
            .inputPreProcessor(0, new FeedForwardToCnnPreProcessor(HEIGHT, WIDTH, CHANNELS))
            .inputPreProcessor(4, new CnnToFeedForwardPreProcessor(2, 2, MEDIUM_HIDDEN))
            .setInputType(InputType.convolutional(HEIGHT, WIDTH, CHANNELS))
            .build();

        return new MultiLayerNetwork(discConf);
    }

    public static void main (String[]args) throws Exception {

        MultiLayerNetwork generator = getGenerator();
        generator.init();

        MultiLayerNetwork discriminator = getDiscriminator();
        discriminator.init();

        GAN gan = new GAN.Builder()
            .generator(generator)
            .discriminator(discriminator)
            .latentDimension(LATENT_DIM)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            .gradientNormalizationThreshold(1.0)
            .updater(new RmsProp.Builder().learningRate(0.0008).rmsDecay(1e-8).build())
            .build();

        Nd4j.getMemoryManager().setAutoGcWindow(15 * 1000);

        int batchSize = 20;
        MnistDataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 42);


        // Sample from latent space once to visualize progress on image generation.
        int numSamples = 9;
        INDArray fakeIn = Nd4j.rand(new int[]{batchSize, LATENT_DIM});
        JFrame frame = GANVisualizationUtils.initFrame();
        JPanel panel = GANVisualizationUtils.initPanel(frame, numSamples);

        for (int i = 0; i < 10; i++) {
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

