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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;


/**
 * Training and visualizing a deep convolutional generative adversarial network (DCGAN) on handwritten digits.
 *
 * @author Max Pumperla, wmeddie
 */
public class MnistDCGANExample {

    private static JFrame frame;
    private static JPanel panel;

    private static final int latentDim = 32;
    private static final int height = 28;
    private static final int width = 28;
    private static final int channels = 1;
    private static final int mediumHidden = 128;
    private static final int largeHidden = 256;


    private static void visualize(INDArray[] samples) {
        if (frame == null) {
            frame = new JFrame();
            frame.setTitle("Viz");
            frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
            frame.setLayout(new BorderLayout());

            panel = new JPanel();

            panel.setLayout(new GridLayout(samples.length / 3, 1, 8, 8));
            frame.add(panel, BorderLayout.CENTER);
            frame.setVisible(true);
        }

        panel.removeAll();

        for (int i = 0; i < samples.length; i++) {
            panel.add(getImage(samples[i]));
        }

        frame.revalidate();
        frame.pack();
    }

    private static JLabel getImage(INDArray tensor) {
        BufferedImage bi = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < 784; i++) {
            int pixel = (int)(((tensor.getDouble(i) + 1) * 2) * 255);
            bi.getRaster().setSample(i % 28, i / 28, 0, pixel);
        }
        ImageIcon orig = new ImageIcon(bi);
        Image imageScaled = orig.getImage().getScaledInstance((8 * 28), (8 * 28), Image.SCALE_REPLICATE);

        ImageIcon scaled = new ImageIcon(imageScaled);

        return new JLabel(scaled);
    }

    public static void main(String[] args) throws Exception {
        MultiLayerConfiguration genConf = new NeuralNetConfiguration.Builder().list()
            .layer(0, new DenseLayer.Builder().nIn(latentDim).nOut(width/2 * height/2 * mediumHidden)
                .activation(Activation.LEAKYRELU).build())
            .layer(1, new Convolution2D.Builder().nIn(mediumHidden).nOut(largeHidden).kernelSize(5, 5)
                .convolutionMode(ConvolutionMode.Same).activation(Activation.LEAKYRELU).build())
            // Up-sampling to 28x28xlargeHidden
            .layer(2, new Deconvolution2D.Builder().nIn(largeHidden).nOut(largeHidden).stride(2,2)
                .kernelSize(5, 5).convolutionMode(ConvolutionMode.Same)
                .activation(Activation.LEAKYRELU).build())
            .layer(3, new Convolution2D.Builder().nIn(largeHidden).nOut(largeHidden).kernelSize(5, 5)
                .convolutionMode(ConvolutionMode.Same).activation(Activation.LEAKYRELU).build())
            .layer(4, new Convolution2D.Builder().nIn(largeHidden).nOut(largeHidden).kernelSize(5, 5)
                .convolutionMode(ConvolutionMode.Same).activation(Activation.LEAKYRELU).build())
            .layer(5, new Convolution2D.Builder().nIn(largeHidden).nOut(channels).kernelSize(7, 7)
                .convolutionMode(ConvolutionMode.Same).activation(Activation.TANH).build())
            .layer(6, new ActivationLayer.Builder().activation(Activation.IDENTITY).build())
            .inputPreProcessor(1,
                new FeedForwardToCnnPreProcessor(height/2,width/2,mediumHidden))
            .inputPreProcessor(6, new CnnToFeedForwardPreProcessor(height, width, channels))
            .setInputType(InputType.feedForward(latentDim))
            .build();

        MultiLayerNetwork generator = new MultiLayerNetwork(genConf);
        generator.init();

        MultiLayerConfiguration discConf = new NeuralNetConfiguration.Builder()
            .updater(new RmsProp.Builder().learningRate(0.0008).rmsDecay(1e-8).build())
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            .gradientNormalizationThreshold(1.0)
            .list()
            .layer(0, new Convolution2D.Builder().nIn(channels).nOut(mediumHidden).kernelSize(3, 3)
                .activation(Activation.LEAKYRELU).build())
            .layer(1, new Convolution2D.Builder().nIn(mediumHidden).nOut(mediumHidden).kernelSize(3, 3).stride(2, 2)
                .activation(Activation.LEAKYRELU).build())
            .layer(2, new Convolution2D.Builder().nIn(mediumHidden).nOut(mediumHidden).kernelSize(3, 3).stride(2, 2)
                .activation(Activation.LEAKYRELU).build())
            .layer(3, new Convolution2D.Builder().nIn(mediumHidden).nOut(mediumHidden).kernelSize(3, 3).stride(2, 2)
                .activation(Activation.LEAKYRELU).build())
            .layer(4, new DropoutLayer.Builder().dropOut(0.6).build())
            .layer(5, new DenseLayer.Builder().nIn(mediumHidden * 2 * 2).nOut(1).activation(Activation.SIGMOID).build())
            .layer(6, new LossLayer.Builder().lossFunction(LossFunctions.LossFunction.XENT).build())
            .inputPreProcessor(0,new FeedForwardToCnnPreProcessor(height, width, channels))
            .inputPreProcessor(4,new CnnToFeedForwardPreProcessor(2, 2, mediumHidden))
            .setInputType(InputType.convolutional(height, width, channels))
            .build();

        MultiLayerNetwork discriminator = new MultiLayerNetwork(discConf);
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


        INDArray fakeIn = Nd4j.rand(new int[]{batchSize,  latentDim});
        INDArray fake = gan.getGenerator().output(fakeIn, false);

        DataSet fakeSet = new DataSet(fake, Nd4j.ones(batchSize, 1));

        for (int i = 0; i< 10; i++) {
            gan.fit(trainData, 1);
            System.out.println("Iteration " + i + " Visualizing...");
            INDArray[] samples = new INDArray[9];
            for (int k = 0; k < 9; k++) {
                INDArray input = fakeSet.get(k).getFeatures();
                samples[k] = gan.getGenerator().output(input, false);
            }
            visualize(samples);
        }
    }
}

