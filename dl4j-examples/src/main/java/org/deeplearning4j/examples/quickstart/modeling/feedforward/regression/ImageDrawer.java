/* *****************************************************************************
 * Copyright (c) 2020 Konduit, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/


package org.deeplearning4j.examples.quickstart.modeling.feedforward.regression;

import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.examples.utils.DownloaderUtility;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.guava.collect.Streams;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.util.Random;

/**
 * Application to show a neural network learning to draw an image.
 * Demonstrates how to feed an NN with externally originated data, in this case an image of the Mona Lisa
 *
 * @author Robert Altena
 *         Many thanks to @tmanthey for constructive feedback and suggestions.
 */

public class ImageDrawer {

    private JFrame mainFrame;
    private MultiLayerNetwork nn; // The neural network.

    private BufferedImage originalImage;
    private JLabel generatedLabel;

    private INDArray xyOut; //x,y grid to calculate the output image. Needs to be calculated once, then re-used.

    private Java2DNativeImageLoader j2dNil; //Datavec class used to read and write images to /from INDArrays.
    private FastRGB rgb; // helper class for fast access to the image pixels.
    private Random random;

    private void init() throws Exception {

        mainFrame = new JFrame("Image drawer example");//creating instance of JFrame
        mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        String localDataPath = DownloaderUtility.DATAEXAMPLES.Download();
        originalImage = ImageIO.read(new File(localDataPath, "Mona_Lisa.png"));

        //start with a blank image of the same size as the original.
        BufferedImage generatedImage = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(), originalImage.getType());

        int width = originalImage.getWidth();
        int height = originalImage.getHeight();

        final JLabel originalLabel = new JLabel(new ImageIcon(originalImage));
        generatedLabel = new JLabel(new ImageIcon(generatedImage));

        originalLabel.setBounds(0, 0, width, height);
        generatedLabel.setBounds(width, 0, width, height);//x axis, y axis, width, height

        mainFrame.add(originalLabel);
        mainFrame.add(generatedLabel);

        mainFrame.setSize(2 * width, height + 25);
        mainFrame.setLayout(null);
        mainFrame.setVisible(true);  // Show UI


        j2dNil = new Java2DNativeImageLoader(); //Datavec class used to write images.
        random = new Random();
        nn = createNN(); // Create the neural network.
        xyOut = calcGrid(); //Create a mesh used to generate the image.

        // read the color channels from the original image.
        rgb = new FastRGB(originalImage);

        SwingUtilities.invokeLater(this::onCalc);
    }

    public static void main(String[] args) throws Exception {
        ImageDrawer imageDrawer = new ImageDrawer();
        imageDrawer.init();
    }

    /**
     * Build the Neural network.
     */
    private static MultiLayerNetwork createNN() {
        int seed = 2345;
        double learningRate = 0.001;
        int numInputs = 2;   // x and y.
        int numHiddenNodes = 1000;
        int numOutputs = 3; //R, G and B value.

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DenseLayer.Builder().nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DenseLayer.Builder().nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DenseLayer.Builder().nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DenseLayer.Builder().nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.L2)
                        .activation(Activation.IDENTITY)
                        .nOut(numOutputs).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        return net;
    }

    /**
     * Training the NN and updating the current graphical output.
     */
    private void onCalc() {
        // Find a reasonable balance between batch size and number of batches per generated redraw.
        int batchSize = 1000; //larger batch size slows the calculation but speeds up the learning per batch
        int numBatches = 10; // Drawing the generated image is slow. Doing multiple batches before redrawing increases speed.
        for (int i = 0; i < numBatches; i++) {
            DataSet ds = generateDataSet(batchSize);
            nn.fit(ds);
        }
        drawImage();
        mainFrame.invalidate();
        mainFrame.repaint();

        SwingUtilities.invokeLater(this::onCalc); //TODO: move training to a worker thread,
    }

    /**
     * Take a batchsize of random samples from the source image.
     * This illustrates how to generate a custom dataset. The normal way of doing this would be to generate a dataset
     * of the entire source image, train om shuffled batches from there.
     *
     * @param batchSize number of sample points to take out of the image.
     * @return DeepLearning4J DataSet.
     */
    private DataSet generateDataSet(int batchSize) {
        int w = originalImage.getWidth();
        int h = originalImage.getHeight();

        float[][] in = new float[batchSize][2];
        float[][] out = new float[batchSize][3];
        final int[] i = {0};
        Streams.forEachPair(
                random.ints(batchSize, 0, w).boxed(),
                random.ints(batchSize, 0, h).boxed(),
                (a, b) -> {
                    final short[] parts = rgb.getRGB(a, b);
                    in[i[0]] = new float[]{((a / (float) w) - 0.5f) * 2f, ((b / (float) h) - 0.5f) * 2f};
                    out[i[0]] = new float[]{parts[0], parts[1], parts[2]};
                    i[0]++;
                }
        );
        final INDArray input = Nd4j.create(in);
        final INDArray labels = Nd4j.create(out).divi(255);
        return new DataSet(input, labels);
    }

    /**
     * Make the Neural network draw the image.
     */
    private void drawImage() {
        int w = originalImage.getWidth();
        int h = originalImage.getHeight();

        INDArray out = nn.output(xyOut); // The raw NN output.
        BooleanIndexing.replaceWhere(out, 0.0, Conditions.lessThan(0.0)); // Clip between 0 and 1.
        BooleanIndexing.replaceWhere(out, 1.0, Conditions.greaterThan(1.0));
        out = out.mul(255).castTo(DataType.INT8); //convert to bytes.

        INDArray r = out.getColumn(0); //Extract the individual color layers.
        INDArray g = out.getColumn(1);
        INDArray b = out.getColumn(2);

        INDArray imgArr = Nd4j.vstack(b, g, r).reshape(3, h, w); // recombine the colors and reshape to image size.

        BufferedImage img = j2dNil.asBufferedImage(imgArr); //update the UI.
        generatedLabel.setIcon(new ImageIcon(img));
    }

    /**
     * The x,y grid to calculate the NN output. Only needs to be calculated once.
     */
    private INDArray calcGrid() {
        int w = originalImage.getWidth();
        int h = originalImage.getHeight();
        INDArray xPixels = Nd4j.linspace(-1.0, 1.0, w, DataType.DOUBLE);
        INDArray yPixels = Nd4j.linspace(-1.0, 1.0, h, DataType.DOUBLE);
        INDArray[] mesh = Nd4j.meshgrid(xPixels, yPixels);

        return Nd4j.vstack(mesh[0].ravel(), mesh[1].ravel()).transpose();
    }


    public class FastRGB {
        int width;
        int height;
        private boolean hasAlphaChannel;
        private int pixelLength;
        private byte[] pixels;

        FastRGB(BufferedImage image) {
            pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
            width = image.getWidth();
            height = image.getHeight();
            hasAlphaChannel = image.getAlphaRaster() != null;
            pixelLength = 3;
            if (hasAlphaChannel)
                pixelLength = 4;
        }

        short[] getRGB(int x, int y) {
            int pos = (y * pixelLength * width) + (x * pixelLength);
            short rgb[] = new short[4];
            if (hasAlphaChannel)
                rgb[3] = (short) (pixels[pos++] & 0xFF); // Alpha
            rgb[2] = (short) (pixels[pos++] & 0xFF); // Blue
            rgb[1] = (short) (pixels[pos++] & 0xFF); // Green
            rgb[0] = (short) (pixels[pos] & 0xFF); // Red
            return rgb;
        }
    }
}
