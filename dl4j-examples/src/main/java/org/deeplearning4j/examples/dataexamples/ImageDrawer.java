/* *****************************************************************************
 * Copyright (c) 2020 Konduit, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.dataexamples;

import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.examples.download.DownloaderUtility;
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

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;

/**
 * Application to show a neural network learning to draw an image.
 * Demonstrates how to feed an NN with externally originated data.
 *
 *  Updates from previous versions:
 *   - Now uses swing. No longer uses JavaFX which caused problems with the OpenJDK.
 *   - All slow java loops in the dataset creation and image drawing are replaced with fast vectorized code.
 *
 * @author Robert Altena
 * Many thanks to @tmanthey for constructive feedback and suggestions.
 */
public class ImageDrawer {

    private JFrame mainFrame;
    private MultiLayerNetwork nn; // The neural network.

    private BufferedImage originalImage;
    private JLabel generatedLabel;

    private INDArray blueMat; // color channels of he original image.
    private INDArray greenMat;
    private INDArray redMat;

    private INDArray xPixels; // x coordinates of the pixels for the NN.
    private INDArray yPixels; // y coordinates of the pixels for the NN.

    private INDArray xyOut; //x,y grid to calculate the output image. Needs to be calculated once, then re-used.

    private Java2DNativeImageLoader j2dNil; //Datavec class used to read and write images to /from INDArrays.


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

        originalLabel.setBounds(0,0, width, height);
        generatedLabel.setBounds(width, 0, width, height);//x axis, y axis, width, height

        mainFrame.add(originalLabel);
        mainFrame.add(generatedLabel);

        mainFrame.setSize(2*width, height +25);
        mainFrame.setLayout(null);
        mainFrame.setVisible(true);  // Show UI


        j2dNil = new Java2DNativeImageLoader(); //Datavec class used to read and write images.
        nn = createNN(); // Create the neural network.
        xyOut = calcGrid(); //Create a mesh used to generate the image.

        // read the color channels from the original image.
        INDArray imageMat = j2dNil.asMatrix(originalImage).castTo(DataType.DOUBLE).div(255.0);
        blueMat = imageMat .tensorAlongDimension(1, 0, 2, 3).reshape(width * height, 1);
        greenMat = imageMat .tensorAlongDimension(2, 0, 2, 3).reshape(width * height, 1);
        redMat = imageMat .tensorAlongDimension(3, 0, 2, 3).reshape(width * height, 1);

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
        int numOutputs = 3 ; //R, G and B value.

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DenseLayer.Builder().nOut(numHiddenNodes )
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DenseLayer.Builder().nOut(numHiddenNodes )
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DenseLayer.Builder().nOut(numHiddenNodes )
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DenseLayer.Builder().nOut(numHiddenNodes )
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer( new OutputLayer.Builder(LossFunctions.LossFunction.L2)
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
    private void onCalc(){
        int batchSize = 1000;
        int numBatches = 10;
        for (int i =0; i< numBatches; i++){
            DataSet ds = generateDataSet(batchSize);
            nn.fit(ds);
        }
        drawImage();
        mainFrame.invalidate();
        mainFrame.repaint();

        SwingUtilities.invokeLater(this::onCalc);
    }

    /**
     * Take a batchsize of random samples from the source image.
     *
     * @param batchSize number of sample points to take out of the image.
     * @return DeepLearning4J DataSet.
     */
    private DataSet generateDataSet(int batchSize) {
        int w = originalImage.getWidth();
        int h = originalImage.getHeight();

        INDArray xindex = Nd4j.rand(batchSize).muli(w-1).castTo(DataType.UINT32);
        INDArray yindex = Nd4j.rand(batchSize).muli(h-1).castTo(DataType.UINT32);

        INDArray xPos = xPixels.get(xindex).reshape(batchSize); // Look up the normalized positions pf the pixels.
        INDArray yPos = yPixels.get(yindex).reshape(batchSize);

        INDArray xy =  Nd4j.vstack(xPos, yPos).transpose(); // Create the array that can be fed into the NN.

        //Look up the correct colors fot our random pixels.
        INDArray xyIndex = yindex.mul(w).add(xindex); //TODO: figure out the 2D version of INDArray.get.
        INDArray b = blueMat.get(xyIndex).reshape(batchSize);
        INDArray g = greenMat.get(xyIndex).reshape(batchSize);
        INDArray r = redMat.get(xyIndex).reshape(batchSize);
        INDArray out = Nd4j.vstack(r, g, b).transpose(); // Create the array that can be used for NN training.

        return new DataSet(xy, out);
    }

    /**
     * Make the Neural network draw the image.
     */
    private void drawImage() {
        int w = originalImage.getWidth();
        int h = originalImage.getHeight();

        INDArray out = nn.output(xyOut); // The raw NN output.
        BooleanIndexing.replaceWhere(out, 0.0, Conditions.lessThan(0.0)); // Cjip between 0 and 1.
        BooleanIndexing.replaceWhere(out, 1.0, Conditions.greaterThan(1.0));
        out = out.mul(255).castTo(DataType.BYTE); //convert to bytes.

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
    private INDArray calcGrid(){
        int w = originalImage.getWidth();
        int h = originalImage.getHeight();
        xPixels = Nd4j.linspace(-1.0, 1.0, w, DataType.DOUBLE);
        yPixels = Nd4j.linspace(-1.0, 1.0, h, DataType.DOUBLE);
        INDArray [] mesh = Nd4j.meshgrid(xPixels, yPixels);

        xPixels = xPixels.reshape(w, 1); // This is a hack to work around a bug in INDArray.get()
        yPixels = yPixels.reshape(h, 1); // in the dataset generation.

        return Nd4j.vstack(mesh[0].ravel(), mesh[1].ravel()).transpose();
    }
}
