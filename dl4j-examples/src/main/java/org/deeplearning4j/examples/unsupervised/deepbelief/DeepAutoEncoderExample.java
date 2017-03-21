package org.deeplearning4j.examples.unsupervised.deepbelief;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * ***** NOTE: This example has not been tuned. It requires additional work to produce sensible results *****
 *
 * @author Adam Gibson
 */
public class DeepAutoEncoderExample {

    private static Logger log = LoggerFactory.getLogger(DeepAutoEncoderExample.class);

    public static void main(String[] args) throws Exception {
        final int numRows = 28;
        final int numColumns = 28;
        int seed = 12345;
        int numSamples = 50000;
        int batchSize = 100;
        int iterations = 1;
        int numEpocs = 20;
        int inSize = numRows * numColumns;
        int hidSize = 1000;

        log.info("Load data....");
        DataSetIterator iter = new MnistDataSetIterator(batchSize, numSamples, true, true, true, seed);
        DataSetIterator test = new MnistDataSetIterator(batchSize, 100, true, false, false, 0);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list()
            .layer(0, new org.deeplearning4j.nn.conf.layers.RBM.Builder().nIn(inSize).nOut(hidSize)
                .biasInit(0.0)
                .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0.0,0.1)).build())
            .layer(1, new org.deeplearning4j.nn.conf.layers.RBM.Builder().nIn(hidSize).nOut(500)
                .biasInit(0.0)
                .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0.0,0.1)).build())
            .layer(2, new org.deeplearning4j.nn.conf.layers.RBM.Builder().nIn(500).nOut(196)
                .biasInit(0.0)
                .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0.0,0.1)).build())
            .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nIn(196).nOut(10).build())
            .pretrain(true).backprop(true)
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(100));

        log.info("Train model....");
        for (int epoch = 0; epoch < numEpocs; epoch++) {
            while(iter.hasNext()) {
                DataSet next = iter.next();
                model.fit(next.getFeatures());
            }
            iter.reset();
        }

        org.deeplearning4j.nn.layers.feedforward.rbm.RBM l0 = (org.deeplearning4j.nn.layers.feedforward.rbm.RBM) model.getLayer(0);
        org.deeplearning4j.nn.layers.feedforward.rbm.RBM l1 = (org.deeplearning4j.nn.layers.feedforward.rbm.RBM) model.getLayer(1);
        org.deeplearning4j.nn.layers.feedforward.rbm.RBM l2 = (org.deeplearning4j.nn.layers.feedforward.rbm.RBM) model.getLayer(2);

        List<INDArray> reconstructedImages = new ArrayList<>();
        List<INDArray> encodedImages = new ArrayList<>();
        List<INDArray> actualImages = new ArrayList<>();
        while(test.hasNext()) {
            DataSet tds = test.next();
            INDArray actualImageData = tds.getFeatureMatrix();
            INDArray encodedImageData;
            INDArray reconstructedImageData;

            /**
             * Encoding test images
             */
            encodedImageData = l0.propUp(actualImageData);
            encodedImageData = l1.propUp(encodedImageData);

            /**
             * top level layer is RBM and requires both positive and negative step
             * may require more than 1 iteration h -> v -> h -> v, but usually 1 is enough
             */
            Pair<INDArray, INDArray> topLevelData = l2.sampleHiddenGivenVisible(encodedImageData);
            encodedImageData = topLevelData.getSecond();
            topLevelData = l2.sampleVisibleGivenHidden(encodedImageData);
            reconstructedImageData = topLevelData.getSecond();

            /**
             * Decoding images
             */
            reconstructedImageData = l1.propDown(reconstructedImageData);
            reconstructedImageData = l0.propDown(reconstructedImageData);

            for (int i = 0; i < reconstructedImageData.rows(); i++) {
                encodedImages.add(encodedImageData.getRow(i));
                actualImages.add(actualImageData.getRow(i));
                reconstructedImages.add(reconstructedImageData.getRow(i));
            }
        }

        MNISTVisualizer encodedVisualizer = new MNISTVisualizer(4.0,encodedImages,"!Encoded", 14);
        encodedVisualizer.visualize();

        MNISTVisualizer reconstructedVisualizer = new MNISTVisualizer(2.0,reconstructedImages,"!Reconstructions", 28);
        reconstructedVisualizer.visualize();

        MNISTVisualizer actualVisualizer = new MNISTVisualizer(2.0,actualImages,"!Originals", 28);
        actualVisualizer.visualize();
    }

    public static class MNISTVisualizer {
        private double imageScale;
        private List<INDArray> digits;  //Digits (as row vectors), one per INDArray
        private String title;
        private int gridWidth;
        private int dim;

        public MNISTVisualizer(double imageScale, List<INDArray> digits, String title, int dim ) {
            this(imageScale, digits, title, 5, dim);
        }

        public MNISTVisualizer(double imageScale, List<INDArray> digits, String title, int gridWidth, int dim ) {
            this.imageScale = imageScale;
            this.digits = digits;
            this.title = title;
            this.gridWidth = gridWidth;
            this.dim = dim;
        }

        public void visualize(){
            JFrame frame = new JFrame();
            frame.setTitle(title);
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

            JPanel panel = new JPanel();
            panel.setLayout(new GridLayout(0,gridWidth));

            List<JLabel> list = getComponents();
            for(JLabel image : list){
                panel.add(image);
            }

            frame.add(panel);
            frame.setVisible(true);
            frame.pack();
        }

        private List<JLabel> getComponents(){
            List<JLabel> images = new ArrayList<>();
            for( INDArray arr : digits ){
                BufferedImage bi = new BufferedImage(dim,dim,BufferedImage.TYPE_BYTE_GRAY);
                for( int i=0; i<dim*dim; i++ ){
                    bi.getRaster().setSample(i % dim, i / dim, 0, (int)(255*arr.getDouble(i)));
                }
                ImageIcon orig = new ImageIcon(bi);
                Image imageScaled = orig.getImage().getScaledInstance((int)(imageScale*dim),(int)(imageScale*dim),Image.SCALE_REPLICATE);
                ImageIcon scaled = new ImageIcon(imageScaled);
                images.add(new JLabel(scaled));
            }
            return images;
        }
    }
}
