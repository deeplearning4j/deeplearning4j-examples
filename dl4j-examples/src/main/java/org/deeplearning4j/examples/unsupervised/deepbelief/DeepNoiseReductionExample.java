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
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Vladimir Sadilovski
 */
public class DeepNoiseReductionExample {

    private static Logger log = LoggerFactory.getLogger(DeepNoiseReductionExample.class);

    public static void main(String[] args) throws Exception {
        final int numRows = 28;
        final int numColumns = 28;
        int seed = 12345;
        int numSamples = 50000;
        int batchSize = 100;
        int iterations = 1;
        int numEpocs = 10;
        int inSize = numRows * numColumns;
        int hidSize = 1000;

        log.info("Load data....");
        DataSetIterator iter = new MnistDataSetIterator(batchSize, numSamples, true, true, true, seed);
        DataSetIterator test = new MnistDataSetIterator(batchSize, 100, true, false, true, seed);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list()
            .layer(0, new org.deeplearning4j.nn.conf.layers.RBM.Builder().lossFunction(LossFunctions.LossFunction.SQUARED_LOSS).nIn(inSize).nOut(hidSize)
                .biasInit(0.0)
                .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0.0,0.1)).build())
            .layer(1, new org.deeplearning4j.nn.conf.layers.RBM.Builder().lossFunction(LossFunctions.LossFunction.SQUARED_LOSS).nIn(hidSize).nOut(500)
                .biasInit(0.0)
                .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0.0,0.1)).build())
//            .layer(2, new org.deeplearning4j.nn.conf.layers.RBM.Builder().lossFunction(LossFunctions.LossFunction.SQUARED_LOSS).nIn(500).nOut(196)
//                .biasInit(0.0)
//                .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0.0,0.1)).build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nIn(196).nOut(10).build())
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
//        org.deeplearning4j.nn.layers.feedforward.rbm.RBM l2 = (org.deeplearning4j.nn.layers.feedforward.rbm.RBM) model.getLayer(2);

        DataSet tds = test.next();
        INDArray actualImageData = tds.getFeatureMatrix();
        INDArray actualImageLables = tds.getLabels();
        INDArray digitThree = null;
        INDArray digitFour = null;
        /**
         * Pick digits 3 and 4 to mix with each other
         */
        for (int i = 0; i < actualImageLables.rows() && (digitThree == null || digitFour == null); i++) {
            if (digitThree == null && actualImageLables.getRow(i).getDouble(3) == 1) {
                digitThree = actualImageData.getRow(i);
            }
            if (digitFour == null && actualImageLables.getRow(i).getDouble(4) == 1) {
                digitFour = actualImageData.getRow(i);
            }
        }

        List<INDArray> bothTypesImages = new ArrayList<>();
        List<INDArray> denoisedImages = new ArrayList<>();
        for (int i = 0; i <= 10; i++) {
            /**
             * Mixing data between two digits, i.e. adding information of one digit to another
             */
            INDArray digitWithNoise = mixData(digitThree, digitFour, i*0.1);
            bothTypesImages.add(digitWithNoise);

            /**
             * De-noising data
             */
            INDArray encodedImageData = l0.propUp(digitWithNoise);
//            encodedImageData = l1.propUp(encodedImageData);

            Pair<INDArray, INDArray> topLevelData = l1.sampleHiddenGivenVisible(encodedImageData);
            encodedImageData = topLevelData.getSecond();
            topLevelData = l1.sampleVisibleGivenHidden(encodedImageData);
            INDArray denoisedDigit = topLevelData.getFirst();

//            denoisedDigit = l1.propDown(denoisedDigit);
            denoisedDigit = l0.propDown(denoisedDigit);
            denoisedImages.add(denoisedDigit);
        }
        bothTypesImages.addAll(denoisedImages);

        MNISTVisualizer corruptedVisualizer = new MNISTVisualizer(2.0,bothTypesImages,"De-noised Example", 11,28);
        corruptedVisualizer.visualize();
    }

    private static INDArray mixData(INDArray data1, INDArray data2, double proportion) {
        INDArray ret = data1.dup();
        Distribution dist = Nd4j.getDistributions().createBinomial(1, Nd4j.create(new double[] {Math.min(proportion, 1)}));
        for (int i = 0; i < data2.size(1); i++) {
            if (dist.sample(new int[] {1}).getInt(0) == 1)
                ret.putScalar(i, data2.getDouble(i));
        }
        return ret;
    }

    public static class MNISTVisualizer {
        private double imageScale;
        private List<INDArray> digits;  //Digits (as row vectors), one per INDArray
        private String title;
        private int gridWidth;
        private int dim;

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
