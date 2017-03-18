package org.deeplearning4j.examples.unsupervised.deepbelief.utils;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by vlad on 3/18/2017.
 * @author Vladimir Sadilovski
 *
 * Visualizer of the results for RBM ref examples
 */
public class RMBVisualizer {
    public static void plot(INDArray label, INDArray reco, String title) {
        List<INDArray> reconstructedImages = new ArrayList<>();
        List<INDArray> actualImages = new ArrayList<>();
        for (int i = 0; i < label.size(0); i++) {
            actualImages.add(label.getRow(i));
            reconstructedImages.add(reco.getRow(i));
        }

        MNISTVisualizer predictedVisualizer = new MNISTVisualizer(2.0,reconstructedImages,"Reconstructions (" + title + ")");
        predictedVisualizer.visualize();

        MNISTVisualizer actualVisualizer = new MNISTVisualizer(2.0,actualImages,"Originals(" + title + ")");
        actualVisualizer.visualize();
    }

    public static void plot(DataSetIterator test, MultiLayerNetwork model) {
        List<INDArray> reconstructedImages = new ArrayList<>();
        List<INDArray> actualImages = new ArrayList<>();
        while(test.hasNext()) {
            DataSet tds = test.next();
            INDArray digitImageData = tds.getFeatureMatrix();
            INDArray reconstructedImage = model.output(digitImageData, false, null, null);
            for (int i = 0; i < digitImageData.rows(); i++) {
                INDArray actualDigitImage = digitImageData.getRow(i);
                actualImages.add(actualDigitImage);
                reconstructedImages.add(reconstructedImage.getRow(i));
            }
        }

        MNISTVisualizer predictedVisualizer = new MNISTVisualizer(2.0,reconstructedImages,"Reconstructions");
        predictedVisualizer.visualize();

        MNISTVisualizer actualVisualizer = new MNISTVisualizer(2.0,actualImages,"Originals");
        actualVisualizer.visualize();

        test.reset();
    }

    public static class MNISTVisualizer {
        private double imageScale;
        private List<INDArray> digits;  //Digits (as row vectors), one per INDArray
        private String title;
        private int gridWidth;

        public MNISTVisualizer(double imageScale, List<INDArray> digits, String title ) {
            this(imageScale, digits, title, 5);
        }

        public MNISTVisualizer(double imageScale, DataSetIterator digitDS, String title) {
            this(imageScale, digitDS, title,5);
        }

        public MNISTVisualizer(double imageScale, INDArray digitImageData, String title) {
            this(imageScale, digitImageData, title,5);
        }

        public MNISTVisualizer(double imageScale, INDArray digitImageData, String title, int gridWidth ) {
            List<INDArray> digits = new ArrayList<>();
            for (int i = 0; i < digitImageData.rows(); i++) {
                INDArray actualDigitImage = digitImageData.getRow(i);
                digits.add(actualDigitImage);
            }

            this.imageScale = imageScale;
            this.digits = digits;
            this.title = title;
            this.gridWidth = gridWidth;
        }

        public MNISTVisualizer(double imageScale, DataSetIterator digitDS, String title, int gridWidth ) {
            List<INDArray> digits = new ArrayList<>();
            while(digitDS.hasNext()) {
                DataSet tds = digitDS.next();
                INDArray digitImageData = tds.getFeatureMatrix();
                for (int i = 0; i < digitImageData.rows(); i++) {
                    INDArray actualDigitImage = digitImageData.getRow(i);
                    digits.add(actualDigitImage);
                }
            }

            this.imageScale = imageScale;
            this.digits = digits;
            this.title = title;
            this.gridWidth = gridWidth;
        }

        public MNISTVisualizer(double imageScale, List<INDArray> digits, String title, int gridWidth ) {
            this.imageScale = imageScale;
            this.digits = digits;
            this.title = title;
            this.gridWidth = gridWidth;
        }

        public void visualize(){
            JFrame frame = new JFrame();
            frame.setTitle(title);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

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
                BufferedImage bi = new BufferedImage(28,28,BufferedImage.TYPE_BYTE_GRAY);
                for( int i=0; i<784; i++ ){
                    bi.getRaster().setSample(i % 28, i / 28, 0, (int)(255*arr.getDouble(i)));
                }
                ImageIcon orig = new ImageIcon(bi);
                Image imageScaled = orig.getImage().getScaledInstance((int)(imageScale*28),(int)(imageScale*28),Image.SCALE_REPLICATE);
                ImageIcon scaled = new ImageIcon(imageScaled);
                images.add(new JLabel(scaled));
            }
            return images;
        }
    }
}
