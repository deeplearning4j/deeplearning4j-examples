/*******************************************************************************
 *
 *
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

package org.deeplearning4j.examples.quickstart.modeling.feedforward.unsupervised;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;

/**Example: Anomaly Detection on MNIST using simple autoencoder without pretraining
 * The goal is to identify outliers digits, i.e., those digits that are unusual or
 * not like the typical digits.
 * This is accomplished in this example by using reconstruction error: stereotypical
 * examples should have low reconstruction error, whereas outliers should have high
 * reconstruction error. The number of epochs here is set to 3. Set to 30 for even better
 * results.
 *
 * @author Alex Black
 */
public class MNISTAutoencoder {

    public static boolean visualize = true;

    public static void main(String[] args) throws Exception {

        //Set up network. 784 in/out (as MNIST images are 28x28).
        //784 -> 250 -> 10 -> 250 -> 784
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaGrad(0.05))
                .activation(Activation.RELU)
                .l2(0.0001)
                .list()
                .layer(new DenseLayer.Builder().nIn(784).nOut(250)
                        .build())
                .layer(new DenseLayer.Builder().nIn(250).nOut(10)
                        .build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(250)
                        .build())
                .layer(new OutputLayer.Builder().nIn(250).nOut(784)
                        .activation(Activation.LEAKYRELU)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(Collections.singletonList(new ScoreIterationListener(10)));

        //Load data and split into training and testing sets. 40000 train, 10000 test
        DataSetIterator iter = new MnistDataSetIterator(100,50000,false);

        List<INDArray> featuresTrain = new ArrayList<>();
        List<INDArray> featuresTest = new ArrayList<>();
        List<INDArray> labelsTest = new ArrayList<>();

        Random r = new Random(12345);
        while(iter.hasNext()){
            DataSet ds = iter.next();
            SplitTestAndTrain split = ds.splitTestAndTrain(80, r);  //80/20 split (from miniBatch = 100)
            featuresTrain.add(split.getTrain().getFeatures());
            DataSet dsTest = split.getTest();
            featuresTest.add(dsTest.getFeatures());
            INDArray indexes = Nd4j.argMax(dsTest.getLabels(),1); //Convert from one-hot representation -> index
            labelsTest.add(indexes);
        }

        //Train model:
        int nEpochs = 3;
        for( int epoch=0; epoch<nEpochs; epoch++ ){
            for(INDArray data : featuresTrain){
                net.fit(data,data);
            }
            System.out.println("Epoch " + epoch + " complete");
        }

        //Evaluate the model on the test data
        //Score each example in the test set separately
        //Compose a map that relates each digit to a list of (score, example) pairs
        //Then find N best and N worst scores per digit
        Map<Integer,List<Pair<Double,INDArray>>> listsByDigit = new HashMap<>();
        for( int i=0; i<10; i++ ) listsByDigit.put(i,new ArrayList<>());

        for( int i=0; i<featuresTest.size(); i++ ){
            INDArray testData = featuresTest.get(i);
            INDArray labels = labelsTest.get(i);
            int nRows = testData.rows();
            for( int j=0; j<nRows; j++){
                INDArray example = testData.getRow(j, true);
                int digit = (int)labels.getDouble(j);
                double score = net.score(new DataSet(example,example));
                // Add (score, example) pair to the appropriate list
                List digitAllPairs = listsByDigit.get(digit);
                digitAllPairs.add(new ImmutablePair<>(score, example));
            }
        }

        //Sort each list in the map by score
        Comparator<Pair<Double, INDArray>> c = new Comparator<Pair<Double, INDArray>>() {
            @Override
            public int compare(Pair<Double, INDArray> o1, Pair<Double, INDArray> o2) {
                return Double.compare(o1.getLeft(),o2.getLeft());
            }
        };

        for(List<Pair<Double, INDArray>> digitAllPairs : listsByDigit.values()){
            Collections.sort(digitAllPairs, c);
        }

        //After sorting, select N best and N worst scores (by reconstruction error) for each digit, where N=5
        List<INDArray> best = new ArrayList<>(50);
        List<INDArray> worst = new ArrayList<>(50);
        for( int i=0; i<10; i++ ){
            List<Pair<Double,INDArray>> list = listsByDigit.get(i);
            for( int j=0; j<5; j++ ){
                best.add(list.get(j).getRight());
                worst.add(list.get(list.size()-j-1).getRight());
            }
        }

        //Visualize by default
        if (visualize) {
            //Visualize the best and worst digits
            MNISTVisualizer bestVisualizer = new MNISTVisualizer(2.0, best, "Best (Low Rec. Error)");
            bestVisualizer.visualize();

            MNISTVisualizer worstVisualizer = new MNISTVisualizer(2.0, worst, "Worst (High Rec. Error)");
            worstVisualizer.visualize();
        }
    }

    public static class MNISTVisualizer {
        private double imageScale;
        private List<INDArray> digits;  //Digits (as row vectors), one per INDArray
        private String title;
        private int gridWidth;

        public MNISTVisualizer(double imageScale, List<INDArray> digits, String title ) {
            this(imageScale, digits, title, 5);
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
