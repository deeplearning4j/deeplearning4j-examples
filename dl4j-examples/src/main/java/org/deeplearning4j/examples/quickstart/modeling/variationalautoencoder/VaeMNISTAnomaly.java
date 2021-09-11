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

package org.deeplearning4j.examples.quickstart.modeling.variationalautoencoder;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.examples.quickstart.modeling.feedforward.unsupervised.MNISTAutoencoder;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.common.primitives.Pair;

import java.io.IOException;
import java.util.*;

/**
 * This example performs unsupervised anomaly detection on MNIST using a variational autoencoder, trained with a Bernoulli
 * reconstruction distribution.
 *
 * For details on the variational autoencoder, see:
 * - Kingma and Welling, 2013 - Auto-Encoding Variational Bayes - https://arxiv.org/abs/1312.6114
 *
 * For the use of VAEs for anomaly detection using reconstruction probability see:
 * - An & Cho, 2015 - Variational Autoencoder based Anomaly Detection using Reconstruction Probability
 *   http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf
 *
 *
 * Unsupervised training is performed on the entire data set at once in this example. An alternative approach would be to
 * train one model for each digit.
 *
 * After unsupervised training, examples are scored using the VAE layer (reconstruction probability). Here, we are using the
 * labels to get the examples with the highest and lowest reconstruction probabilities for each digit for plotting. In a general
 * unsupervised anomaly detection situation, these labels would not be available, and hence highest/lowest probabilities
 * for the entire data set would be used instead.
 *
 * @author Alex Black
 */
public class VaeMNISTAnomaly {

    public static boolean visualize = true;
    public static void main(String[] args) throws IOException {
        int minibatchSize = 128;
        int rngSeed = 12345;
        int nEpochs = 5;                    //Total number of training epochs
        int reconstructionNumSamples = 16;  //Reconstruction probabilities are estimated using Monte-Carlo techniques; see An & Cho for details

        //MNIST data for training
        DataSetIterator trainIter = new MnistDataSetIterator(minibatchSize, true, rngSeed);

        //Neural net configuration
        Nd4j.getRandom().setSeed(rngSeed);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .updater(new Adam(1e-3))
            .weightInit(WeightInit.XAVIER)
            .l2(1e-4)
            .list()
            .layer(new VariationalAutoencoder.Builder()
                .activation(Activation.LEAKYRELU)
                .encoderLayerSizes(256, 256)                    //2 encoder layers, each of size 256
                .decoderLayerSizes(256, 256)                    //2 decoder layers, each of size 256
                .pzxActivationFunction(Activation.IDENTITY)     //p(z|data) activation function
                //Bernoulli reconstruction distribution + sigmoid activation - for modelling binary data (or data in range 0 to 1)
                .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
                .nIn(28 * 28)                                   //Input size: 28x28
                .nOut(32)                                       //Size of the latent variable space: p(z|x) - 32 values
                .build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new ScoreIterationListener(100));

        //Fit the data (unsupervised training)
        for( int i=0; i<nEpochs; i++ ){
            net.pretrain(trainIter);        //Note use of .pretrain(DataSetIterator) not fit(DataSetIterator) for unsupervised training
            System.out.println("Finished epoch " + (i+1) + " of " + nEpochs);
        }


        //Perform anomaly detection on the test set, by calculating the reconstruction probability for each example
        //Then add pair (reconstruction probability, INDArray data) to lists and sort by score
        //This allows us to get best N and worst N digits for each digit type

        DataSetIterator testIter = new MnistDataSetIterator(minibatchSize, false, rngSeed);

        //Get the variational autoencoder layer:
        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae
            = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(0);

        Map<Integer,List<Pair<Double,INDArray>>> listsByDigit = new HashMap<>();
        for( int i=0; i<10; i++ ) listsByDigit.put(i, new ArrayList<>());

        //Iterate over the test data, calculating reconstruction probabilities
        while(testIter.hasNext()){
            DataSet ds = testIter.next();
            INDArray features = ds.getFeatures();
            INDArray labels = Nd4j.argMax(ds.getLabels(), 1);   //Labels as integer indexes (from one hot), shape [minibatchSize, 1]
            int nRows = features.rows();

            //Calculate the log probability for reconstructions as per An & Cho
            //Higher is better, lower is worse
            INDArray reconstructionErrorEachExample = vae.reconstructionLogProbability(features, reconstructionNumSamples);    //Shape: [minibatchSize, 1]

            for( int j=0; j<nRows; j++){
                INDArray example = features.getRow(j, true);
                int label = (int)labels.getDouble(j);
                double score = reconstructionErrorEachExample.getDouble(j);
                listsByDigit.get(label).add(new Pair<>(score, example));
            }
        }

        //Sort data by score, separately for each digit
        Comparator<Pair<Double, INDArray>> c = new Comparator<Pair<Double, INDArray>>() {
            @Override
            public int compare(Pair<Double, INDArray> o1, Pair<Double, INDArray> o2) {
                //Negative: return highest reconstruction probabilities first -> sorted from best to worst
                return -Double.compare(o1.getFirst(),o2.getFirst());
            }
        };

        for(List<Pair<Double, INDArray>> list : listsByDigit.values()){
            Collections.sort(list, c);
        }

        //Select the 5 best and 5 worst numbers (by reconstruction probability) for each digit
        List<INDArray> best = new ArrayList<>(50);
        List<INDArray> worst = new ArrayList<>(50);

        List<INDArray> bestReconstruction = new ArrayList<>(50);
        List<INDArray> worstReconstruction = new ArrayList<>(50);
        for( int i=0; i<10; i++ ){
            List<Pair<Double,INDArray>> list = listsByDigit.get(i);
            for( int j=0; j<5; j++ ){
                INDArray b = list.get(j).getSecond();
                INDArray w = list.get(list.size()-j-1).getSecond();

                LayerWorkspaceMgr mgr = LayerWorkspaceMgr.noWorkspaces();
                vae.setInput(b, mgr);
                INDArray pzxMeanBest = vae.preOutput(false, mgr);
                INDArray reconstructionBest = vae.generateAtMeanGivenZ(pzxMeanBest);

                vae.setInput(w, mgr);
                INDArray pzxMeanWorst = vae.preOutput(false, mgr);
                INDArray reconstructionWorst = vae.generateAtMeanGivenZ(pzxMeanWorst);

                best.add(b);
                bestReconstruction.add(reconstructionBest);
                worst.add(w);
                worstReconstruction.add(reconstructionWorst);
            }
        }

        //plot by default
        if (visualize) {
            //Visualize the best and worst digits
            MNISTAutoencoder.MNISTVisualizer bestVisualizer = new MNISTAutoencoder.MNISTVisualizer(2.0, best, "Best (Highest Rec. Prob)");
            bestVisualizer.visualize();

            MNISTAutoencoder.MNISTVisualizer bestReconstructions = new MNISTAutoencoder.MNISTVisualizer(2.0, bestReconstruction, "Best - Reconstructions");
            bestReconstructions.visualize();

            MNISTAutoencoder.MNISTVisualizer worstVisualizer = new MNISTAutoencoder.MNISTVisualizer(2.0, worst, "Worst (Lowest Rec. Prob)");
            worstVisualizer.visualize();

            MNISTAutoencoder.MNISTVisualizer worstReconstructions = new MNISTAutoencoder.MNISTVisualizer(2.0, worstReconstruction, "Worst - Reconstructions");
            worstReconstructions.visualize();
        }
    }
}
