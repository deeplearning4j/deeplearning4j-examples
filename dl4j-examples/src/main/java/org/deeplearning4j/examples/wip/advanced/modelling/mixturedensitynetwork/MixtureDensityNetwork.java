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

package org.deeplearning4j.examples.wip.advanced.modelling.mixturedensitynetwork;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.impl.LossMixtureDensity;

import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Random;

/**
 * This is an example of using a mixture density network to learn the
 * distribution of a dataset instead of trying to converge directly onto
 * the dataset.  This is particularly useful when your data represents
 * a mixture of multivariate gaussian distributions.  Traditional cost
 * functions will converge on the mean of the dataset whereas a mixture
 * density network will fit a series of gaussians to that dataset
 * and extract the parameters "alpha" (relative strength of that gaussian),
 * the "sigma" or "standard-deviation" of that gaussian, and "mu", the
 * mean of that distribution.
 *
 * For a more detailed explanation of this, see Christopher Bishop's paper.
 *
 * Bishop CM. Mixture density networks,
 * Neural Computing Research Group Report:
 * NCRG/94/004, Aston University, Birmingham, 1994
 * https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-ncrg-94-004.pdf
 *
 * @author Jonathan Arney
 */
public class MixtureDensityNetwork {
    public static void main(String[] args) {

        final int inputSize = 1;
        final int outputLabels = 2;

        // Number of gaussian mixtures to
        // attempt to fit.
        final int mixturesToFit = 2;

        final int hiddenLayerSize = 20 * mixturesToFit;

        Random rng = new Random();

        NumberFormat formatter = new DecimalFormat("#0.0000");
        LossMixtureDensity loss = LossMixtureDensity.builder()
                            .gaussians(mixturesToFit)
                            .labelWidth(outputLabels)
                            .build();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rng.nextInt())
                .weightInit(WeightInit.XAVIER)
                .updater(new Nadam())
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(inputSize)
                        .nOut(hiddenLayerSize)
                        .activation(Activation.TANH)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(hiddenLayerSize)
                        .nOut(hiddenLayerSize)
                        .activation(Activation.TANH)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(hiddenLayerSize)
                        .nOut(mixturesToFit * (2+outputLabels))
                        .activation(Activation.IDENTITY)
                        .lossFunction(loss)
                        .build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new PerformanceListener(32));

        int trainingEpochs = 500;

        DataSetIterator iter = new GaussianMixtureIterator(mixturesToFit);
        PrintStream logOutput = System.out;

        //Do training, and then generate and print samples from network
        for (int i = 0; i < trainingEpochs; i++) {
            logOutput.println("Epoch number " + i);
            while (iter.hasNext()) {
                DataSet ds = iter.next();
                net.fit(ds);
            }

            iter.reset();	//Reset iterator for another epoch

            INDArray in = Nd4j.zeros(1).reshape(1,1);

            // Output what the network
            // has learned.  Go from -5 to 5
            // and print out the gaussian mixtures.
            for (int j = 0; j < 11; j++) {
                double input = (.1 * j - 0.5)*10;
                in.putScalar(0, input);
                INDArray output = net.activateSelectedLayers(0, net.getnLayers()-1, in);

                LossMixtureDensity.MixtureDensityComponents mixtures = loss.extractComponents(output);

                System.out.print("" + formatter.format(input) + "\t");
                for (int mixtureNumber = 0; mixtureNumber < mixturesToFit; mixtureNumber++) {
                    System.out.print("a" + mixtureNumber + " = " + formatter.format(mixtures.getAlpha().getDouble(0, mixtureNumber)) +
                                     " s" + mixtureNumber + " = " + formatter.format(mixtures.getSigma().getDouble(0, mixtureNumber)) +
                            " m" + mixtureNumber + " = ("
                    );
                    for (int labelNumber = 0;  labelNumber < outputLabels; labelNumber++) {
                        if (labelNumber != 0) {
                            System.out.print(",");
                        }
                        System.out.print(
                                formatter.format(mixtures.getMu().getDouble(0, mixtureNumber, labelNumber))
                        );
                    }
                    System.out.print(") ");
                }
                System.out.println();
            }
        }
    }
}
