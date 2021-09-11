/* *****************************************************************************
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
import org.deeplearning4j.examples.utils.VAEPlotUtil;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.RmsProp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A simple example of training a variational autoencoder on MNIST.
 * This example intentionally has a small hidden state Z (2 values) for visualization on a 2-grid.
 *
 * After training, this example plots 2 things:
 * 1. The MNIST digit reconstructions vs. the latent space
 * 2. The latent space values for the MNIST test set, as training progresses (every N minibatches)
 *
 * Note that for both plots, there is a slider at the top - change this to see how the reconstructions and latent
 * space changes over time.
 *
 * @author Alex Black
 */
public class VaeMNIST2dPlots {
    public static boolean visualize = true;
    private static final Logger log = LoggerFactory.getLogger(VaeMNIST2dPlots.class);

    public static void main(String[] args) throws IOException {
        int minibatchSize = 128;
        int rngSeed = 12345;
        int nEpochs = 20;                   //Total number of training epochs

        //Plotting configuration
        int plotEveryNMinibatches = 100;    //Frequency with which to collect data for later plotting
        double plotMin = -5;                //Minimum values for plotting (x and y dimensions)
        double plotMax = 5;                 //Maximum values for plotting (x and y dimensions)
        int plotNumSteps = 16;              //Number of steps for reconstructions, between plotMin and plotMax

        //MNIST data for training
        DataSetIterator trainIter = new MnistDataSetIterator(minibatchSize, true, rngSeed);

        //Neural net configuration
        Nd4j.getRandom().setSeed(rngSeed);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .updater(new RmsProp(1e-3))
            .weightInit(WeightInit.XAVIER)
            .l2(1e-4)
            .list()
            .layer(new VariationalAutoencoder.Builder()
                .activation(Activation.LEAKYRELU)
                .encoderLayerSizes(256, 256)        //2 encoder layers, each of size 256
                .decoderLayerSizes(256, 256)        //2 decoder layers, each of size 256
                .pzxActivationFunction(Activation.IDENTITY)  //p(z|data) activation function
                .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction()))     //Bernoulli distribution for p(data|z) (binary or 0 to 1 data only)
                .nIn(28 * 28)                       //Input size: 28x28
                .nOut(2)                            //Size of the latent variable space: p(z|x). 2 dimensions here for plotting, use more in general
                .build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        //Get the variational autoencoder layer
        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae
            = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(0);


        //Test data for plotting
        DataSet testdata = new MnistDataSetIterator(10000, false, rngSeed).next();
        INDArray testFeatures = testdata.getFeatures();
        INDArray testLabels = testdata.getLabels();
        INDArray latentSpaceGrid = getLatentSpaceGrid(plotMin, plotMax, plotNumSteps);              //X/Y grid values, between plotMin and plotMax

        //Lists to store data for later plotting
        List<INDArray> latentSpaceVsEpoch = new ArrayList<>(nEpochs + 1);
        INDArray latentSpaceValues = vae.activate(testFeatures, false, LayerWorkspaceMgr.noWorkspaces());                     //Collect and record the latent space values before training starts
        latentSpaceVsEpoch.add(latentSpaceValues);
        List<INDArray> digitsGrid = new ArrayList<>();


        //Add a listener to the network that, every N=100 minibatches:
        // (a) collect the test set latent space values for later plotting
        // (b) collect the reconstructions at each point in the grid
        net.setListeners(new PlottingListener(100, testFeatures, latentSpaceGrid, latentSpaceVsEpoch, digitsGrid));

        //Perform training
        for (int i = 0; i < nEpochs; i++) {
            log.info("Starting epoch {} of {}",(i+1),nEpochs);
            net.pretrain(trainIter);    //Note use of .pretrain(DataSetIterator) not fit(DataSetIterator) for unsupervised training
        }

        //plot by default
        if (visualize) {
            //Plot MNIST test set - latent space vs. iteration (every 100 minibatches by default)
            VAEPlotUtil.plotData(latentSpaceVsEpoch, testLabels, plotMin, plotMax, plotEveryNMinibatches);

            //Plot reconstructions - latent space vs. grid
            double imageScale = 2.0;        //Increase/decrease this to zoom in on the digits
            VAEPlotUtil.MNISTLatentSpaceVisualizer v = new VAEPlotUtil.MNISTLatentSpaceVisualizer(imageScale, digitsGrid, plotEveryNMinibatches);
            v.visualize();
        }
    }


    //This simply returns a 2d grid: (x,y) for x=plotMin to plotMax, and y=plotMin to plotMax
    private static INDArray getLatentSpaceGrid(double plotMin, double plotMax, int plotSteps) {
        INDArray data = Nd4j.create(plotSteps * plotSteps, 2);
        INDArray linspaceRow = Nd4j.linspace(plotMin, plotMax, plotSteps, DataType.FLOAT);
        for (int i = 0; i < plotSteps; i++) {
            data.get(NDArrayIndex.interval(i * plotSteps, (i + 1) * plotSteps), NDArrayIndex.point(0)).assign(linspaceRow);
            int yStart = plotSteps - i - 1;
            data.get(NDArrayIndex.interval(yStart * plotSteps, (yStart + 1) * plotSteps), NDArrayIndex.point(1)).assign(linspaceRow.getDouble(i));
        }
        return data;
    }

    private static class PlottingListener extends BaseTrainingListener {

        private final int plotEveryNMinibatches;
        private final INDArray testFeatures;
        private final INDArray latentSpaceGrid;
        private final List<INDArray> latentSpaceVsEpoch;
        private final List<INDArray> digitsGrid;
        private PlottingListener(int plotEveryNMinibatches, INDArray testFeatures, INDArray latentSpaceGrid,
                                 List<INDArray> latentSpaceVsEpoch, List<INDArray> digitsGrid){
            this.plotEveryNMinibatches = plotEveryNMinibatches;
            this.testFeatures = testFeatures;
            this.latentSpaceGrid = latentSpaceGrid;
            this.latentSpaceVsEpoch = latentSpaceVsEpoch;
            this.digitsGrid = digitsGrid;
        }

        @Override
        public void iterationDone(Model model, int iterationCount, int epoch) {
            if(!(model instanceof org.deeplearning4j.nn.layers.variational.VariationalAutoencoder)){
                return;
            }

            org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae
                = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder)model;

            //Every N=100 minibatches:
            // (a) collect the test set latent space values for later plotting
            // (b) collect the reconstructions at each point in the grid
            if (iterationCount % plotEveryNMinibatches == 0) {
                INDArray latentSpaceValues = vae.activate(testFeatures, false, LayerWorkspaceMgr.noWorkspaces());
                latentSpaceVsEpoch.add(latentSpaceValues);

                INDArray out = vae.generateAtMeanGivenZ(latentSpaceGrid);
                digitsGrid.add(out);
            }
        }
    }
}
