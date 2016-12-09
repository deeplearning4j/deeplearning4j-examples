package org.deeplearning4j.examples.unsupervised.variational;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.examples.feedforward.anomalydetection.MNISTAnomalyExample;
import org.deeplearning4j.examples.unsupervised.variational.plot.PlotUtil;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A simple example of training a variational autoencoder on MNIST.
 * This example intentionally has a small hidden state Z (2 values) so this can be visualized in a 2-grid.
 *
 * This example plots 2 things:
 * 1. The MNIST digit reconstructions vs. the latent space
 * 2. The latent space values for the MNIST test set, as training progresses (every N minibatches)
 *
 * @author Alex Black
 */
public class VariationalAutoEncoderExample {

    public static void main(String[] args) throws IOException {

        int minibatchSize = 128;
        int totalExamples = 60000;
        boolean binarizeMnistImages = false;
        int rngSeed = 12345;

        //Total number of training epochs
        int nEpochs = 5;

        //Frequency with which to collect data for later plotting
        int plottingLatentSpaceEveryNMinibatches = 100;

        DataSetIterator trainIter = new MnistDataSetIterator(minibatchSize, totalExamples, binarizeMnistImages, true, true, rngSeed);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .iterations(1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(5e-5)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .weightInit(WeightInit.XAVIER)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new VariationalAutoencoder.Builder()
                .activation("leakyrelu")
                .encoderLayerSizes(256, 256)
                .decoderLayerSizes(256, 256)
                .pzxActivationFunction("tanh")
                .reconstructionDistribution(new BernoulliReconstructionDistribution("sigmoid"))
//                .reconstructionDistribution(new GaussianReconstructionDistribution("tanh"))
                .nIn(28*28)
                .nOut(2)
                .build())
            .pretrain(true).backprop(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
//        net.setListeners(new ScoreIterationListener(100), new StatsListener(new InMemoryStatsStorage(), 10));
        net.setListeners(new ScoreIterationListener(100));

        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(0);


        //Test data for plotting
        DataSet testdata = new MnistDataSetIterator(10000, false, rngSeed).next();
        INDArray testFeatures = testdata.getFeatures();
        INDArray testLabels = testdata.getLabels();

        INDArray latentSpaceGrid = getLatentSpaceGrid();


        List<INDArray> latentSpaceVsEpoch = new ArrayList<>(nEpochs+1);
        INDArray latentSpaceValues = vae.activate(testFeatures, false);     //Collect the latent space values before the
        latentSpaceVsEpoch.add(latentSpaceValues);

        int iterationCount = 0;
        INDArray lastOut = null;
        for( int i=0; i<nEpochs; i++ ){
            while(trainIter.hasNext()){
                DataSet ds = trainIter.next();
                net.fit(ds);

                //Every N minibatches: collect the test set latent space values for later plotting
                if(iterationCount++ % plottingLatentSpaceEveryNMinibatches == 0){
                    latentSpaceValues = vae.activate(testFeatures, false);
                    latentSpaceVsEpoch.add(latentSpaceValues);
                }
                //Every N minibatches: Also collect the reconstructions
                INDArray out = vae.generateAtMeanGivenZ(latentSpaceGrid);
                lastOut = out;
            }

            trainIter.reset();

        }

        PlotUtil.plotData(latentSpaceVsEpoch,testLabels);


        List<INDArray> list = new ArrayList<>();
        for( int i=0; i<lastOut.size(0); i++ ){
            list.add(lastOut.getRow(i));
        }

        PlotUtil.MNISTLatentSpaceVisualizer v = new PlotUtil.MNISTLatentSpaceVisualizer(2.0,list,"Test");
        v.visualize();
    }

    private static INDArray getLatentSpaceGrid(){
        int min = -1;
        int max = 1;
        int nSteps = 15;

        INDArray data = Nd4j.create(nSteps*nSteps, 2);
        INDArray linspaceRow = Nd4j.linspace(min, max, nSteps);
        for( int i=0; i<nSteps; i++ ){
            data.get(NDArrayIndex.interval(i*nSteps, (i+1)*nSteps), NDArrayIndex.point(0)).assign(linspaceRow.getDouble(i));
            data.get(NDArrayIndex.interval(i*nSteps, (i+1)*nSteps), NDArrayIndex.point(1)).assign(linspaceRow);
        }

        return data;
    }

}
