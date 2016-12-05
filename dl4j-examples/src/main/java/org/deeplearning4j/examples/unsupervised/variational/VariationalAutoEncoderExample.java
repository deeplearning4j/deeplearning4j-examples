package org.deeplearning4j.examples.unsupervised.variational;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.examples.feedforward.anomalydetection.MNISTAnomalyExample;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A simple example on MNIST.
 *
 * This example intentionally has a small hidden state Z (2 values) so this can be visualized in a 2-grid.
 *
 * @author Alex Black
 */
public class VariationalAutoEncoderExample {

    public static void main(String[] args) throws IOException {

        int minibatchSize = 128;
        int totalExamples = 60000;
        boolean binarizeMnistImages = true;
        int rngSeed = 12345;

        int nEpochs = 5;

        DataSetIterator trainIter = new MnistDataSetIterator(minibatchSize, totalExamples, binarizeMnistImages, true, true, rngSeed);
//        DataSetIterator trainIter = new MnistDataSetIterator(minibatchSize, true, rngSeed);


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .learningRate(1e-5)
            .updater(Updater.NESTEROVS).momentum(0.9)
//            .regularization(true).l2(1e-3)
            .list()
            .layer(0, new VariationalAutoencoder.Builder()
                .activation("leakyrelu")
                .encoderLayerSizes(256, 256)
                .decoderLayerSizes(256, 256)
                .pzxActivationFunction("identity")
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


//        System.out.println(Arrays.toString(net.params().data().asFloat()));
//        List<float[]> temp = new ArrayList<>();
//        temp.add(net.params().data().asFloat());
        for( int i=0; i<nEpochs; i++ ){
            net.fit(trainIter);
            trainIter.reset();
//            System.out.println(Arrays.toString(net.params().data().asFloat()));
//            temp.add(net.params().data().asFloat());
        }

//        for(float[] f : temp){
//            System.out.println(Arrays.toString(f));
//        }

        int min = -6;
        int max = 6;
        int nSteps = 20;

        INDArray data = Nd4j.create(nSteps*nSteps, 2);
        INDArray linspaceRow = Nd4j.linspace(min, max, nSteps);
        for( int i=0; i<nSteps; i++ ){
            data.get(NDArrayIndex.interval(i*nSteps, (i+1)*nSteps), NDArrayIndex.point(0)).assign(linspaceRow.getDouble(i));
            data.get(NDArrayIndex.interval(i*nSteps, (i+1)*nSteps), NDArrayIndex.point(1)).assign(linspaceRow);
        }

        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(0);
        INDArray out = vae.generateAtMeanGivenZ(data);

        List<INDArray> list = new ArrayList<>();
        for( int i=0; i<out.size(0); i++ ){
            list.add(out.getRow(i));
        }

        MNISTAnomalyExample.MNISTVisualizer v = new MNISTAnomalyExample.MNISTVisualizer(2.0,list,"Test",nSteps);
        v.visualize();
    }

}
