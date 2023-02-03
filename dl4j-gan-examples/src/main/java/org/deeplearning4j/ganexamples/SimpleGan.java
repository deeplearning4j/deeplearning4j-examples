package org.deeplearning4j.ganexamples;

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
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 *          *****    ********        *****************
 *   z ---- * G *----* G(z) * ------ * discriminator * ---- fake
 *          *****    ********        *               *
 *   x ----------------------------- ***************** ---- real
 *
 * @author zdl
 */
public class SimpleGan {

    public static void main(String[] args) throws Exception {

        /**
         *Build the discriminator
         */
        MultiLayerConfiguration discriminatorConf = new NeuralNetConfiguration.Builder().seed(12345)
            .weightInit(WeightInit.XAVIER).updater(new RmsProp(0.001))
            .list()
            .layer(0, new DenseLayer.Builder().nIn(28 * 28).nOut(512).activation(Activation.RELU).build())
            .layer(1, new DenseLayer.Builder().activation(Activation.RELU)
                .nIn(512).nOut(256).build())
            .layer(2, new DenseLayer.Builder().activation(Activation.RELU)
                .nIn(256).nOut(128).build())
            .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                .activation(Activation.SIGMOID).nIn(128).nOut(1).build()).build();


        MultiLayerConfiguration ganConf = new NeuralNetConfiguration.Builder().seed(12345)
            .weightInit(WeightInit.XAVIER)
            //generator
            .updater(new RmsProp(0.001)).list()
            .layer(0, new DenseLayer.Builder().nIn(20).nOut(256).activation(Activation.RELU).build())
            .layer(1, new DenseLayer.Builder().activation(Activation.RELU)
                .nIn(256).nOut(512).build())
            .layer(2, new DenseLayer.Builder().activation(Activation.RELU)
                .nIn(512).nOut(28 * 28).build())
            //Freeze the discriminator parameter
            .layer(3, new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(new DenseLayer.Builder().nIn(28 * 28).nOut(512).activation(Activation.RELU).build()))
            .layer(4, new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(new DenseLayer.Builder().nIn(512).nOut(256).activation(Activation.RELU).build()))
            .layer(5, new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(new DenseLayer.Builder().nIn(256).nOut(128).activation(Activation.RELU).build()))
            .layer(6, new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                .activation(Activation.SIGMOID).nIn(128).nOut(1).build())).build();


        MultiLayerNetwork discriminatorNetwork = new MultiLayerNetwork(discriminatorConf);
        discriminatorNetwork.init();
        System.out.println(discriminatorNetwork.summary());
        discriminatorNetwork.setListeners(new ScoreIterationListener(1));

        MultiLayerNetwork ganNetwork = new MultiLayerNetwork(ganConf);
        ganNetwork.init();
        ganNetwork.setListeners(new ScoreIterationListener(1));
        System.out.println(ganNetwork.summary());

        DataSetIterator train = new MnistDataSetIterator(30, true, 12345);

        INDArray labelD = Nd4j.vstack(Nd4j.ones(30, 1), Nd4j.zeros(30, 1));
        INDArray labelG = Nd4j.ones(30, 1);
        MNISTVisualizer mnistVisualizer = new MNISTVisualizer(1, "Gan");
        for (int i = 1; i <= 100000; i++) {
            if (!train.hasNext()) {
                train.reset();
            }
            INDArray trueImage = train.next().getFeatures();
            INDArray z = Nd4j.rand(new NormalDistribution(), new long[]{30, 20});
            List<INDArray> ganFeedForward = ganNetwork.feedForward(z, false);
            INDArray fakeImage = ganFeedForward.get(3);
            INDArray trainDiscriminatorFeatures = Nd4j.vstack(trueImage, fakeImage);
            //Training discriminator
            discriminatorNetwork.fit(trainDiscriminatorFeatures, labelD);
            copyDiscriminatorParam(discriminatorNetwork, ganNetwork);
            //Training generator
            ganNetwork.fit(z, labelG);
            if (i % 1000 == 0) {
                List<INDArray> indArrays = ganNetwork.feedForward(Nd4j.rand(new NormalDistribution(), new long[]{30, 20}), false);
                List<INDArray> list = new ArrayList<>();
                INDArray indArray = indArrays.get(3);
                for (int j = 0; j < indArray.size(0); j++) {
                    list.add(indArray.getRow(j));
                }
                mnistVisualizer.setDigits(list);
                mnistVisualizer.visualize();
            }
        }
    }

    public static void copyDiscriminatorParam(MultiLayerNetwork discriminatorNetwork, MultiLayerNetwork ganNetwork) {
        for (int i = 0; i <= 3; i++) {
            ganNetwork.getLayer(i + 3).setParams(discriminatorNetwork.getLayer(i).params());
        }
    }
}
