package org.deeplearning4j.examples.convolution;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Arrays;

/**
 * Created by agibsonccc on 9/21/15.
 */
public class Cifar {
    public static void main(String[] args) {
        DataSet dataSet = new DataSet();
        // dataSet.load(new File("/home/agibsonccc/Downloads/cifar-train.bin"));
        dataSet.load(new File("cifar-small.bin"));
        System.out.println(Arrays.toString(dataSet.getFeatureMatrix().shape()));
        int nChannels = 3;
        int outputNum = 10;
        int numSamples = 2000;
        int batchSize = 500;
        int iterations = 10;
        int splitTrainNum = (int) (batchSize*.8);
        int seed = 123;
        int listenerFreq = iterations/5;

        //setup the network
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .batchSize(batchSize)
                .iterations(iterations).regularization(true)
                .l1(1e-1).l2(2e-4).useDropConnect(true)
                .constrainGradientToUnitNorm(true).miniBatch(true)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nOut(5).dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())

                .layer(1, new SubsamplingLayer
                        .Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nOut(10).dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(3, new SubsamplingLayer
                        .Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(4, new DenseLayer.Builder().nOut(100).activation("relu")
                        .build())

                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);

        new ConvolutionLayerSetup(builder,32,32,nChannels);
        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.fit(dataSet);

    }


}
