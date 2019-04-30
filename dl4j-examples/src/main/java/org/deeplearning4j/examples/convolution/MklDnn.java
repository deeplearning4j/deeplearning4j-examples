package org.deeplearning4j.examples.convolution;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


public class MklDnn {

    public static INDArray randomOneHot(long examples, long nOut){
        return randomOneHot(examples, nOut, new Random(12345));
    }

    public static INDArray randomOneHot(long examples, long nOut, long rngSeed){
        return randomOneHot(examples, nOut, new Random(rngSeed));
    }

    public static INDArray randomOneHot(long examples, long nOut, Random rng){
        INDArray arr = Nd4j.create(examples, nOut);
        for( int i=0; i<examples; i++ ){
            // FIXME: int cast
            arr.putScalar(i, rng.nextInt((int) nOut), 1.0);
        }
        return arr;
    }

    public static void main(String[] args){
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.RELU)
            .activation(Activation.LEAKYRELU)
            .updater(new Adam(0.01))
            .convolutionMode(ConvolutionMode.Same)
            .l2(0.001)
            .list()
            .layer(new ConvolutionLayer.Builder()
                .kernelSize(3, 50)
                .stride(1, 50)
                .nIn(1)
                .nOut(100)
                .build())
            .layer(new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.7)
                .build())
            .layer(new OutputLayer.Builder()
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(100)
                .nOut(10)
                .build())
            .build();
        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();
        List<long[]> shapes = Arrays.asList(
            new long[]{64,1,368,50},
            new long[]{64,1,512,50},
            new long[]{64,1,512,50},
            new long[]{64,1,512,50},
            new long[]{64,1,512,50},
            new long[]{64,1,512,50},
            new long[]{64,1,368,50},
            new long[]{64,1,443,50},
            new long[]{64,1,436,50},
            new long[]{64,1,469,50},
            new long[]{64,1,376,50},
            new long[]{64,1,403,50},
            new long[]{64,1,350,50},
            new long[]{64,1,419,50},
            new long[]{64,1,441,50},
            new long[]{64,1,512,50},
            new long[]{64,1,402,50});
        List<DataSet> l = new ArrayList<>();
        for( long[] s : shapes){
            l.add(new DataSet(Nd4j.rand(DataType.FLOAT, s), randomOneHot(s[0], 10).castTo(DataType.FLOAT)));
        }
        net.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<100; i++ ){
            net.fit(new ExistingDataSetIterator(l));
            System.out.println("EPOCH: " + i);
        }
    }
}
