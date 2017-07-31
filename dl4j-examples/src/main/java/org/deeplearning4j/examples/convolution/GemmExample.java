package org.deeplearning4j.examples.convolution;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by raver119 on 07.07.17.
 */
public class GemmExample {
    public static void main(String[] args) throws Exception {
        Nd4j.getRandom().setSeed(12345);
        int nOut = 1;
        int nIn = 4;
        int minibatch = 5;

        INDArray input = Nd4j.rand(minibatch, nIn);
        INDArray eps = Nd4j.rand(minibatch, nOut);


        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .trainingWorkspaceMode(WorkspaceMode.NONE)
            .inferenceWorkspaceMode(WorkspaceMode.NONE)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(nIn).nOut(nOut).activation(Activation.TANH)
                .build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        //System.out.println(net.getFlattenedGradients());


        for( int i=0; i<1; i++ ) {
            net.output(input);
            net.backpropGradient(eps);
            System.out.println(net.getFlattenedGradients());
        }


//        //This *should* be equivalent to BaseLayer.backpropGradient, but doesn't seem to reproduce the problem
//        INDArray weightGrad = Nd4j.create(new int[]{nIn,nOut},'f');
//        INDArray z = net.getLayer(0).preOutput(inputCopy);
//        INDArray delta = new ActivationTanH().backprop(z, epsCopy).getFirst();
//        Nd4j.gemm(input, delta, weightGrad, true, false, 1.0, 0.0);
//        System.out.println(Arrays.toString(weightGrad.data().asDouble()));
    }
}
