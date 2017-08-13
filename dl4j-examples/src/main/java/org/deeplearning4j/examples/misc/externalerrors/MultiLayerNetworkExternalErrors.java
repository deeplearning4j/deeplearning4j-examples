package org.deeplearning4j.examples.misc.externalerrors;

import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This example: shows how to train a MultiLayerNetwork where the errors come from an external source, instead
 * of using an Output layer and a labels array.
 * <p>
 * Possible use cases for this are reinforcement learning and testing/development of new algorithms.
 * <p>
 * For some uses cases, the following alternatives may be worth considering:
 * - Implement a custom loss function
 * - Implement a custom (output) layer
 * <p>
 * Both of these alternatives are available in DL4J
 *
 * @author Alex Black
 */
public class MultiLayerNetworkExternalErrors {

    public static void main(String[] args) {

        //Create the model
        int nIn = 4;
        int nOut = 3;
        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS)
            .learningRate(0.1)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(3).build())
            .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
            .backprop(true).pretrain(false)
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        //Calculate gradient with respect to an external error
        int minibatch = 32;
        INDArray input = Nd4j.rand(minibatch, nIn);
        INDArray output = model.output(input);          //Do forward pass. Normally: calculate the error based on this

        INDArray externalError = Nd4j.rand(minibatch, nOut);
        Pair<Gradient, INDArray> p = model.backpropGradient(externalError);  //Calculate backprop gradient based on error array

        //Update the gradient: apply learning rate, momentum, etc
        //This modifies the Gradient object in-place
        Gradient gradient = p.getFirst();
        int iteration = 0;
        model.getUpdater().update(model, gradient, iteration, minibatch);

        //Get a row vector gradient array, and apply it to the parameters to update the model
        INDArray updateVector = gradient.gradient();
        model.params().subi(updateVector);
    }

}
