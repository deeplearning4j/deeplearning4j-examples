package org.deeplearning4j.examples.misc.customlayers;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

/**
 * Created by Alex on 26/08/2016.
 */
public class CustomLayerTests {

    public static void main(String[] args){

        int nIn = 5;
        int nOut = 8;

        //Let's create a network with our custom layer

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .updater(Updater.RMSPROP).rmsDecay(0.95)
            .regularization(true).l2(0.03)
            .list()
            .layer(0, new DenseLayer.Builder().activation("tanh").nIn(nIn).nOut(6).build())    //Standard DenseLayer
            .layer(1, new CustomLayer.Builder()
                .activation("tanh")                                                             //Property inherited from FeedForwardLayer
                .secondActivationFunction("sigmoid")                                            //Custom property we defined for our layer
                .nIn(6).nOut(7)                                                                 //nIn and nOut also inherited from FeedForwardLayer
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)                //Standard OutputLayer
                .activation("softmax").nIn(7).nOut(nOut).build())
            .pretrain(false).backprop(true).build();


        //Let's run some basic checks on the configuration:
        double customLayerL2 = config.getConf(1).getLayer().getL2();
        System.out.println("l2 coefficient for custom layer: " + customLayerL2);                //As expected: custom layer inherits the global L2 parameter configuration
        Updater customLayerUpdater = config.getConf(1).getLayer().getUpdater();
        System.out.println("Updater for custom layer: " + customLayerUpdater);                  //As expected: custom layer inherits the global Updater configuration

        //Let's check that the JSON and YAML configuration works, with the custom layer
        // If there were problems with serialization, you'd get an exception during deserialization ("No suitable constructor found..." or similar)
        String configAsJson = config.toJson();
        String configAsYaml = config.toYaml();
        MultiLayerConfiguration fromJson = MultiLayerConfiguration.fromJson(configAsJson);
        MultiLayerConfiguration fromYaml = MultiLayerConfiguration.fromYaml(configAsYaml);

        System.out.println("JSON configuration works: " + config.equals(fromJson));
        System.out.println("YAML configuration works: " + config.equals(fromYaml));


        //Let's run some more basic tests. First, check that the forward and backward pass methods don't throw any exceptions:



        //Finally (and perhaps most importantly) we want to run some gradient checks
        // Gradient checks are necessary to ensure that your implementation is correct: without them, you could easily have
        // a subtle error, and not even know it
        //DL4J comes with a gradient check method that you can use

        //There are a few things to note when doing gradient checks:
        //1. It is necessary to use double precision for ND4J. Single precision (float - the default) isn't sufficiently accurate due to numerical issues
        //2. It is necessary to set the updater to None, or equivalently use SGD updater
        //   (Reason: we are testing the gradients, not the updates (i.e., gradients after they have been modified with learning rate, momentum etc))

        

    }

}
