package org.deeplearning4j.examples.misc.modelsaving;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

/**
 * A very simple example for saving and loading a MultiLayerNetwork
 *
 * @author Alex Black
 */
public class SaveLoadMultiLayerNetwork {

    public static void main(String[] args) throws Exception {
        //Define a simple MultiLayerNetwork:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS)
            .learningRate(0.1)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).activation(Activation.TANH).build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(3).nOut(3).build())
            .backprop(true).pretrain(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();


        //Save the model
        File locationToSave = new File("MyMultiLayerNetwork.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, saveUpdater);

        //Load the model
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(locationToSave);


        System.out.println("Saved and loaded parameters are equal:      " + net.params().equals(restored.params()));
        System.out.println("Saved and loaded configurations are equal:  " + net.getLayerWiseConfigurations().equals(restored.getLayerWiseConfigurations()));
    }

}
