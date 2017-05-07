package org.deeplearning4j.examples.userInterface;

import org.deeplearning4j.examples.userInterface.util.ActivationsListener;
import org.deeplearning4j.examples.userInterface.util.UIExampleUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * An example of how to attach Deeplearning4j's training UI to a network and view activations.
 *
 * To change the UI port (usually not necessary) - set the org.deeplearning4j.ui.port system property
 * i.e., run the example and pass the following to the JVM, to use port 9001: -Dorg.deeplearning4j.ui.port=9001
 *
 * @author Donald A. Smith, Alex Black
 */
public class ActivationsListenerExample {

    public static void main(String[] args){

        //Get our network and training data
        MultiLayerNetwork net = UIExampleUtils.getMnistNetwork();
        DataSetIterator trainData = UIExampleUtils.getMnistData();

        System.out.println();
        for(Layer layer:net.getLayers()) {
            System.out.println(layer);
        }
        System.out.println();
        net.setListeners(new ActivationsListener(net,20));
       // net.setListeners(new ScoreIterationListener(10));
        net.fit(trainData);

    }
}

