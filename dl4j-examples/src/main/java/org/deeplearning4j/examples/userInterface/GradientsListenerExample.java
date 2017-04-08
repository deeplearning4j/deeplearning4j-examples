package org.deeplearning4j.examples.userInterface;

import org.deeplearning4j.examples.userInterface.util.GradientsListener;
import org.deeplearning4j.examples.userInterface.util.UIExampleUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 *
 * An example of how to view gradients in a running network using a
 * TrainingListener and JavaFX 3D.
 *
 * Note: if you're using Java 7 or earlier, you need to set the
 * environment variable JAVAFX_HOME to the directory of the JavaFX SDK.
 *
 * @author Donald A. Smith, Alex Black
 */
public class GradientsListenerExample {

    public static void main(String[] args){

        //Get our network and training data
        MultiLayerNetwork net = UIExampleUtils.getMnistNetwork();
        DataSetIterator trainData = UIExampleUtils.getMnistData();

        System.out.println();
        for(Layer layer:net.getLayers()) {
            System.out.println(layer);
        }
        System.out.println();
        net.setListeners(new GradientsListener(net,100));
        net.fit(trainData);

    }
}

