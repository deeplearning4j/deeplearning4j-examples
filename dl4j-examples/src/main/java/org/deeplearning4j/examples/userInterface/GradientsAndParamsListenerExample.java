package org.deeplearning4j.examples.userInterface;

import org.deeplearning4j.examples.userInterface.util.GradientsAndParamsListener;
import org.deeplearning4j.examples.userInterface.util.UIExampleUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * An example of how to view params and gradients for a samples of neurons
 * in a running network using a TrainingListener and JavaFX 3D.
 * Params include weights and biases.
 *
 * You can navigate in 3d space by dragging the mouse or by the arrow keys.
 *
 * Red gradients are large positive. Blue gradients are large negative.
 * Large positive weights or biases cause a large radius.
 * Large negative weights or biases cause a small radius.
 *
 * The slider on the bottom of the window adjusts the mapping of gradients to colors.
 *
 * Note: if you're using Java 7 or earlier, you need to set the
 * environment variable JAVAFX_HOME to the directory of the JavaFX SDK.
 *
 * @author Donald A. Smith, Alex Black
 */
public class GradientsAndParamsListenerExample {

    public static void main(String[] args){

        //Get our network and training data
        MultiLayerNetwork net = UIExampleUtils.getMnistNetwork();
        DataSetIterator trainData = UIExampleUtils.getMnistData();

        System.out.println();
        for(Layer layer:net.getLayers()) {
            System.out.println(layer);
        }
        System.out.println();
        net.setListeners(new GradientsAndParamsListener(net,100));
        net.fit(trainData);

    }
}

