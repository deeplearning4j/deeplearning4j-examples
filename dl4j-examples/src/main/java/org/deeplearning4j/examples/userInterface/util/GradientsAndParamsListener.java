package org.deeplearning4j.examples.userInterface.util;

import javafx.application.Application;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;
//import javafx.

/**
 * Use JavaFX to visualize network gradients and params during backward pass.
 *
 * @author Donald A. Smith
 */
public class GradientsAndParamsListener implements TrainingListener {
    private boolean invoked = false;
    private final MultiLayerNetwork network;
    private final int sampleSizePerLayer;
    private GradientsAndParamsViewer viewer;

    public GradientsAndParamsListener(MultiLayerNetwork network, int sampleSizePerLayer) {
        this.network = network;
        this.sampleSizePerLayer = sampleSizePerLayer;
    }

    private void initializeViewer() {
        GradientsAndParamsViewer.initialize(network, sampleSizePerLayer);
        new Thread(() -> Application.launch(GradientsAndParamsViewer.class)).start();
        int count = 0;
        int sleepMls = 20;
        while (GradientsAndParamsViewer.staticInstance == null) {
            count++;
            try {
                Thread.sleep(sleepMls);
            } catch (InterruptedException exc) {
                Thread.interrupted();
            }
        }
        System.out.println("GradientsViewer started up after " + (count * sleepMls) / 1000.0 + " seconds");
        viewer = GradientsAndParamsViewer.staticInstance;
        describeLayers();
    }

    @Override
    public boolean invoked() {
        if (!invoked) {
            System.out.println("Invoked");
        }
        return invoked;
    }

    @Override
    public void invoke() {
        System.out.println("invoke()");
        invoked = true;
    }

    public static String toString(int[] values) {
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for (int i = 0; i < values.length; i++) {
            if (i > 0) {
                sb.append(",");
            }
            sb.append(values[i]);
        }
        sb.append(']');
        return sb.toString();
    }

    private void describeLayers() {
        /*
For GradientsListenerExample (whose network is from UIExampleUtils.getMnistNetwork()):
0: 520 params, input shape = [64,1,28,28], input rank = 4, activation shape = [64,20,24,24]
1: 0 params, input shape = [64,20,24,24], input rank = 4, activation shape = [64,20,12,12]
2: 25050 params, input shape = [64,20,12,12], input rank = 4, activation shape = [64,50,8,8]
3: 0 params, input shape = [64,50,8,8], input rank = 4, activation shape = [64,50,4,4]
4: 400500 params, input shape = [64,800], input rank = 2, activation shape = [64,500]
5: 5010 params, input shape = [64,500], input rank = 2, activation shape = [64,10]

For MelodyModelingExample:
GradientsLister: describeLayers:  miniBatchSize = 32
0: 208600 params, input shape = [32,59,50], activation shape = [32,200,50]
1: 321400 params, input shape = [32,200,50], activation shape = [32,200,50]
2: 11859 params, input shape = [32,200,50], activation shape = [32,59,50]
         */
        System.out.println("\nGradientsListener: describeLayers:  miniBatchSize = " + network.getInputMiniBatchSize());
        for (Layer layer : network.getLayers()) {
            INDArray input = layer.input();
            INDArray activation = layer.activate();
            System.out.println(layer.getIndex() + ": " + layer.numParams() +
                    " params, input shape = " + toString(input.shape())
                    + ", activation shape = " + toString(activation.shape()) // matches the input of the next layer
                // + ", input mini-batch size = " + network.getInputMiniBatchSize()
                // + ", gradient  = " + gradient // toString(gradient.gradient().shape())
            );
        }
        System.out.println();
    }

    @Override
    public void iterationDone(Model model, int iteration) {
    }

    @Override
    public void onEpochStart(Model model) {

    }

    @Override
    public void onEpochEnd(Model model) {

    }

    @Override
    public void onForwardPass(Model model, List<INDArray> activations) {

    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {

    }

    @Override
    public void onGradientCalculation(Model model) {

    }

    @Override
    public void onBackwardPass(Model model) {
        if (viewer == null) {
            initializeViewer();
        }
        viewer.requestBackwardPassUpdate(model);
    }
}
