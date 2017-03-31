package org.deeplearning4j.examples.userInterface.util;

import javafx.application.Application;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;
//import javafx.

/**
 * Use JavaFX to visualize network activations
 *
 * @author Donald A. Smith
 */
public class ActivationsListener implements IterationListener {
    private boolean invoked=false;
    private final MultiLayerNetwork network;
    private final int sampleSizePerLayer;
    private ActivationsViewer viewer;
    public ActivationsListener(MultiLayerNetwork network, int sampleSizePerLayer) {
        this.network=network;
        this.sampleSizePerLayer=sampleSizePerLayer;
    }
    private void initializeViewer() {
        ActivationsViewer.initialize(network,sampleSizePerLayer);
        new Thread(() -> Application.launch(ActivationsViewer.class)).start();
        int count=0;
        int sleepMls=20;
        while (ActivationsViewer.staticInstance==null) {
            count++;
            try{Thread.sleep(sleepMls);} catch (InterruptedException exc) {Thread.interrupted();}
        }
        System.out.println("ActivationsViewer started up after " + (count*sleepMls)/1000.0 + " seconds");
        viewer= ActivationsViewer.staticInstance;
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
        invoked=true;
    }
    public static String toString(int [] values) {
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for(int i=0;i<values.length;i++) {
            if (i>0) {
                sb.append(",");
            }
            sb.append(values[i]);
        }
        sb.append(']');
        return sb.toString();
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        if (iteration==0) {
            initializeViewer();
            describeLayers();
        }
        if (iteration>0 && iteration%1==0) {
            viewer.requestIterationUpdate(iteration);
        }
    }
    private void describeLayers() {
        /*
0: 520 params, input shape = [64,1,28,28], input rank = 4, activation shape = [64,20,24,24]
1: 0 params, input shape = [64,20,24,24], input rank = 4, activation shape = [64,20,12,12]
2: 25050 params, input shape = [64,20,12,12], input rank = 4, activation shape = [64,50,8,8]
3: 0 params, input shape = [64,50,8,8], input rank = 4, activation shape = [64,50,4,4]
4: 400500 params, input shape = [64,800], input rank = 2, activation shape = [64,500]
5: 5010 params, input shape = [64,500], input rank = 2, activation shape = [64,10]
         */
        System.out.println("\nActivationsListener layers:");
        for(Layer layer:network.getLayers()) {
            INDArray input= layer.input();
            INDArray activation = layer.activate();
            //Gradient gradient=layer.gradient(); //null
            System.out.println(layer.getIndex() + ": " + layer.numParams() +
                    " params, input shape = " + toString(input.shape())
                //    + ", input rank = " + input.rank()  // You can infer this from the shape above
                    + ", activation shape = " + toString(activation.shape()) // matches the input of the next layer
                // + ", input mini-batch size = " + network.getInputMiniBatchSize()
                // + ", gradient  = " + gradient // toString(gradient.gradient().shape())
            );
        }
        System.out.println();
    }
}
