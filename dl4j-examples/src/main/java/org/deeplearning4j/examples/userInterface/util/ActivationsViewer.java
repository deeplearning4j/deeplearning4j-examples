package org.deeplearning4j.examples.userInterface.util;

import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.paint.Color;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.Sphere;
import javafx.stage.Modality;
import javafx.stage.Stage;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;


/**
 * Created by Don Smith on 3/27/2017.
 */
public class ActivationsViewer extends Application {
    private static int WIDTH = 1400;
    private static int HEIGHT = 900;
    public static ActivationsViewer staticInstance; // output
    public static MultiLayerNetwork network;  //input
    private static int sampleSizePerLayer;    //input
    private static int numberOfLayers;        //input
    //...
    private static NumberFormat numberFormat = NumberFormat.getInstance();

    static {
        numberFormat.setMaximumFractionDigits(2);
        numberFormat.setMinimumFractionDigits(2);
    }

    //...
    private Stage stage;
    private final Group root = new Group();
    private List<int[]>[] sampleCoordinatesByLayer;
    private ActivationShape[][] shapesByLayerAndSample;
    private final List<ActivationShape> allActivationShapes = new ArrayList<>();
    private volatile CountDownLatch needUpdate;

    //....
    private class ActivationShape extends Sphere {
        private final int[] coordinatesInLayerInput;
        private final int layerIndex;

        public ActivationShape(int layerIndex, int sampleIndex, int[] coordinates) {
            super(10);
            this.layerIndex = layerIndex;
            //double d=network.getLayer(layerIndex).input().getDouble(coordinates);
            this.coordinatesInLayerInput = coordinates;
            double deltaWidth = WIDTH / sampleSizePerLayer;
            double deltaHeight = HEIGHT / numberOfLayers;
            setTranslateX(deltaWidth / 2 + sampleIndex * deltaWidth);
            setTranslateY(deltaHeight / 2 + layerIndex * deltaHeight);
            updateFromNeuralInput();
            allActivationShapes.add(this);
        }

        public void updateFromNeuralInput() {
            double d = 0;
            INDArray input = network.getLayer(layerIndex).input();
            d = -10 * input.getDouble(coordinatesInLayerInput);
            double hue = 360.0 / (1 + Math.exp(d));
            PhongMaterial material = new PhongMaterial(Color.hsb(hue, 1, 1));
            this.setMaterial(material);
            //  System.out.println("d = " + d +", hue = " + hue);
        }
    }

    //......
    public ActivationsViewer() {

    }

    public static void initialize(MultiLayerNetwork network, int sampleSizePerLayer) {
        ActivationsViewer.network = network;
        ActivationsViewer.sampleSizePerLayer = sampleSizePerLayer;
        ActivationsViewer.numberOfLayers = network.getnLayers();
    }

    private void makeLayerViews() {
        sampleCoordinatesByLayer = new List[numberOfLayers];
        chooseSampleCoordinates();
        makeInputShapes();
    }

    //-----
    private void makeInputShapes() {
        shapesByLayerAndSample = new ActivationShape[numberOfLayers][sampleSizePerLayer];
        for (int layerIndex = 0; layerIndex < numberOfLayers; layerIndex++) {
            for (int i = 0; i < sampleSizePerLayer; i++) {
                ActivationShape activationShape = new ActivationShape(layerIndex, i, sampleCoordinatesByLayer[layerIndex].get(i));
                shapesByLayerAndSample[layerIndex][i] = activationShape;
                root.getChildren().add(activationShape);

            }
        }
    }

    //.....
    private void chooseSampleCoordinates() { // But the 0th index must be 0, because some mini-batches are smaller
        Random random = new Random(1234);
        for (int layerIndex = 0; layerIndex < sampleCoordinatesByLayer.length; layerIndex++) {
            List<int[]> list = new ArrayList<>();
            sampleCoordinatesByLayer[layerIndex] = list;
            Layer layer = network.getLayer(layerIndex);
            INDArray input = layer.input();
            int[] shape = input.shape();
            int repeats = 0; // used to avoid the (rare) case in which we try to add many duplicates.
            for (int r = 0; r < sampleSizePerLayer; r++) {
                int[] sampleCoordinates = new int[shape.length];
                for (int i = 1; i < shape.length; i++) {
                    sampleCoordinates[i] = random.nextInt(shape[i]);
                }
                if (isDuplicate(sampleCoordinates, list) && repeats < 20) {
                    r--;
                    repeats++;
                } else {
                    list.add(sampleCoordinates);
                }
            }
        }
        showSampleCoordinates();
    }

    private void showSampleCoordinates() {
        System.out.println("Sample coordindates");
        for (int layer = 0; layer < sampleCoordinatesByLayer.length; layer++) {
            System.out.println(layer);
            for (int[] sample : sampleCoordinatesByLayer[layer]) {
                System.out.println("  " + ActivationsListener.toString(sample));
            }
        }
        System.out.println();
    }

    private static boolean isDuplicate(int[] array, List<int[]> list) {
        for (int[] ar : list) {
            if (equal(ar, array)) {
                return true;
            }
        }
        return false;
    }

    private static boolean equal(int[] a1, int[] a2) {
        if (a1.length != a2.length) {
            throw new IllegalStateException();
        }
        for (int i = 0; i < a1.length; i++) {
            if (a1[i] != a2[i]) {
                return false;
            }
        }
        return true;
    }

    /*
    ActivationsListener layers:
    0: 520 params, input shape = [64,1,28,28], activation shape = [64,20,24,24]   64 is batch size, 20 is nOut, 20*24*24=11520
    1: 0 params, input shape = [64,20,24,24], activation shape = [64,20,12,12]   max pooling,  20*12*12=2880
    2: 25050 params, input shape = [64,20,12,12], activation shape = [64,50,8,8]  64 is batch size, 25 is nOut, 50*8*8=3200
    3: 0 params, input shape = [64,50,8,8], activation shape = [64,50,4,4]       max pooling, 5*4*4=800
    4: 400500 params, input shape = [64,800], activation shape = [64,500]
    5: 5010 params, input shape = [64,500], activation shape = [64,10]
     */
    private long start = System.currentTimeMillis();

    public void requestIterationUpdate(int iteration) {
        needUpdate = new CountDownLatch(1); // The method animate() returns unless needUpdate is non-null.
        try {
            needUpdate.await();
        } catch (InterruptedException exc) {
            System.err.println("Warning: interrupted in requestIterationUpdate of ActivationsViewer");
            Thread.interrupted();
        }
        // TODO: I think is there a race condition here. It's possible that the UI thread runs again before we set
        // needUpdate to null.  We could overcome this by copying the state into shared variables and letting
        // the UI thread reference those variables.
        needUpdate = null;
    }

    private void updateInternal() {
        for (ActivationShape activationShape : allActivationShapes) {
            activationShape.updateFromNeuralInput();
        }
        root.requestLayout();
    }

    private void animate() {
        final AnimationTimer timer = new AnimationTimer() {
            @Override
            public void handle(long nowInNanoSeconds) {
                if (needUpdate != null) {
                    updateInternal();
                    needUpdate.countDown();
                }
            }
        };
        timer.start();
    }

    @Override
    public void start(Stage primaryStage) throws Exception {
        stage = new Stage();
        stage.initModality(Modality.NONE);
        stage.setOnCloseRequest(r -> System.exit(0));
        stage.setTitle("Network viewer");
        stage.setMinWidth(850);
        stage.setMinHeight(600);
        Scene scene = new Scene(root, WIDTH, HEIGHT, true);
        scene.setFill(Color.DIMGREY);
        stage.setScene(scene);
        staticInstance = this;
        makeLayerViews();
        stage.show();
        animate();
    }
    //--------------------------------------------
}
