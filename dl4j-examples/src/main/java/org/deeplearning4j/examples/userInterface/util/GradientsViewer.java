package org.deeplearning4j.examples.userInterface.util;

import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.event.*;
import javafx.event.Event;
import javafx.geometry.Insets;
import javafx.scene.DepthTest;
import javafx.scene.Group;
import javafx.scene.PerspectiveCamera;
import javafx.scene.Scene;
import javafx.scene.control.Slider;
import javafx.scene.input.InputEvent;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.PickResult;
import javafx.scene.layout.Background;
import javafx.scene.layout.BackgroundFill;
import javafx.scene.layout.CornerRadii;
import javafx.scene.paint.Color;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.Cylinder;
import javafx.scene.shape.Sphere;
import javafx.scene.text.Text;
import javafx.scene.transform.Rotate;
import javafx.scene.transform.Translate;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.Window;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.text.NumberFormat;
import java.util.*;
import java.util.List;

import static javafx.scene.input.KeyCode.Q;
import static javafx.scene.input.KeyCode.R;
import static org.nd4j.linalg.util.ArrayUtil.sum;


/**
 * A JavaFX application for viewing gradients in 3D.
 * Blue = negative gradient. Red = positive gradient. Green = gradient near zero.
 * The slider adjusts the sensitivity of the color mapping for gradients, on a logarithmic scale.
 *
 * You can navigate in 3d space with the mouse and with Up and Down arrow keys.
 * The Left and Right arrow keys control the slider.
 *
 * @author Donald A. Smith
 *
 */
public class GradientsViewer extends Application {
    private static double INITIAL_GRADIENT_FACTOR=100.0; // to make colors visible
    public static final int FRAME_COUNT_TO_COLLECT=200;
    public static boolean captureScreenImages = false; // Set to true to capture images to /tmp/images/gradients/
    private static int WIDTH=1400;
    private static int HEIGHT=900;
    static GradientsViewer staticInstance; // output
    static MultiLayerNetwork network;  //input
    private static int sampleSizePerLayer;    //input
    private int numberOfLayers;        //input
    private final Slider slider = new Slider();
    private final Random random = new Random(1234);
    //...
    private static NumberFormat numberFormat= NumberFormat.getInstance();
    static {
        numberFormat.setMaximumFractionDigits(2);
        numberFormat.setMinimumFractionDigits(2);
    }
    //...
    private Stage stage;
    private boolean sampleCoordinatesNeedToBeChosen=true;
    private double mousePosX, mousePosY, mouseOldX, mouseOldY, mouseDeltaX, mouseDeltaY;
    private double gradientFactor=INITIAL_GRADIENT_FACTOR;
    private final Group root = new Group();
    private final ShapesGroup shapesGroup = new ShapesGroup();
    private final List<GradientShape> allGradientShapes = new ArrayList<>();
    private List<Text> texts = new ArrayList<Text>();

    private static Robot robot; // For frame capture
    private int frameCount=0;

    private Sphere selectedSphere;
    private final PerspectiveCamera camera = new PerspectiveCamera(true);
    private static double cameraInitialX = WIDTH/2;
    private static double cameraInitialY = HEIGHT/2;
    private static double cameraInitialZ = -1600;
    private static final double CAMERA_NEAR_CLIP = 0.1;
    private static final double CAMERA_FAR_CLIP = 20000.0;
    private static final long FIVE_SECOND_IN_NANO_SECONDS=5000000000L;
    private java.awt.Rectangle screenBounds;
    private Set<String> shapes = new HashSet<>();

    //....
    private class GradientShape extends Sphere{
        private final String mapKey;
        private final int [] coordinateInIndArray;
        private volatile double lastGradient=-1;
        private boolean initialized=false;
        private volatile boolean needsUpdate=true;
        public GradientShape(String mapKey, int layerIndex,int sampleIndex, List<Integer> listOfCoordinates, int numberOfSamples) {
            super(10);
            this.coordinateInIndArray=new int[listOfCoordinates.size()];
            for(int i=0;i<listOfCoordinates.size();i++) {
                coordinateInIndArray[i] = listOfCoordinates.get(i).intValue();
            }
            this.mapKey =mapKey;
            //double deltaWidth=(0.0+WIDTH)/sampleSizePerLayer;
            double deltaWidth=(0.0+WIDTH)/Math.min(numberOfSamples,10);
            double deltaHeight= (0.0+HEIGHT)/numberOfLayers;
            //setTranslateX(deltaWidth/2 + sampleIndex*deltaWidth);
            setTranslateX(deltaWidth/2 + (sampleIndex%Math.min(numberOfSamples,10))*deltaWidth);
            setTranslateY(HEIGHT-(deltaHeight/2+layerIndex*deltaHeight));
            setTranslateZ(120*(sampleIndex/10));
            allGradientShapes.add(this);
        }

        public void updateFromNeuralGradient(Map<String, INDArray> map) {
            INDArray array = map.get(mapKey);
            if (shapes.add(GradientsListener.toString(array.shape()))) {
                System.out.println(" " + shapes);
            }
            double gradient = array.getDouble(coordinateInIndArray);
            if (gradient!= lastGradient) {
                lastGradient=gradient;
                needsUpdate=true;
            }

            //  System.out.println("d = " + d +", hue = " + hue);
        }
        public void applyUpdatesToShapesAndRefresh() {
            if (!initialized) {
                shapesGroup.getChildren().add(this);
                initialized=true;
            }
            if (needsUpdate) {
                //double hue=360.0/(1+ Math.exp(gradientFactor*lastGradient));
                double hue = 260 - 260.0/(1+ Math.exp(gradientFactor*lastGradient));
                // Blue is negative, Green is near zero, Red is positive.
                // The reason we don't let hues vary from 0 to 360 is that both ends are red. That is, 0=360=red.
                PhongMaterial material = new PhongMaterial(Color.hsb(hue,1,1));
                this.setMaterial(material);
                needsUpdate=false;
            }
        }
    }
    // -------------------------
    private static class ShapesGroup extends Group {
        final Translate t = new Translate(0.0, 0.0, 0.0);
        final Rotate rx = new Rotate(0, 0, 0, 0, Rotate.X_AXIS);
        final Rotate ry = new Rotate(0, 0, 0, 0, Rotate.Y_AXIS);
        final Rotate rz = new Rotate(0, 0, 0, 0, Rotate.Z_AXIS);

        public ShapesGroup() {
            super();
            this.getTransforms().addAll(t, rx, ry, rz);
        }
    }

    //......
    public GradientsViewer() {

    }
    public static void initialize(MultiLayerNetwork network, int sampleSizePerLayer) {
        GradientsViewer.network=network;
        GradientsViewer.sampleSizePerLayer=sampleSizePerLayer;
    }
    private void buildCamera() {
        root.getChildren().add(camera);
        camera.setNearClip(CAMERA_NEAR_CLIP);
        camera.setFarClip(CAMERA_FAR_CLIP);
        camera.setTranslateX(cameraInitialX);
        camera.setTranslateY(cameraInitialY);
        camera.setTranslateZ(cameraInitialZ);
    }
    private void buildSlider() {
        slider.setTranslateX(WIDTH/4);
        slider.setTranslateY(HEIGHT-60);
        slider.setMin(-10);
        slider.setValue(5);
        slider.setShowTickLabels(true);
        slider.setShowTickMarks(true);
        slider.setMajorTickUnit(1);
        BackgroundFill backgroundFill = new BackgroundFill(Color.AQUAMARINE, CornerRadii.EMPTY, Insets.EMPTY);
        Background background = new Background(backgroundFill);
        slider.setBackground(background);
        slider.setMax(10); // logarithmic scale
        slider.setMinWidth(WIDTH/2);
        root.getChildren().add(slider);

        // We add this to prevent the slider from processing the key event
        EventHandler filter = new EventHandler<InputEvent>() {
            public void handle (InputEvent event){
                System.out.println("Filtering out event " + event.getEventType());
                handleKeyEvent(event);
                event.consume();
            }
        };
        root.addEventFilter(KeyEvent.KEY_PRESSED, filter);
    }
    private void handleMouse(Scene scene) {
        scene.setOnMousePressed((MouseEvent me) -> {
            mousePosX = me.getSceneX();
            mousePosY = me.getSceneY();
            mouseOldX = me.getSceneX();
            mouseOldY = me.getSceneY();
            // this is done after clicking and the rotations are apparently
            // performed in coordinates that are NOT rotated with the camera.
            // (pls activate the two lines below for clicking)
            // cameraXform.rx.setAngle(-90.0);
            // cameraXform.ry.setAngle(180.0);
            PickResult pr = me.getPickResult();
            if (pr.getIntersectedNode() instanceof Sphere) {
                selectedSphere = (Sphere) pr.getIntersectedNode();
            }
            if (pr.getIntersectedNode() instanceof Cylinder) {
            }
        });
        scene.setOnMouseReleased((MouseEvent me) -> {
        });
        scene.setOnMouseDragExited((MouseEvent me) -> {
        });
        scene.setOnMouseDragged((MouseEvent me) -> {
            mouseOldX = mousePosX;
            mouseOldY = mousePosY;
            mousePosX = me.getSceneX();
            mousePosY = me.getSceneY();
            mouseDeltaX = (mousePosX - mouseOldX);
            mouseDeltaY = (mousePosY - mouseOldY);
            if (me.isPrimaryButtonDown()) {
                // this is done when the mouse is dragged and each rotation is
                // performed in coordinates, that are rotated with the camera.
                shapesGroup.ry.setAngle(shapesGroup.ry.getAngle() - mouseDeltaX * 0.2);
                shapesGroup.rx.setAngle(shapesGroup.rx.getAngle() + mouseDeltaY * 0.2);

                // world.ry.setAngle(world.ry.getAngle() - mouseDeltaX * 0.2);
                // world.rx.setAngle(world.rx.getAngle() + mouseDeltaY * 0.2);
            } else if (me.isSecondaryButtonDown()) {
                shapesGroup.t.setZ(shapesGroup.t.getZ() + mouseDeltaY);
                shapesGroup.t.setX(shapesGroup.t.getX() + mouseDeltaX);
            }
        });
    }
    private void handleKeyEvent(Event event) {
        KeyEvent ke = (KeyEvent) event;
        // System.out.println(ke.getCharacter() + " " + ke.getCode());
        switch (ke.getCode()) {
            case Q:
                System.exit(0);
                break;
            case R:
                shapesGroup.t.setX(0);
                shapesGroup.t.setY(0);
                shapesGroup.t.setZ(0);
                shapesGroup.rx.setAngle(0);
                shapesGroup.ry.setAngle(0);
                shapesGroup.rz.setAngle(0);
                break;
            case LEFT:
                shapesGroup.t.setX(shapesGroup.t.getX()-10);
                break;
            case RIGHT:
                shapesGroup.t.setX(shapesGroup.t.getX()+10);
                break;
            case UP:
                if (ke.isShiftDown()) {
                    shapesGroup.setTranslateY(shapesGroup.getTranslateY() - 10);
                } else {
                    shapesGroup.setTranslateZ(shapesGroup.getTranslateZ()-10);
                }
                break;
            case DOWN:
                if (ke.isShiftDown()) {
                    shapesGroup.setTranslateY(shapesGroup.getTranslateY() + 10);
                } else {
                    shapesGroup.setTranslateZ(shapesGroup.getTranslateZ()+10);
                }
                break;
            case PAGE_UP:
                break;
            case PAGE_DOWN:
            case C:
                if (ke.isShiftDown()) {
                } else {
                }
                break;
            default:
        }
    }

    private void handleKeyEvents(Scene scene) {
        // We do it this way to prevent the slider from consuming the LEFT and RIGHT arrow events.
        EventHandler<? super KeyEvent> handler = new EventHandler() {
            @Override
            public void handle(Event event) {
                handleKeyEvent(event);
            }
        };
        scene.setOnKeyPressed(handler);

    }
    //-------------------
    private void initializeCapture() {
        try {
            robot = new Robot();
            Window window=stage; // .getOwner();
            if (window==null) {
                System.err.println("null window");
                System.exit(1);
            }
            screenBounds = new java.awt.Rectangle((int)window.getX(),(int)window.getY(),(int)window.getWidth(), (int)window.getHeight());
        } catch (Exception exc) {
            exc.printStackTrace();
            System.exit(1);
        }
    }
    private static void mkdir(String path) {
        File dir = new File(path);
        if (dir.exists()) {
            if (!dir.isDirectory()) {
                throw new RuntimeException(path + " is not a directory");
            }
        } else {
            if (!dir.mkdir()) {
                throw new RuntimeException("Could not create " + path);
            }
        }
    }

    private void capture() {
        mkdir("/tmp"); // On windows
        mkdir("/tmp/images");
        mkdir("/tmp/images/gradients");
        try {
            BufferedImage image = robot.createScreenCapture(screenBounds);
            File file = new File("/tmp/images/gradients/image" + frameCount + ".jpg");
            ImageIO.write(image, "JPEG",file);
            frameCount++;
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }
    //---------
    private static int product(final int[] array) {
        int value=1;
        for(int v:array) {
            value*= v;
        }
        return value;
    }

    private List<Integer> chooseRandomSample(int[] shape) {
        List<Integer> list = new ArrayList<>(shape.length);
        for(int i=0;i<shape.length;i++) {
            list.add(random.nextInt(shape[i]));
        }
        return  list;
    }
    //.....
    private void chooseSampleCoordinates(Map<String, INDArray> map) { //[0_b, 0_W, 2_b, 2_W, 4_W, 4_b, 5_W, 5_b]

        int layerIndex = 0;
        numberOfLayers=map.size();
        System.out.println("Entering chooseSampleCoordinates with map.size() = " + map.size() + " and sampleSizePerLayer= " + sampleSizePerLayer);
        for (String key : map.keySet()) {
            INDArray array = map.get(key);
            int[] shape = array.shape();
            System.out.println("  Gradient map shape for " + key + " is " + GradientsListener.toString(shape));

            int sampleLengthWeWillUse = Math.min(sampleSizePerLayer,product(shape));
            Set<List<Integer>> chosen = new HashSet<>();
            // TODO: this could be slow if product(shape) is close to sampleSizePerLayer
            while (chosen.size() < sampleLengthWeWillUse) {
                chosen.add(chooseRandomSample(shape));
            }
            int sampleIndex=0;
            for (List<Integer> coordinates:chosen) {
//   public GradientShape(String mapKey, int layerIndex,int sampleIndex, int coordinateInIndArray) {
                new GradientShape(key, layerIndex, sampleIndex, coordinates,sampleLengthWeWillUse);
                sampleIndex++;
            }
            layerIndex++;
        }
        System.out.println();
    }
    // We can't update the JavaFX UI components from this thread, so we just store the updates in the GradientShape's variables. Later,
    // the animation handler will apply the updates to the JavaFX shapes themselves.
    public void requestBackwardPassUpdate(Model model) {
        Gradient gradient = model.gradient();
        // INDArray gradientArray = gradient.gradient();
        // gradientArrayShape = [1,541859]
        Map<String, INDArray> map= gradient.gradientForVariable();

        if (sampleCoordinatesNeedToBeChosen) {
            chooseSampleCoordinates(map);
            System.out.println("Created " + allGradientShapes.size() + " shapes");
            //
            int layerIndex=0;
            double deltaY = (0.0+HEIGHT)/map.size();
            for(String key: map.keySet()) {
                Text text = new Text(10,  (HEIGHT-deltaY/2) - deltaY * layerIndex, key);
                texts.add(text);
                layerIndex++;
            }
            sampleCoordinatesNeedToBeChosen=false;
        }
        // gradientForVariable keys= [0_b, 0_W, 2_b, 2_W, 4_W, 4_b, 5_W, 5_b]
        for(GradientShape input: allGradientShapes) {
            input.updateFromNeuralGradient(map);
        }
    }
    private long startTime=System.nanoTime();
    private void animate() {
        final AnimationTimer timer = new AnimationTimer() {
            @Override
            public void handle(long nowInNanoSeconds) {
                if (texts!=null && !texts.isEmpty()) {
                    for(Text text: texts) {
                        shapesGroup.getChildren().add(text);
                    }
                    texts=null;
                }
                gradientFactor = Math.pow(2,slider.getValue());
                for(GradientShape input: allGradientShapes) {
                    input.applyUpdatesToShapesAndRefresh();
                }
                shapesGroup.requestLayout();
                if (captureScreenImages && nowInNanoSeconds-startTime> FIVE_SECOND_IN_NANO_SECONDS && frameCount<FRAME_COUNT_TO_COLLECT) {
                    if (frameCount==0) {
                        initializeCapture();
                        System.out.println("Starting capture");
                    }
                    capture();
                    if (frameCount==100) {
                        System.out.println("Ending capture");
                    }
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
        stage.setTitle("Gradients visualization for a sample of neurons.  Use the " +
            "mouse to navigate in 3D.  Blue is -; Red is +.  The slider adjusts the mapping of gradients to colors.");
        stage.setMinWidth(850);
        stage.setMinHeight(600);
        buildCamera();
        buildSlider();
        root.setDepthTest(DepthTest.ENABLE);
        root.getChildren().add(shapesGroup);
        shapesGroup.setTranslateZ(100);
        Scene scene = new Scene(root, WIDTH, HEIGHT, true);
        scene.setCamera(camera);
        scene.setFill(Color.DIMGREY);
        handleMouse(scene);
        handleKeyEvents(scene);
        stage.setScene(scene);
        staticInstance=this;
        stage.show();
        animate();
    }
    //--------------------------------------------
}

