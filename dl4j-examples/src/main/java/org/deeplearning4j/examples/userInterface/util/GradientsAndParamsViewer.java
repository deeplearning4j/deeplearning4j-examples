package org.deeplearning4j.examples.userInterface.util;

import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.event.Event;
import javafx.event.EventHandler;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.DepthTest;
import javafx.scene.Group;
import javafx.scene.PerspectiveCamera;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.input.InputEvent;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.PickResult;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.Sphere;
import javafx.scene.text.*;
import javafx.scene.transform.Rotate;
import javafx.scene.transform.Translate;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import javafx.stage.Window;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import java.awt.Robot;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.File;
import java.text.NumberFormat;
import java.util.*;
import java.util.List;

/**
 * A JavaFX application for viewing gradients and params in 3D.
 * Blue = negative gradient. Red = positive gradient. Green = gradient near zero.
 * The slider adjusts the sensitivity of the color mapping for gradients, on a logarithmic scale.
 * <p>
 * The radii of the spheres reflect the size of the weight or bias on the parameter:
 * negative params lead to small radii, while positive params lead to large radii.
 * <p>
 * You can navigate in 3d space by dragging the mouse or by the arrow keys.
 * <p>
 * Note: there are two layers in the visualization for each layer in the network:
 * one layer for the bias and another layer for the weights.
 *
 * @author Donald A. Smith
 */
public class GradientsAndParamsViewer extends Application {
    private static double INITIAL_GRADIENT_FACTOR = 100.0; // to make colors visible
    private static final double SELECTED_RADIUS = 30;
    public static final int FRAME_COUNT_TO_COLLECT = 200;
    private static final double DEFAULT_RADIUS_LOGIT_FACTOR = 100.0;
    private static final String PAUSE_TEXT="Pause     ";
    public static boolean captureScreenImages = false; // Set to true to capture images to /tmp/images/gradients/
    private static int WIDTH = 1400;
    private static int HEIGHT = 900;
    static GradientsAndParamsViewer staticInstance; // output
    static MultiLayerNetwork network;  //input
    private static int sampleSizePerLayer;    //input
    private int numberOfLayers;        //input
    private final Slider slider = new Slider();
    private final Random random = new Random(1234);
    private static NumberFormat numberFormatLonger = NumberFormat.getNumberInstance();
    //...
    private static NumberFormat numberFormat = NumberFormat.getInstance();

    static {
        numberFormat.setMaximumFractionDigits(2);
        numberFormat.setMinimumFractionDigits(2);
        numberFormatLonger.setMaximumFractionDigits(8);
        numberFormatLonger.setMinimumFractionDigits(8);
    }

    //...
    private Stage stage;
    private boolean sampleCoordinatesNeedToBeChosen = true;
    private double mousePosX, mousePosY, mouseOldX, mouseOldY, mouseDeltaX, mouseDeltaY;
    private double gradientFactor = INITIAL_GRADIENT_FACTOR;
    private final Group root = new Group();
    private final ShapesGroup shapesGroup = new ShapesGroup();
    private final List<GradientParamShape> allGradientParamShapes = new ArrayList<>();
    private List<Text> texts = new ArrayList<Text>(); // for layer names

    private static Robot robot; // For frame capture
    private int frameCount = 0;

    private GradientParamShape selectedGradientParamShape;
    private final PerspectiveCamera camera = new PerspectiveCamera(true);
    private static double cameraInitialX = WIDTH / 2;
    private static double cameraInitialY = HEIGHT / 2;
    private static double cameraInitialZ = -1600;
    private static final double CAMERA_NEAR_CLIP = 0.1;
    private static final double CAMERA_FAR_CLIP = 20000.0;
    private static final long FIVE_SECOND_IN_NANO_SECONDS = 5000000000L;
    private Rectangle screenBounds;
    private double minParam = Double.MAX_VALUE;
    private double maxParam = Double.NEGATIVE_INFINITY;
    private double radiusLogitFactor = DEFAULT_RADIUS_LOGIT_FACTOR;
    private Text textForSelectedGradientParam;
    final Button pauseButton = new Button(PAUSE_TEXT);
    final Button stepButton =  new Button("Step");
    //final Button saveButton = new Button("Save");
    private volatile boolean paused=false;
    private volatile boolean stepping=false;
    private Stage layerStage=null;
    private final Label helpLabelForLayers = new Label("Click on a bias/weight param name to view details");
    private final TextField learningRateTextField = new TextField("");
    private final TextField activationFunctionTextField = new TextField("");
    private final TextField updaterTextField = new TextField("");
    //........
    private class GradientParamShape extends Sphere {
        private final String mapKey;
        private final int[] coordinateInIndArray;
        private volatile double lastGradient = 1000.1;
        private volatile double lastParam = 1000.1;
        private boolean initialized = false;
        private boolean selected = false;
        private volatile boolean needsUpdate = true;

        public GradientParamShape(String mapKey, int layerIndex, int sampleIndex, List<Integer> listOfCoordinates, int numberOfSamples) {
            super(10);
            this.setUserData(this);
            this.coordinateInIndArray = new int[listOfCoordinates.size()];
            for (int i = 0; i < listOfCoordinates.size(); i++) {
                coordinateInIndArray[i] = listOfCoordinates.get(i).intValue();
            }
            this.mapKey = mapKey;
            //double deltaWidth=(0.0+WIDTH)/sampleSizePerLayer;
            double deltaWidth = (0.0 + WIDTH) / Math.min(numberOfSamples, 10);
            double deltaHeight = (0.0 + HEIGHT) / numberOfLayers;
            setTranslateX(deltaWidth / 2 + (sampleIndex % Math.min(numberOfSamples, 10)) * deltaWidth);
            setTranslateY(HEIGHT - (deltaHeight / 2 + layerIndex * deltaHeight));
            setTranslateZ(120 * (sampleIndex / 10));
            allGradientParamShapes.add(this);
        }

        private double getGradient() {
            return lastGradient;
        }

        private double getParam() {
            return lastParam;
        }

        private void select() {
            selected = true;
        }

        private void unselect() {
            selected=false;
        }

        private void updateFromNeuralGradientAndParams(Map<String, INDArray> gradientMap, Map<String, INDArray> paramMap) {
            INDArray gradientArray = gradientMap.get(mapKey);
            INDArray paramArray = paramMap.get(mapKey);
            double gradient = gradientArray.getDouble(coordinateInIndArray);
            double param = paramArray.getDouble(coordinateInIndArray);
            if (gradient != lastGradient) {
                lastGradient = gradient;
                needsUpdate = true;
            }
            if (param != lastParam) {
                lastParam = param;
                needsUpdate = true;
                if (lastParam < minParam) {
                    minParam = lastParam;
                }
                if (lastParam > maxParam) {
                    maxParam = lastParam;
                }
            }
        }

        public void applyUpdatesToShapesAndRefresh() {
            if (!initialized) {
                shapesGroup.getChildren().add(this);
                initialized = true;
            }
            if (needsUpdate || stepping || paused) {
                //double hue=360.0/(1+ Math.exp(gradientFactor*lastGradient));
                double hue = 260 - 260.0 / (1 + Math.exp(gradientFactor * lastGradient));
                // Blue is negative, Green is near zero, Red is positive.
                // The reason we don't let hues vary from 0 to 360 is that both ends are red. That is, 0=360=red.
                PhongMaterial material = new PhongMaterial(Color.hsb(hue, 1, 1));
                this.setMaterial(material);
                double newRadius = selected? SELECTED_RADIUS : (2 + 10.0 / (1.0 + Math.exp(-lastParam * radiusLogitFactor)));
                //newRadius = 5+ 10*Math.pow(newRadius,3);
                // The animation becomes very slow if we update the radius too often. Hence the next if.
                if (Math.abs(newRadius - getRadius()) > 0.5) {
                    this.setRadius(newRadius);
                }
                needsUpdate = false;
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
    public GradientsAndParamsViewer() {

    }

    public static void initialize(MultiLayerNetwork network, int sampleSizePerLayer) {
        GradientsAndParamsViewer.network = network;
        GradientsAndParamsViewer.sampleSizePerLayer = sampleSizePerLayer;
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
        slider.setTranslateX(WIDTH / 4);
        slider.setTranslateY(HEIGHT - 60);
        slider.setMin(-10);
        slider.setMax(50); // logarithmic scale
        slider.setValue(5);
        slider.setShowTickLabels(false);
        slider.setShowTickMarks(true);
        slider.setMajorTickUnit(5);
        slider.setTooltip(new Tooltip("Modify the sensitivity of the mapping from gradients to colors."));
        BackgroundFill backgroundFill = new BackgroundFill(Color.AQUAMARINE, CornerRadii.EMPTY, Insets.EMPTY);
        Background background = new Background(backgroundFill);
        slider.setBackground(background);
        slider.setMinWidth(WIDTH / 2);
        root.getChildren().add(slider);

        // We add this to prevent the slider from processing the key event
        EventHandler filter = new EventHandler<InputEvent>() {
            public void handle(InputEvent event) {
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
                Sphere sphere = (Sphere) pr.getIntersectedNode();
                GradientParamShape newGradientParamShape = (GradientParamShape) sphere.getUserData();
                if (selectedGradientParamShape != newGradientParamShape) {
                    if (selectedGradientParamShape != null) {
                        selectedGradientParamShape.unselect();
                    }
                    newGradientParamShape.select();
                    selectedGradientParamShape = newGradientParamShape;
                }
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
                shapesGroup.t.setX(shapesGroup.t.getX() - 10);
                break;
            case RIGHT:
                shapesGroup.t.setX(shapesGroup.t.getX() + 10);
                break;
            case UP:
                if (ke.isShiftDown()) {
                    shapesGroup.setTranslateY(shapesGroup.getTranslateY() - 10);
                } else {
                    shapesGroup.setTranslateZ(shapesGroup.getTranslateZ() - 10);
                }
                break;
            case DOWN:
                if (ke.isShiftDown()) {
                    shapesGroup.setTranslateY(shapesGroup.getTranslateY() + 10);
                } else {
                    shapesGroup.setTranslateZ(shapesGroup.getTranslateZ() + 10);
                }
                break;

            case P:
                if (paused) {
                    paused=false;
                    stepping=false;
                } else {
                    paused=true;
                    stepping=false;
                }
                break;
            case S: // step
                paused=false;
                stepping=true;
                break;
            case C:
                if (paused) {
                    paused=false;
                    stepping=false;
                }
                break;
            case PAGE_UP:
                radiusLogitFactor *= 1.1;
                System.out.println("radiusLogitFactor = " + radiusLogitFactor);
                break;
            case PAGE_DOWN:
                radiusLogitFactor /= 1.1;
                System.out.println("radiusLogitFactor = " + radiusLogitFactor);
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
            Window window = stage; // .getOwner();
            if (window == null) {
                System.err.println("null window");
                System.exit(1);
            }
            screenBounds = new Rectangle((int) window.getX(), (int) window.getY(), (int) window.getWidth(), (int) window.getHeight());
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
            ImageIO.write(image, "JPEG", file);
            frameCount++;
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    //---------
    private static int product(final int[] array) {
        int value = 1;
        for (int v : array) {
            value *= v;
        }
        return value;
    }

    private List<Integer> chooseRandomSample(int[] shape) {
        List<Integer> list = new ArrayList<>(shape.length);
        for (int i = 0; i < shape.length; i++) {
            list.add(random.nextInt(shape[i]));
        }
        return list;
    }

    //.....
    private void chooseSampleCoordinates(Map<String, INDArray> map) { //[0_b, 0_W, 2_b, 2_W, 4_W, 4_b, 5_W, 5_b]
        int layerIndex = 0;
        numberOfLayers = map.size();
        System.out.println("Entering chooseSampleCoordinates with map.size() = " + map.size() + " and sampleSizePerLayer= " + sampleSizePerLayer);
        for (String key : map.keySet()) {
            INDArray array = map.get(key);
            int[] shape = array.shape();
            System.out.println("  Gradient map shape for " + key + " is " + GradientsAndParamsListener.toString(shape));

            int sampleLengthWeWillUse = Math.min(sampleSizePerLayer, product(shape));
            Set<List<Integer>> chosen = new HashSet<>();
            // TODO: this could be slow if product(shape) is close to sampleSizePerLayer
            while (chosen.size() < sampleLengthWeWillUse) {
                chosen.add(chooseRandomSample(shape));
            }
            int sampleIndex = 0;
            for (List<Integer> coordinates : chosen) {
                new GradientParamShape(key, layerIndex, sampleIndex, coordinates, sampleLengthWeWillUse);
                sampleIndex++;
            }
            layerIndex++;
        }
        System.out.println();
    }
    private static String pretty(String string) {
        StringBuilder sb = new StringBuilder();
        int col=0;
        for(int i=0;i<string.length();i++) {
            char ch=string.charAt(i);
            if (col>120 && ch==',') {
                sb.append("\n  ");
                col=0;
            }
            sb.append(ch);
            col++;
        }
        return sb.toString();
    }
    private void makeLayerStage() {
        int layerStageWidth=350;
        int layerStageHeight=200;
        int vboxPaneSpacing=20;
        int hboxSpacing=30;
        int textFieldColumnCountForDoubles=8;
        layerStage=new Stage(StageStyle.DECORATED);
        //final Group layerStageGroup = new Group();
        VBox vboxPane = new VBox(vboxPaneSpacing);
        vboxPane.setAlignment(Pos.CENTER);

        HBox hboxLearningRate = new HBox(hboxSpacing);
        HBox hboxMomentum = new HBox(hboxSpacing);
        HBox hboxActivationFunction = new HBox(hboxSpacing);
        HBox hboxUpdater = new HBox(hboxSpacing);

        hboxLearningRate.setAlignment(Pos.CENTER);
        hboxMomentum.setAlignment(Pos.CENTER);
        hboxActivationFunction.setAlignment(Pos.CENTER);
        hboxUpdater.setAlignment(Pos.CENTER);

        vboxPane.getChildren().addAll(hboxLearningRate,hboxMomentum,hboxActivationFunction, hboxUpdater);

        Scene layerScene = new Scene(vboxPane,layerStageWidth,layerStageHeight);
        vboxPane.setBackground(new Background( new BackgroundFill(Color.LIGHTBLUE, CornerRadii.EMPTY, Insets.EMPTY)));

        Label learningRateTitleLabel = new Label("Learning Rate: " );
        Label momentumTitleLabel = new Label("Momentum: ");
        Label activationFunctionLabel = new Label("Activation function: ");
        Label updatedLabel = new Label("Updater: ");

        learningRateTextField.setPrefColumnCount(textFieldColumnCountForDoubles);

        // Modifying the learning rate or momentum is trickier than just setting the values in the configuration,
        // so we disable editing and don't create action handlers.
        learningRateTextField.setEditable(false);
        activationFunctionTextField.setEditable(false);
        updaterTextField.setEditable(false);

        hboxLearningRate.getChildren().addAll(learningRateTitleLabel, learningRateTextField);
        hboxActivationFunction.getChildren().addAll(activationFunctionLabel,activationFunctionTextField);
        hboxUpdater.getChildren().addAll(updatedLabel,updaterTextField);

        layerStage.setScene(layerScene);
        layerStage.setAlwaysOnTop(true);
        layerStage.setOnCloseRequest(r -> {});
        layerStage.show();
    }
    private void showLayerData(String key, Layer layer, MultiLayerNetwork network, String param) {
        org.deeplearning4j.nn.conf.layers.BaseLayer conf = (BaseLayer)layer.conf().getLayer();
        System.out.println("For " + key + ", conf = " + pretty(conf.toString()));
        if (layerStage == null || !layerStage.isShowing()) {
            makeLayerStage();
        }
        layerStage.setTitle(key + " (" + layer.type() + ")");
        double learningRate = conf.getLearningRateByParam(param);
        learningRateTextField.setText(""+ learningRate);
        activationFunctionTextField.setText(conf.getActivationFn().toString());
        updaterTextField.setText(conf.getIUpdater().toString());

        layerStage.setTitle(key + ": " + layer.type());
        layerStage.requestFocus();
    }
    public void requestBackwardPassUpdate(Model model) {
    // We can't update the JavaFX UI components from this thread, so we just store the updates in the GradientParamShape's variables. Later,
    // the animation handler will apply the updates to the JavaFX shapes themselves.
        while (paused) {
            try {
                Thread.sleep(200);
            } catch (InterruptedException exc) {
                Thread.interrupted();
                System.err.println("Interrupted!");
            }
        }
        Gradient gradient = model.gradient();
        Map<String, INDArray> gradientMap = gradient.gradientForVariable();
        Map<String, INDArray> paramMap = model.paramTable();
        final MultiLayerNetwork multiLayerNetwork = (MultiLayerNetwork) model;
        if (sampleCoordinatesNeedToBeChosen) {
            chooseSampleCoordinates(gradientMap);
            System.out.println("Created " + allGradientParamShapes.size() + " shapes");
            assert (gradientMap.keySet().equals(paramMap.keySet()));
            //
            int layerIndex = 0;
            double deltaY = (0.0 + HEIGHT) / gradientMap.size();
            for (String key : gradientMap.keySet()) {
                Text text = new Text(15, (HEIGHT - deltaY / 2) - deltaY * layerIndex, key);
                text.setFill(Color.GOLD);
                text.setFont(new Font(18));
                texts.add(text);
                text.setOnMouseClicked(e -> {
                    System.out.println("Clicked " + key + ", model.getClass() = " + model.getClass());
                    // Examples of a key are "2_W" and "5_B", where 2 and 5 are the layer numbers, respectively.
                    String parts[]= key.split("_");
                    if (parts.length==2) {
                        Layer layer = multiLayerNetwork.getLayer(Integer.parseInt(parts[0]));
                        String param=parts[1];
                        showLayerData(key, layer, multiLayerNetwork, param);
                    }
                });
                layerIndex++;
            }
            sampleCoordinatesNeedToBeChosen = false;
        }
        // gradientForVariable keys= [0_b, 0_W, 2_b, 2_W, 4_W, 4_b, 5_W, 5_b]
        for (GradientParamShape input : allGradientParamShapes) {
            input.updateFromNeuralGradientAndParams(gradientMap, paramMap);
        }
        if (stepping) {
            paused=true;
        }
    }

    private long startTime = System.nanoTime();

    private void createTextForSelectedGradientParam() {
        textForSelectedGradientParam = new Text("(Click on a node to track its gradient and bias/weight in real time.)");
        textForSelectedGradientParam.setTranslateX(50);
        textForSelectedGradientParam.setTranslateY(40);
        Font font = new Font(20);
        textForSelectedGradientParam.setFont(font);
        textForSelectedGradientParam.setFill(Color.GREENYELLOW);
        root.getChildren().add(textForSelectedGradientParam);
    }

    private void buildPauseButton() {
        pauseButton.setTranslateX(WIDTH-110);
        pauseButton.setTranslateY(30);
        pauseButton.setTextFill(Color.GREEN);
        BackgroundFill backgroundFill = new BackgroundFill(Color.BLACK, CornerRadii.EMPTY, Insets.EMPTY);
        Background background = new Background(backgroundFill);
        pauseButton.setBackground(background);
        root.getChildren().add(pauseButton);
        pauseButton.setOnAction(action -> {
            if (paused) {
                paused=false;
                stepping=false;
            } else {
                paused=true;
                stepping=false;
            }
        });
    }
    private void buildStepButton() {
        stepButton.setTranslateX(WIDTH-110);
        stepButton.setTranslateY(60);
        stepButton.setTextFill(Color.GREEN);
        BackgroundFill backgroundFill = new BackgroundFill(Color.BLACK, CornerRadii.EMPTY, Insets.EMPTY);
        Background background = new Background(backgroundFill);
        stepButton.setBackground(background);
        root.getChildren().add(stepButton);
        stepButton.setOnAction(action -> {
           paused=false;
           stepping=true;
        });
    }
    private void doPauseLogicInUIThread() {
        if (paused) {
            pauseButton.setText("Continue");
            pauseButton.setTextFill(Color.RED);
        } else {
            pauseButton.setText(PAUSE_TEXT);
            pauseButton.setTextFill(Color.GREEN);
        }
        if (stepping) {
            stepButton.setTextFill(Color.YELLOW);
        } else {
            stepButton.setTextFill(Color.GREEN);
        }
    }
    private static String format(final double d) {
        if (d>=0) {
            return "+" + numberFormatLonger.format(d);
        } else {
            return numberFormatLonger.format(d);
        }
    }
    private void addHelpLabelForLayers() {
        helpLabelForLayers.setTranslateX(-165);
        helpLabelForLayers.setTranslateY(HEIGHT/2);
        helpLabelForLayers.setRotationAxis(Rotate.Z_AXIS);
        helpLabelForLayers.setRotate(-90);
        helpLabelForLayers.setFont(new Font(15));
        helpLabelForLayers.setTextFill(Color.ANTIQUEWHITE);
        shapesGroup.getChildren().add(helpLabelForLayers);
    }

    private void animate() {
        final AnimationTimer timer = new AnimationTimer() {
            @Override
            public void handle(long nowInNanoSeconds) {
                doPauseLogicInUIThread();
                if (texts != null && !texts.isEmpty()) {
                    for (Text text : texts) {
                        shapesGroup.getChildren().add(text);
                    }
                    texts = null;
                }
                gradientFactor = Math.pow(2, slider.getValue());
                for (GradientParamShape input : allGradientParamShapes) {
                    input.applyUpdatesToShapesAndRefresh();
                }
                shapesGroup.requestLayout();
                if (captureScreenImages && nowInNanoSeconds - startTime > FIVE_SECOND_IN_NANO_SECONDS && frameCount < FRAME_COUNT_TO_COLLECT) {
                    if (frameCount == 0) {
                        initializeCapture();
                        System.out.println("Starting capture");
                    }
                    capture();
                    if (frameCount == 100) {
                        System.out.println("Ending capture");
                    }
                }
                if (selectedGradientParamShape != null) {
                    String g = format(selectedGradientParamShape.getGradient());
                    String p = format(selectedGradientParamShape.getParam());
                    textForSelectedGradientParam.setText("param = " + p + ", gradient = " + g);
                }
            }
        };
        timer.start();
    }

    @Override
    public void start(Stage stage) throws Exception {
        stage.setOnCloseRequest(r -> System.exit(0));
        stage.setTitle("Gradient & parameters visualization for a sample of neurons. " +
            "The mouse & arrow keys navigate in 3D. Blue is - gradient; red is +; slider adjusts colors." +
            " Large radius is positive param; small radius is negative.");
        stage.setMinWidth(850);
        stage.setMinHeight(600);
        buildCamera();
        buildSlider();
        buildPauseButton();
        buildStepButton();
        addHelpLabelForLayers();
        createTextForSelectedGradientParam();
        root.setDepthTest(DepthTest.ENABLE);
        root.getChildren().add(shapesGroup);
        shapesGroup.setTranslateZ(100);
        Scene scene = new Scene(root, WIDTH, HEIGHT, true);
        scene.setCamera(camera);
        scene.setFill(Color.DIMGREY);
        handleMouse(scene);
        handleKeyEvents(scene);
        stage.setScene(scene);
        staticInstance = this;
        stage.show();
        animate();
    }
    //--------------------------------------------
}
