package org.deeplearning4j.examples.convolution.mnist;

import javafx.application.Application;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.input.KeyCode;
import javafx.scene.input.MouseButton;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.StrokeLineCap;
import javafx.stage.Stage;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Test UI for MNIST classifier.
 * Run the MnistClassifier first to build the model.
 *
 * @author jesuino
 * @author fvaleri
 */
@SuppressWarnings("restriction")
public class MnistClassifierUI extends Application {

  private static final String basePath = System.getProperty("java.io.tmpdir") + "/mnist";
  private final int canvasWidth = 150;
  private final int canvasHeight = 150;
  private MultiLayerNetwork net; // trained model

  public MnistClassifierUI() throws IOException {
    File model = new File(basePath + "/minist-model.zip");
    if (!model.exists())
      throw new IOException("Can't find the model");
    net = MultiLayerNetwork.load(model, true);
  }

  public static void main(String[] args) throws Exception {
    launch();
  }

  @Override
  public void start(Stage stage) throws Exception {
    Canvas canvas = new Canvas(canvasWidth, canvasHeight);
    GraphicsContext ctx = canvas.getGraphicsContext2D();

    ImageView imgView = new ImageView();
    imgView.setFitHeight(100);
    imgView.setFitWidth(100);
    ctx.setLineWidth(10);
    ctx.setLineCap(StrokeLineCap.SQUARE);
    Label lblResult = new Label();

    HBox hbBottom = new HBox(10, imgView, lblResult);
    hbBottom.setAlignment(Pos.CENTER);
    VBox root = new VBox(5, canvas, hbBottom);
    root.setAlignment(Pos.CENTER);

    Scene scene = new Scene(root, 520, 300);
    stage.setScene(scene);
    stage.setTitle("Draw a digit and hit enter (right-click to clear)");
    stage.setResizable(false);
    stage.show();

    canvas.setOnMousePressed(e -> {
      ctx.setStroke(Color.WHITE);
      ctx.beginPath();
      ctx.moveTo(e.getX(), e.getY());
      ctx.stroke();
    });
    canvas.setOnMouseDragged(e -> {
      ctx.setStroke(Color.WHITE);
      ctx.lineTo(e.getX(), e.getY());
      ctx.stroke();
    });
    canvas.setOnMouseClicked(e -> {
      if (e.getButton() == MouseButton.SECONDARY) {
        clear(ctx);
      }
    });
    canvas.setOnKeyReleased(e -> {
      if (e.getCode() == KeyCode.ENTER) {
        BufferedImage scaledImg = getScaledImage(canvas);
        imgView.setImage(SwingFXUtils.toFXImage(scaledImg, null));
        try {
          predictImage(scaledImg, lblResult);
        } catch (Exception e1) {
          e1.printStackTrace();
        }
      }
    });
    clear(ctx);
    canvas.requestFocus();
  }

  private void clear(GraphicsContext ctx) {
    ctx.setFill(Color.BLACK);
    ctx.fillRect(0, 0, 300, 300);
  }

  private BufferedImage getScaledImage(Canvas canvas) {
    WritableImage writableImage = new WritableImage(canvasWidth, canvasHeight);
    canvas.snapshot(null, writableImage);
    Image tmp = SwingFXUtils.fromFXImage(writableImage, null).getScaledInstance(28, 28, Image.SCALE_SMOOTH);
    BufferedImage scaledImg = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
    Graphics graphics = scaledImg.getGraphics();
    graphics.drawImage(tmp, 0, 0, null);
    graphics.dispose();
    return scaledImg;
  }

  private void predictImage(BufferedImage img, Label lbl) throws IOException {
    NativeImageLoader loader = new NativeImageLoader(28, 28, 1, true);
    INDArray image = loader.asRowVector(img);
    ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
    scaler.transform(image);
    INDArray output = net.output(image);
    lbl.setText("Prediction: " + net.predict(image)[0] + "\n " + output);
  }

}
