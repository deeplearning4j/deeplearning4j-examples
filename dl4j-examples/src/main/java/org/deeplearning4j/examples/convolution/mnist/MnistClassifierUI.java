/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.convolution.mnist;

import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javafx.scene.text.Font;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Test UI for MNIST classifier. User can painting digit by using mouse and predict value using a trained model. <br>
 * Run the {@link MnistClassifier} first to build the model.
 *
 * @author jesuino
 * @author fvaleri
 * @author dariuszzbyrad
 */
public class MnistClassifierUI extends Application {
    private static final Logger LOGGER = LoggerFactory.getLogger(MnistClassifier.class);
    private static final String BASE_PATH = System.getProperty("java.io.tmpdir") + "/mnist";

    private static final int CANVAS_WIDTH = 150;
    private static final int CANVAS_HEIGHT = 150;
    private static final int IMAGE_WIDTH = 28;
    private static final int IMAGE_HEIGHT = 28;

    private MultiLayerNetwork net; // trained model

    public MnistClassifierUI() throws IOException {
        File model = new File(BASE_PATH + "/minist-model.zip");
        if (!model.exists())
            throw new IOException("Can't find the model");

        net = ModelSerializer.restoreMultiLayerNetwork(model);
    }

    public static void main(String[] args) {
        launch();
    }

    @Override
    public void start(Stage stage) {
        Canvas canvas = new Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
        GraphicsContext ctx = canvas.getGraphicsContext2D();

        ImageView imgView = new ImageView();
        imgView.setFitHeight(100);
        imgView.setFitWidth(100);
        ctx.setLineWidth(10);
        ctx.setLineCap(StrokeLineCap.SQUARE);

        Label lblResult = new Label();
        lblResult.setMaxWidth(360);
        lblResult.setWrapText(true);
        lblResult.setFont(new Font("Arial", 20));

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
        WritableImage writableImage = new WritableImage(CANVAS_WIDTH, CANVAS_HEIGHT);
        canvas.snapshot(null, writableImage);
        Image tmp = SwingFXUtils.fromFXImage(writableImage, null).getScaledInstance(IMAGE_WIDTH, IMAGE_HEIGHT, Image.SCALE_SMOOTH);
        BufferedImage scaledImg = new BufferedImage(IMAGE_WIDTH, IMAGE_HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
        Graphics graphics = scaledImg.getGraphics();
        graphics.drawImage(tmp, 0, 0, null);
        graphics.dispose();
        return scaledImg;
    }

    private void predictImage(BufferedImage img, Label lbl) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(IMAGE_HEIGHT, IMAGE_WIDTH, 1, true);
        INDArray image = loader.asRowVector(img);
        ImagePreProcessingScaler imageScaler = new ImagePreProcessingScaler();
        imageScaler.transform(image);

        String output = generateOutputWithResult(net, image);
        lbl.setText(output);
    }

    private String generateOutputWithResult(MultiLayerNetwork net, INDArray image) {
        INDArray output = net.output(image);
        int predictedDigit = net.predict(image)[0];
        double probability = output.getDouble(predictedDigit) * 100;
        LOGGER.info("Prediction: {}", output);
        return String.format("Prediction: %s with probability: %.1f%%", predictedDigit, probability);
    }
}
