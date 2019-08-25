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

package com.deeplearning4java.activity;

import android.graphics.Bitmap;
import android.media.Image;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;

import com.lavajaw.deeplearning4java.model.Dl4jModel;

import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.BoxImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.primitives.Pair;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static com.lavajaw.deeplearning4java.utils.Utils.makeBitmapFromMat;
import static com.lavajaw.deeplearning4java.utils.Utils.makeMatFromImage;

public class BaseActivity extends AppCompatActivity {

    public static final int width = 640;
    public static final int height = 480;

    private static MultiLayerNetwork multiLayerNetwork;
    private static ComputationGraph computationGraph;
    private static NativeImageLoader loader;
    private static DataNormalization scalar;

    public void loadDL4JModel(String path, LoadModelListener listener) {
        loader = new NativeImageLoader(111, 111, 1);
        scalar = new ImagePreProcessingScaler(0, 1);
        HandlerThread peopleDetectionThread = new HandlerThread("Load model");
        peopleDetectionThread.start();
        Handler handlerPeopleDetection = new Handler(peopleDetectionThread.getLooper());
        handlerPeopleDetection.post(new Runnable() {
            @Override
            public void run() {
                multiLayerNetwork = null;
                computationGraph = null;
                try {
                    Log.i(Dl4jModel.class.getSimpleName(), "load model");
                    multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(path);
                    listener.loadFinished(multiLayerNetwork);
                } catch (RuntimeException e) {
                    try {
                        computationGraph = ModelSerializer.restoreComputationGraph(path);
                        listener.loadFinished(computationGraph);
                    } catch (Exception e1) {
                        listener.loadFailed(e1);
                        e1.printStackTrace();
                    }
                } catch (Exception ex) {
                    listener.loadFailed(ex);
                    ex.printStackTrace();
                }
            }
        });
    }

    public void runImages(Image image, ResultListener resultListener) {
        double[] results;
        if (multiLayerNetwork != null || computationGraph != null) {
            Bitmap bitmap = makeBitmapFromMat(makeMatFromImage(image, width, height));
            INDArray indArray;
            try {
                indArray = loader.asMatrix(bitmap);
            } catch (IOException e) {
                e.printStackTrace();
                return;
            }
            scalar.transform(indArray);
            if (multiLayerNetwork != null) {
                INDArray outputMLN = multiLayerNetwork.output(indArray);
                results = outputMLN.toDoubleVector();
                resultListener.resultValue(results);
            } else if (computationGraph != null) {
                INDArray[] outputCG = computationGraph.output(indArray);
                results = outputCG[0].toDoubleVector();
                resultListener.resultValue(results);
            }
        }
    }

    public NeuralNetwork getModel() {
        if (multiLayerNetwork != null) {
            return multiLayerNetwork;
        }
        return computationGraph;
    }

    public void doImageTransformations() {
        ImageTransform initialResize = new ResizeImageTransform(1280, 960);
        ImageTransform initialBox = new BoxImageTransform(960, 960);
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(initialResize, 1.0),
                new Pair<>(initialBox, 1.0)
        );
    }

    public interface LoadModelListener {
        void loadFinished(MultiLayerNetwork multiLayerNetwork);

        void loadFinished(ComputationGraph computationGraph);

        void loadFailed(Exception e);
    }

    public interface ResultListener {
        void resultValue(double[] results);
    }
}
