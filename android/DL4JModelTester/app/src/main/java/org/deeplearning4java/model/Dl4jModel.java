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

package com.deeplearning4java.model;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.widget.TextView;

import com.lavajaw.deeplearning4java.commons.PrefManager;

import org.bytedeco.javacv.FrameFilter;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.BoxImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.common.primitives.Pair;
import org.opencv.core.Mat;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.lavajaw.deeplearning4java.utils.Utils.arrayMaximum;
import static com.lavajaw.deeplearning4java.utils.Utils.getIndexOfLargestValue;
import static java.io.File.separator;
import static org.opencv.android.Utils.matToBitmap;

public class Dl4jModel {

    private static volatile Dl4jModel instance;

    private MultiLayerNetwork multiLayerNetwork;
    private ComputationGraph computationGraph;
    private NativeImageLoader loader;
    private DataNormalization scalar;

    public static synchronized Dl4jModel getInstance() {
        if (instance == null) {
            synchronized (Dl4jModel.class) {
                if (instance == null) {
                    instance = new Dl4jModel();
                }
            }
        }
        return instance;
    }

    private Dl4jModel() {
        int height = 100;
        int width = 100;
        int channels = 3;
        ImageTransform initialResize = new ResizeImageTransform(1280, 960);
        ImageTransform initialBox = new BoxImageTransform(960, 960);
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(initialResize, 1.0),
                new Pair<>(initialBox, 1.0)
        );
        loader = new NativeImageLoader(height, width, channels);
        scalar = new ImagePreProcessingScaler(0, 1);
    }

    public void onMatReady(Mat mat, Activity activity, final TextView textView) {
        double[] results;
        if (multiLayerNetwork != null || computationGraph != null) {
            Bitmap bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
            matToBitmap(mat, bitmap);
            INDArray image;
            try {
                image = loader.asMatrix(bitmap);
            } catch (IOException e) {
                e.printStackTrace();
                return;
            }
            scalar.transform(image);
            if (multiLayerNetwork != null) {
                INDArray outputMLN = multiLayerNetwork.output(image);
                results = new double[]{outputMLN.getDouble(0, 0), outputMLN.getDouble(0, 1), outputMLN.getDouble(0, 2), outputMLN.getDouble(0, 3)};
                writeOutput(activity, textView, results);
            } else if (computationGraph != null) {
                INDArray[] outputCG = computationGraph.output(image);
                results = new double[]{outputCG[0].getDouble(0, 0), outputCG[1].getDouble(0, 1), outputCG[2].getDouble(0, 2), outputCG[3].getDouble(0, 3)};
                writeOutput(activity, textView, results);
            }
        }
    }

    private static void writeOutput(Activity activity, TextView textView, double[] results){
        final DecimalFormat df2 = new DecimalFormat(".##");
        if (results != null) {
            activity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    textView.setText(String.format("Result %s %s", String.valueOf(df2.format(arrayMaximum(results))), String.valueOf(getIndexOfLargestValue(results))));
                }
            });
            Log.i(Dl4jModel.class.getSimpleName(), String.format("Result %s %s", String.valueOf(df2.format(arrayMaximum(results))), String.valueOf(getIndexOfLargestValue(results))));
        }
    }
}
