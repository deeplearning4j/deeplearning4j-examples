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

package org.nd4j.examples;

import org.deeplearning4j.examples.download.DownloaderUtility;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

/**
 * Shows tensorflow import using mnist.
 * For the trained graph, please look at the python files under
 * Mnist/
 *
 * mnist_tf.py was used to train the graph
 *
 * @author Fariz Rahman
 */
public class Mnist {
    private static SameDiff sd;
    public static String dataLocalPath;

    public static void loadModel(String filepath) throws Exception{
        File file = new File(filepath);
        if (!file.exists()){
            file = new File(filepath);
        }

        sd = TFGraphMapper.importGraph(file);

        if (sd == null) {
            throw new Exception("Error loading model : " + file);
        }
    }

    public static INDArray predict(INDArray arr){
        INDArray batchedArr = Nd4j.expandDims(arr, 0);
        sd.associateArrayWithVariable(batchedArr, sd.variables().get(0));
        INDArray out = sd.outputSingle(null, "");
        return Nd4j.squeeze(out, 0);
    }

    public static INDArray predictBatch(INDArray arr){
        sd.associateArrayWithVariable(arr, sd.variables().get(0));
        return sd.outputSingle(null, "");
    }

    public static INDArray predict (String filepath) throws IOException{
        File file = new File(filepath);
        if (!file.exists()){
            file = new File(filepath);
        }

        BufferedImage img = ImageIO.read(file);
        double data[] = new double[28 * 28];
        for(int i = 0; i < 28; i++){
            for(int j = 0; j < 28; j++){
                Color color = new Color(img.getRGB(i, j));
                int r = color.getRed();
                int g = color.getGreen();
                int b = color.getBlue();
                double greyScale = (r + g + b) / 3;
                greyScale /= 255.;
                data[i * 28 + j] = greyScale;
            }
        }

        INDArray arr = Nd4j.create(data).reshape(1, 28, 28);
        arr = Nd4j.pile(arr, arr);
        sd.associateArrayWithVariable(arr, sd.variables().get(0));
        INDArray output = sd.outputSingle(null, "").get(NDArrayIndex.point(0));
        System.out.println(Arrays.toString(output.reshape(10).toDoubleVector()));
        return output;

    }

    public static int predictionToLabel(INDArray prediction){
        return Nd4j.argMax(prediction.reshape(10)).getInt(0);
    }


    public static void main(String[] args) throws Exception{
        dataLocalPath = DownloaderUtility.TFIMPORTEXAMPLES.Download();
        loadModel("Mnist/mnist.pb");
        for(int i = 1; i < 11; i++){
            String file = "Mnist/images/img_%d.jpg";
            file = String.format(file, i);
            INDArray prediction = predict(file);
            int label = predictionToLabel(prediction);
            System.out.println(file + "  ===>  " + label);
        }

    }
}
