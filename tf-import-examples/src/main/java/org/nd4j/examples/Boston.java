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

import org.apache.commons.io.FileUtils;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.examples.download.DownloaderUtility;

import java.io.File;

import static java.nio.charset.Charset.defaultCharset;


public class Boston {

    private static SameDiff sd;
    private static double mean;
    private static double std;
    public static String dataLocalPath;

    public static void loadModel(String filepath) throws Exception{
        File file = new File(filepath);
        if (!file.exists()){
            file = new File(dataLocalPath,filepath);
        }

        sd = TFGraphMapper.getInstance().importGraph(file);

        if (sd == null) {
            throw new Exception("Error loading model : " + file);
        }
    }

    private static void loadStats() throws Exception {
        File file = new File(dataLocalPath,"Boston/stats.txt");
        String contents = FileUtils.readFileToString(file,defaultCharset());
        String stats[] = contents.split(",");
        mean = Double.parseDouble(stats[0]);
        std = Double.parseDouble(stats[1]);
    }

    public static INDArray getSampleData() {

        /*
        * The dataset contains 13 different features:

                Per capita crime rate.
                The proportion of residential land zoned for lots over 25,000 square feet.
                The proportion of non-retail business acres per town.
                Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
                Nitric oxides concentration (parts per 10 million).
                The average number of rooms per dwelling.
                The proportion of owner-occupied units built before 1940.
                Weighted distances to five Boston employment centers.
                Index of accessibility to radial highways.
                Full-value property-tax rate per $10,000.
                Pupil-teacher ratio by town.
                1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
                Percentage lower status of the population.

        * */

        double sampleData[] = new double[]{7.8750e-02, 4.5000e+01, 3.4400e+00, 0.0000e+00, 4.3700e-01, 6.7820e+00,
        4.1100e+01, 3.7886e+00, 5.0000e+00, 3.9800e+02, 1.5200e+01, 3.9387e+02, 6.6800e+00};

        INDArray arr = Nd4j.create(sampleData).reshape(13);

        // Normalize
        arr.subi(mean);
        arr.divi(std);
        return arr;
    }


    public static double predict(INDArray arr){
        arr = Nd4j.expandDims(arr, 0);  // add batch dimension
        sd.associateArrayWithVariable(arr, sd.variables().get(0));
        INDArray outArr = sd.execAndEndResult();
        double pred = outArr.getDouble(0);
        return pred;
    }

    public static void main(String[] args) throws Exception{
        dataLocalPath = DownloaderUtility.TFIMPORTEXAMPLES.Download();
        loadModel("Boston/boston.pb");
        loadStats();
        INDArray sampleData = getSampleData();
        double prediction = predict(sampleData); // in $1000
        System.out.println(String.format("Predicted price = $%d",prediction * 1000));
    }
}
