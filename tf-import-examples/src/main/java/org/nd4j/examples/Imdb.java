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
import org.deeplearning4j.examples.download.DownloaderUtility;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

/**
 * Tensorflow import for IMDB.
 *
 * @author Fariz Rahman
 */
public class Imdb {

    private static SameDiff sd;
    private static final int maxlen = 256;
    private static Map<String, Integer> wordIndex;
    public static String dataLocalPath;

    private static void loadWordIndex() throws Exception {
        wordIndex = new HashMap<>();
        File file = new File(dataLocalPath,"Imdb/word_index.txt");
        String content = FileUtils.readFileToString(file);
        String[] lines = content.split("\n");
        for(int i=0; i < lines.length - 1; i++){
            String line = lines[i];
            String[] kv = line.split(",");
            String k = kv[0];
            int v = Integer.parseInt(kv[1]);
            wordIndex.put(k, v);

        }
    }

    private static INDArray encodeText(String text) throws Exception {
        String[] words = text.split(" ");
        double arr[] = new double[maxlen];
        int pads = 256 - words.length;
        for(int i = 0; i<pads; i++){
            arr[i] = (double)wordIndex.get("<PAD>");
        }
        for(int i=0; i<words.length; i++){
            if(wordIndex.containsKey(words[i]) ){
                arr[pads + i] = (double)wordIndex.get(words[i]);
            }
            else {
                arr[pads + i] = (double)wordIndex.get("<UNK>");
            }
        }
        INDArray indArr = Nd4j.create(arr).reshape(256);
        return indArr;
    }

    public static void loadModel(String filepath) throws Exception{
        File file = new File(filepath);
        if (!file.exists()){
            file = new File(filepath);
        }
        sd = TFGraphMapper.getInstance().importGraph(file);
        if (sd == null){
            throw new Exception("Error loading model : " + file);
        }
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
        loadModel("Imdb/imdb.pb");
        loadWordIndex();
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        while(true){
            System.out.println("Enter review : ");
            String review = reader.readLine();
            INDArray arr = encodeText(review);
            double prediction = predict(arr);
            System.out.println(String.format("Sentiment prediction : %d", prediction));
        }

    }
}
