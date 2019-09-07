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

package org.datavec.transform.basic;


import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;

import java.io.File;
import java.net.URL;
import java.util.List;

/**
 * Another basic preprocessing and filtering example. We download the raw iris dataset from the internet.
 * The file we downloaded has 2 problems:
 * 1. an empty line at the bottom.
 * 2. the labels are in the form of strings. We want an integer.
 * The output matches the iris.txt file you can find in the resources of the datavec and dl4j examples under ~/dl4j-examples-data/datavec-examples/IrisData
**/
public class BasicTransformExample {

    public static  void main(String[] args) throws Exception {

        String filename = "iris.data";
        URL url = new URL("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data");

        File irisText = new File(filename);
        if (!irisText.exists()){
            FileUtils.copyURLToFile(url, irisText);
        }

        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("DataVec Example");

        JavaSparkContext sc = new JavaSparkContext(conf);
        JavaRDD<String> stringData = sc.textFile(filename);

        //Take out empty lines.
        RecordReader rr = new CSVRecordReader();
        JavaRDD<List<Writable>> parsedInputData = stringData.filter((x) -> !x.isEmpty()).map(new StringToWritablesFunction(rr));

        // Print the original text file.  Not the empty line at the bottom,
        List<String> inputDataCollected = stringData.collect();
        System.out.println("\n\n---- Original Data ----");
        for(String s : inputDataCollected) System.out.println("'" + s + "'");

        //
        JavaRDD<String> processedAsString = parsedInputData.map(new WritablesToStringFunction(","));
        List<String> inputDataParsed = processedAsString.collect();
        System.out.println("\n\n---- Parsed Data ----");
        for(String s : inputDataParsed) System.out.println("'" + s + "'");

        // the String to label conversion. Define schema and transform:
        Schema schema = new Schema.Builder()
            .addColumnsDouble("Sepal length", "Sepal width", "Petal length", "Petal width")
            .addColumnCategorical("Species", "Iris-setosa", "Iris-versicolor", "Iris-virginica")
            .build();

        TransformProcess tp = new TransformProcess.Builder(schema)
            .categoricalToInteger("Species")
            .build();

        // do the transformation.
        JavaRDD<List<Writable>> processedData = SparkTransformExecutor.execute(parsedInputData, tp);

        // This is where we print the final result (which you would save to a text file.
        processedAsString = processedData.map(new WritablesToStringFunction(","));
        inputDataParsed = processedAsString.collect();
        System.out.println("\n\n---- Parsed and filtered data ----");
        for(String s : inputDataParsed) System.out.println(s);

    }
}
