/*******************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.deeplearning4j.datapipelineexamples.tablesaw;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import org.deeplearning4j.datapipelineexamples.utils.DownloaderUtility;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;
import tech.tablesaw.api.CategoricalColumn;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.io.csv.CsvReadOptions;

import java.io.File;
import java.util.Arrays;
import java.util.stream.Collectors;

/**
 * This example uses the table saw library to prepare csv data for conversion to a neural network.
 * If you would like more information on tablesaw, please look at the table saw quickstart:
 * https://jtablesaw.github.io/tablesaw/gettingstarted
 *
 * This example leverages tablesaw to load a csv and convert it to a dataset object.
 *
 * @author Adam Gibson
 */
public class TablesawCSVExample {

    public static void main(String...args) throws Exception {
        //download the data
        String directory = DownloaderUtility.IRISDATA.Download();
        //note our downloaded csv has no headers, so we want auto generated column names
        CsvReadOptions csvReadOptions = CsvReadOptions
                .builder(new File(directory, "iris.txt")).header(false).build();
        Table table = Table.read().csv(csvReadOptions);
        System.out.println(table.columnNames());
        //Convert the data without the label column to get just the raw input data out.
        Table justLabel = Table.create(table.column(4));
        Table withoutLabel = table.removeColumns(table.column(4));
        //convert the data to a double array filtering the column without
        double[][] data = Arrays.stream(withoutLabel.columnArray())
                .map(column -> (DoubleColumn) column)
                .map(input -> input.asList())
                .map(input -> Doubles.toArray(input))
                .collect(Collectors.toList())
                .toArray(new double[table.columnNames().size()][]);

        //create the data from the array and print the data
        INDArray arr = Nd4j.create(data);
        System.out.println(arr.toStringFull());

        //print the categories
        CategoricalColumn<?> objects = justLabel.categoricalColumn(0);
        System.out.println("List " + objects.asList());
        System.out.println(objects.countByCategory());


        //create an ndarray of the outcomes converted to categorical 0 1 labels
        int[] outcomes = Ints.toArray(justLabel.longColumn(0).asList());
        INDArray labels = FeatureUtil.toOutcomeMatrix(outcomes, 3);


        //create a dataset object containing the input and the labels
        DataSet dataSet = new DataSet(arr,labels);



    }


}
