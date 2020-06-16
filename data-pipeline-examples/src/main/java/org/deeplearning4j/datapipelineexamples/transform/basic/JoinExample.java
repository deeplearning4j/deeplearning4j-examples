/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

package org.deeplearning4j.datapipelineexamples.transform.basic;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.join.Join;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datapipelineexamples.utils.DownloaderUtility;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.joda.time.DateTimeZone;

import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 * This example shows how to perform joins in DataVec
 * Joins are analogous to join operations in databases/SQL: data from multiple sources are combined together, based
 * on some common key that appears in both sources.
 *
 * This example loads data from two CSV files. It is some mock customer data
 *
 * @author Alex Black
 */
public class JoinExample {

    public static String dataLocalPath;


    public static void main(String[] args) throws Exception {

        dataLocalPath = DownloaderUtility.JOINEXAMPLE.Download();
        String customerInfoPath = new File(dataLocalPath,"CustomerInfo.csv").getAbsolutePath();
        String purchaseInfoPath = new File(dataLocalPath,"CustomerPurchases.csv").getAbsolutePath();

        //First: Let's define our two data sets, and their schemas

        Schema customerInfoSchema = new Schema.Builder()
            .addColumnLong("customerID")
            .addColumnString("customerName")
            .addColumnCategorical("customerCountry", Arrays.asList("USA","France","Japan","UK"))
            .build();

        Schema customerPurchasesSchema = new Schema.Builder()
            .addColumnLong("customerID")
            .addColumnTime("purchaseTimestamp", DateTimeZone.UTC)
            .addColumnLong("productID")
            .addColumnInteger("purchaseQty")
            .addColumnDouble("unitPriceUSD")
            .build();



        //Spark Setup
        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("DataVec Join Example");
        JavaSparkContext sc = new JavaSparkContext(conf);

        //Load the data:
        RecordReader rr = new CSVRecordReader();
        JavaRDD<List<Writable>> customerInfo = sc.textFile(customerInfoPath).map(new StringToWritablesFunction(rr));
        JavaRDD<List<Writable>> purchaseInfo = sc.textFile(purchaseInfoPath).map(new StringToWritablesFunction(rr));
            //Collect data for later printing
        List<List<Writable>> customerInfoList = customerInfo.collect();
        List<List<Writable>> purchaseInfoList = purchaseInfo.collect();

        //Let's join these two data sets together, by customer ID
        Join join = new Join.Builder(Join.JoinType.Inner)
            .setJoinColumns("customerID")
            .setSchemas(customerInfoSchema, customerPurchasesSchema)
            .build();

        JavaRDD<List<Writable>> joinedData = SparkTransformExecutor.executeJoin(join, customerInfo, purchaseInfo);
        List<List<Writable>> joinedDataList = joinedData.collect();

        //Stop spark, and wait a second for it to stop logging to console
        sc.stop();
        Thread.sleep(2000);


        //Print the original data
        System.out.println("\n\n----- Customer Information -----");
        System.out.println("Source file: " + customerInfoPath);
        System.out.println(customerInfoSchema);
        System.out.println("Customer Information Data:");
        for(List<Writable> line : customerInfoList){
            System.out.println(line);
        }


        System.out.println("\n\n----- Purchase Information -----");
        System.out.println("Source file: " + purchaseInfoPath);
        System.out.println(customerPurchasesSchema);
        System.out.println("Purchase Information Data:");
        for(List<Writable> line : purchaseInfoList){
            System.out.println(line);
        }


        //Print the joined data
        System.out.println("\n\n----- Joined Data -----");
        System.out.println(join.getOutputSchema());
        System.out.println("Joined Data:");
        for(List<Writable> line : joinedDataList){
            System.out.println(line);
        }
    }

}
