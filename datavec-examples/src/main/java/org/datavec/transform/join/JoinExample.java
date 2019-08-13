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

package org.datavec.transform.join;

import org.apache.commons.io.FilenameUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.join.Join;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.joda.time.DateTimeZone;
import org.nd4j.resources.Downloader;

import java.io.File;
import java.io.IOException;
import java.net.URL;
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

    public static final String DATA_LOCAL_PATH;

    static {
        final String DATA_URL = "https://deeplearning4jblob.blob.core.windows.net/dl4j-examples/datavec-examples/JoinExample.zip";
        final String MD5 = "cbd6232cf1463d68ff24807d5dd8b530";
        final int DOWNLOAD_RETRIES = 10;
        final String DOWNLOAD_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "JoinExample.zip");
        final String EXTRACT_DIR = FilenameUtils.concat(System.getProperty("user.home"), "dl4j-examples-data/datavec-examples");
        DATA_LOCAL_PATH = FilenameUtils.concat(EXTRACT_DIR,"JoinExample");
        if (!new File(DATA_LOCAL_PATH).exists()) {
            try {
                System.out.println("_______________________________________________________________________");
                System.out.println("Downloading data (1KB) to \n\t" + DATA_LOCAL_PATH);
                System.out.println("_______________________________________________________________________");
                Downloader.downloadAndExtract("files",
                    new URL(DATA_URL),
                    new File(DOWNLOAD_PATH),
                    new File(EXTRACT_DIR),
                    MD5,
                    DOWNLOAD_RETRIES);
            }
            catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            System.out.println("_______________________________________________________________________");
            System.out.println("Example data present in \n\t" + DATA_LOCAL_PATH);
            System.out.println("_______________________________________________________________________");
        }

    }

    public static void main(String[] args) throws Exception {

        String customerInfoPath = new File(DATA_LOCAL_PATH,"CustomerInfo.csv").getAbsolutePath();
        String purchaseInfoPath = new File(DATA_LOCAL_PATH,"CustomerPurchases.csv").getAbsolutePath();

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
