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

package org.deeplearning4j.tinyimagenet;

import com.beust.jcommander.Parameter;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.patent.utils.JCommanderUtils;
import org.deeplearning4j.spark.util.SparkDataUtils;
import org.deeplearning4j.spark.util.SparkUtils;

/**
 * This file is for preparing the training data for the tiny imagenet CNN example.
 * Either this class OR PreprocessLocal (but not both) must be run before training can be run on a cluster via TrainSpark.
 *
 * PreprocessSpark requires that the tiny imagenet source image files (.jpg format) available on network storage that
 * Spark can access, such as HDFS, Azure blob storage, S3 etc.
 *
 * To get these image files, you have two options:
 *
 * Option 1: Direct download
 * Step 1: Download https://deeplearning4jblob.blob.core.windows.net/datasets/tinyimagenet_200_dl4j.v1.zip
 * Step 2: Extract files locally
 * Step 3: Copy contents (in their existing train/test subdirectories) to remote storage (for example, using Hadoop FS utils or similar)
 *
 * Option 2: Use TinyImageNetFetcher to download
 * Step 1: Run {@code new TinyImageNetFetcher().downloadAndExtract()} to download the files
 * Step 2: Copy the contents of the following directory to remote storage (for example, using Hadoop FS utils or similar)
 *     Windows:  C:\Users\<username>\.deeplearning4j\data\TINYIMAGENET_200
 *     Linux:    ~/.deeplearning4j/data/TINYIMAGENET_200
 *
 * After completing the steps of option 1 or option 2, then run this script to preprocess the data.
 *
 * @author Alex Black
 */
public class PreprocessSpark {

    @Parameter(names = {"--sourceDir"}, description = "Directory to get source image files", required = true)
    public String sourceDir;

    @Parameter(names = {"--saveDir"}, description = "Directory to save the preprocessed data files on remote storage (for example, HDFS)", required = true)
    private String saveDir;

    @Parameter(names = {"--batchSize"}, description = "Batch size for saving the data", required = false)
    private int batchSize = 32;

    public static void main(String[] args) throws Exception {
        new PreprocessSpark().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        JCommanderUtils.parseArgs(this, args);

        SparkConf conf = new SparkConf();
        conf.setAppName("DL4JTinyImageNetSparkPreproc");
        JavaSparkContext sc = new JavaSparkContext(conf);

        //Create training set
        JavaRDD<String> filePathsTrain = SparkUtils.listPaths(sc, sourceDir + "/train", true, NativeImageLoader.ALLOWED_FORMATS);
        SparkDataUtils.createFileBatchesSpark(filePathsTrain, saveDir, batchSize, sc);

        //Create test set
        JavaRDD<String> filePathsTest = SparkUtils.listPaths(sc, sourceDir + "/test", true, NativeImageLoader.ALLOWED_FORMATS);
        SparkDataUtils.createFileBatchesSpark(filePathsTest, saveDir, batchSize, sc);


        System.out.println("----- Data Preprocessing Complete -----");
    }

}
