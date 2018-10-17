package org.deeplearning4j.tinyimagenet;

import com.beust.jcommander.Parameter;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.deeplearning4j.datasets.fetchers.TinyImageNetFetcher;
import org.deeplearning4j.patent.utils.JCommanderUtils;
import org.deeplearning4j.spark.util.SparkDataUtils;
import org.deeplearning4j.spark.util.SparkUtils;

import java.io.File;

//import org.deeplearning4j.spark.util.SparkDataUtils;

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

        JavaSparkContext sc = new JavaSparkContext();

        //List files:
        JavaRDD<String> filePathsTrain = SparkUtils.listPaths(sc, sourceDir + "/train");
        SparkDataUtils.createFileBatchesSpark(filePathsTrain, saveDir, batchSize, sc);

        JavaRDD<String> filePathsTest = SparkUtils.listPaths(sc, sourceDir + "/test");
        SparkDataUtils.createFileBatchesSpark(filePathsTest, saveDir, batchSize, sc);


        System.out.println("----- Data Preprocessing Complete -----");
    }

}
