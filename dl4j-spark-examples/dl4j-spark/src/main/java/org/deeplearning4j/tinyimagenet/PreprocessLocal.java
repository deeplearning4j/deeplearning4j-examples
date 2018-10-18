package org.deeplearning4j.tinyimagenet;

import com.beust.jcommander.Parameter;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.deeplearning4j.datasets.fetchers.TinyImageNetFetcher;
import org.deeplearning4j.patent.utils.JCommanderUtils;
import org.deeplearning4j.spark.util.SparkDataUtils;

import java.io.File;

public class PreprocessLocal {

    @Parameter(names = {"--localSaveDir"}, description = "Directory to save the preprocessed data files on your local drive", required = true)
    private String localSaveDir = null;

    @Parameter(names = {"--batchSize"}, description = "Batch size for saving the data", required = false)
    private int batchSize = 32;

    public static void main(String[] args) throws Exception {
        new PreprocessLocal().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        JCommanderUtils.parseArgs(this, args);

        //First, ensure we have the required data:
        TinyImageNetFetcher f = new TinyImageNetFetcher();
        f.downloadAndExtract();

        //Preprocess the training set
        File baseDirTrain = DL4JResources.getDirectory(ResourceType.DATASET, f.localCacheName() + "/train");
        File saveDirTrain = new File(localSaveDir, "train");
        if(!saveDirTrain.exists())
            saveDirTrain.mkdirs();
        SparkDataUtils.createFileBatchesLocal(baseDirTrain, NativeImageLoader.ALLOWED_FORMATS, true, saveDirTrain, batchSize);

        //Preprocess the test set
        File baseDirTest = DL4JResources.getDirectory(ResourceType.DATASET, f.localCacheName() + "/test");
        File saveDirTest = new File(localSaveDir, "test");
        if(!saveDirTest.exists())
            saveDirTest.mkdirs();
        SparkDataUtils.createFileBatchesLocal(baseDirTest, NativeImageLoader.ALLOWED_FORMATS, true, saveDirTest, batchSize);

        System.out.println("----- Data Preprocessing Complete -----");
    }

}
