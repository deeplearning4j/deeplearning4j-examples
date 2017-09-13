package org.deeplearning4j.examples.multigpu.vgg16.dataHelpers;

import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;

/**
 * Iterator for featurized data.
 * @author susaneraly on 3/10/17.
 */
public class FlowerDataSetIteratorFeaturized {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FlowerDataSetIteratorFeaturized.class);
    static String featureExtractorLayer = FeaturizedPreSave.featurizeExtractionLayer;

    public static void setup(String featureExtractorLayerArg) {
        featureExtractorLayer = featureExtractorLayerArg;
    }

    public static DataSetIterator trainIterator() throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        runFeaturize();
        DataSetIterator existingTrainingData = new ExistingMiniBatchDataSetIterator(new File("trainFolder"),"flowers-"+featureExtractorLayer+"-train-%d.bin");
        DataSetIterator asyncTrainIter = new AsyncDataSetIterator(existingTrainingData);
        return asyncTrainIter;
    }
    public static DataSetIterator testIterator() {
        DataSetIterator existingTestData = new ExistingMiniBatchDataSetIterator(new File("testFolder"),"flowers-"+featureExtractorLayer+"-test-%d.bin");
        DataSetIterator asyncTestIter = new AsyncDataSetIterator(existingTestData);
        return asyncTestIter;
    }

    private static void runFeaturize() throws InvalidKerasConfigurationException, IOException, UnsupportedKerasConfigurationException {
        File trainDir = new File("trainFolder","flowers-"+featureExtractorLayer+"-train-0.bin");
        if (!trainDir.isFile()) {
            log.info("\n\tFEATURIZED DATA NOT FOUND. \n\t\tRUNNING \"FeaturizedPreSave\" first to do presave of featurized data");
            FeaturizedPreSave.main(null);
        }
    }
}
