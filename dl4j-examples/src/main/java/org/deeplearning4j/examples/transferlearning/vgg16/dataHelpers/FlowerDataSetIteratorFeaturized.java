package org.deeplearning4j.examples.transferlearning.vgg16.dataHelpers;

import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

/**
 * @author susaneraly on 3/10/17.
 */
public class FlowerDataSetIteratorFeaturized {

    static String featureExtractorLayer = FeaturizedPreSave.featurizeExtractionLayer;

    public static void setup(String featureExtractorLayerArg) {
        featureExtractorLayer = featureExtractorLayerArg;
    }

    public static DataSetIterator trainIterator() {
        DataSetIterator existingTrainingData = new ExistingMiniBatchDataSetIterator(new File("trainFolder"),"flowers-"+featureExtractorLayer+"-train-%d.bin");
        DataSetIterator asyncTrainIter = new AsyncDataSetIterator(existingTrainingData);
        return asyncTrainIter;
    }
    public static DataSetIterator testIterator() {
        DataSetIterator existingTestData = new ExistingMiniBatchDataSetIterator(new File("testFolder"),"flowers-"+featureExtractorLayer+"-test-%d.bin");
        DataSetIterator asyncTestIter = new AsyncDataSetIterator(existingTestData);
        return asyncTestIter;
    }
}
