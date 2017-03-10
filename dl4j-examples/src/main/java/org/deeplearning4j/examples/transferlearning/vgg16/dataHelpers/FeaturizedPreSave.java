package org.deeplearning4j.examples.transferlearning.vgg16.dataHelpers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModelHelper;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;

/**
 * @author susaneraly on 2/28/17.
 */
@Slf4j
public class FeaturizedPreSave {

    private static final int trainPerc = 80;
    protected static final int batchSize = 15;
    public static final String featurizeExtractionLayer = "fc2";

    public static void main(String [] args) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {

        TrainedModelHelper modelImportHelper = new TrainedModelHelper(TrainedModels.VGG16);
        ComputationGraph vgg16 = modelImportHelper.loadModel();
        log.info(vgg16.summary());
        TransferLearningHelper transferLearningHelper = new TransferLearningHelper(vgg16, featurizeExtractionLayer);
        log.info(vgg16.summary());

        FlowerDataSetIterator.setup(batchSize,trainPerc);
        DataSetIterator trainIter = FlowerDataSetIterator.trainIterator();
        DataSetIterator testIter = FlowerDataSetIterator.testIterator();

        int trainDataSaved = 0;
        while(trainIter.hasNext()) {
            DataSet currentFeaturized = transferLearningHelper.featurize(trainIter.next());
            saveToDisk(currentFeaturized,trainDataSaved,true);
            trainDataSaved++;
        }

        int testDataSaved = 0;
        while(testIter.hasNext()) {
            DataSet currentFeaturized = transferLearningHelper.featurize(testIter.next());
            saveToDisk(currentFeaturized,testDataSaved,false);
            testDataSaved++;
        }

        log.info("Finished pre saving featurized test and train data");
    }

    public static void saveToDisk(DataSet currentFeaturized, int iterNum, boolean isTrain) {
        File fileFolder = isTrain ? new File("trainFolder"): new File("testFolder");
        if (iterNum == 0) {
            fileFolder.mkdirs();
        }
        String fileName = "flowers-" + featurizeExtractionLayer + "-";
        fileName += isTrain ? "train-" : "test-";
        fileName += iterNum + ".bin";
        currentFeaturized.save(new File(fileFolder,fileName));
        log.info("Saved " + (isTrain?"train ":"test ") + "dataset #"+ iterNum);
    }
}
