package org.deeplearning4j.examples.modelimport;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import java.util.Random;

/**
 * Created by susaneraly on 2/28/17.
 */
@Slf4j
public class TransferLearningExamplePreSave {

    protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    protected static final int numClasses = 5;

    protected static final int batchSize = 15;
    protected static final long seed = 12345;
    public static final Random rng = new Random(seed);

    protected static int height = 224;
    protected static int width = 224;
    protected static int channels = 3;

    public static void main(String [] args) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {

        /*
            Step I: Print a summary of the imported model. Note layer names, nIn, nOuts etc
         */
        TrainedModelHelper modelImportHelper = new TrainedModelHelper(TrainedModels.VGG16);
        ComputationGraph vgg16 = modelImportHelper.loadModel();
        log.info(vgg16.summary());

        /*
            Set up dataset with the train and test split
            Set up the training dataset iterator
         */
        File parentDir = new File("/Users/susaneraly/flower_photos");
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        recordReader.initialize(trainData);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        trainIter.setPreProcessor(TrainedModels.VGG16.getPreProcessor());

        /*
            We want to featurize our inputs and save to disk so we can iterate quickly with only the unfrozen layers.
            We set up a transfer learning helper instance
         */
        TransferLearningHelper transferLearningHelper = new TransferLearningHelper(vgg16,"block5_pool");
        log.info(vgg16.summary());


        /*
            We use the transfer learning helper and save our minibatches to disk
         */
        File trainFolder = new File("trainFolder");
        trainFolder.mkdirs();
        int trainDataSaved = 0;
        while(trainIter.hasNext()) {
            DataSet currentFeaturized = transferLearningHelper.featurize(trainIter.next());
            currentFeaturized.save(new File(trainFolder,"flowers-train-" + trainDataSaved + ".bin"));
            log.info("Saved train dataset #"+trainDataSaved);
            trainDataSaved++;
        }

        /*
            Repeat the same for the test data
         */
        recordReader.reset();
        recordReader.initialize(testData);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        testIter.setPreProcessor(TrainedModels.VGG16.getPreProcessor());
        File testFolder = new File("testFolder");
        testFolder.mkdirs();
        int testDataSaved = 0;
        while(testIter.hasNext()) {
            DataSet currentFeaturized = transferLearningHelper.featurize(testIter.next());
            currentFeaturized.save(new File(testFolder,"flowers-test-" + testDataSaved + ".bin"));
            log.info("Saved test dataset #"+testDataSaved);
            testDataSaved++;
        }


        log.info("Finished pre saving featurized test and train data");

    }
}
