package org.deeplearning4j.examples.dataexamples;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Random;

/**
 * This code example is featured in this youtube video
 * https://www.youtube.com/watch?v=zrTSs715Ylo
 *
 * This differs slightly from the Video Example,
 * The Video example had the data already downloaded
 * This example includes code that downloads the data
 *
 * Data is downloaded from
 * wget http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
 * followed by tar xzvf mnist_png.tar.gz
 *
 * This examples builds on the MnistImagePipelineExample
 * by Loading the previously saved Neural Net
 */
public class MnistImagePipelineExampleLoad {
    private static Logger log = LoggerFactory.getLogger(MnistImagePipelineExampleLoad.class);

    /** Data URL for downloading */
    public static final String DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

    /** Location to save and extract the training/testing data */
    public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_Mnist/");

    public static void main(String[] args) throws Exception {
        // image information
        // 28 * 28 grayscale
        // grayscale implies single channel
        int height = 28;
        int width = 28;
        int channels = 1;
        int rngseed = 123;
        Random randNumGen = new Random(rngseed);
        int batchSize = 128;
        int outputNum = 10;

        /*
        This class downloadData() downloads the data
        stores the data in java's tmpdir 15MB download compressed
        It will take 158MB of space when uncompressed
        The data can be downloaded manually here
        http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
        */
        MnistImagePipelineExample.downloadData();

        // Define the File Paths
        File trainData = new File(DATA_PATH + "/mnist_png/training");
        File testData = new File(DATA_PATH + "/mnist_png/testing");

        // Define the FileSplit(PATH, ALLOWED FORMATS,random)
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS,randNumGen);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS,randNumGen);

        // Extract the parent path as the image label
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);

        // Initialize the record reader
        // add a listener, to extract the name
        recordReader.initialize(train);
        //recordReader.setListeners(new LogRecordListener());

        // DataSet Iterator
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);

        // Scale pixel values to 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        // Build Our Neural Network
        log.info("******LOAD TRAINED MODEL******");
        // Where the saved model would be if
        // MnistImagePipelineSave has been run
        File locationToSave = new File("trained_mnist_model.zip");

        if (locationToSave.exists()) {
            log.info("\n######Saved Model Found######\n");
        } else {
            log.info("\n\n#######File not found!#######");
            log.info("This example depends on running ");
            log.info("MnistImagePipelineExampleSave");
            log.info("Run that Example First");
            log.info("#############################\n\n");
            System.exit(0);
        }

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        model.getLabels();

        // Test the Loaded Model with the test data
        recordReader.initialize(test);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);

        // Create Eval object with 10 possible classes
        Evaluation eval = new Evaluation(outputNum);

        while (testIter.hasNext()) {
            DataSet next = testIter.next();
            INDArray output = model.output(next.getFeatures());
            eval.eval(next.getLabels(),output);
        }

        log.info(eval.stats());
    }

}
