package org.deeplearning4j.examples.dataExamples;

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

import java.io.File;
import java.util.Random;

/**
 * /**
 *  This code example is featured in this youtube video
 *
 *  https://www.youtube.com/watch?v=zrTSs715Ylo
 *
 *
 * Instructions
 * You must download the data for this example to work
 * The datafile is a 15M download that uncompresses to a 273MB directory
 * The Data Directory mnist_png will have two child directories training and testing
 * The training and testing directories will have directories 0-9 with
 * 28 * 28 PNG images of handwritten images
 *
 *
 *
 *  You Must Download the data first
 *
 *  Either..
 *  wget http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
 *  followed by tar xzvf mnist_png.tar.gz
 *
 *  OR
 *  git clone https://github.com/myleott/mnist_png.git
 *  cd mnist_png
 *  tar xvf mnist_png.tar.gz
 *
 *  Once you have downloaded the data verify that the
 *  following lines reflect the location of the mnist_png directory
 *
 *  File trainData = new File("/tmp/mnist_png/training");
 *  File testData = new File("/tmp/mnist_png/testing");
 *
 *  This examples builds on the MnistImagePipelineExample
 *  by Loading the previously saved Neural Net
 */
public class MnistImagePipelineExampleLoad {
    private static Logger log = LoggerFactory.getLogger(MnistImagePipelineExampleLoad.class);

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
        int numEpochs = 15;

        // Define the File Paths
        File trainData = new File("/Users/tomhanlon/SkyMind/java/dl4j-examples62/dl4j-examples/src/main/resources/mnist_png/training");
        File testData = new File("/Users/tomhanlon/SkyMind/java/dl4j-examples62/dl4j-examples/src/main/resources/mnist_png/testing");

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
        // Details

        // Where to save model
        File locationToSave = new File("trained_mnist_model.zip");

        // boolean save Updater
        //boolean saveUpdater = false;

        // ModelSerializer needs modelname, saveUpdater, Location

        //ModelSerializer.writeModel(model,locationToSave,saveUpdater);



        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);


        model.getLabels();


        //recordReader.reset();

        recordReader.initialize(test);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);

        // Create Eval object with 10 possible classes
        Evaluation eval = new Evaluation(outputNum);




        while(testIter.hasNext()){
            DataSet next = testIter.next();
            INDArray output = model.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(),output);

        }

        log.info(eval.stats());







    }
}
