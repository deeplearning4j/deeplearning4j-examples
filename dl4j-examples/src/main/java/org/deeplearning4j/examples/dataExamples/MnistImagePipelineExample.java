package org.deeplearning4j.examples.dataExamples;



import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

/**
 * Created by tomhanlon on 11/7/16.
 * This code example is featured in this youtube video
 * https://www.youtube.com/watch?v=GLC8CIoHDnI
 *
 * Instructions
 * You must download the data for this example to work
 * The datafile is a 15M download that uncompresses to a 273MB directory
 * The Data Directory mnist_png will have two child directories training and testing
 * The training and testing directories will have directories 0-9 with
 * 28 * 28 PNG images of handwritten images
 *
 * The code here shows how to use a ParentPathLabelGenerator to label the images as
 * they are read into the RecordReader
 *
 * The pixel values are scaled to values between 0 and 1 using
 *  ImagePreProcessingScaler
 *
 *  In this example a loop steps through 3 images and prints the DataSet to
 *  the terminal. The expected output is the 28* 28 matrix of scaled pixel values
 *  the list with the label for that image
 *  and a list of the label values
 *
 *  This example also applies a Listener to the RecordReader that logs the path of each image read
 *  You would not want to do this in production
 *  The reason it is done here is to show that a handwritten image 3 (for example)
 *  was read from directory 3,
 *  has a matrix with the shown values
 *  Has a label value corresponding to 3
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
 */

public class MnistImagePipelineExample {
    private static Logger log = LoggerFactory.getLogger(MnistImagePipelineExample.class);

    public static void main(String[] args) throws Exception {
        // image information
        // 28 * 28 grayscale
        // grayscale implies single channel
        int height = 28;
        int width = 28;
        int channels = 1;
        int rngseed = 123;
        Random randNumGen = new Random(rngseed);
        int batchSize = 1;
        int outputNum = 10;

        // Define the File Paths
        File trainData = new File("/tmp/mnist_png/training");
        File testData = new File("/tmp/mnist_png/testing");

        // Define the FileSplit(PATH, ALLOWED FORMATS,random)

        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS,randNumGen);
        FileSplit test = new FileSplit(testData,NativeImageLoader.ALLOWED_FORMATS,randNumGen);

        // Extract the parent path as the image label

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);

        // Initialize the record reader
        // add a listener, to extract the name

        recordReader.initialize(train);

        // The LogRecordListener will log the path of each image read
        // used here for informaiton purposes,
        // It will show up in the output with this format
        // o.d.a.r.l.i.LogRecordListener - Reading /tmp/mnist_png/training/4/36384.png

        recordReader.setListeners(new LogRecordListener());

        // DataSet Iterator

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);

        // Scale pixel values to 0-1

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        // In production you would loop through all the data
        // in this example the loop is just through 3
        // images for demonstration purposes
        for (int i = 1; i< 3; i++){
            DataSet ds = dataIter.next();
            System.out.println(ds);
            System.out.println(dataIter.getLabels());

        }

    }
}
