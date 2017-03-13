package org.deeplearning4j.examples.dataexamples;



import org.apache.commons.io.FilenameUtils;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.utilities.DataUtilities;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Random;

/**
 * Created by tom hanlon on 11/7/16.
 * This code example is featured in this youtube video
 * https://www.youtube.com/watch?v=GLC8CIoHDnI
 *
 * This differs slightly from the Video Example,
 * The Video example had the data already downloaded
 * This example includes code that downloads the data
 *
 * Instructions
 * Downloads a directory containing a testing and a training folder
 * each folder has 10 directories 0-9
 * in each directory are 28 * 28 grayscale pngs of handwritten digits
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
 */

public class MnistImagePipelineExample {

    /** Data URL for downloading */
    public static final String DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

    /** Location to save and extract the training/testing data */
    public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_Mnist/");



    private static Logger log = LoggerFactory.getLogger(MnistImagePipelineExample.class);

    public static void main(String[] args) throws Exception {
        /*
        image information
        28 * 28 grayscale
        grayscale implies single channel
        */
        int height = 28;
        int width = 28;
        int channels = 1;
        int rngseed = 123;
        Random randNumGen = new Random(rngseed);
        int batchSize = 1;
        int outputNum = 10;



        /*
        This class downloadData() downloads the data
        stores the data in java's tmpdir
        15MB download compressed
        It will take 158MB of space when uncompressed
        The data can be downloaded manually here
        http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
         */


        downloadData();

        // Define the File Paths
        File trainData = new File(DATA_PATH + "/mnist_png/training");
        File testData = new File(DATA_PATH + "/mnist_png/testing");

        // Define the FileSplit(PATH, ALLOWED FORMATS,random)


        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        // Extract the parent path as the image label

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        // Initialize the record reader
        // add a listener, to extract the name

        recordReader.initialize(train);

        // The LogRecordListener will log the path of each image read
        // used here for information purposes,
        // If the whole dataset was ingested this would place 60,000
        // lines in our logs
        // It will show up in the output with this format
        // o.d.a.r.l.i.LogRecordListener - Reading /tmp/mnist_png/training/4/36384.png

        recordReader.setListeners(new LogRecordListener());

        // DataSet Iterator

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

        // Scale pixel values to 0-1

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        // In production you would loop through all the data
        // in this example the loop is just through 3
        // images for demonstration purposes
        for (int i = 1; i < 3; i++) {
            DataSet ds = dataIter.next();
            System.out.println(ds);
            System.out.println(dataIter.getLabels());

        }

    }

    /*
    Everything below here has nothing to do with your RecordReader,
    or DataVec, or your Neural Network
    The classes downloadData, getMnistPNG(),
    and extractTarGz are for downloading and extracting the data
     */

    private static void downloadData() throws Exception {
        //Create directory if required
        File directory = new File(DATA_PATH);
        if(!directory.exists()) directory.mkdir();

        //Download file:
        String archizePath = DATA_PATH + "/mnist_png.tar.gz";
        File archiveFile = new File(archizePath);
        String extractedPath = DATA_PATH + "mnist_png";
        File extractedFile = new File(extractedPath);

        if( !archiveFile.exists() ){
            System.out.println("Starting data download (15MB)...");
            getMnistPNG();
            //Extract tar.gz file to output directory
            DataUtilities.extractTarGz(archizePath, DATA_PATH);
        } else {
            //Assume if archive (.tar.gz) exists, then data has already been extracted
            System.out.println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath());
            if( !extractedFile.exists()){
                //Extract tar.gz file to output directory
                DataUtilities.extractTarGz(archizePath, DATA_PATH);
            } else {
                System.out.println("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
            }
        }
    }

    public static void getMnistPNG() throws IOException {
        String tmpDirStr = System.getProperty("java.io.tmpdir");
        String archizePath = DATA_PATH + "/mnist_png.tar.gz";

        if (tmpDirStr == null) {
            throw new IOException("System property 'java.io.tmpdir' does specify a tmp dir");
        }
        String url = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";
        File f = new File(archizePath);
        File dir = new File(tmpDirStr);
        if (!f.exists()) {
            HttpClientBuilder builder = HttpClientBuilder.create();
            CloseableHttpClient client = builder.build();
            try (CloseableHttpResponse response = client.execute(new HttpGet(url))) {
                HttpEntity entity = response.getEntity();
                if (entity != null) {
                    try (FileOutputStream outstream = new FileOutputStream(f)) {
                        entity.writeTo(outstream);
                        outstream.flush();
                        outstream.close();
                    }
                }

            }
            System.out.println("Data downloaded to " + f.getAbsolutePath());
        } else {
            System.out.println("Using existing directory at " + f.getAbsolutePath());
        }

    }


    }


