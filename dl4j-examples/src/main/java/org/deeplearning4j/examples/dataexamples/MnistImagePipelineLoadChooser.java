package org.deeplearning4j.examples.dataexamples;

import org.apache.commons.io.FilenameUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 * This code example is featured in this youtube video
 * http://www.youtube.com/watch?v=DRHIpeJpJDI
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
 * by giving the user a file chooser to test an image of their choice
 * against the Nueral Net, will the network think this cat is an 8 or a 1
 * 
 * Seriously you can test anything, but obviously the network was trained 
 * on handwritten images 0-9 white digit, black background, so it will work 
 * better with stuff closer to what it was designed for
 */
public class MnistImagePipelineLoadChooser {
  private static Logger log = LoggerFactory.getLogger(MnistImagePipelineLoadChooser.class);
  
  /** Location to save and extract the training/testing data */
  public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_Mnist/");

  /*
  Create a popup window to allow you to chose an image file to test against the
  trained Neural Network
  Chosen images will be automatically
  scaled to 28*28 grayscale
  */
  public static String fileChose() {
    JFileChooser fc = new JFileChooser();
    int ret = fc.showOpenDialog(null);
    if (ret == JFileChooser.APPROVE_OPTION) {
      File file = fc.getSelectedFile();
      String filename = file.getAbsolutePath();
      return filename;
    } else {
      return null;
    }
  }

  public static void main(String[] args) throws Exception {
    int height = 28;
    int width = 28;
    int channels = 1;

    // recordReader.getLabels()
    // In this version Labels are always in order
    // So this is no longer needed
    //List<Integer> labelList = Arrays.asList(2,3,7,1,6,4,0,5,8,9);
    List<Integer> labelList = Arrays.asList(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);

    // pop up file chooser
    String filechose = fileChose().toString();

    //LOAD NEURAL NETWORK

    // Where to save model
    File locationToSave = new File(DATA_PATH + "trained_mnist_model.zip");
    // Check for presence of saved model
    if (locationToSave.exists()) {
      log.info("Saved Model Found!");
    } else {
      log.error("File not found!");
      log.error("This example depends on running MnistImagePipelineExampleSave, run that example first");
      System.exit(0);
    }

    MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

    log.info("TEST YOUR IMAGE AGAINST SAVED NETWORK");
    // FileChose is a string we will need a file
    File file = new File(filechose);

    // Use NativeImageLoader to convert to numerical matrix
    NativeImageLoader loader = new NativeImageLoader(height, width, channels);

    // Get the image into an INDarray
    INDArray image = loader.asMatrix(file);

    // 0-255
    // 0-1
    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    scaler.transform(image);
    
    // Pass through to neural Net
    INDArray output = model.output(image);

    log.info("The file chosen was " + filechose);
    log.info("The neural nets prediction (list of probabilities per label)");
    //log.info("## List of Labels in Order## ");
    // In new versions labels are always in order
    log.info(output.toString());
    log.info(labelList.toString());
  }

}
