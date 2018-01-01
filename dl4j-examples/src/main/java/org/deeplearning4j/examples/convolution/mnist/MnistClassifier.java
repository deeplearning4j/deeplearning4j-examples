package org.deeplearning4j.examples.convolution.mnist;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.utilities.DataUtilities;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Handwritten digits image classification on MNIST dataset (99% accuracy).
 * This example will download 15 Mb of data on the first run.
 * Supervised learning best modeled by CNN.
 *
 * @author hanlon
 * @author agibsonccc
 * @author fvaleri
 */
public class MnistClassifier {

  private static final Logger log = LoggerFactory.getLogger(MnistClassifier.class);
  private static final String basePath = System.getProperty("java.io.tmpdir") + "/mnist";
  private static final String dataUrl = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

  public static void main(String[] args) throws Exception {
    int height = 28;
    int width = 28;
    int channels = 1; // single channel for grayscale images
    int outputNum = 10; // 10 digits classification
    int batchSize = 54;
    int nEpochs = 1;
    int iterations = 1;

    int seed = 1234;
    Random randNumGen = new Random(seed);

    log.info("Data load and vectorization...");
    String localFilePath = basePath + "/mnist_png.tar.gz";
    if (DataUtilities.downloadFile(dataUrl, localFilePath))
      log.debug("Data downloaded from {}", dataUrl);
    if (!new File(basePath + "/mnist_png").exists())
      DataUtilities.extractTarGz(localFilePath, basePath);

    // vectorization of train data
    File trainData = new File(basePath + "/mnist_png/training");
    FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
    ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // parent path as the image label
    ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
    trainRR.initialize(trainSplit);
    DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

    // pixel values from 0-255 to 0-1 (min-max scaling)
    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    scaler.fit(trainIter);
    trainIter.setPreProcessor(scaler);

    // vectorization of test data
    File testData = new File(basePath + "/mnist_png/testing");
    FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
    ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
    testRR.initialize(testSplit);
    DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
    testIter.setPreProcessor(scaler); // same normalization for better results

    log.info("Network configuration and training...");
    Map<Integer, Double> lrSchedule = new HashMap<>();
    lrSchedule.put(0, 0.06); // iteration #, learning rate
    lrSchedule.put(300, 0.05);
    lrSchedule.put(500, 0.029);
    lrSchedule.put(700, 0.0044);
    lrSchedule.put(1000, 0.0011);

    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed)
        .iterations(iterations)
        .regularization(true).l1(0.0005)
        .learningRate(.01)
        .learningRateDecayPolicy(LearningRatePolicy.Schedule)
        .learningRateSchedule(lrSchedule) // overrides the rate set in learningRate
        .weightInit(WeightInit.XAVIER)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(Updater.NESTEROVS)
        .list()
        .layer(0, new ConvolutionLayer.Builder(5, 5)
            .nIn(channels)
            .stride(1, 1)
            .nOut(20)
            .activation(Activation.IDENTITY)
            .build())
        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2, 2)
            .stride(2, 2)
            .build())
        .layer(2, new ConvolutionLayer.Builder(5, 5)
            .stride(1, 1) // nIn need not specified in later layers
            .nOut(50)
            .activation(Activation.IDENTITY)
            .build())
        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2, 2)
            .stride(2, 2)
            .build())
        .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
            .nOut(500).build())
        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(outputNum)
            .activation(Activation.SOFTMAX)
            .build())
        .setInputType(InputType.convolutionalFlat(28, 28, 1)) // InputType.convolutional for normal image
        .backprop(true).pretrain(false).build();

    MultiLayerNetwork net = new MultiLayerNetwork(conf);
    net.init();
    net.setListeners(new ScoreIterationListener(10));
    log.debug("Total num of params: {}", net.numParams());

    // evaluation while training (the score should go down)
    for (int i = 0; i < nEpochs; i++) {
      net.fit(trainIter);
      log.info("Completed epoch {}", i);
      Evaluation eval = net.evaluate(testIter);
      log.info(eval.stats());
      trainIter.reset();
      testIter.reset();
    }

    ModelSerializer.writeModel(net, new File(basePath + "/minist-model.zip"), true);
  }

}
