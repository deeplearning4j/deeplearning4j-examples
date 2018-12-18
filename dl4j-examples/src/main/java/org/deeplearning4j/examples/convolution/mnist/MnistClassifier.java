package org.deeplearning4j.examples.convolution.mnist;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.examples.utilities.DataUtilities;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

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
    public static int batchSize = 54;
    public static int nEpochs = 1;
    public static int seed = 123;

    public static int channels = 1;
    public static int width = 28;
    public static int height = 28;
    public static int classes = 10;

  public static void main(String[] args) throws Exception {
    Random randNumGen = new Random(seed);

    log.info("Loading data...");
    DataSetIterator trainMnist = new MnistDataSetIterator(batchSize, true, seed);
    DataSetIterator testMnist = new MnistDataSetIterator(batchSize, false, seed);

    log.info("Creating network...");
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed)
        .l2(0.0005)
        .updater(new AdaDelta())
        .weightInit(WeightInit.XAVIER)
        .list()
        .layer(new ConvolutionLayer.Builder(5, 5)
            .nIn(channels)
            .stride(1, 1)
            .nOut(20)
            .activation(Activation.IDENTITY)
            .build())
        .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2, 2)
            .stride(2, 2)
            .build())
        .layer(new ConvolutionLayer.Builder(5, 5)
            .stride(1, 1) // nIn need not specified in later layers
            .nOut(50)
            .activation(Activation.IDENTITY)
            .build())
        .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2, 2)
            .stride(2, 2)
            .build())
        .layer(new DenseLayer.Builder().activation(Activation.RELU)
            .nOut(500).build())
        .layer( new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(classes)
            .activation(Activation.SOFTMAX)
            .build())
        .setInputType(InputType.convolutionalFlat(height, width, channels)) // InputType.convolutional for normal image
        .build();

    MultiLayerNetwork net = new MultiLayerNetwork(conf);
    net.init();
    net.setListeners(new ScoreIterationListener(10), new EvaluativeListener(testMnist, 1, InvocationType.EPOCH_END));
    log.debug("Total num of params: {}", net.numParams());

    // evaluation while training (the score should go down)
    net.fit(trainMnist, nEpochs);

    log.info("Saving model to disk...");
    String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
    net.save(new File(basePath + "mnist-model.zip"));
  }

}
