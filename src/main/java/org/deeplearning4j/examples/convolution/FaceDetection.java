package org.deeplearning4j.examples.convolution;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.LimitFileSplit;
import org.canova.image.loader.BaseImageLoader;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * MSRA-CFW Dataset of Celebrity Faces on the Web is a data set created by MicrosoftResearch.
 * This is based of of the thumbnails data set which is a smaller subset. It includes 2215 images
 * and 10 classifications with each image only including one face.
 *
 * More information and the data set can be found at: http://research.microsoft.com/en-us/projects/msra-cfw/
 *
 */
public class FaceDetection {
    private static final Logger log = LoggerFactory.getLogger(FaceDetection.class);

    public final static int NUM_IMAGES = 2215; // # examples per person range 50 to 700
    public final static int NUM_LABELS = 10;
    public final static int WIDTH = 50; // size varies
    public final static int HEIGHT = 50;
    public final static int CHANNELS = 3;

    public static void main(String[] args) {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
        
        boolean appendLabels = true;
        int numExamples = 100;
        int batchSize = 20;

        int iterations = 5;
        int splitTrainNum = (int) (batchSize*.8);
        int seed = 123;
        int listenerFreq = batchSize;
        SplitTestAndTrain trainTest;
        DataSet trainInput;
        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();
        DataSet dsNext;

        // TODO setup to download and untar the example - currently need to manually download
        log.info("Load data....");
        File mainPath = new File(BaseImageLoader.BASE_DIR, "thumbnails_features_deduped_sample");
        RecordReader recordReader = new ImageRecordReader(WIDTH, HEIGHT, CHANNELS, appendLabels);
        try {
            recordReader.initialize(new LimitFileSplit(mainPath, BaseImageLoader.ALLOWED_FORMATS, numExamples, NUM_LABELS, null, new Random(123)));
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, -1, NUM_LABELS);

        List<String> labels = dataIter.getLabels();

        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation("relu")
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.01)
                .momentum(0.9)
                .regularization(true)
                .l2(1e-3)
                .updater(Updater.NESTEROVS)
                .useDropConnect(true)
//                .list(11)
//                .layer(0, new ConvolutionLayer.Builder(7, 7)
//                        .name("cnn1")
//                        .nIn(CHANNELS)
//                        .stride(1, 1)
//                        .nOut(48)
//                        .build())
//                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
//                        .name("pool1")
//                        .build())
//                .layer(2, new LocalResponseNormalization.Builder().build())
//                .layer(3, new ConvolutionLayer.Builder(5, 5)
//                        .name("cnn2")
//                        .stride(1, 1)
//                        .nOut(128)
//                        .build())
//                .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
//                        .name("pool2")
//                        .build())
//                .layer(5, new LocalResponseNormalization.Builder().build())
//                .layer(6, new ConvolutionLayer.Builder(3, 3)
//                        .name("cnn3")
//                        .stride(1, 1)
//                        .nOut(192)
//                        .build())
//                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
//                        .name("pool3")
//                        .build())
//                .layer(8, new DenseLayer.Builder()
//                        .name("ffn1")
//                        .nOut(256)
//                        .dropOut(0.5)
//                        .build())
//                .layer(9, new DenseLayer.Builder()
//                        .name("ffn1")
//                        .nOut(256)
//                        .dropOut(0.5)
//                        .build())
//                .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .nOut(NUM_LABELS)
//                        .activation("softmax")
//                        .build())
                .list(9)
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn1")
                        .nIn(CHANNELS)
                        .stride(1, 1)
                        .padding(1, 1)
                        .nOut(128)
                        .build())
                .layer(1, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn2")
                        .stride(1, 1)
                        .padding(1, 1)
                        .nOut(128)
                        .build())
                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool1")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool2")
                        .build())
                .layer(4, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn3")
                        .stride(1, 1)
                        .padding(1, 1)
                        .nOut(64)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool3")
                        .stride(1, 1)
                        .build())
                .layer(6, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(400)
                        .dropOut(0.5)
                        .build())
                .layer(7, new DenseLayer.Builder()
                        .name("ffn3")
                        .nOut(200)
                        .dropOut(0.5)
                        .build())
                .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(NUM_LABELS)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false)
                .cnnInputSize(HEIGHT, WIDTH, CHANNELS);

        MultiLayerNetwork model = new MultiLayerNetwork(builder.build());
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Train model....");
        while(dataIter.hasNext()) {
            dsNext = dataIter.next();
            dsNext.scale();
            trainTest = dsNext.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            trainInput = trainTest.getTrain(); // get feature matrix and labels for training
            testInput.add(trainTest.getTest().getFeatureMatrix());
            testLabels.add(trainTest.getTest().getLabels());
            model.fit(trainInput);
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(labels);
        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }
        INDArray output = model.output(testInput.get(0));
        eval.eval(testLabels.get(0), output);
        log.info(eval.stats());
        log.info("****************Example finished********************");


    }

}
