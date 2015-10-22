package org.deeplearning4j.examples.video;

import org.canova.api.conf.Configuration;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.records.reader.impl.CSVSequenceRecordReader;
import org.canova.api.split.InputSplit;
import org.canova.api.split.NumberedFileInputSplit;
import org.canova.codec.reader.CodecRecordReader;
import org.deeplearning4j.datasets.canova.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Example: Combine convolutional neural network layer and
 * recurrent (LSTM) layer to classify the frames of a video
 * <p>
 * Specifically, each video contains a number of shapes (circles, squares, lines, arcs)
 * These shapes are present (though moving) in a number of frames
 * To make things more, difficult, each video frame contains, in addition to the target shape,
 * (a) background noise
 * (b) a number of 'distractor' shapes (which are present for one frame only)
 * the idea is that the classification problem is not possible to solve in isolation
 * (i.e., it's not possible to determine the target shape by looking at a single video frame)
 * <p>
 * *******************************************************
 * WARNING: THIS EXAMPLE GENERATES A VERY LARGE DATA SET
 * This is a synthetic video data set. Typical settings result
 * in several gigabytes of video files.
 * Furthermore, this data generation can take a long time.
 * It is recommended that you customize the data generation
 * location, and/or run this on a machine with sufficient
 * hard drive space.
 * This examples does NOT automatically delete this data set
 * after the example is complete.
 * *******************************************************
 *
 * @author Alex Black
 */
public class VideoClassificationExample {

    public static final int N_VIDEOS_TO_GENERATE = 100;
    public static final int V_WIDTH = 128;
    public static final int V_HEIGHT = 128;
    public static final int V_NFRAMES = 150;

    public static void main(String[] args) throws Exception {

        int miniBatchSize = 5;
        boolean generateData = false;

//        String tempDir = System.getProperty("java.io.tmpdir");
        String tempDir = "c:/Temp/";
        String outputDirectory = tempDir + "DL4JVideoShapesExample/";


        if (generateData) {
            System.out.println("Starting data generation...");
            generateData(outputDirectory);
            System.out.println("Data generation complete");
        }

        //Set up network architecture:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .regularization(true).l2(0.001) //l2 regularization on all layers
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.1)
                .list(3)
                .layer(0, new ConvolutionLayer.Builder(8, 8)
                        .nIn(3) //3 channels: RGB
                        .nOut(6)
                        .stride(4, 4)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .updater(Updater.RMSPROP)
                        .build())
                .layer(1, new GravesLSTM.Builder()
                        .activation("tanh")
                        .nIn(31 * 31 * 6)   //128x128: (128-8)/4+1 = 31
                        .nOut(200)  //200 LSTM units/neurons
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.08, 0.08)) //Use in conjunction with WeightInit.DISTRIBUTION
                        .updater(Updater.RMSPROP)
                        .build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax")
                        .nIn(200)
                        .nOut(4)    //4 possible shapes: circle, square, arc, line
                        .updater(Updater.RMSPROP)
                        .build())
                .inputPreProcessor(0, new ComposableInputPreProcessor( //Video data: time series. Preprocess for CNN
                        new RnnToFeedForwardPreProcessor(),
                        new FeedForwardToCnnPreProcessor(V_HEIGHT, V_WIDTH, 3)))
                .inputPreProcessor(1, new ComposableInputPreProcessor(
                        new CnnToFeedForwardPreProcessor(),
                        new FeedForwardToRnnPreProcessor()))
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(1)));

        int testStartIdx = (int) (0.8 * N_VIDEOS_TO_GENERATE);  //80% in train, 20% in test
        int nTrain = testStartIdx;
        int nTest = N_VIDEOS_TO_GENERATE - nTrain;

        //Conduct learning
        System.out.println("Starting training...");
        int nTrainEpochs = 5;
        for (int i = 0; i < nTrainEpochs; i++) {
            //Use 80% of the generated videos to train, 20% to test
            DataSetIterator trainData = getDataSetIterator(outputDirectory, 0, nTrain - 1, miniBatchSize);
            net.fit(trainData);
            System.out.println("Epoch " + i + " complete");
        }

        //Evaluate the network's classification peformance:
        System.out.println("Starting evaluation:");
        DataSetIterator testData = getDataSetIterator(outputDirectory, testStartIdx, nTest, nTest);
        DataSet dsTest = testData.next();
        INDArray predicted = net.activate(dsTest.getFeatureMatrix(), false);
        INDArray actual = dsTest.getLabels();

        Map<Integer, String> labelMap = new HashMap<>();
        labelMap.put(0, "circle");
        labelMap.put(1, "square");
        labelMap.put(2, "arc");
        labelMap.put(3, "line");

        Evaluation evaluation = new Evaluation(labelMap);
        evaluation.evalTimeSeries(actual, predicted);

        System.out.println("Test set evaluation:");
        System.out.println(evaluation.stats());
    }

    private static void generateData(String path) throws Exception {
        File f = new File(path);
        if (!f.exists()) f.mkdir();

        VideoGenerator.generateVideoData(path, "shapes", N_VIDEOS_TO_GENERATE,
                V_NFRAMES, V_WIDTH, V_HEIGHT,
                3,  //Number of shapes per video. Switches from one shape to another over time
                true,   //Background noise. Significantly increases video file size
                3,  //Number of distractors per frame ('distractors' are shapes show for one frame only)
                12345L);    //Seed, for reproducability when generating data

    }

    private static DataSetIterator getDataSetIterator(String dataDirectory, int startIdx, int nExamples, int miniBatchSize) throws Exception {
        //Here, our data and labels are in separate files
        //videos: shapes_0.mp4, shapes_1.mp4, etc
        //labels: shapes_0.txt, shapes_1.txt, etc. One time step per line

        SequenceRecordReader featuresTrain = getFeaturesReader(dataDirectory, startIdx, nExamples);
        SequenceRecordReader labelsTrain = getLabelsReader(dataDirectory, startIdx, nExamples);

        SequenceRecordReaderDataSetIterator trainIter =
                new SequenceRecordReaderDataSetIterator(featuresTrain, labelsTrain, miniBatchSize, 4, false);
        return trainIter;
    }

    private static SequenceRecordReader getFeaturesReader(String path, int startIdx, int num) throws Exception {
        //InputSplit is used here to define what the file paths look like
        InputSplit is = new NumberedFileInputSplit(path + "shapes_%d.mp4", startIdx, startIdx + num - 1);

        Configuration conf = new Configuration();
        conf.set(CodecRecordReader.RAVEL, "true");
        conf.set(CodecRecordReader.START_FRAME, "0");
        conf.set(CodecRecordReader.TOTAL_FRAMES, String.valueOf(V_NFRAMES));
        conf.set(CodecRecordReader.ROWS, String.valueOf(V_WIDTH));
        conf.set(CodecRecordReader.COLUMNS, String.valueOf(V_HEIGHT));
        CodecRecordReader crr = new CodecRecordReader();
        crr.initialize(conf, is);
        return crr;
    }

    private static SequenceRecordReader getLabelsReader(String path, int startIdx, int num) throws Exception {
        InputSplit isLabels = new NumberedFileInputSplit(path + "shapes_%d.txt", startIdx, startIdx + num - 1);
        CSVSequenceRecordReader csvSeq = new CSVSequenceRecordReader();
        csvSeq.initialize(isLabels);
        return csvSeq;
    }
}
