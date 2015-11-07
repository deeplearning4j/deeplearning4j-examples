package org.deeplearning4j.examples.convolution;

import org.apache.commons.io.FileUtils;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageNetRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.SamplingDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.convolution.sampleNetStructure.AlexNet;
import org.deeplearning4j.examples.convolution.sampleNetStructure.LeNet;
import org.deeplearning4j.examples.convolution.sampleNetStructure.VGGNet;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;

import static org.junit.Assert.assertTrue;

/**
 *
 * Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
 * Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei.
 * (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. arXiv:1409.0575, 2014.
 *

 * Created by nyghtowl on 9/24/15.
 */
public class CNNImageNetExample {
    private static final Logger log = LoggerFactory.getLogger(CNNImageNetExample.class);

    public static void main(String[] args) throws Exception {
        // libraries like Caffe scale to 256?
        final int numRows = 224;  // TODO should be 224 based on VGG and AlexNet original paper
        final int numColumns = 224;
        int nChannels = 3;
        int outputNum = 1860;
        int numBatches = 1; // TODO - total training amount for CSL is 1281167
        int batchSize = 20;
        int iterations = 5;
        int nTrainEpochs = 1;
        int seed = 123;
        int listenerFreq = 1;
        int splitTrainNum = (int) (batchSize * .9);
        MultiLayerNetwork model = null;
        String modelType = "LeNet";
        boolean gradientCheck = false;
        boolean train = true;
        DataSetIterator dataIter;
        AsyncDataSetIterator asyncIter;

        String basePath = System.getProperty("user.home") + File.separator + "Documents" + File.separator + "skymind" + File.separator + "imagenet" + File.separator;
        String dataPath = basePath + "sample-pics" + File.separator;
        String labelPath = basePath + "cls-loc-labels.csv";

        List<File> someFiles = new ArrayList<>();
        Random rnd = new Random();
        rnd.setSeed(seed);
        boolean val;

        File fileBase = new File(dataPath);
        final List<String> allForm = Arrays.asList("jpg", "jpeg", "JPG", "JPEG");
        File[] paths = fileBase.listFiles(new FileFilter() {
            @Override
            public boolean accept(File pathname) {
                if (pathname.toString().endsWith("jpeg") || pathname.toString().endsWith("JPEG")) {
                    return true;
                }
                return false;
            }
        });
// TODO need to recursively loop through dir - haven't found a method yet that does this
        for(File p: paths) {
            System.out.println(p.getName());
            if(rnd.nextBoolean())
                someFiles.add(new File(p.getPath()));
        }



        log.info("Load data....");
        RecordReader recordReader = new ImageNetRecordReader(numColumns, numRows, nChannels, true, labelPath);
//        recordReader.initialize(new FileSplit(new File(dataPath)));
        recordReader.initialize(dataPath, seed, numBatches*splitTrainNum);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, numRows * numColumns * nChannels, 1860);
        Evaluation eval = new Evaluation(recordReader.getLabels());

        log.info("Build model....");
        switch (modelType) {
            case "LeNet":
                model = new LeNet(numRows, numColumns, nChannels, outputNum, seed, iterations).init();
                break;
            case "AlexNet":
                model = new AlexNet(numRows, numColumns, nChannels, outputNum, seed, iterations).init();
                break;
            case "VGGNet":
                model = new VGGNet(numRows, numColumns, nChannels, outputNum, seed, iterations).init();
                break;
        }

//        model.setListeners(Arrays.asList(new ScoreIterationListener(listenerFreq), new HistogramIterationListener(listenerFreq)));
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        if (gradientCheck) gradientCheck(dataIter, model);


        if (train) {
            log.info("Train model....");

            for (int i = 0; i < nTrainEpochs; i++) {
                //TODO need dataIter that loops through set number of examples like SamplingIter but takes iter vs dataset
                dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, numRows * numColumns * nChannels, 1860);
//                asyncIter = new AsyncDataSetIterator(dataIter, 1); TODO doesn't have next(num)
                for (int j = 0; j < numBatches; j++)
                    model.fit(dataIter.next(batchSize*splitTrainNum));
                System.out.println("Epoch " + i + " complete");

                //Evaluate classification performance:
                eval = evaluatePerformance(model, dataIter, batchSize*(1-splitTrainNum), numBatches, eval);
            }

            //        SplitTestAndTrain trainTest;
//        DataSet trainInput;
//        List<INDArray> testInput = new ArrayList<>();
//        List<INDArray> testLabels = new ArrayList<>();
//        Map<Integer, String> testLabelMap = new LinkedHashMap<>();

            // Small sample run - can't scale too far
//            for(int i = 0; i < numBatches; i++) {
//                imgNet = dataIter.next();
//                imgNet.normalizeZeroMeanZeroUnitVariance();
//                trainTest = imgNet.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
//                trainInput = trainTest.getTrain(); // get feature matrix and labels for training
//                testInput.add(trainTest.getTest().getFeatureMatrix());
//                testLabels.add(trainTest.getTest().getLabels());
//                model.fit(trainInput);
//            }
            // Small test setup
//            Evaluation eval = new Evaluation(dataIter.labelMap());
//            for(int i = 0; i < testInput.size(); i++) {
//                INDArray output = model.output(testInput.get(i));
//                eval.eval(testLabels.get(i), output);
//            }

            log.info(eval.stats());

            log.info("****************Example finished********************");
        }
    }


    private static void gradientCheck(DataSetIterator dataIter, MultiLayerNetwork model){
        DataSet imgNet;
        log.info("Gradient Check....");

        imgNet = dataIter.next();
        String name = new Object() {
        }.getClass().getEnclosingMethod().getName();

        model.setInput(imgNet.getFeatures());
        model.setLabels(imgNet.getLabels());
        model.computeGradientAndScore();
        double scoreBefore = model.score();
        for (int j = 0; j < 1; j++)
            model.fit(imgNet);
        model.computeGradientAndScore();
        double scoreAfter = model.score();
//            String msg = name + " - score did not (sufficiently) decrease during learning (before=" + scoreBefore + ", scoreAfter=" + scoreAfter + ")";
//            assertTrue(msg, scoreAfter < 0.8 * scoreBefore);
        for (int j = 0; j < model.getnLayers(); j++)
            System.out.println("Layer " + j + " # params: " + model.getLayer(j).numParams());

        double default_eps = 1e-6;
        double default_max_rel_error = 0.25;
        boolean print_results = true;
        boolean return_on_first_failure = false;
        boolean useUpdater = true;

        boolean gradOK = GradientCheckUtil.checkGradients(model, default_eps, default_max_rel_error,
                print_results, return_on_first_failure, imgNet.getFeatures(), imgNet.getLabels(), useUpdater);

        assertTrue(gradOK);

    }

    private static Evaluation evaluatePerformance(MultiLayerNetwork model, DataSetIterator iter, int numExamples, int numBatches, Evaluation eval){
        log.info("Evaluate model....");
        DataSet imgNet;
        INDArray output;

        //TODO setup iterator to randomize and split test and train
        for(int i=0; i < numBatches; i++){
            imgNet = iter.next();
            output = model.output(imgNet.getFeatureMatrix());
            eval.eval(imgNet.getLabels(), output);
        }
        return eval;
    }

}
