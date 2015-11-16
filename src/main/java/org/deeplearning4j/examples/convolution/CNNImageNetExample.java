package org.deeplearning4j.examples.convolution;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.LimitFileSplit;
import org.canova.image.recordreader.ImageNetRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.convolution.sampleNetStructure.AlexNet;
import org.deeplearning4j.examples.convolution.sampleNetStructure.LeNet;
import org.deeplearning4j.examples.convolution.sampleNetStructure.VGGNetA;
import org.deeplearning4j.examples.convolution.sampleNetStructure.VGGNetD;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ParamAndGradientIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

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
        final int numRows = 224;
        final int numColumns = 224;
        int nChannels = 3;
        int outputNum = 1860;
        int numBatches = 1; // TODO - total training amount for CSL is 1281167
        int batchSize = 6;
        int iterations = 2;
        int numTrainEpochs = 2;
        int seed = 123;
        int listenerFreq = 1;
        MultiLayerNetwork model = null;
        String modelType = "VGGNetA";
        boolean gradientCheck = false;
        boolean train = true;
        DataSetIterator dataIter;
        int totalNumExamples = batchSize*numBatches;
        int splitTrainNum = (int) (batchSize * .8);
        int numTestExamples = totalNumExamples/(numBatches) - splitTrainNum;

        String basePath = System.getProperty("user.home") + File.separator + "Documents" + File.separator + "skymind" + File.separator + "imagenet" + File.separator;
        String dataPath = basePath + "sample-pics" + File.separator;
        String labelPath = basePath + "cls-loc-labels.csv";
        String[] allForms = {"jpg", "jpeg", "JPG", "JPEG"};


        log.info("Load data....");
        RecordReader recordReader = new ImageNetRecordReader(numColumns, numRows, nChannels, true, labelPath);
        recordReader.initialize(new LimitFileSplit(new File(dataPath), allForms, totalNumExamples, 2, Pattern.quote("_"), 0, new Random(123)));
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
            case "VGGNetA":
                model = new VGGNetA(numRows, numColumns, nChannels, outputNum, seed, iterations).init();
                break;
            case "VGGNetD":
                model = new VGGNetD(numRows, numColumns, nChannels, outputNum, seed, iterations).init();
                break;
        }

        IterationListener paramListener = ParamAndGradientIterationListener.builder()
                .outputToFile(true)
                .file(new File(System.getProperty("java.io.tmpdir") + "/paramAndGradTest.txt"))
                .outputToConsole(true).outputToLogger(false)
                .iterations(1).printHeader(true)
                .printMean(false)
                .printMinMax(false)
                .printMeanAbsValue(true)
                .delimiter("\t").build();

        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

//        model.setListeners(Arrays.asList((IterationListener) new HistogramIterationListener(listenerFreq)));
//        model.setListeners(Arrays.asList(new ScoreIterationListener(listenerFreq), paramListener));
        model.setListeners(Arrays.asList((IterationListener)  new ScoreIterationListener(listenerFreq)));

        if (gradientCheck) gradientCheck(dataIter, model);
        
        if (train) {
            log.info("Train model....");

            //TODO need dataIter that loops through set number of examples like SamplingIter but takes iter vs dataset
            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, numRows * numColumns * nChannels, 1860);
            MultipleEpochsIterator epochIter = new MultipleEpochsIterator(numTrainEpochs, dataIter);
////                asyncIter = new AsyncDataSetIterator(dataIter, 1); TODO doesn't have next(num)

            for (int i = 0; i < numTrainEpochs; i++) {
                for (int j = 0; j < numBatches; j++)
                    model.fit(epochIter.next(splitTrainNum));
                eval = evaluatePerformance(model, epochIter, numTestExamples, numBatches, eval);// TODO split out eval for its own dataset
            }

//                dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, numRows * numColumns * nChannels, 1860);

//            for (int i = 0; i < numTrainEpochs; i++) {
//                for (int j = 0; j < numBatches; j++)
//                    model.fit(dataIter.next(splitTrainNum));
//                System.out.println("Epoch " + i + " complete");

//                //Evaluate classification performance:
//                eval = evaluatePerformance(model, dataIter, numTestExamples, numBatches, eval);
//            }
            log.info(eval.stats());

            log.info("****************Example finished********************");
        }
    }


    private static Evaluation evaluatePerformance(MultiLayerNetwork model, MultipleEpochsIterator iter, int numExamples, int numBatches, Evaluation eval){
        log.info("Evaluate model....");
        DataSet imgNet;
        INDArray output;

        //TODO setup iterator to randomize and split test and train
        for(int i=0; i < numBatches; i++){
            imgNet = iter.next(numExamples);
            output = model.output(imgNet.getFeatureMatrix());
            eval.eval(imgNet.getLabels(), output);
        }
        return eval;
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


}
