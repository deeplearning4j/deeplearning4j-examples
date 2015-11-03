package org.deeplearning4j.examples.convolution;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageNetRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.convolution.sampleNetStructure.AlexNet;
import org.deeplearning4j.examples.convolution.sampleNetStructure.LeNet;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

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
        // libraries like Caffe scale to 256
        final int numRows = 120;  // TODO should be 224 based on VGG and AlexNet original paper
        final int numColumns = 120;
        int nChannels = 3;
        int outputNum = 1860; // TODO currently testing 1 category but there are 1300 options
        int numBatches = 1; // TODO - total training amount for CSL is 1281167
        int batchSize = 60;
        int iterations = 5;
        int seed = 123;
        int listenerFreq = 1;
        int splitTrainNum = (int) (batchSize*.5);
        boolean gradientCheck = false;
        boolean train = true;

//        Collection<Writable> imNet;
        SplitTestAndTrain trainTest;
        DataSet trainInput;
        DataSet imgNet;
        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();

        String basePath = System.getProperty("user.home") + File.separator + "Documents" + File.separator + "skymind" + File.separator + "imagenet" + File.separator;
        String dataPath = basePath + "dogs_resize" + File.separator;
        String labelPath = basePath + "cls-loc-labels.csv";

        //////////// TODO remove this when interface updated

        Map<String,String> labelIdMap = new LinkedHashMap<>();

        BufferedReader br = new BufferedReader(new FileReader(labelPath));
        String line;

        while((line = br.readLine())!=null){
            String row[] = line.split(",");
            labelIdMap.put(row[0], row[1]);
        }
        List<String> labels = new ArrayList<>(labelIdMap.values());

        Map<Integer,String> testLabelMap = new LinkedHashMap<>();

        int j = 0;
        for (String label : labels){
            testLabelMap.put(j, label);
            j++;
        }

        ////////////

        log.info("Load data....");
        RecordReader recordReader = new ImageNetRecordReader(numColumns, numRows, nChannels, true, labelPath);
        recordReader.initialize(new FileSplit(new File(dataPath)));
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, numRows*numColumns*nChannels, 1860);

        log.info("Build model....");
//        MultiLayerNetwork model = new LeNet(numRows, numColumns, nChannels, outputNum, seed, iterations).init();
        MultiLayerNetwork model = new AlexNet(numRows, numColumns, nChannels, outputNum, seed, iterations).init();
//        MultiLayerNetwork model = new VGGNet(numRows, numColumns, nChannels, outputNum, seed, iterations).init();

//        model.setListeners(Arrays.asList(new ScoreIterationListener(listenerFreq), new HistogramIterationListener(listenerFreq)));
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));


        log.info("Gradient Check....");
        if(gradientCheck) {

            imgNet = dataIter.next();
            String name = new Object() {
            }.getClass().getEnclosingMethod().getName();

            model.setInput(imgNet.getFeatures());
            model.setLabels(imgNet.getLabels());
            model.computeGradientAndScore();
            double scoreBefore = model.score();
            for (j = 0; j < 1; j++)
                model.fit(imgNet);
            model.computeGradientAndScore();
            double scoreAfter = model.score();
//            String msg = name + " - score did not (sufficiently) decrease during learning (before=" + scoreBefore + ", scoreAfter=" + scoreAfter + ")";
//            assertTrue(msg, scoreAfter < 0.8 * scoreBefore);
            for (j = 0; j < model.getnLayers(); j++)
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


        log.info("Train model....");
        if(train) {
        for(int i = 0; i < numBatches; i++) {

            imgNet = dataIter.next();
            imgNet.normalizeZeroMeanZeroUnitVariance();
            trainTest = imgNet.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            trainInput = trainTest.getTrain(); // get feature matrix and labels for training
            testInput.add(trainTest.getTest().getFeatureMatrix());
            testLabels.add(trainTest.getTest().getLabels());
            model.fit(trainInput);
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(testLabelMap);

        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }

        log.info(eval.stats());
        }
        log.info("****************Example finished********************");

    }

}
