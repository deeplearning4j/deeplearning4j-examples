package org.deeplearning4j.examples.mlp;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.mlp.sampleNetStructure.CMGSNet;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


/**
 * Deep, Big, Simple Neural Nets Excel on Handwritten Digit Recognition
 * 2010 paper by Cireșan, Meier, Gambardella, and Schmidhuber
 * They achieved 99.65 accuracy
 */
public class MLPMnistCMGSExample {

    private static final Logger LOG = LoggerFactory.getLogger(MLPMnistCMGSExample.class);


    public static void main(String[] args) throws Exception {

        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int numSamples = 60000;
        int batchSize = 500;
        int iterations = 50;
        int seed = 123;
        int listenerFreq = 10;
        int splitTrainNum = (int) (batchSize*.8);
        DataSet mnist;
        SplitTestAndTrain trainTest;
        DataSet trainInput;
        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();


        LOG.info("Load data....");
        DataSetIterator iter = new MnistDataSetIterator(batchSize,numSamples);

        LOG.info("Build model....");
        MultiLayerNetwork model = new CMGSNet(numRows, numColumns, outputNum, seed, iterations).init();

        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        LOG.info("Train model....");
        while(iter.hasNext()) {
            mnist = iter.next();
            trainTest = mnist.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            trainInput = trainTest.getTrain(); // get feature matrix and labels for training
            testInput.add(trainTest.getTest().getFeatureMatrix());
            testLabels.add(trainTest.getTest().getLabels());
            model.fit(trainInput);
        }

        LOG.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }


        LOG.info(eval.stats());
        LOG.info("****************Example finished********************");

    }

}
