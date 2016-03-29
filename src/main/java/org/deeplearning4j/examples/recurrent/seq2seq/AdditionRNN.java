package org.deeplearning4j.examples.recurrent.seq2seq;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;
import java.util.Arrays;

/**
 * Created by susaneraly on 3/27/16.
 */
public class AdditionRNN {

    //Random number generator seed, for reproducability
    public static final int seed = 12345;

    public static final int NUM_DIGITS = 3;
    public static final int FEATURE_VEC_SIZE = 12;

    //Tweak these to tune - dataset size = batchSize * totalBatches
    public static final int batchSize = 100;
    public static final int totalBatches = 500;
    public static final int nEpochs = 200;
    public static final int nIterations = 1;
    public static final int numHiddenNodes = 128;


    public static void main(String[] args) throws Exception {
        //Training data iterator
        CustomSequenceIterator iterator = new CustomSequenceIterator(seed, batchSize, totalBatches, NUM_DIGITS,12);

        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
            .regularization(true).l2(0.000005)
            //.weightInit(WeightInit.XAVIER)
            .weightInit(WeightInit.DISTRIBUTION)
            .dist(new UniformDistribution(-0.08, 0.08))
            .learningRate(0.5)
            .updater(Updater.ADAM)
            //.updater(Updater.RMSPROP)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(nIterations)
            .graphBuilder()
            .addInputs("additionIn", "sumOut")
            .setInputTypes(InputType.recurrent(FEATURE_VEC_SIZE), InputType.recurrent(FEATURE_VEC_SIZE))
            .addLayer("encoder", new GravesLSTM.Builder().nIn(FEATURE_VEC_SIZE).nOut(numHiddenNodes).activation("softsign").build(),"additionIn")
            .addVertex("lastTimeStep", new LastTimeStepVertex("additionIn"), "encoder")
            .addVertex("duplicateTimeStep", new DuplicateToTimeSeriesVertex("sumOut"), "lastTimeStep")
            .addLayer("decoder", new GravesLSTM.Builder().nIn(FEATURE_VEC_SIZE + numHiddenNodes).nOut(numHiddenNodes).activation("softsign").build(), "sumOut", "duplicateTimeStep")
            .addLayer("output", new RnnOutputLayer.Builder().nIn(numHiddenNodes).nOut(FEATURE_VEC_SIZE).activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT).build(), "decoder")
            .setOutputs("output")
            .pretrain(false).backprop(true)
            .build();

        ComputationGraph net = new ComputationGraph(configuration);
        net.init();
        net.setListeners(new ScoreIterationListener(1),new HistogramIterationListener(1));
        //Train model:
        int iEpoch = 0;
        //int testSize = (int) 0.2 * batchSize * totalBatches;
        while (iEpoch < nEpochs) {
            System.out.printf("* = * = * = ** EPOCH %d ** = * = * = * = * = *\n",iEpoch);
            net.fit(iterator);
            //MultiDataSet testData = iterator.generateTest(testSize+1);
            //System.out.println(testData.getFeatures(0));
            //System.out.println(testData.getFeatures(1));
            iterator.reset();
            iEpoch++;
        }
        System.out.println("* = * = * = * = * = * EPOCHS COMPLETE * = * = * = * = * = * = * = *");

    }
}

