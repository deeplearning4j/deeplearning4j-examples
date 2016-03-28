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
    public static final Random rng = new Random(seed);
    //A batch size of any more 1 was giving nans with earlier release
    public static final int batchSize = 100;
    public static final int totalBatches = 5000;
    public static final int nIterations = 20;
    public static final int NUM_DIGITS = 6;

    //This is the number of steps the rnn encoder/decoder is unrolled to
    //running with NUM_DIGITS * 5 gives NANs immediately, running with NUM_DIGITS * 7 gives some numbers
    //I was seeing more scores with NUM_DIGITS * 5 with an earlier release
    //Keras example uses 128. Below is 120. Score goes down, but interspersed with Nans.
    public static final int featureSizeVector = 12;
    public static int timeStepEncoder;
    public static int timeStepDecoder;


    public static void main(String[] args) throws Exception {
        //Training data iterator
        CustomSequenceIterator iterator = new CustomSequenceIterator(rng, batchSize, totalBatches, NUM_DIGITS,12);

        timeStepEncoder = NUM_DIGITS * 2 + 1 + 1;
        timeStepDecoder = NUM_DIGITS + 1 + 1;
        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
            .regularization(true).l2(0.0025)
            //.weightInit(WeightInit.XAVIER)
            .weightInit(WeightInit.DISTRIBUTION)
            .dist(new UniformDistribution(-0.08, 0.08))
            .learningRate(0.5)
            .updater(Updater.ADAM)
            //.updater(Updater.RMSPROP)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(nIterations)
            .graphBuilder()
            .addInputs("additionIn", "sumOut")
            .setInputTypes(InputType.recurrent(timeStepEncoder), InputType.recurrent(timeStepDecoder))
            .addLayer("encoder", new GravesLSTM.Builder().nIn(featureSizeVector).nOut(featureSizeVector).activation("softsign").build(),"additionIn")
            .addVertex("lastTimeStep", new LastTimeStepVertex("additionIn"), "encoder")
            .addVertex("duplicateTimeStep", new DuplicateToTimeSeriesVertex("sumOut"), "lastTimeStep")
            .addLayer("decoder", new GravesLSTM.Builder().nIn(2*featureSizeVector).nOut(featureSizeVector).activation("softsign").build(), "sumOut", "duplicateTimeStep")
            .addLayer("output", new RnnOutputLayer.Builder().nIn(featureSizeVector).nOut(featureSizeVector).activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT).build(), "decoder")
            .setOutputs("output")
            .pretrain(false).backprop(true)
            .build();

        ComputationGraph net = new ComputationGraph(configuration);
        net.init();
        net.setListeners(new ScoreIterationListener(1),new HistogramIterationListener(1));
        int i = 0;
        while (iterator.hasNext()) {
            net.fit(iterator);
            i++;
            System.out.println("Batch num:"+i);

        }

        // I have not called predict on model, WIP - generate test set from iterator and check accuracy with each batch
        System.out.println("* = * = * = * = * = * DONE!! * = * = * = * = * = *");
    }
}

