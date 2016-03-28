package org.deeplearning4j.examples.recurrent.seq2seq;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
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
    public static final int batchSize = 1;
    public static final int totalBatches = 2000;
    public static final int NUM_DIGITS = 3;

    //This is the number of steps the rnn encoder/decoder is unrolled to
    //running with NUM_DIGITS * 5 gives NANs immediately, running with NUM_DIGITS * 7 gives some numbers
    //I was seeing more scores with NUM_DIGITS * 5 with an earlier release
    //Keras example uses 128. Below is 120. Score goes down, but interspersed with Nans.
    public static final int TIME_STEPS = NUM_DIGITS * 40;

    public static void main(String[] args) throws Exception {
        //Training data iterator
        CustomSequenceIterator iterator = new CustomSequenceIterator(rng, batchSize, totalBatches, NUM_DIGITS);

        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
            .regularization(true).l2(0.1)
            .weightInit(WeightInit.XAVIER)
            .learningRate(0.000001)
            .updater(Updater.ADAM)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .graphBuilder()
            .addInputs("additionIn", "sumOut")
            .setInputTypes(InputType.recurrent(NUM_DIGITS*2+1), InputType.recurrent((NUM_DIGITS+1)+1))
            .addLayer("encoder", new GravesLSTM.Builder().nIn(NUM_DIGITS*2+1).nOut(TIME_STEPS).activation("softsign").build(),"additionIn")
            .addVertex("lastTimeStep", new LastTimeStepVertex("additionIn"), "encoder")
            .addVertex("duplicateTimeStep", new DuplicateToTimeSeriesVertex("sumOut"), "lastTimeStep")
            .addLayer("decoder", new GravesLSTM.Builder().nIn( (NUM_DIGITS + 1) + 1 + TIME_STEPS).nOut(TIME_STEPS).activation("softsign").build(), "sumOut", "duplicateTimeStep")
            .addLayer("output", new RnnOutputLayer.Builder().nIn(TIME_STEPS).nOut(NUM_DIGITS + 1).activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT).build(), "decoder")
            .setOutputs("output")
            .pretrain(false).backprop(true)
            .build();

        ComputationGraph net = new ComputationGraph(configuration);
        net.init();
        net.setListeners(new ScoreIterationListener(1),new HistogramIterationListener(1));
        net.fit(iterator);

        // I have not called predict on model, WIP - generate test set from iterator and check accuracy with each batch
        System.out.println("* = * = * = * = * = * DONE!! * = * = * = * = * = *");
    }
}

