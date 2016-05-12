package org.deeplearning4j.examples.recurrent.seq2seq;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

/**
 * Created by susaneraly on 3/27/16.
 */
public class AdditionRNN {

    /*
        This example is modeled off the sequence to sequence RNNs described in http://arxiv.org/abs/1410.4615
        Specifically, a sequence to sequence NN is build for the addition operation
        Two numbers and the addition operator are encoded as a sequence and passed through an "encoder" RNN
        The output from the last time step of the encoder RNN is reinterpreted as a time series and passed through the "decoder" RNN
        The result is the output of the decoder RNN which in training is the sum, encoded as a sequence.
        Note: This example has not been tuned.
     */

    //Random number generator seed, for reproducability
    public static final int seed = 1234;

    public static final int NUM_DIGITS = 3;
    public static final int FEATURE_VEC_SIZE = 12;

    //Tweak these to tune - dataset size = batchSize * totalBatches
    public static final int batchSize = 100;
    public static final int totalBatches = 500;
    public static final int nEpochs = 5;
    public static final int nIterations = 1;
    public static final int numHiddenNodes = 128;

    //Currently the sequences are implemented as length = max length
    //This is a placeholder for an enhancement
    public static final int timeSteps = NUM_DIGITS * 2 + 1;

    public static void main(String[] args) throws Exception {
        //Training data iterator
        CustomSequenceIterator iterator = new CustomSequenceIterator(seed, batchSize, totalBatches, NUM_DIGITS,timeSteps);

        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
            //.regularization(true).l2(0.000005)
            .weightInit(WeightInit.XAVIER)
            .learningRate(0.5)
            .updater(Updater.RMSPROP)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(nIterations)
            .seed(seed)
            .graphBuilder()
            .addInputs("additionIn", "sumOut")
            .setInputTypes(InputType.recurrent(FEATURE_VEC_SIZE), InputType.recurrent(FEATURE_VEC_SIZE))
            .addLayer("encoder", new GravesLSTM.Builder().nIn(FEATURE_VEC_SIZE).nOut(numHiddenNodes).activation("softsign").build(),"additionIn")
            .addVertex("lastTimeStep", new LastTimeStepVertex("additionIn"), "encoder")
            .addVertex("duplicateTimeStep", new DuplicateToTimeSeriesVertex("sumOut"), "lastTimeStep")
            .addLayer("decoder", new GravesLSTM.Builder().nIn(FEATURE_VEC_SIZE+numHiddenNodes).nOut(numHiddenNodes).activation("softsign").build(), "sumOut","duplicateTimeStep")
            .addLayer("output", new RnnOutputLayer.Builder().nIn(numHiddenNodes).nOut(FEATURE_VEC_SIZE).activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT).build(), "decoder")
            .setOutputs("output")
            .pretrain(false).backprop(true)
            .build();

        ComputationGraph net = new ComputationGraph(configuration);
        net.init();
        //net.setListeners(new ScoreIterationListener(200),new HistogramIterationListener(200));
        //net.setListeners(new ScoreIterationListener(1));
        //net.setListeners(new HistogramIterationListener(200));
        //Train model:
        int iEpoch = 0;
        int testSize = 200;
        while (iEpoch < nEpochs) {
            System.out.printf("* = * = * = ** EPOCH %d ** = * = * = * = * = *\n",iEpoch);
            net.fit(iterator);

            MultiDataSet testData = iterator.generateTest(testSize);
            ArrayList<int[]> testNums = iterator.testFeatures();
            int[] testnum1 = testNums.get(0);
            int[] testnum2 = testNums.get(1);
            int[] testSums = iterator.testLabels();
            INDArray[] prediction_array = net.output(new INDArray[]{testData.getFeatures(0),testData.getFeatures(1)});
            INDArray predictions = prediction_array[0];
            INDArray answers = Nd4j.argMax(predictions,1);

            encode_decode(testnum1,testnum2,testSums,answers);

            iterator.reset();
            iEpoch++;
        }
        System.out.println("* = * = * = * = * = * EPOCHS COMPLETE * = * = * = * = * = * = * = *");

    }

    private static void encode_decode(int[] num1, int[] num2, int[] sum, INDArray answers) {

        int nTests = answers.size(0);
        int wrong = 0;
        int correct = 0;
        for (int iTest=0; iTest < nTests; iTest++) {
            int aDigit = NUM_DIGITS;
            int thisAnswer = 0;
            while (aDigit >= 0) {
                int thisDigit = (int) answers.getDouble(iTest,aDigit);
                //System.out.println(thisDigit);
                if (thisDigit > 9) {
                    // Need support to check if spaces are predicted correctly
                    break;
                }
                thisAnswer += thisDigit * (int) Math.pow(10,NUM_DIGITS-aDigit);
                aDigit--;
            }
            if (thisAnswer != sum[iTest]) {
                System.out.println(num1[iTest]+"+"+num2[iTest]+"!="+thisAnswer);
                wrong ++;
            }
            else {
                System.out.println(num1[iTest]+"+"+num2[iTest]+"=="+thisAnswer);
                correct ++;
            }
        }
        double randomAcc = Math.pow(10,-1*(NUM_DIGITS+1)) * 100;
        System.out.println("*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*==*=*=*=*=*");
        System.out.println("WRONG: "+wrong);
        System.out.println("CORRECT: "+correct);
        System.out.println("Note randomly guessing digits in succession gives lower than a accuracy of:"+randomAcc+"%");
        System.out.println("Since the correct number of digits have to also be predicted the number above will be even lower");
        System.out.println("*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*");
    }

}

