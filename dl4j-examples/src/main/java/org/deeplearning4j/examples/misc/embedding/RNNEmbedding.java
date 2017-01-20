package org.deeplearning4j.examples.misc.embedding;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/** Feed-forward layer that expects single integers per example as input (class numbers, in range 0 to numClass-1)
 * as input. This input has shape [numExamples,1] instead of [numExamples,numClasses] for the equivalent one-hot representation.
 * Mathematically, EmbeddingLayer is equivalent to using a DenseLayer with a one-hot representation for the input; however,
 * it can be much more efficient with a large number of classes (as a dense layer + one-hot input does a matrix multiply
 * with all but one value being zero).<br>
 * <b>Note</b>: can only be used as the first layer for a network<br>
 * <b>Note 2</b>: For a given example index i, the output is activationFunction(weights.getRow(i) + bias), hence the
 * weight rows can be considered a vector/embedding for each example.
 *
 * @author Alex Black
 */
public class RNNEmbedding {
    public static void main(String[] args) throws Exception {

        int nClassesIn = 10;
        int batchSize = 3;
        int timeSeriesLength = 8;
        INDArray inEmbedding = Nd4j.create(batchSize, 1, timeSeriesLength);
        INDArray outLabels = Nd4j.create(batchSize, 4, timeSeriesLength);

        Random r = new Random(12345);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < timeSeriesLength; j++) {
                int classIdx = r.nextInt(nClassesIn);
                inEmbedding.putScalar(new int[]{i, 0, j}, classIdx);
                int labelIdx = r.nextInt(4);
                outLabels.putScalar(new int[]{i, labelIdx, j}, 1.0);
            }
        }

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .activation(Activation.RELU)
            .list()
            .layer(0, new EmbeddingLayer.Builder().nIn(nClassesIn).nOut(5).build())
            .layer(1, new GravesLSTM.Builder().nIn(5).nOut(7).activation(Activation.SOFTSIGN).build())
            .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(7).nOut(4).activation(Activation.SOFTMAX).build())
            .inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
            .inputPreProcessor(1, new FeedForwardToRnnPreProcessor())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setInput(inEmbedding);
        net.setLabels(outLabels);

        net.computeGradientAndScore();
        System.out.println(net.score());


    }
}
