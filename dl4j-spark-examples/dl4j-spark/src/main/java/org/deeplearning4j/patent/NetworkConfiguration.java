package org.deeplearning4j.patent;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * 1D convolutional network for patent classification
 * example. Assumes inputs are sequences of Google News
 * word vectors (300 dimensions).
 *
 * For inspiration in architecture, we looked at
 *
 * - Zhang and Wallace, IJCNLP 2017: https://www.aclweb.org/anthology/I17-1026
 * - CODE: https://github.com/bwallace/CNN-for-text-classification/blob/master/CNN_text.py
 */
public class NetworkConfiguration {
    private static final int W2V_VECTOR_SIZE = 300;
    private static final int[] ngramFilters = new int[]{3, 4, 5};
    private static final int numFilters = 128;
    private static final double dropoutRetain = 0.9;
    private static final int numClasses = 398;

    public static ComputationGraphConfiguration getConf() {
        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(0.01))
                .weightInit(WeightInit.RELU)
                .graphBuilder()
                .addInputs("in");

        String[] poolNames = new String[ngramFilters.length];
        int i = 0;
        for (int ngram : ngramFilters) {
            String filterName = String.format("ngram%d", ngram);
            poolNames[i] = String.format("pool%d", ngram);
            builder = builder.addLayer(filterName, new Convolution1DLayer.Builder()
                    .nOut(numFilters)
                    .kernelSize(ngram)
                    .activation(Activation.RELU)
                    .build(), "in")
                    .addLayer(poolNames[i], new GlobalPoolingLayer.Builder(PoolingType.MAX).build(), filterName);
            i++;
        }
        return builder.addVertex("concat", new MergeVertex(), poolNames)
                .addLayer("predict", new DenseLayer.Builder().nOut(numClasses).dropOut(dropoutRetain)
                        .activation(Activation.SOFTMAX).build(), "concat")
                .addLayer("loss", new LossLayer.Builder(LossFunctions.LossFunction.MCXENT).build(), "predict")
                .setOutputs("loss")
                .setInputTypes(InputType.recurrent(W2V_VECTOR_SIZE, 1000))
                .build();
    }
}
