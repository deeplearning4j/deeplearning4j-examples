package org.deeplearning4j.examples.convolution;

import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class AlexExample {
    private static int height = 224;
    private static int width = 224;
    private static int channels = 3;
    private static int numLabels = 1000;
    private static long seed = 42;
    private static int iterations = 1;


    public static void main(String[] args) throws Exception {

        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .trainingWorkspaceMode(WorkspaceMode.SINGLE)
            .seed(seed)
            .weightInit(WeightInit.DISTRIBUTION)
            .dist(new NormalDistribution(0.0, 0.01))
            .activation(Activation.RELU)
            .updater(Updater.SGD)
            .iterations(iterations)
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(1e-2)
//                .momentum(0.9)
            .biasLearningRate(1e-2*2)
            .learningRateDecayPolicy(LearningRatePolicy.Step)
            .lrPolicyDecayRate(0.1)
            .lrPolicySteps(100000)
            .regularization(true)
            .convolutionMode(ConvolutionMode.Same)
            .dropOut(0.5)
            .l2(5 * 1e-4)
            .miniBatch(false)
            .list()
            .layer(0, new ConvolutionLayer.Builder(new int[]{11,11}, new int[]{4, 4}, new int[]{2,2})
                .name("cnn1")
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                .convolutionMode(ConvolutionMode.Truncate)
                .nIn(channels)
                .nOut(64)
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{2,2}, new int[]{1,1})
                .convolutionMode(ConvolutionMode.Truncate)
                .name("maxpool1")
                .build())
            .layer(2, new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{2,2}, new int[]{2,2}) // TODO: fix input and put stride back to 1,1
                .convolutionMode(ConvolutionMode.Truncate)
                .name("cnn2")
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                .nOut(192)
                .biasInit(nonZeroBias)
                .build())
            .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3}, new int[]{2, 2})
                .name("maxpool2")
                .build())
            .layer(4, new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1}, new int[]{1,1})
                .name("cnn3")
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                .nOut(384)
                .build())
            .layer(5, new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1}, new int[]{1,1})
                .name("cnn4")
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                .nOut(256)
                .biasInit(nonZeroBias)
                .build())
            .layer(6, new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1}, new int[]{1,1})
                .name("cnn5")
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                .nOut(256)
                .biasInit(nonZeroBias)
                .build())
            .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{7,7}) // TODO: fix input and put stride back to 2,2
                .name("maxpool3")
                .build())
            .layer(8, new DenseLayer.Builder()
                .name("ffn1")
                .nIn(256)
                .nOut(4096)
                .dist(new GaussianDistribution(0, 0.005))
                .biasInit(nonZeroBias)
                .dropOut(dropOut)
                .build())
            .layer(9, new DenseLayer.Builder()
                .name("ffn2")
                .nOut(4096)
                .dist(new GaussianDistribution(0, 0.005))
                .biasInit(nonZeroBias)
                .dropOut(dropOut)
                .build())
            .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("output")
                .nOut(numLabels)
                .activation(Activation.SOFTMAX)
                .build())
            .backprop(true)
            .pretrain(false)
            .setInputType(InputType.convolutional(height,width,channels))
            .build();


        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();


        String json = conf.toJson();
        MultiLayerConfiguration fromJson = MultiLayerConfiguration.fromJson(json);

        System.out.println(json);

        ParallelWrapper pw = new ParallelWrapper.Builder(network)
            .prefetchBuffer(2)
            .workers(2)
            .averagingFrequency(1)
            .reportScoreAfterAveraging(true)
            .build();

        List<DataSet> l = new ArrayList<>();
        for( int i=0; i<4; i++ ){
            l.add(new DataSet(Nd4j.rand(new int[]{2,channels,height,width}), Nd4j.zeros(2, numLabels)));
        }

        DataSetIterator iter = new ExistingDataSetIterator(l);

        System.out.println("SIZE: " + l.size());

        pw.setListeners(new PerformanceListener(1, true));
        pw.fit(iter);

//        network.fit(iter);
    }
}
