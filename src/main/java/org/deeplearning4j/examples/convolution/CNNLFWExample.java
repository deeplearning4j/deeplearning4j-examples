package org.deeplearning4j.examples.convolution;

import org.canova.image.loader.LFWLoader;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Labeled Faces in the Wild is a data set created by Erik Learned-Miller, Gary Huang, Aruni RoyChowdhury,
 * Haoxiang Li, Gang Hua. This is used to study unconstrained face recognition. There are over 13K images.
 * Each face has been labeled with the name of the person pictured.
 *
 * 5749 unique classes (different people)
 * 1680 people have 2+ photos
 *
 * References:
 * General information is at http://vis-www.cs.umass.edu/lfw/. Architecture partially based on DeepFace:
 * http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf
 *
 * Note: this is a sparse dataset with only 1 example for many of the faces; thus, performance is low.
 * Ideally train on a larger dataset like celebs to get params and/or generate variations of the image examples.
 *
 * Currently set to only use the subset images, names starting with A.
 * Switch to NUM_LABELS & NUM_IMAGES and set subset to false to use full dataset.
 */

public class CNNLFWExample {
    private static final Logger log = LoggerFactory.getLogger(CNNMnistExample.class);

    public static void main(String[] args) {


        final int numRows = 40;
        final int numColumns = 40;
        int nChannels = 3;
        int outputNum = LFWLoader.NUM_LABELS;
        int numSamples = 1000; // LFWLoader.NUM_IMAGES;
        boolean useSubset = false;
        int batchSize = 200;// numSamples/10;
        int iterations = 5;
        int splitTrainNum = (int) (batchSize*.8);
        int seed = 123;
        int listenerFreq = iterations/5;
        DataSet lfwNext;
        SplitTestAndTrain trainTest;
        DataSet trainInput;
        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();

        log.info("Load data....");
        DataSetIterator lfw = new LFWDataSetIterator(batchSize, numSamples, new int[] {numRows, numColumns, nChannels}, outputNum, useSubset, new Random(seed));

        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation("relu")
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.01)
                .momentum(0.9)
                .regularization(true)
                .updater(Updater.ADAGRAD)
                .useDropConnect(true)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(4, 4)
                        .name("cnn1")
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool1")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn2")
                        .stride(1,1)
                        .nOut(40)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool2")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn3")
                        .stride(1,1)
                        .nOut(60)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool3")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(2, 2)
                        .name("cnn3")
                        .stride(1,1)
                        .nOut(80)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(160)
                        .dropOut(0.5)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder,numRows,numColumns,nChannels);

        MultiLayerNetwork model = new MultiLayerNetwork(builder.build());
        model.init();

        log.info("Train model....");
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

        while(lfw.hasNext()) {
            lfwNext = lfw.next();
            lfwNext.scale();
            trainTest = lfwNext.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            trainInput = trainTest.getTrain(); // get feature matrix and labels for training
            testInput.add(trainTest.getTest().getFeatureMatrix());
            testLabels.add(trainTest.getTest().getLabels());
            model.fit(trainInput);
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(lfw.getLabels());
        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }
        INDArray output = model.output(testInput.get(0));
        eval.eval(testLabels.get(0), output);
        log.info(eval.stats());
        log.info("****************Example finished********************");

    }

}
