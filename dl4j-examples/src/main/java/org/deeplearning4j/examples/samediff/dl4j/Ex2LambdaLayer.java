package org.deeplearning4j.examples.samediff.dl4j;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.samediff.dl4j.layers.L2NormalizeLambdaLayer;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

/**
 *
 * This example: how to implement a simple custom DL4J lambda layer using SameDiff.
 *
 * The lambda layer (see L2NormalizeLamdaLayer) implements "out = in / l2Norm(in)" on a per-example basis.
 *
 * Lamda layers are those that do not have any parameters, but require only the implementation of a single method
 * to define the forward pass. However, lambda layers have a number of optional methods, that may be implemented
 * for additional functionality, including:
 * - getOutputType(int layerIndex, InputType inputType)
 * - applyGlobalConfigToLayer(NeuralNetConfiguration.Builder globalConfig)
 *
 *
 * @author Alex Black
 */
public class Ex2LambdaLayer {

    public static void main(String[] args) throws Exception {
        int networkNumOutputs = 10;         //For MNIST - 10 classes

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .updater(new Adam(1e-1))
            .seed(12345)
            .activation(Activation.TANH)
            .convolutionMode(ConvolutionMode.Same)
            .list()
            //Add some standard DL4J layers:
            .layer(new ConvolutionLayer.Builder().nIn(1).nOut(16).kernelSize(2,2).stride(1,1).build())
            .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).build())
            .layer(new ConvolutionLayer.Builder().nIn(16).nOut(8).kernelSize(2,2).stride(1,1).build())
            .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).build())
            //Add custom SameDiff lambda layer:
            .layer(new L2NormalizeLambdaLayer(1,2,3))
            //Add standard DL4J output layer:
            .layer(new OutputLayer.Builder().nIn(7*7*8).nOut(networkNumOutputs).activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
            .inputPreProcessor(0, new FeedForwardToCnnPreProcessor(28, 28, 1))
            .inputPreProcessor(5, new CnnToFeedForwardPreProcessor(7,7,8))
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(50));


        //Train and evaluate the network with the custom SameDiff layer
        //Note that training and evaluation is the same as with built-in layers
        DataSetIterator train = new MnistDataSetIterator(32, true, 12345);
        net.fit(train, 1);  //Train for 1 epoch

        DataSetIterator test = new MnistDataSetIterator(32, false, 12345);
        Evaluation e = net.evaluate(test);
        System.out.println(e.stats());

        //Also: validate correctness of the network/layer
        validateLayer();
    }

    public static void validateLayer() throws Exception {

        /*
        When implementing a custom layer, it is a good idea to validate the implementation.
        Specifically:
        1. Checking that model serialization (i.e., saving and loading) works
        2. Performing gradient checks

        Note that gradient checks require ND4J to be set to double precision, and the updater needs to be set
        to NoOp, or equivalently SGD with learning rate of 1.0
         */

        System.out.println("===== Starting Layer Validation =====");
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);

        //Define network: smaller than before, so gradient checks are quicker
        int[] inputShape = new int[]{3, 1, 5, 5};       //[minibatch, channels, height, width]
        int networkNOut = 3;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .updater(new NoOp())
            .seed(12345)
            .activation(Activation.TANH)
            .convolutionMode(ConvolutionMode.Same)
            .list()
            //Add a standard DL4J layer:
            .layer(new ConvolutionLayer.Builder().nIn(1).nOut(4).kernelSize(2,2).stride(1,1).build())
            //Add custom SameDiff lambda layer:
            .layer(new L2NormalizeLambdaLayer(1,2,3))
            //Add standard DL4J output layer:
            .layer(new OutputLayer.Builder().nIn(5*5*4).nOut(networkNOut).activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
            .inputPreProcessor(2, new CnnToFeedForwardPreProcessor(5, 5, 4))
            .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        //Check model serialization:
        File f = new File(FileUtils.getTempDirectory(), "SameDiffExample2Model.zip");
        net.save(f);
        MultiLayerNetwork loaded = MultiLayerNetwork.load(f, true);

        Nd4j.getRandom().setSeed(12345);
        INDArray testFeatures = Nd4j.rand(inputShape);
        INDArray fromOriginalNet = net.output(testFeatures);
        INDArray fromLoadedNet = loaded.output(testFeatures);
        if(!fromOriginalNet.equals(fromLoadedNet)){
            throw new IllegalStateException("Saved and loaded nets should have equal outputs!");
        }

        //Create some random labels for gradient checks:
        INDArray testLabels = Nd4j.eye(3);  //3x3, with 1s along the dimension. Here, minibatch size 3, networkNumOutputs = 3


        //Check gradients
        boolean print = true;                                                                   //Whether to print status for each parameter during testing
        boolean return_on_first_failure = false;                                                //If true: terminate test on first failure
        double gradient_check_epsilon = 1e-6;                                                   //Epsilon value used for gradient checks
        double max_relative_error = 1e-5;                                                       //Maximum relative error allowable for each parameter
        double min_absolute_error = 1e-8;                                                      //Minimum absolute error, to avoid failures on 0 vs 1e-30, for example.

        boolean gradOk = GradientCheckUtil.checkGradients(net, gradient_check_epsilon, max_relative_error, min_absolute_error, print,
            return_on_first_failure, testFeatures, testLabels);
        if(!gradOk){
            throw new IllegalStateException("Gradient check failed");
        }
    }
}
