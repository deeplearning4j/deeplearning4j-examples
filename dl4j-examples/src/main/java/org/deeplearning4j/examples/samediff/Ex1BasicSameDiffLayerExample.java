package org.deeplearning4j.examples.samediff;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.samediff.layers.MinimalSameDiffDense;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
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
 * This example: how to implement a custom DL4J layer using SameDiff.
 *
 * The layer itself is a standard dense (fully connected) layer with an activation function.
 *
 * The idea here
 *
 *
 * @author Alex Black
 */
public class Ex1BasicSameDiffLayerExample {

    public static void main(String[] args) throws Exception {

        int networkNumInputs = 28*28;       //For MNIST - 28x28 pixels
        int networkNumOutputs = 10;         //For MNIST - 10 classes
        int layerSize = 128;                //128 units for the SameDiff layers

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .updater(new Adam(1e-1))
            .seed(12345)
            .list()
            //Add two custom layers:
            .layer(new MinimalSameDiffDense(networkNumInputs, layerSize, Activation.TANH, WeightInit.XAVIER))
            .layer(new MinimalSameDiffDense(layerSize, layerSize, Activation.TANH, WeightInit.XAVIER))
            //Combine with a standard DL4J output layer
            .layer(new OutputLayer.Builder().nIn(layerSize).nOut(networkNumOutputs).activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
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
        int networkNumInputs = 4;
        int layerSize = 5;
        int networkNumOutputs = 3;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .updater(new NoOp())    //Required for gradient checks
            .seed(12345)
            .list()
            //Add two custom layers:
            .layer(new MinimalSameDiffDense(networkNumInputs, layerSize, Activation.TANH, WeightInit.XAVIER))
            .layer(new MinimalSameDiffDense(layerSize, layerSize, Activation.TANH, WeightInit.XAVIER))
            //Combine with a standard DL4J output layer
            .layer(new OutputLayer.Builder().nIn(layerSize).nOut(networkNumOutputs).activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
            .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        //Check model serialization:
        File f = new File(FileUtils.getTempDirectory(), "SameDiffExample1Model.zip");
        net.save(f);
        MultiLayerNetwork loaded = MultiLayerNetwork.load(f, true);

        Nd4j.getRandom().setSeed(12345);
        INDArray testFeatures = Nd4j.rand(3, networkNumInputs);
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
