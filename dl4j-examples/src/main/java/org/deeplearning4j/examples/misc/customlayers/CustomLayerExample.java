package org.deeplearning4j.examples.misc.customlayers;

import org.deeplearning4j.examples.misc.customlayers.layer.CustomLayer;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * Custom layer example. This example shows the use and some basic testing for a custom layer implementation.
 * For more details, see the CustomLayerExampleReadme.md file
 *
 * @author Alex Black
 */
public class CustomLayerExample {

    static{
        //Double precision for the gradient checks. See comments in the doGradientCheck() method
        // See also http://nd4j.org/userguide.html#miscdatatype
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    public static void main(String[] args) throws IOException {
        runInitialTests();
        doGradientCheck();
    }

    private static void runInitialTests() throws IOException {
        /*
        This method shows the configuration and use of the custom layer.
        It also shows some basic sanity checks and tests for the layer.
        In practice, these tests should be implemented as unit tests; for simplicity, we are just printing the results
         */

        System.out.println("----- Starting Initial Tests -----");

        int nIn = 5;
        int nOut = 8;

        //Let's create a network with our custom layer

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .updater( new RmsProp(0.95))
            .weightInit(WeightInit.XAVIER)
            .regularization(true).l2(0.03)
            .list()
            .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(nIn).nOut(6).build())     //Standard DenseLayer
            .layer(1, new CustomLayer.Builder()
                .activation(Activation.TANH)                                                    //Property inherited from FeedForwardLayer
                .secondActivationFunction(Activation.SIGMOID)                                   //Custom property we defined for our layer
                .nIn(6).nOut(7)                                                                 //nIn and nOut also inherited from FeedForwardLayer
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)                //Standard OutputLayer
                .activation(Activation.SOFTMAX).nIn(7).nOut(nOut).build())
            .pretrain(false).backprop(true).build();


        //First:  run some basic sanity checks on the configuration:
        double customLayerL2 = ((BaseLayer)config.getConf(1).getLayer()).getL2();
        System.out.println("l2 coefficient for custom layer: " + customLayerL2);                //As expected: custom layer inherits the global L2 parameter configuration
        IUpdater customLayerUpdater = ((BaseLayer)config.getConf(1).getLayer()).getIUpdater();
        System.out.println("Updater for custom layer: " + customLayerUpdater);                  //As expected: custom layer inherits the global Updater configuration

        //Second: We need to ensure that that the JSON and YAML configuration works, with the custom layer
        // If there were problems with serialization, you'd get an exception during deserialization ("No suitable constructor found..." for example)
        String configAsJson = config.toJson();
        String configAsYaml = config.toYaml();
        MultiLayerConfiguration fromJson = MultiLayerConfiguration.fromJson(configAsJson);
        MultiLayerConfiguration fromYaml = MultiLayerConfiguration.fromYaml(configAsYaml);

        System.out.println("JSON configuration works: " + config.equals(fromJson));
        System.out.println("YAML configuration works: " + config.equals(fromYaml));

        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();


        //Third: Let's run some more basic tests. First, check that the forward and backward pass methods don't throw any exceptions
        // To do this: we'll create some simple test data
        int minibatchSize = 5;
        INDArray testFeatures = Nd4j.rand(minibatchSize, nIn);
        INDArray testLabels = Nd4j.zeros(minibatchSize, nOut);
        Random r = new Random(12345);
        for( int i=0; i<minibatchSize; i++ ){
            testLabels.putScalar(i,r.nextInt(nOut),1);  //Random one-hot labels data
        }

        List<INDArray> activations = net.feedForward(testFeatures);
        INDArray activationsCustomLayer = activations.get(2);                                   //Activations index 2: index 0 is input, index 1 is first layer, etc.
        System.out.println("\nActivations from custom layer:");
        System.out.println(activationsCustomLayer);
        net.fit(new DataSet(testFeatures, testLabels));


        //Finally, let's check the model serialization process, using ModelSerializer:
        ModelSerializer.writeModel(net, new File("CustomLayerModel.zip"), true);
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(new File("CustomLayerModel.zip"));

        System.out.println();
        System.out.println("Original and restored networks: configs are equal: " + net.getLayerWiseConfigurations().equals(restored.getLayerWiseConfigurations()));
        System.out.println("Original and restored networks: parameters are equal: " + net.params().equals(restored.params()));
    }


    private static void doGradientCheck(){
        /*
        Gradient checks are one of the most important components of implementing a layer
        They are necessary to ensure that your implementation is correct: without them, you could easily have a subtle
         error, and not even know it.

        Deeplearning4j comes with a gradient check utility that you can use to check your layers.
        This utility works for feed-forward layers, CNNs, RNNs etc.
        For more details on gradient checks, and some references, see the Javadoc for the GradientCheckUtil class:
        https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/gradientcheck/GradientCheckUtil.java

        There are a few things to note when doing gradient checks:
        1. It is necessary to use double precision for ND4J. Single precision (float - the default) isn't sufficiently
           accurate for reliably performing gradient checks
        2. It is necessary to set the updater to None, or equivalently use both the SGD updater and learning rate of 1.0
           Reason: we are testing the raw gradients before they have been modified with learning rate, momentum, etc.
        */

        System.out.println("\n\n\n----- Starting Gradient Check -----");

        Nd4j.getRandom().setSeed(12345);
        int nIn = 3;
        int nOut = 2;

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .updater(Updater.NONE).learningRate(1.0)
            .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1))              //Larger weight init than normal can help with gradient checks
            .regularization(true).l2(0.03)
            .list()
            .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(nIn).nOut(3).build())    //Standard DenseLayer
            .layer(1, new CustomLayer.Builder()
                .activation(Activation.TANH)                                                    //Property inherited from FeedForwardLayer
                .secondActivationFunction(Activation.SIGMOID)                                   //Custom property we defined for our layer
                .nIn(3).nOut(3)                                                                 //nIn and nOut also inherited from FeedForwardLayer
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)                //Standard OutputLayer
                .activation(Activation.SOFTMAX).nIn(3).nOut(nOut).build())
            .pretrain(false).backprop(true).build();
        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();

        boolean print = true;                                                                   //Whether to print status for each parameter during testing
        boolean return_on_first_failure = false;                                                //If true: terminate test on first failure
        double gradient_check_epsilon = 1e-8;                                                   //Epsilon value used for gradient checks
        double max_relative_error = 1e-5;                                                       //Maximum relative error allowable for each parameter
        double min_absolute_error = 1e-10;                                                      //Minimum absolute error, to avoid failures on 0 vs 1e-30, for example.

        //Create some random input data to use in the gradient check
        int minibatchSize = 5;
        INDArray features = Nd4j.rand(minibatchSize, nIn);
        INDArray labels = Nd4j.zeros(minibatchSize, nOut);
        Random r = new Random(12345);
        for( int i=0; i<minibatchSize; i++ ){
            labels.putScalar(i,r.nextInt(nOut),1);  //Random one-hot labels data
        }

        //Print the number of parameters in each layer. This can help to identify the layer that any failing parameters
        // belong to.
        for( int i=0; i<3; i++ ){
            System.out.println("# params, layer " + i + ":\t" + net.getLayer(i).numParams());
        }

        GradientCheckUtil.checkGradients(net, gradient_check_epsilon, max_relative_error, min_absolute_error, print,
            return_on_first_failure, features, labels);
    }

}
