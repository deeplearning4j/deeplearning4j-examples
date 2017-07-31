package org.deeplearning4j.examples.convolution;

import org.deeplearning4j.datasets.iterator.AsyncShieldDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 9/16/15.
 */
public class LenetMnistCGExample {
    private static final Logger log = LoggerFactory.getLogger(LenetMnistCGExample.class);

    public static void main(String[] args) throws Exception {
        int nChannels = 1; // Number of input channels
        int outputNum = 10; // The number of possible outcomes
        int batchSize = 64; // Test batch size
        int nEpochs = 10; // Number of training epochs
        int iterations = 1; // Number of training iterations
        int seed = 123; //

        //CudaEnvironment.getInstance().getConfiguration().allowCrossDeviceAccess(false);

        /*
            Create an iterator using the batch size for one iteration
         */
        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);

        /*
            Construct the neural network
         */
        log.info("Build model....");
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations) // Training iterations as above
                .regularization(true).l2(0.0005)
                /*
                    Uncomment the following for learning decay and bias
                 */
                .learningRate(.01)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .trainingWorkspaceMode(WorkspaceMode.SINGLE)
                .cacheMode(CacheMode.DEVICE)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn1", new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .dropOut(0.95)
                        .activation(Activation.IDENTITY)
                        .build(),"input")
                .addLayer("pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build(),"cnn1")
                .addLayer("cnn2", new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build(),"pool1")
                .addLayer("pool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build(),"cnn2")
                .addLayer("dense", new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build(),"pool2")
                .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build(),"dense")
                .setOutputs("output")
                .setInputTypes(InputType.convolutionalFlat(28,28,1)) //See note below
                .backprop(true).pretrain(false).build();

        /*
        Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
        (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
            and the dense layer
        (b) Does some additional configuration validation
        (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
            layer based on the size of the previous layer (but it won't override values manually set by the user)

        InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
        For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
        MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
        row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
        */

        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        Nd4j.getMemoryManager().setAutoGcWindow(1000000);

        // ParallelWrapper will take care of load balancing between GPUs.
        ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
            // DataSets prefetching options. Set this value with respect to number of actual devices
            .prefetchBuffer(4)

            // set number of workers equal or higher then number of available devices. x1-x2 are good values to start with
            .workers(2)

            // rare averaging improves performance, but might reduce model accuracy
            .averagingFrequency(5)

            // if set to TRUE, on every averaging model score will be reported
            .reportScoreAfterAveraging(false)

            // optinal parameter, set to false ONLY if your system has support P2P memory access across PCIe (hint: AWS do not support P2P)
            //.useLegacyAveraging(false)

            .workspaceMode(WorkspaceMode.SINGLE)

            //.useMQ(true)

            .build();

        log.info("Train model....");
        nEpochs = 1;
        model.setListeners(new PerformanceListener(50, true));//, new EvaluativeListener(mnistTest, 50));
        for( int i=0; i<nEpochs; i++ ) {
            long time1 = System.currentTimeMillis();
            model.fit((mnistTrain));
            //wrapper.fit(mnistTrain);
            long time2 = System.currentTimeMillis();



            log.info("*** Completed epoch {}; {} ms ***", i, time2 - time1);

        }

        Nd4j.getWorkspaceManager().printAllocationStatisticsForCurrentThread();



        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
//        ROC roceval = new ROC(outputNum);
        model.doEvaluation(mnistTest, eval);
        while(mnistTest.hasNext()){
            DataSet ds = mnistTest.next();
            INDArray output = model.output(false,ds.getFeatureMatrix())[0];
            eval.eval(ds.getLabels(), output);
        }
        log.info(eval.stats());
        mnistTest.reset();

        log.info("****************Example finished********************");

    }
}
