package org.deeplearning4j.examples.convolution;

import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
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
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

//import org.deeplearning4j.nn.conf.LearningRatePolicy;

/**
 * Created by agibsonccc on 9/16/15.
 */
public class LenetMnistExample {
    private static final Logger log = LoggerFactory.getLogger(LenetMnistExample.class);

    public static void main(String[] args) throws Exception {
        int nChannels = 1;
        int outputNum = 10;
        int batchSize = 256;
        int nEpochs = 10;
        int iterations = 1;
        int seed = 123;



        CudaEnvironment.getInstance().getConfiguration()
            // key option enabled
            .allowMultiGPU(true)

            // we're allowing larger memory caches
            .setMaximumDeviceCache(3L * 1024L * 1024L * 1024L)

            .setMaximumHostCache(6L * 1024L * 1024L * 1024L)

            .setMaximumGridSize(512)

            .setMaximumBlockSize(512)

            // cross-device access is used for faster model averaging over pcie
            .allowCrossDeviceAccess(true);

        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);

        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(0.01)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation("identity")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn needed be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation("identity")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        // The builder needs the dimensions of the image along with the number of channels. these are 28x28 images in one channel
        new ConvolutionLayerSetup(builder,28,28,1);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        log.info("Train model....");
        //model.setListeners(new PerformanceListener(200, true));

        ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
            // DataSets prefetching options. Set this value with respect to number of actual devices
            .prefetchBuffer(24)

            // set number of workers equal or higher then number of available devices. x1-x2 are good values to start with
            .workers(2)

            // rare averaging improves performance, but might reduce model accuracy
            .averagingFrequency(3)

            // if set to TRUE, on every averaging model score will be reported
            .reportScoreAfterAveraging(true)

            // optinal parameter, set to fals ONLY if your system has support P2P memory access across PCIe (hint: AWS do not support P2P)
            .useLegacyAveraging(false)

            .build();

        for( int i=0; i<nEpochs; i++ ) {
            long time1 = System.currentTimeMillis();
            wrapper.fit(mnistTrain);
            long time2 = System.currentTimeMillis();
            log.info("*** Completed epoch {}, time elapsed: {} ***", i, (time2 - time1));
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        while(mnistTest.hasNext()){
            DataSet ds = mnistTest.next();
            INDArray output = model.output(ds.getFeatureMatrix(), false);
            eval.eval(ds.getLabels(), output);
        }
        log.info(eval.stats());
        mnistTest.reset();

        log.info("****************Example finished********************");
    }
}
