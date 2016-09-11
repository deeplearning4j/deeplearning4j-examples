package org.deeplearning4j.examples.convolution;

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
//import org.nd4j.jita.allocator.impl.AtomicAllocator;
//import org.nd4j.jita.concurrency.EventsProvider;
//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.cpu.nativecpu.ops.NativeOpExecutioner;
//import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
//import org.nd4j.linalg.jcublas.ops.executioner.CudaGridExecutioner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

//import org.nd4j.jita.conf.CudaEnvironment;
//import org.nd4j.jita.perf.OpDashboard;
import org.deeplearning4j.nn.conf.LearningRatePolicy;

/**
 * Created by agibsonccc on 9/16/15.
 */
public class LenetMnistExample {
    private static final Logger log = LoggerFactory.getLogger(LenetMnistExample.class);

    public static void main(String[] args) throws Exception {
/*
        CudaEnvironment.getInstance().getConfiguration()
            .allowMultiGPU(true)
            //.(true)
            .setMaximumGridSize(512)
            .setMaximumBlockSize(512)
            .setMinimumBlockSize(384)
            .setVerbose(false)
            .enableDebug(false);
        */

        int nChannels = 1;
        int outputNum = 10;
        int batchSize = 64;
        int nEpochs = 10;
        int iterations = 1;
        int seed = 123;

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
/*
        ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
            .averagingFrequency(1)
            .prefetchBuffer(12)
            .workers(2)
            .reportScoreAfterAveraging(false)
            .useLegacyAveraging(false)
            .build();
*/

        log.info("Train model....");
        model.setListeners(new PerformanceListener(10));

        //((NativeOpExecutioner) Nd4j.getExecutioner()).getLoop().setOmpNumThreads(8);

        long timeX = System.currentTimeMillis();
//        nEpochs = 2;
        for( int i=0; i<nEpochs; i++ ) {
            long time1 = System.currentTimeMillis();
            model.fit(mnistTrain);
            //wrapper.fit(mnistTrain);
            long time2 = System.currentTimeMillis();
            log.info("*** Completed epoch {}, Time elapsed: {} ***", i, (time2 - time1));
        }
        long timeY = System.currentTimeMillis();

        log.info("Training complete in: {} ms", (timeY - timeX));
        log.info("Model score: {}", model.score());
  /*
        log.info("New events: {}", AtomicAllocator.getInstance().getFlowController().getEventsProvider().getEventsNumber());
        log.info("Cached events: {}", AtomicAllocator.getInstance().getFlowController().getEventsProvider().getCachedNumber());

        if (Nd4j.getExecutioner() instanceof CudaGridExecutioner) {
            long meta = ((CudaGridExecutioner) Nd4j.getExecutioner()).getMetaCounter();
            long tots = ((CudaGridExecutioner) Nd4j.getExecutioner()).getExecutionCounter();
            log.info("Total metaOps launched: {}", meta);
            log.info("Total Ops launched: {}", tots);
        }

        OpDashboard.getInstance().printOutDashboard();
*/
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
