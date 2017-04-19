package org.deeplearning4j.examples.convolution;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.AsyncMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncShieldMultiDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.NoSuchElementException;
import java.util.Random;

/**
 * Created by Alex on 15/04/2017.
 */
@Slf4j
public class AlexMDSI {

    public static void main(String[] args){
        int inputSize = 32;
        int layerSize = 128;
        int vectorSize = 128;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .workspaceMode(WorkspaceMode.SEPARATE)
//                .workspaceMode(WorkspaceMode.NONE)
            .updater(Updater.ADAM).adamMeanDecay(0.9).adamVarDecay(0.999)
            .weightInit(WeightInit.XAVIER)
            .regularization(true).l2(0.01)
            .learningRate(0.05)
            .graphBuilder()
            .addInputs("in")
            .addLayer("blstm1", new GravesBidirectionalLSTM.Builder()
                .nIn(inputSize).nOut(layerSize)
                .activation(Activation.TANH)
                .build(), "in")
            .addLayer("blstm2", new GravesBidirectionalLSTM.Builder()
                .nIn(layerSize).nOut(layerSize)
                .activation(Activation.TANH)
                .build(), "blstm1")
            .addLayer("poolMax", new GlobalPoolingLayer.Builder(PoolingType.MAX).build(), "blstm2")
            .addLayer("poolAvg", new GlobalPoolingLayer.Builder(PoolingType.AVG).build(), "blstm2")
            .addLayer("dense", new DenseLayer.Builder()
                .nIn(2*layerSize).nOut(vectorSize)
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .build(), "poolMax", "poolAvg")
            .addLayer("out", new OutputLayer.Builder()
                .nIn(vectorSize).nOut(10)
                .activation(Activation.SIGMOID)
                .lossFunction(LossFunctions.LossFunction.XENT)
                .epsilon(1e-6)
                .l2Bias(1e-2)
                .build(), "dense")
            .setOutputs("out")
            .build();

        ComputationGraph g = new ComputationGraph(conf);
        g.init();
        g.setListeners(new PerformanceListener(1, true));

        int minibatch = 48;
        MultiDataSetIterator mds = new RandomMultiDataSetIterator(
            minibatch, 512, new int[]{minibatch,inputSize, 16},
            new int[]{minibatch,inputSize, 256},
            new int[]{minibatch, 10},
            true, true, 0, 0, 12345);

        Nd4j.getMemoryManager().setAutoGcWindow(2000);

        MultiDataSet ds = mds.next();

        log.info("DS size: {}", ds.getMemoryFootprint());

        Nd4j.getMemoryManager().setAutoGcWindow(1000000);

        // ParallelWrapper will take care of load balancing between GPUs.
        ParallelWrapper wrapper = new ParallelWrapper.Builder(g)
            // DataSets prefetching options. Set this value with respect to number of actual devices
            .prefetchBuffer(10)

            // set number of workers equal or higher then number of available devices. x1-x2 are good values to start with
            .workers(2)

            // rare averaging improves performance, but might reduce model accuracy
            .averagingFrequency(5)

            // if set to TRUE, on every averaging model score will be reported
            .reportScoreAfterAveraging(false)

            // optinal parameter, set to false ONLY if your system has support P2P memory access across PCIe (hint: AWS do not support P2P)
            .useLegacyAveraging(false)

            .workspaceMode(WorkspaceMode.SEPARATE)

            .useMQ(true)

            .build();


        //wrapper.fit(mds);
        //g.fit(mds);
        g.fit(new AsyncMultiDataSetIterator(mds, 20, true));
        //g.fit(new AsyncShieldMultiDataSetIterator(mds));

    }

    public static class RandomMultiDataSetIterator implements MultiDataSetIterator {


        private final int minibatch;
        private final int maxMinibatches;
        private final int[] featuresShapeMin;
        private final int[] featuresShapeMax;
        private final int[] labelsShape;
        private final boolean featuresTimeSeriesMask;
        private final boolean labelsTimeSeriesMask;
        private final long delayMsMin;
        private final long delayMsMax;
        private final long rngSeed;

        private int cursor;
        private Random jRng;

        public RandomMultiDataSetIterator(int minibatch, int maxMinibatches, int[] featuresShapeMin,
                                          int[] featuresShapeMax, int[] labelsShape,
                                          boolean featuresTimeSeriesMask, boolean labelsTimeSeriesMask,
                                          long delayMsMin, long delayMsMax, long rngSeed) {
            this.minibatch = minibatch;
            this.maxMinibatches = maxMinibatches;
            this.featuresShapeMin = featuresShapeMin;
            this.featuresShapeMax = featuresShapeMax;
            this.labelsShape = labelsShape;
            this.featuresTimeSeriesMask = featuresTimeSeriesMask;
            this.labelsTimeSeriesMask = labelsTimeSeriesMask;
            this.delayMsMin = delayMsMin;
            this.delayMsMax = delayMsMax;
            this.rngSeed = rngSeed;
            this.jRng = new Random(rngSeed);
        }


        @Override
        public MultiDataSet next(int num) {
            if(!hasNext()) throw new NoSuchElementException();

            long start = System.currentTimeMillis();

            int[] fShape = new int[featuresShapeMin.length];
            for( int i=0; i<fShape.length; i++ ){
                if(featuresShapeMin[i] == featuresShapeMax[i]){
                    fShape[i] = featuresShapeMin[i];
                } else {
                    fShape[i] = featuresShapeMin[i] + jRng.nextInt(featuresShapeMax[i]-featuresShapeMin[i]);
                }
            }

            INDArray f = Nd4j.rand(fShape);
            INDArray l = Nd4j.rand(labelsShape);
            INDArray fm = null;
            INDArray lm = null;

//            log.info("Features shape: " + Arrays.toString(fShape));

            if(featuresTimeSeriesMask){
                int length = fShape[2];
                fm = Nd4j.zeros(minibatch, length);
                for( int i=0; i<minibatch; i++ ){
                    int thisLength = jRng.nextInt(length);
                    thisLength = Math.max(1, thisLength);
                    fm.get(NDArrayIndex.point(i), NDArrayIndex.interval(0,thisLength)).assign(1);
                }
            }

            if(labelsTimeSeriesMask){
                //TODO do this properly
                lm = Nd4j.getExecutioner().exec(new BernoulliDistribution(
                    Nd4j.createUninitialized(l.shape(), l.ordering()), 0.5));
            }

            if(delayMsMin > 0){
                long sleepTime;
                if(delayMsMin == delayMsMax){
                    sleepTime = delayMsMin - (System.currentTimeMillis() - start);
                } else {
                    long r = jRng.nextInt((int)(delayMsMax-delayMsMin));
                    sleepTime = delayMsMin - (System.currentTimeMillis() - start) + r;
                }

                if(sleepTime > 0){
                    try{
                        Thread.sleep(sleepTime);
                    } catch (InterruptedException e){
                        throw new RuntimeException(e);
                    }
                }
            }

            return new org.nd4j.linalg.dataset.MultiDataSet(
                new INDArray[]{f},
                new INDArray[]{l},
                (fm != null ? new INDArray[]{fm} : null),
                (lm != null ? new INDArray[]{lm} : null));
        }

        @Override
        public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
            throw new UnsupportedOperationException();
        }

        @Override
        public boolean resetSupported() {
            return true;
        }

        @Override
        public boolean asyncSupported() {
            return true;
        }

        @Override
        public void reset() {
            cursor = 0;
        }

        @Override
        public boolean hasNext() {
            return cursor < maxMinibatches;
        }

        @Override
        public MultiDataSet next() {
            return next(minibatch);
        }
    }

}
