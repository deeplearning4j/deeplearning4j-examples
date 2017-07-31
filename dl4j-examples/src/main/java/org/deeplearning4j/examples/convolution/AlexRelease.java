package org.deeplearning4j.examples.convolution;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.AsyncMultiDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;


/**
 * Created by raver119 on 10.07.17.
 */
@Slf4j
public class AlexRelease {

    public static void main(String... args) throws Exception {
        Nd4j.getMemoryManager().setAutoGcWindow(500);
        MultiDataSetIterator testDataAsync = getTestData();

        String currCheckpointPath = "./trainedNet_epoch14.bin";
        ComputationGraph net = ModelSerializer.restoreComputationGraph(new File(currCheckpointPath), false);
        net.getConfiguration().setInferenceWorkspaceMode(WorkspaceMode.SINGLE);
        net.getConfiguration().setTrainingWorkspaceMode(WorkspaceMode.SINGLE);
        //net.setCacheMode(CacheMode.NONE);
        log.info("Cache mode set to {}, inference workspace mode to {}", net.getConfiguration().getCacheMode(), net.getConfiguration().getInferenceWorkspaceMode());


        //Perform evaluation, if necessary - last epoch, or every N iterations
        log.info("Starting evaluation/inference: Cache mode set to {}, train workspace mode {}, inference workspace mode to {}",
            net.getConfiguration().getCacheMode(), net.getConfiguration().getTrainingWorkspaceMode(), net.getConfiguration().getInferenceWorkspaceMode());

        int nOutputs = net.getNumOutputArrays();
        Nd4j.getWorkspaceManager().printAllocationStatisticsForCurrentThread();
        Evaluation eval = new Evaluation();
        ROC roc = new ROC(0);
        net.doEvaluation(testDataAsync, eval, roc);
        /*
        while (testDataAsync.hasNext()) {
            testDataAsync.next();
        }
        */
        log.info("Finished");
    }


    public static AsyncMultiDataSetIterator getTestData(){
        return new AsyncMultiDataSetIterator(new TestIter());
    }


    private static class TestIter implements MultiDataSetIterator {

        private int[][] fShape = new int[][]{
            {8,182,8471},
            {8,182,8513},
            {8,182,8443},
            {8,182,8195},
            {8,182,7635},
            {8,182,8734},
            {8,182,8165},
            {8,182,8971},
            {8,182,8987},
            {8,182,8582},
            {8,182,7810},
            {8,182,8207},
            {8,182,8714},
            {8,182,7964},
            {8,182,7390},
            {8,182,7977},
            {8,182,6638},
            {8,182,8272},
            {8,182,9111},
            {8,182,8011},
            {8,182,8145},
            {8,182,7810},
            {8,182,8951},
            {8,182,7975},
            {8,182,7967},
            {8,182,7560},
            {8,182,9009},
            {8,182,7547},
            {8,182,8429},
            {8,182,8937},
            {8,182,8456},
            {8,182,6729},
            {8,182,7552},
            {8,182,8666},
            {8,182,8251},
            {8,182,7298},
            {8,182,7548},
            {8,182,8275},
            {8,182,7757},
            {8,182,7864},
            {8,182,8384},
            {8,182,8477},
            {8,182,7595},
            {8,182,8405},
            {4,182,4918}};

        private int[][] lShape = new int[][]{
            {8,2,8471},
            {8,2,8513},
            {8,2,8443},
            {8,2,8195},
            {8,2,7635},
            {8,2,8734},
            {8,2,8165},
            {8,2,8971},
            {8,2,8987},
            {8,2,8582},
            {8,2,7810},
            {8,2,8207},
            {8,2,8714},
            {8,2,7964},
            {8,2,7390},
            {8,2,7977},
            {8,2,6638},
            {8,2,8272},
            {8,2,9111},
            {8,2,8011},
            {8,2,8145},
            {8,2,7810},
            {8,2,8951},
            {8,2,7975},
            {8,2,7967},
            {8,2,7560},
            {8,2,9009},
            {8,2,7547},
            {8,2,8429},
            {8,2,8937},
            {8,2,8456},
            {8,2,6729},
            {8,2,7552},
            {8,2,8666},
            {8,2,8251},
            {8,2,7298},
            {8,2,7548},
            {8,2,8275},
            {8,2,7757},
            {8,2,7864},
            {8,2,8384},
            {8,2,8477},
            {8,2,7595},
            {8,2,8405},
            {4,2,4918}};

        private int pos = 0;

        @Override
        public MultiDataSet next(int num) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
            throw new UnsupportedOperationException();
        }

        @Override
        public MultiDataSetPreProcessor getPreProcessor() {
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
            pos = 0;
        }

        @Override
        public boolean hasNext() {
            return pos < fShape.length;
        }

        @Override
        public MultiDataSet next() {
            if(!hasNext()){
                throw new RuntimeException();
            }
            MultiDataSet mds = new org.nd4j.linalg.dataset.MultiDataSet(
                new INDArray[]{Nd4j.rand('f', fShape[pos])},
                new INDArray[]{Nd4j.rand('f', lShape[pos])});

            pos++;

            return mds;
        }
    }
}
