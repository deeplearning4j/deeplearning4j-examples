package org.deeplearning4j.tinyimagenet;

import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

public class ParallelWrapperVersion {

    public static void main(String[] args) throws Exception {

        //Create the data pipeline
        int batchSize = 32;
        DataSetIterator iter = new TinyImageNetDataSetIterator(batchSize);
        iter.setPreProcessor(new ImagePreProcessingScaler());   //Scale 0-255 valued pixels to 0-1 range

        //Create the network
        ComputationGraph net = TrainSpark.getNetwork();


        ParallelWrapper wrapper = new ParallelWrapper.Builder(net)
            .prefetchBuffer(4)
            .workers(2)
            .averagingFrequency(1)
            .reportScoreAfterAveraging(true)
            .build();

        wrapper.setListeners(new PerformanceListener(1, true));

        Nd4j.getMemoryManager().setAutoGcWindow(10000);

        wrapper.fit(iter);

    }

}
