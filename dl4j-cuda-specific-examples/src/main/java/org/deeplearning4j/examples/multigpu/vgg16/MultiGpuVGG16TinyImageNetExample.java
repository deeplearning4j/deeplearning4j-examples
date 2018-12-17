package org.deeplearning4j.examples.multigpu.vgg16;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.fetchers.TinyImageNetFetcher;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold.AdaptiveThresholdAlgorithm;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;

import java.io.File;
import java.io.FilenameFilter;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;


/**
 * This is modified version of original LenetMnistExample, made compatible with multi-gpu environment
 * for the TinyImageNet dataset and VGG16.
 *
 * @author Justin Long (crockpotveggies)
 */
public class MultiGpuVGG16TinyImageNetExample {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(MultiGpuVGG16TinyImageNetExample.class);

    // for GPU you usually want to have higher batchSize
    public static int batchSize = 16;
    public static int nEpochs = 10;
    public static int seed = 123;

    public static void main(String[] args) throws Exception {

        VGG16 zooModel = VGG16.builder()
            .numClasses(TinyImageNetFetcher.NUM_LABELS)
            .seed(seed)
            .inputShape(new int[]{TinyImageNetFetcher.INPUT_CHANNELS, 224, 224})
            .updater(new AdaDelta())
            .build();
        ComputationGraph vgg16 = zooModel.init();
        vgg16.setListeners(new PerformanceListener(1, true));

        log.info("Loading data....");
        DataSetIterator trainIter = new TinyImageNetDataSetIterator(batchSize, new int[]{224,224}, DataSetType.TRAIN);
        DataSetIterator testIter = new TinyImageNetDataSetIterator(batchSize, new int[]{224,224}, DataSetType.TEST);

        log.info("Fitting normalizer...");
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        // ParallelWrapper will take care of load balancing between GPUs.
        ParallelWrapper wrapper = new ParallelWrapper.Builder(vgg16)
            // DataSets prefetching options. Set this value with respect to number of actual devices
            .prefetchBuffer(24)

            // set number of workers equal to number of available devices
            .workers(Nd4j.getAffinityManager().getNumberOfDevices())

            // use gradient sharing, a more effective distributed training method
            .trainingMode(ParallelWrapper.TrainingMode.SHARED_GRADIENTS)

            .thresholdAlgorithm(new AdaptiveThresholdAlgorithm())

            .build();

        log.info("Train model....");
        long timeX = System.currentTimeMillis();

        for( int i=0; i<nEpochs; i++ ) {
            long time = System.currentTimeMillis();
            wrapper.fit(trainIter);
            time = System.currentTimeMillis() - time;
            log.info("*** Completed epoch {}, time: {} ***", i, time);
        }
        long timeY = System.currentTimeMillis();

        log.info("*** Training complete, time: {} ***", (timeY - timeX));

        log.info("Evaluate model....");
        Evaluation eval = vgg16.evaluate(testIter);
        log.info(eval.stats());

        log.info("****************Example finished********************");
    }
}
