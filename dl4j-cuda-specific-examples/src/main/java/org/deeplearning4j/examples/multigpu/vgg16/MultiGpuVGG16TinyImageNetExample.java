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
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
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

    public static void main(String[] args) throws Exception {
        // temp workaround for backend initialization
        CudaEnvironment.getInstance().getConfiguration()
            // key option enabled
            .allowMultiGPU(true)
            // we're allowing larger memory caches
            .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)
            // cross-device access is used for faster model averaging over pcie
            .allowCrossDeviceAccess(true);

        int nChannels = 1;
        int outputNum = 10;

        // for GPU you usually want to have higher batchSize
        int batchSize = 16;
        int nEpochs = 10;
        int iterations = 1;
        int seed = 123;
        Random rng = new Random(seed);

        VGG16 zooModel = new VGG16(200, seed, 1);
        int[] inputShape = zooModel.metaData().getInputShape()[0];
        MultiLayerNetwork vgg16 = zooModel.init();
        vgg16.setListeners(new PerformanceListener(1, true));

        log.info("Load data....");
        String dataPath = "/home/justin/Datasets/tiny-imagenet-200/train/";
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        FileSplit fileSplit = new FileSplit(new File(dataPath), NativeImageLoader.ALLOWED_FORMATS, rng);
        RandomPathFilter randomFilter = new RandomPathFilter(rng, NativeImageLoader.ALLOWED_FORMATS);

      /*
       * Data Setup -> train test split
       */
        log.info("Splitting data for production....");
        InputSplit[] split = fileSplit.sample(randomFilter, 0.8, 0.2);
        InputSplit trainData = split[0];
        InputSplit testData = split[1];
        log.info("Total training images is " + trainData.length());
        log.info("Total test images is " + testData.length());

        log.info("Calculating labels...");
        File file = new File(dataPath);
        String[] directories = file.list(new FilenameFilter() {
            @Override
            public boolean accept(File current, String name) {
                return new File(current, name).isDirectory();
            }
        });

        log.info("Initializing RecordReader and pipelines....");
        List<Pair<ImageTransform, Double>> pipeline = new LinkedList<>();
        pipeline.add(new Pair<>(new ResizeImageTransform(inputShape[1], inputShape[2]), 1.0));
        pipeline.add(new Pair<>(new FlipImageTransform(1), 0.5));
        ImageTransform combinedTransform = new PipelineImageTransform(pipeline, false);

        ImageRecordReader trainRR = new ImageRecordReader(inputShape[1], inputShape[2], inputShape[0], combinedTransform);
        trainRR.setLabels(Arrays.asList(directories));
        trainRR.initialize(trainData);
        ImageRecordReader testRR = new ImageRecordReader(inputShape[1], inputShape[2], inputShape[0], combinedTransform );
        testRR.setLabels(Arrays.asList(directories));
        testRR.initialize(testData);

        log.info("Total dataset labels: "+ directories.length);
        log.info("Total training labels: " + trainRR.getLabels().size());
        log.info("Total test labels: " + testRR.getLabels().size());

        log.info("Creating RecordReader iterator....");
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, 200);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, 200);

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

            .build();

        log.info("Train model....");
        long timeX = System.currentTimeMillis();

        // optionally you might want to use MultipleEpochsIterator instead of manually iterating/resetting over your iterator
        //MultipleEpochsIterator mnistMultiEpochIterator = new MultipleEpochsIterator(nEpochs, mnistTrain);

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
