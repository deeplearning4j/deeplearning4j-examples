package org.deeplearning4j.examples.convolution;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.berkeley.Pair;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.RandomCropTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.parallelism.MagicQueue;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class JustinExample {

    public static void main(String[] args) throws Exception {
        Random rng = new Random(119);
        int batchSize = 32;
        int inputDimension = 96;
        int resizeDimension = 96;

        log.info("Load data....");
        /**cd
         * Data Setup -> organize and limit data file paths:
         *  - mainPath = path to image files
         *  - fileSplit = define basic dataset split with limits on format
         *  - pathFilter = define additional file load filter to limit size and balance batch content
         **/
        log.info("Loading paths....");
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File("/Users/raver119/Downloads/");
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        RandomPathFilter randomFilter = new RandomPathFilter(rng, NativeImageLoader.ALLOWED_FORMATS);

        /**
         * Data Setup -> train test split
         *  - inputSplit = define train and test split
         **/
        InputSplit[] split;
        log.info("Splitting data for debugging....");
        split = fileSplit.sample(randomFilter, 0.998, 0.002);
        InputSplit trainData = split[0];
        InputSplit testData = split[1];
        log.info("Total training images is "+trainData.length());
        log.info("Total test images is "+testData.length());

        // step 1
        log.info("Initializing RecordReader and pipelines....");
        List<Pair<ImageTransform, Double>> pipeline = new LinkedList<>();
        pipeline.add(new Pair<>(new RandomCropTransform(resizeDimension,resizeDimension), 1.0));
        pipeline.add(new Pair<>(new FlipImageTransform(1), 0.5));
        ImageTransform combinedTransform = new ProbabilisticPipelineTransform(pipeline, false);

        ImageRecordReader trainRR = new ImageRecordReader(inputDimension, inputDimension, 3, labelMaker, combinedTransform);
        trainRR.initialize(trainData);
        ImageRecordReader testRR = new ImageRecordReader(inputDimension, inputDimension, 3, labelMaker, combinedTransform);
        testRR.setLabels(trainRR.getLabels());
        testRR.initialize(testData);

        int numClasses = trainRR.getLabels().size();
        int testClasses = testRR.getLabels().size();
        log.info("Total training labels: "+numClasses);
        log.info("Total test labels: "+testClasses);

        log.info("Creating RecordReader iterator....");
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, numClasses);

        MagicQueue q = new MagicQueue.Builder().setNumberOfBuckets(2).setCapacityPerFlow(4).setMode(MagicQueue.Mode.SEQUENTIAL).setType(MagicQueue.Type.DS).build();

        DataSetIterator adsi = new AsyncDataSetIterator(trainIter, 8, q, true);

        AtomicLong cnt = new AtomicLong(0);
        while (true) {
            while (adsi.hasNext()) {
                DataSet ds = adsi.next();
            }

            adsi.reset();

            if (cnt.incrementAndGet() % 100 == 0)
                log.info("{} iterations passed...", cnt.get());
        }

    }
}
