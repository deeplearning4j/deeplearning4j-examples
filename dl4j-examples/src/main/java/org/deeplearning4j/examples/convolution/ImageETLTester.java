package org.deeplearning4j.examples.convolution;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by raver119 on 19.04.17.
 */
@Slf4j
public class ImageETLTester {
    protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    public static final Random randNumGen = new Random(119);

    public static void main(String[] args) throws Exception {
        File parentDir = new File("/home/raver119/Downloads/umdfaces_aligned_96_condensed/umdfaces_aligned_96_condensed");
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        //BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
        RandomPathFilter pathFilter = new RandomPathFilter(randNumGen, allowedExtensions);

        //Split the image files into train and test. Specify the train test split as 80%,20%
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        ImageRecordReader recordReader = new ImageRecordReader(96, 96, 3,labelMaker);
        recordReader.initialize(trainData);

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 10, 1, 244);

        List<Long> nanos = new ArrayList<>();

        for (int i = 0; i < 10; i++) {
            while (dataIter.hasNext()) {
                dataIter.next();
            }
            dataIter.reset();
            log.info("warm-up iteration {} passed...", i);
        }


        while (dataIter.hasNext()) {
            long time1 = System.nanoTime();
            dataIter.next();
            long time2 = System.nanoTime();

            nanos.add(time2 - time1);
        }

        Collections.sort(nanos);
        log.info("p50: {}", nanos.get(nanos.size() / 2));
    }
}
