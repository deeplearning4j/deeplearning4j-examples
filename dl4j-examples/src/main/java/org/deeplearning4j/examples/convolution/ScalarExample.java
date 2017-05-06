package org.deeplearning4j.examples.convolution;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang.SerializationUtils;
import org.bytedeco.javacpp.Pointer;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by raver119 on 27.04.17.
 */
@Slf4j
public class ScalarExample {
    public static void main(String[] args) throws Exception {

        INDArray features = Nd4j.create(32, 3, 224, 224);
        INDArray labels = Nd4j.create(32, 200);
        File tmp = File.createTempFile("12dadsad","dsdasds");
        float[] array = new float[33 * 3 * 224 * 224];
        while (true) {
/*
            try (FileOutputStream fos = new FileOutputStream(tmp); BufferedOutputStream bos = new BufferedOutputStream(fos)) {
                SerializationUtils.serialize(array, fos);
            }

            try (FileInputStream fis = new FileInputStream(tmp); BufferedInputStream bis = new BufferedInputStream(fis)) {
                long time1 = System.currentTimeMillis();
                float[] arrayR = (float[]) SerializationUtils.deserialize(bis);
                long time2 = System.currentTimeMillis();

                log.info("Load time: {}", time2 - time1);
            }
*/

            DataSet ds = new DataSet(features, labels);
            ds.save(tmp);

            long time1 = System.currentTimeMillis();
            ds.load(tmp);
            long time2 = System.currentTimeMillis();

            log.info("Load time: {}", time2 - time1);
        }

        /*
        List<String> list = Arrays.asList("five and seven should be unavailable".split(" "));
        for (String str:list) {
            log.info("Word: [{}]", str);
        }

        Random rng = new Random();

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File("/home/raver119/Downloads/umdfaces_aligned_96_condensed/umdfaces_aligned_96_condensed/");
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        RandomPathFilter randomFilter = new RandomPathFilter(rng, NativeImageLoader.ALLOWED_FORMATS);

        ImageRecordReader trainRR = new ImageRecordReader(96, 96, 3, labelMaker, null);

        trainRR.initialize(fileSplit);

        RecordReaderDataSetIterator rrdsi = new RecordReaderDataSetIterator(trainRR, 72);
        while (true) {
            while (rrdsi.hasNext()) {
                long time1 = System.currentTimeMillis();
                rrdsi.next();
                long time2 = System.currentTimeMillis();

                log.info("Runtime: {} ms", time2 - time1);
            }
            rrdsi.reset();
        }
        */

        /*
        while (true) {
            INDArray array = Nd4j.create(128, 3, 224, 224);
            long time1 = System.currentTimeMillis();
            for (int i = 0; i < 128; i++) {
                fillNDArray(array.tensorAlongDimension(i, 1, 2, 3), i);
            }
            long time2 = System.currentTimeMillis();
            log.info("Compilation time: {} ms", time2 - time1);
        }*/
    }


    protected static void fillNDArray(INDArray view, double value) {
        Pointer pointer = view.data().pointer();
        int[] shape = view.shape();

        for (int c = 0; c < shape[0]; c++) {
            for (int h = 0; h < shape[1]; h++) {
                for (int w = 0; w < shape[2]; w++) {
                    view.putScalar(c, h, w, (float) value);
                }
            }
        }

        /*
        if (pointer instanceof FloatPointer) {
            FloatIndexer idx = FloatIndexer.create((FloatPointer) pointer, new long[]{view.shape()[0], view.shape()[1], view.shape()[2]}, new long[]{view.stride()[0], view.stride()[1], view.stride()[2]});
            for (long c = 0; c < shape[0]; c++) {
                for (long h = 0; h < shape[1]; h++) {
                    for (long w = 0; w < shape[2]; w++) {
                        idx.put(c, h, w, (float) value);
                    }
                }
            }
        } else {
            DoubleIndexer idx = DoubleIndexer.create((DoublePointer) pointer, new long[]{view.shape()[0], view.shape()[1], view.shape()[2]}, new long[]{view.stride()[0], view.stride()[1], view.stride()[2]});
            for (long c = 0; c < shape[0]; c++) {
                for (long h = 0; h < shape[1]; h++) {
                    for (long w = 0; w < shape[2]; w++) {
                        idx.put(c, h, w, value);
                    }
                }
            }
        }
        */
    }
}
