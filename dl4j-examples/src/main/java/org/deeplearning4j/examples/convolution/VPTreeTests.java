package org.deeplearning4j.examples.convolution;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class VPTreeTests {

    public static void main(String[] args) throws Exception {

        log.info("max physical bytes: {}", System.getProperty("org.bytedeco.javacpp.maxphysicalbytes"));
        log.info("max bytes: {}", System.getProperty("org.bytedeco.javacpp.maxbytes"));

        INDArray matrix = Nd4j.rand(3700000, 500);
/*
        for (int r = 0; r < matrix.rows(); r++) {
            matrix.getRow(r).assign(r);
        }
*/
        log.info("Starting VPTree creation... {}", Arrays.toString(matrix.shape()));

        long time1 = System.currentTimeMillis();
        VPTree vpTree = new VPTree(matrix, false, 1);
        long time2 = System.currentTimeMillis();

        log.info("VPTree created: {} ms", time2 - time1);

        List<DataPoint> points = new ArrayList<>();
        List<Double> distances = new ArrayList<>();
/*
        vpTree.search(Nd4j.create(10).assign(matrix.rows() - 1), 5, points, distances);

        for (int k = 0; k < points.size(); k++) {
            log.info("Point: {}; Distance: {}", points.get(k), distances.get(k));
        }
*/
        log.info("Sleeping...");

        int cnt = 0;
        while (1 > cnt) {
            Thread.sleep(1000);
        }

        vpTree.getItems();
    }
}
