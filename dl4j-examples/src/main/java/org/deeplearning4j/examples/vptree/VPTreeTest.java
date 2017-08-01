package org.deeplearning4j.examples.vptree;

import org.deeplearning4j.clustering.vptree.VPTree;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

public class VPTreeTest {
    private static Logger log = LoggerFactory.getLogger(VPTreeTest.class);

    public static void main(String...args) {
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
    }

}
