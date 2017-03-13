package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * --- Nd4j Example 13: Large Matrix ---
 *
 * In this example, we'll see operations with a large matrix
 *
 * @author Adam Gibson
 */

public class Nd4jEx13_LargeMatrices {

    private static Logger log = LoggerFactory.getLogger(Nd4jEx13_LargeMatrices.class);

    public static void main(String[] args) {
        INDArray n = Nd4j.linspace(1,10000000,10000000);
        System.out.println("MMUL" + n.mmul(n.transpose()));

    }

}
