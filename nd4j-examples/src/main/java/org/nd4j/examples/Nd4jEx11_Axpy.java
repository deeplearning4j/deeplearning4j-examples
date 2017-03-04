package org.nd4j.examples;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * --- Nd4j Example 11: Axpy ---
 * *
 * @author Adam Gibson
 */
public class Nd4jEx11_Axpy {

    public static void main(String[] args) {

        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        INDArray arr = Nd4j.create(300);
        double numTimes = 10000000;
        double total = 0;

        for(int i = 0; i < numTimes; i++) {
            long start = System.nanoTime();
            Nd4j.getBlasWrapper().axpy(new Integer(1), arr,arr);
            long after = System.nanoTime();
            long add = Math.abs(after - start);
            System.out.println("Took " + add);
            total += Math.abs(after - start);
        }
        System.out.println("Avg time " + (total / numTimes));
    }
}
