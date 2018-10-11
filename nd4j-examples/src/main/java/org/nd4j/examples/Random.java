package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Random {

    public static void main(String[] args) throws Exception {
        int[] shapes = new int[] {16, 32, 48, 64, 96, 128, 256, 512, 1024};

        for (int shapeR: shapes) {
            for (int shapeC: shapes) {
                long[] shape = new long[] {shapeR, shapeC};

                System.out.println("Trying N: {" + shapeR + ", " + shapeC + "}");
                INDArray arrayN = Nd4j.rand(shape);

                Nd4j.getExecutioner().commit();

                System.out.println("Trying U: {" + shapeR + ", " + shapeC + "}");
                INDArray arrayU = Nd4j.rand(shape);

                Nd4j.getExecutioner().commit();
            }
        }
    }
}
