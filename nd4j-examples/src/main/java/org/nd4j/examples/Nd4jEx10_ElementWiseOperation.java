package org.nd4j.examples;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * --- Nd4j Example 10: Element Wise Operation ---
 *
 * In this example, we'll see ways to manipulate INDArray
 *
 * Created by cvn on 9/6/14.
 */


public class Nd4jEx10_ElementWiseOperation {

    public static void main(String[] args) {
        //Nd4j.dtype = DataBuffer.DOUBLE;
        INDArray nd1 = Nd4j.create(new double[]{1,2,3,4,5,6},new int[]{2,3});
        System.out.println("nd1:\n"+nd1);

        //create nd-array variable ndv to be able to print result of nondestructive operations. add scalar to matrix and assign ndv the sum.

        INDArray ndv = nd1.add(1);

        System.out.println("nd1.add(1):\n"+ndv);

        ndv = nd1.mul(5);
        System.out.println("nd1.mul(5):\n"+ndv);

        ndv = nd1.sub(3);
        System.out.println("nd1.sub(3):\n"+ndv);

        ndv = nd1.div(2);
        System.out.println("nd1.div(2):\n"+ndv);

        //add column vector to matrix

        INDArray nd2 = Nd4j.create(new double[]{10,20},new int[]{2,1}); //vector as column
        System.out.println("nd2:\n"+nd2);

        ndv = nd1.addColumnVector(nd2);

        System.out.println("nd1.addColumnVector(nd2):\n"+ndv);

        // add row vector to matrix

        INDArray nd3 = Nd4j.create(new double[]{30,40,50},new int[]{1, 3}); //vector as row
        System.out.println("nd3:\n"+nd3);

        ndv = nd1.addRowVector(nd3);

        System.out.println("nd1.addRowVector(nd3):\n"+ndv);

        //multiply two matrices of equal dimensions elementwise.

        INDArray nd4 = Nd4j.create(new double[]{1,2,1,2,1,2},new int[]{2,3});
        System.out.println("nd4:\n"+nd4);

        ndv = nd1.mul(nd4);

        System.out.println("nd1.mul(nd4):\n"+ndv);


    }

}
