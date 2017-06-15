package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


/**
 * --- Nd4j Example 7: Matrix Operation ---
 *
 * In this example, we'll see ways to multiply matrices
 *
 * Created by cvn on 9/7/14.
 */

public class Nd4jEx7_MatrixOperation {

    public static void main(String[] args) {

        //See other examples for creating INDArrays from Java arrays
        //See also http://nd4j.org/userguide
        INDArray nd = Nd4j.create(new float[]{1,2},new int[]{1, 2});        //vector as row
        INDArray nd2 = Nd4j.create(new float[]{3,4},new int[]{2, 1});       //vector as column
        INDArray nd3 = Nd4j.create(new float[][]{{1,2},{3,4}});
        INDArray nd4 = Nd4j.create(new float[][]{{3,4},{5,6}});

        System.out.println(nd);
        System.out.println(nd2);
        System.out.println(nd3);


        System.out.println("Creating nd array with data type " + Nd4j.dataType());
        //create nd-array variable to show result of nondestructive operations. matrix multiply row vector by column vector to obtain dot product.
        //assign product to nd-array variable.

        INDArray ndv = nd.mmul(nd2);

        System.out.println(ndv);

        //multiply a row by a 2 x 2 matrix

        ndv = nd.mmul(nd4);
        System.out.println(ndv);

        //multiply two 2 x 2 matrices

        ndv = nd3.mmul(nd4);
        System.out.println(ndv);

        //now switch the position of the matrices in the equation to obtain different result. matrix multiplication is not commutative.

        ndv = nd4.mmul(nd3);
        System.out.println(ndv);

        // switch the row and column vector to obtain the outer product

        ndv = nd2.mmul(nd);
        System.out.println(ndv);

    }
}
