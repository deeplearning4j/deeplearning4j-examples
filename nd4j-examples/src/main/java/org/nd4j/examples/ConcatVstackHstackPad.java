package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * --- ND4J Example 1: INDArray Concat, HStack, VStack, and Pad ---
 *
 * In this example, we'll see some basic operations for INDArrays
 *
 * @author Tom Hanlon
 */
public class ConcatVstackHstackPad {

    public static void main(String[] args) {

        /*
        Before we begin, let's review what an INDArray is:
        A INDArray is a multi-dimensional array of numbers: a vector, matrix, or tensor for example.
        Internally, it may store single precision or double precision floating point values for each entry.

        Here, we'll see how you can get some basic information about INDArrays. In later examples, we'll see
        the different ways to create INDArrays, and more operations we can do on them.
         */

        //Let's start by creating a basic 2d array: a matrix with 3 rows and 5 columns. All elements are 0.0
        int nRows = 2;
        int nColumns = 2;
        INDArray zeros = Nd4j.zeros(nRows, nColumns);

        // Create one of all ones

        INDArray ones = Nd4j.ones(nRows, nColumns);

        System.out.println("#### zeros ####");
        System.out.println(zeros);
        System.out.println("### ONES ####");
        System.out.println(ones);

        INDArray combined = Nd4j.concat(0,zeros,ones);

        System.out.println("### COMBINED dimension 0####");
        System.out.println(combined);


        INDArray combined2 = Nd4j.concat(1,zeros,ones);

        System.out.println("### COMBINED dimension 1 ####");
        System.out.println(combined2);

        //Padding
        INDArray padded = Nd4j.pad(ones, new int[]{1,1}, Nd4j.PadMode.CONSTANT );
        System.out.println("### Padded ####");
        System.out.println(padded);

        //hstack

        INDArray hstack = Nd4j.hstack(ones,zeros);
        System.out.println("### HSTACK ####");
        System.out.println(hstack);

        //vstack
        INDArray vstack = Nd4j.vstack(ones,zeros);
        System.out.println("### VSTACK ####");
        System.out.println(vstack);



    }

}
