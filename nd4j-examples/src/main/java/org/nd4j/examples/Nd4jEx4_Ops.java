package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sin;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

/**
 * --- Nd4j Example 4: Additional Operations with INDArrays ---
 *
 * In this example, we'll see ways to manipulate INDArray
 *
 * @author Alex Black
 */
public class Nd4jEx4_Ops {

    public static void main(String[] args){

        /*
        ND4J defines a wide variety of operations. Here we'll see how to use some of them:
        - Elementwise operations:   add, multiply, divide, subtract, etc
            add, mul, div, sub,
            INDArray.add(INDArray), INDArray.mul(INDArray), etc
        - Matrix multiplication:    mmul
        - Row/column vector ops:    addRowVector, mulColumnVector, etc
        - Element-wise transforms, like tanh, scalar max operations, etc
         */

        //First, let's see how in-place vs. copy operations work
        //Consider the calls:   myArray.add(1.0)    vs  myArray.addi(1.0)
        // i.e., "add" vs. "addi"   ->  the "i" means in-place.
        //In practice: the in-place ops modify the original array; the others ("copy ops") make a copy
        INDArray originalArray = Nd4j.linspace(1,15,15).reshape('c',3,5);       //As per example 3
        INDArray copyAdd = originalArray.add(1.0);
        System.out.println("Same object returned by add:    " + (originalArray == copyAdd));
        System.out.println("Original array after originalArray.add(1.0):\n" + originalArray);
        System.out.println("copyAdd array:\n" + copyAdd);

            //Let's do the same thing with the in-place add operation:
        INDArray inPlaceAdd = originalArray.addi(1.0);
        System.out.println();
        System.out.println("Same object returned by addi:    " + (originalArray == inPlaceAdd));    //addi returns the exact same Java object
        System.out.println("Original array after originalArray.addi(1.0):\n" + originalArray);
        System.out.println("inPlaceAdd array:\n" + copyAdd);


        //Let's recreate our our original array for the next section, and create another one:
        originalArray = Nd4j.linspace(1,15,15).reshape('c',3,5);
        INDArray random = Nd4j.rand(3,5);               //See example 2; we have a 3x5 with uniform random (0 to 1) values



        //We can perform element-wise operations. Note that the array shapes must match here
        // add vs. addi works in exactly the same way as for scalars
        INDArray added = originalArray.add(random);
        System.out.println("\n\n\nRandom values:\n" + random);
        System.out.println("Original plus random values:\n" + added);


        //Matrix multiplication is easy:
        INDArray first = Nd4j.rand(3,4);
        INDArray second = Nd4j.rand(4,5);
        INDArray mmul = first.mmul(second);
        System.out.println("\n\n\nShape of mmul array:      " + Arrays.toString(mmul.shape()));     //3x5 output as expected


        //We can do row-wise ("for each row") and column-wise ("for each column") operations
        //Again, inplace vs. copy ops work the same way (i.e., addRowVector vs. addiRowVector)
        INDArray row = Nd4j.linspace(0,4,5);
        System.out.println("\n\n\nRow:\n" + row);
        INDArray mulRowVector = originalArray.mulRowVector(row);        //For each row in 'originalArray', do an element-wise multiplication with the row vector
        System.out.println("Result of originalArray.mulRowVector(row)");
        System.out.println(mulRowVector);


        //Element-wise transforms are things like 'tanh' and scalar max values. These can be applied in a few ways:
        System.out.println("\n\n\n");
        System.out.println("Random array:\n" + random);     //Again, note the limited printing precision, as per example 2
        System.out.println("Element-wise tanh on random array:\n" + Transforms.tanh(random));
        System.out.println("Element-wise power (x^3.0) on random array:\n" + Transforms.pow(random,3.0));
        System.out.println("Element-wise scalar max (with scalar 0.5):\n" + Transforms.max(random,0.5));
            //We can perform this in a more verbose way, too:
        INDArray sinx = Nd4j.getExecutioner().execAndReturn(new Sin(random.dup()));
        System.out.println("Element-wise sin(x) operation:\n" + sinx);
    }
}
