package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * --- Nd4j Example 5: Accumulation/Reduction Operations ---
 *
 * In this example, we'll see ways to reduce INDArrays - for example, perform sum and max operations
 *
 * @author Alex Black
 */
public class Nd4jEx5_Accumulations {

    public static void main(String[] args){

        /*
        There are two types of accumulation/reduction operations:
        - Whole array operations                    ->  returns a scalar value
        - Operations along one or more dimensions   ->  returns an array

        Furthermore, there are two classes of accumulations:
        - Standard accumulations:   Accumulations that return a real-value - for example, min, max, sum, etc.
        - Index accumulations:      Accumulations that return an integer index - for example argmax

         */

        INDArray originalArray = Nd4j.linspace(1,15,15).reshape('c',3,5);       //As per example 3
        System.out.println("Original array: \n" + originalArray);

        //First, let's consider whole array reductions:
        double minValue = originalArray.minNumber().doubleValue();
        double maxValue = originalArray.maxNumber().doubleValue();
        double sum = originalArray.sumNumber().doubleValue();
        double avg = originalArray.meanNumber().doubleValue();
        double stdev = originalArray.stdNumber().doubleValue();

        System.out.println("minValue:       " + minValue);
        System.out.println("maxValue:       " + maxValue);
        System.out.println("sum:            " + sum);
        System.out.println("average:        " + avg);
        System.out.println("standard dev.:  " + stdev);


        //Second, let's perform the same along dimension 0.
        //In this case, the output is a [1,5] array; each output value is the min/max/mean etc of the corresponding column:
        INDArray minAlong0 = originalArray.min(0);
        INDArray maxAlong0 = originalArray.max(0);
        INDArray sumAlong0 = originalArray.sum(0);
        INDArray avgAlong0 = originalArray.mean(0);
        INDArray stdevAlong0 = originalArray.std(0);

        // Index Accumulation operations
        // IAMax returns index of max value along specified dimension
        INDArray idxOfMaxInEachColumn = Nd4j.getExecutioner().exec(new IAMax(originalArray),0);
        INDArray idxOfMaxInEachRow = Nd4j.getExecutioner().exec(new IAMax(originalArray),1);


        System.out.println("\n\n\n");
        System.out.println("min along dimension 0:  " + minAlong0);
        System.out.println("max along dimension 0:  " + maxAlong0);
        System.out.println("sum along dimension 0:  " + sumAlong0);
        System.out.println("avg along dimension 0:  " + avgAlong0);
        System.out.println("Index of max dimension 0:  " + idxOfMaxInEachColumn);
        System.out.println("Index of max dimension 1:  " + idxOfMaxInEachRow);


        //If we had instead performed these along dimension 1, we would instead get a [3,1] array out
        //In this case, each output value would be the reduction of the values in each column
        //Again, note that when this is printed it looks like a row vector, but is in facta column vector
        INDArray avgAlong1 = originalArray.mean(1);
        System.out.println("\n\navg along dimension 1:  " + avgAlong1);
        System.out.println("Shape of avg along d1:  " + Arrays.toString(avgAlong1.shape()));



        //Index accumulations return an integer value.
        INDArray argMaxAlongDim0 = Nd4j.argMax(originalArray,0);                            //Index of the max value, along dimension 0
        System.out.println("\n\nargmax along dimension 0:   " + argMaxAlongDim0);
        INDArray argMinAlongDim0 = Nd4j.getExecutioner().exec(new IMin(originalArray),0);   //Index of the min value, along dimension 0
        System.out.println("argmin along dimension 0:   " + argMinAlongDim0);




    }

}
