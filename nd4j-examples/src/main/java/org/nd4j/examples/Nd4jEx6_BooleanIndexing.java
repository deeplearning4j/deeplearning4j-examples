package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 * --- Nd4j Example 6: Boolean Indexing ---
 *
 * In this example, we'll see ways to use boolean indexing to perform some simple conditional element-wise operations
 *
 * @author Alex Black
 */
public class Nd4jEx6_BooleanIndexing {

    public static void main(String[] args){

        int nRows = 3;
        int nCols = 5;
        long rngSeed = 12345;
        //Generate random numbers between -1 and +1
        INDArray random = Nd4j.rand(nRows, nCols, rngSeed).muli(2).subi(1);

        System.out.println("Array values:");
        System.out.println(random);

        //For example, we can conditionally replace values less than 0.0 with 0.0:
        INDArray randomCopy = random.dup();
        BooleanIndexing.replaceWhere(randomCopy, 0.0, Conditions.lessThan(0.0));
        System.out.println("After conditionally replacing negative values:\n" + randomCopy);

        //Or conditionally replace NaN values:
        INDArray hasNaNs = Nd4j.create(new double[]{1.0,1.0,Double.NaN,1.0});
        BooleanIndexing.replaceWhere(hasNaNs,0.0, Conditions.isNan());
        System.out.println("hasNaNs after replacing NaNs with 0.0:\n" + hasNaNs);

        //Or we can conditionally copy values from one array to another:
        randomCopy = random.dup();
        INDArray tens = Nd4j.valueArrayOf(nRows, nCols, 10.0);
        BooleanIndexing.replaceWhere(randomCopy, tens, Conditions.lessThan(0.0));
        System.out.println("Conditionally copying values from array 'tens', if original value is less than 0.0\n" + randomCopy);


        //One simple task is to count the number of values that match the condition
        MatchCondition op = new MatchCondition(random, Conditions.greaterThan(0.0));
        int countGreaterThanZero = Nd4j.getExecutioner().exec(op,Integer.MAX_VALUE).getInt(0);  //MAX_VALUE = "along all dimensions" or equivalently "for entire array"
        System.out.println("Number of values matching condition 'greater than 0': " + countGreaterThanZero);
    }

}
