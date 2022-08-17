package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

/**
 * --- Nd4j Example 3: Getting and setting parts of INDArrays ---
 *
 * In this example, we'll see ways to obtain and manipulate subsets of INDArray
 *
 * @author Alex Black
 */
public class Nd4jEx3_GettingAndSettingSubsets {

    public static void main(String[] args){

        //Let's start by creating a 3x5 INDArray with manually specified values
        // To do this, we are starting with a 1x15 array, and perform a 'reshape' operation to convert it to a 3x5 INDArray
        INDArray originalArray = Nd4j.linspace(1,15,15).reshape('c',3,5);
        System.out.println("Original Array:");
        System.out.println(originalArray);

        //We can use getRow and getColumn operations to get a row or column respectively:
        INDArray firstRow = originalArray.getRow(0);
        INDArray lastColumn = originalArray.getColumn(4);
        System.out.println();
        System.out.println("First row:\n" + firstRow);
        System.out.println("Last column:\n" + lastColumn);
        //Careful of the printing here: lastColumn looks like a row vector when printed, but it's really a column vector
        System.out.println("Shapes:         " + Arrays.toString(firstRow.shape()) + "\t" + Arrays.toString(lastColumn.shape()));

        //A key concept in ND4J is the idea of views: one INDArray may point to the same locations in memory as other arrays
        //For example, getRow and getColumn are both views of originalArray
        //Consequently, changes to one results in changes to the other:
        firstRow.addi(1.0);             //In-place addition operation: changes the values of both firstRow AND originalArray:
        System.out.println("\n\n");
        System.out.println("firstRow, after addi operation:");
        System.out.println(firstRow);
        System.out.println("originalArray, after firstRow.addi(1.0) operation: (note it is modified, as firstRow is a view of originalArray)");
        System.out.println(originalArray);



        //Let's recreate our our original array for the next section...
        originalArray = Nd4j.linspace(1,15,15).reshape('c',3,5);


        //We can select arbitrary subsets, using INDArray indexing:
        //All rows, first 3 columns (note that internal here is columns 0 inclusive to 3 exclusive)
        INDArray first3Columns = originalArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0,3));
        System.out.println("first 3 columns:\n" + first3Columns);
        //Again, this is also a view:
        first3Columns.addi(100);
        System.out.println("originalArray, after first3Columns.addi(100)");
        System.out.println(originalArray);



        //Let's recreate our our original array for the next section...
        originalArray = Nd4j.linspace(1,15,15).reshape('c',3,5);


        //We can similarly set arbitrary subsets.
        //Let's set the 3rd column (index 2) to zeros:
        INDArray zerosColumn = Nd4j.zeros(3,1);
        originalArray.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(2)}, zerosColumn);     //All rows, column index 2
        System.out.println("\n\n\nOriginal array, after put operation:\n" + originalArray);



        //Let's recreate our our original array for the next section...
        originalArray = Nd4j.linspace(1,15,15).reshape('c',3,5);


        //Sometimes, we don't want this in-place behaviour. In this case: just add a .dup() operation at the end
        //the .dup() operation - aka 'duplicate' - creates a new and separate array
        INDArray firstRowDup = originalArray.getRow(0).dup();   //We now have a copy of the first row. i.e., firstRowDup is NOT a view of originalArray
        firstRowDup.addi(100);
        System.out.println("\n\n\n");
        System.out.println("firstRowDup, after .addi(100):\n" + firstRowDup);
        System.out.println("originalArray, after firstRowDup.addi(100): (note it is unmodified)\n" + originalArray);
    }
}
