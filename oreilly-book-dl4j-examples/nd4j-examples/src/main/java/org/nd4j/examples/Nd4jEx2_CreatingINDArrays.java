package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * --- Nd4j Example 2: Creating INDArrays ---
 *
 * In this example, we'll see a number of different ways to create INDArrays
 *
 * @author Alex Black
 */
public class Nd4jEx2_CreatingINDArrays {

    public static void main(String[] args){

        //Here, we'll see how to create INDArrays with different scalar value initializations
        int nRows = 3;
        int nColumns = 5;
        INDArray allZeros = Nd4j.zeros(nRows, nColumns);
        System.out.println("Nd4j.zeros(nRows, nColumns)");
        System.out.println(allZeros);

        INDArray allOnes = Nd4j.ones(nRows, nColumns);
        System.out.println("\nNd4j.ones(nRows, nColumns)");
        System.out.println(allOnes);

        INDArray allTens = Nd4j.valueArrayOf(nRows, nColumns, 10.0);
        System.out.println("\nNd4j.valueArrayOf(nRows, nColumns, 10.0)");
        System.out.println(allTens);



        //We can also create INDArrays from double[] and double[][] (or, float/int etc Java arrays)
        double[] vectorDouble = new double[]{1,2,3};
        INDArray rowVector = Nd4j.create(vectorDouble);
        System.out.println("rowVector:              " + rowVector);
        System.out.println("rowVector.shape():      " + Arrays.toString(rowVector.shape()));    //1 row, 3 columns

        INDArray columnVector = Nd4j.create(vectorDouble, new int[]{3,1});  //Manually specify: 3 rows, 1 column
        System.out.println("columnVector:           " + columnVector);      //Note for printing: row/column vectors are printed as one line
        System.out.println("columnVector.shape():   " + Arrays.toString(columnVector.shape()));    //3 row, 1 columns

        double[][] matrixDouble = new double[][]{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}};
        INDArray matrix = Nd4j.create(matrixDouble);
        System.out.println("\nINDArray defined from double[][]:");
        System.out.println(matrix);



        //It is also possible to create random INDArrays:
        //Be aware however that by default, random values are printed with truncated precision using INDArray.toString()
        int[] shape = new int[]{nRows, nColumns};
        INDArray uniformRandom = Nd4j.rand(shape);
        System.out.println("\n\n\nUniform random array:");
        System.out.println(uniformRandom);
        System.out.println("Full precision of random value at position (0,0): " + uniformRandom.getDouble(0,0));

        INDArray gaussianMeanZeroUnitVariance = Nd4j.randn(shape);
        System.out.println("\nN(0,1) random array:");
        System.out.println(gaussianMeanZeroUnitVariance);

        //We can make things repeatable using RNG seed:
        long rngSeed = 12345;
        INDArray uniformRandom2 = Nd4j.rand(shape, rngSeed);
        INDArray uniformRandom3 = Nd4j.rand(shape, rngSeed);
        System.out.println("\nUniform random arrays with same fixed seed:");
        System.out.println(uniformRandom2);
        System.out.println();
        System.out.println(uniformRandom3);



        //Of course, we aren't restricted to 2d. 3d or higher is easy:
        INDArray threeDimArray = Nd4j.ones(3,4,5);      //3x4x5 INDArray
        INDArray fourDimArray = Nd4j.ones(3,4,5,6);     //3x4x5x6 INDArray
        INDArray fiveDimArray = Nd4j.ones(3,4,5,6,7);   //3x4x5x6x7 INDArray
        System.out.println("\n\n\nCreating INDArrays with more dimensions:");
        System.out.println("3d array shape:         " + Arrays.toString(threeDimArray.shape()));
        System.out.println("4d array shape:         " + Arrays.toString(fourDimArray.shape()));
        System.out.println("5d array shape:         " + Arrays.toString(fiveDimArray.shape()));



        //We can create INDArrays by combining other INDArrays, too:
        INDArray rowVector1 = Nd4j.create(new double[]{1,2,3});
        INDArray rowVector2 = Nd4j.create(new double[]{4,5,6});

        INDArray vStack = Nd4j.vstack(rowVector1, rowVector2);      //Vertical stack:   [1,3]+[1,3] to [2,3]
        INDArray hStack = Nd4j.hstack(rowVector1, rowVector2);      //Horizontal stack: [1,3]+[1,3] to [1,6]
        System.out.println("\n\n\nCreating INDArrays from other INDArrays, using hstack and vstack:");
        System.out.println("vStack:\n" + vStack);
        System.out.println("hStack:\n" + hStack);


        //There's some other miscellaneous methods, too:
        INDArray identityMatrix = Nd4j.eye(3);
        System.out.println("\n\n\nNd4j.eye(3):\n" + identityMatrix);
        INDArray linspace = Nd4j.linspace(1,10,10);                 //Values 1 to 10, in 10 steps
        System.out.println("Nd4j.linspace(1,10,10):\n" + linspace);
        INDArray diagMatrix = Nd4j.diag(rowVector2);                //Create square matrix, with rowVector2 along the diagonal
        System.out.println("Nd4j.diag(rowVector2):\n" + diagMatrix);

    }

}
