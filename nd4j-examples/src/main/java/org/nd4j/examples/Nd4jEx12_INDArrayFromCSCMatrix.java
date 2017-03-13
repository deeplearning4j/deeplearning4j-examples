package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * --- Nd4j Example 12: INDArray from CSC matrix ---
 *
 * In this example, we'll see ways to create a INDArray from a CSC representation matrix
 *
 * @author Audrey Loeffel
 */

public class Nd4jEx12_INDArrayFromCSCMatrix {

    public static void main(String[] args) {

        System.out.println("-------- CSC to INDArray -------");
        int[] rowIndices = {0, 0, 1, 0, 1, 0, 2};
        int[] columnPointers = {0, 1, 3, 5};
        double[] dataCSC = {11, 12, 22, 13, 23, 14, 34};
        int[] shapeCSC = {5, 4};
        INDArray matrix = createFromCSC(dataCSC, rowIndices, columnPointers, shapeCSC);
        System.out.println("nd array is: \n" + matrix);
        int[] matrixCSCShape = matrix.shape();
        System.out.println("Its shape is [" + matrixCSCShape[0] +","+ matrixCSCShape[1]+"]");

        System.out.println("-------- CSC with empty column to INDArray -------");
        int[] rowIndices2 = {1, 2};
        int[] columnPointers2 = {0, 0, 1};
        double[] dataCSC2 = {22, 33};
        int[] shapeCSC2 = {4, 3};
        INDArray matrix2 = createFromCSC(dataCSC2, rowIndices2, columnPointers2, shapeCSC2);
        System.out.println("nd array is: \n" + matrix2);
        int[] matrixCSCShape2 = matrix2.shape();
        System.out.println("Its shape is [" + matrixCSCShape2[0] +","+ matrixCSCShape2[1]+"]");
    }

    private static INDArray createFromCSC(double[] data, int[] rowIndices, int[] columnPointers, int[] shape){
        INDArray result = Nd4j.zeros(shape);
        int columns = shape[1];
        int dataIdx = 0;
        for(int i = 0; i < columns; i++){
            for(int k = dataIdx; k < (i == columnPointers.length-1 ? rowIndices.length : columnPointers[i+1]); k++, dataIdx++){
                int j = rowIndices[k];
                result.put(j, i, data[k]);
                //System.out.println("i = "+i+", k = "+k+ ", data[k] = "+data[k]+"\n matrix = "+result.toString());
            }
        }
        return result;
    }
}
