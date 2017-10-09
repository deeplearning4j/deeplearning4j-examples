package org.nd4j.examples.numpy_cheatsheat;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.io.IOException;
import java.util.Arrays;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

/**
 * These are common functions that most python numpy users use for their daily work.
 * I've provided examples for all such users who are coming from the numpy environment to ND4J
 * You can view the cheatsheat and see the implementations and use cases here
 *
 * Following is the link to the cheatsheat I've implemented
 * https://www.dataquest.io/blog/images/cheat-sheets/numpy-cheat-sheet.pdf
 *
 * @author Shams Ul Azeem
 */

public class NumpyCheatSheat {
    public static void main(String[] args) {
        /* A. IMPORTING/EXPORTING */
        // 1. np.loadtxt('file.txt') - From a text file
        INDArray readFromText = null;
        try {
            readFromText = Nd4j.readNumpy(makeResourcePath("/numpy_cheatsheet/file.txt"));
            print("Read from text", readFromText);
        } catch (IOException e) {
            e.printStackTrace();
        }
        // 2. np.genfromtxt('file.csv',delimiter=',') - From a CSV file
        INDArray readFromCSV = null;
        try {
            readFromCSV = Nd4j.readNumpy(makeResourcePath("/numpy_cheatsheet/file.csv"), ",");
            print("Read from csv", readFromCSV);
        } catch (IOException e) {
            e.printStackTrace();
        }
        // 3. np.savetxt('file.txt',arr,delimiter=' ') - Writes to a text file
        try {
            if (readFromText != null) {
                Nd4j.writeNumpy(readFromText, makeResourcePath("/numpy_cheatsheet/saveFile.txt"), " "); //This method is deprecated but it's the closest to the numpy one
                System.out.println("Printed array into a text file");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        // 4. np.savetxt('file.csv',arr,delimiter=',') - Writes to a CSV file
        try {
            if (readFromCSV != null) {
                Nd4j.writeNumpy(readFromCSV, makeResourcePath("/numpy_cheatsheet/saveFile.csv"), ","); //This method is deprecated but it's the closest to the numpy one
                System.out.println("Printed array into a csv file");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        /* B. CREATING ARRAYS */
        // 1. np.array([1,2,3]) - One dimensional array
        INDArray oneDArray = Nd4j.create(new float[]{1,2,3,4,5,6} , new int[]{6});
        print("One Dimensional Array", oneDArray);
        // 2. np.array([(1,2,3),(4,5,6)]) - Two dimensional array
        INDArray twoDArray = Nd4j.create(new float[]{1,2,3,4,5,6} , new int[]{2,3});
        print("Two Dimensional Array", twoDArray);
        // 3. np.zeros(3) - 1D array of length 3 all values 0
        INDArray oneDZeros = Nd4j.zeros(3);
        print("One dimensional zeros", oneDZeros);
        // 4. np.ones((3,4)) - 3x4 array with all values 1
        INDArray threeByFourOnes = Nd4j.ones(3, 4);
        print("3x4 ones", threeByFourOnes);
        // 5. np.eye(5) - 5x5 array of 0 with 1 on diagonal (Identity matrix)
        INDArray fiveByFiveIdentity = Nd4j.eye(5);
        print("5x5 Identity", fiveByFiveIdentity);
        // 6. np.linspace(0,100,6) - Array of 6 evenly divided values from 0 to 100
        INDArray zeroToHundredLinspaceOfSix = Nd4j.linspace(0, 100, 6);
        print("Zero to Hundred With linspace interval 6", zeroToHundredLinspaceOfSix);
        // 7. np.arange(0,10,3) - Array of values from 0 to less than 10 with step 3 (eg [0,3,6,9])
        INDArray stepOfThreeTillTen = CustomOperations.arange(-10, -20, -0.4);
        print("ARange", stepOfThreeTillTen);
        // 8. np.full((2,3),8) - 2x3 array with all values 8
        INDArray allEights = Nd4j.valueArrayOf(new int[] {2,3}, 8);
        print("2x3 Eights", allEights);
        // 9. np.random.rand(4,5) - 4x5 array of random floats between 0-1
        INDArray fourByFiveRandomZeroToOne = Nd4j.rand(new int[] {4, 5});
        print("4x5 Random between zero and one", fourByFiveRandomZeroToOne);
        // 10. np.random.rand(6,7)*100 - 6x7 array of random floats between 0-100
        INDArray sixBySevenRandomZeroToHundred = Nd4j.rand(new int[] {6, 7}).mul(100);
        print("6x7 Random between zero and hundred", sixBySevenRandomZeroToHundred);
        // 11. np.random.randint(5,size=(2,3)) - 2x3 array with random ints between 0-4
        INDArray twoByThreeRandIntZeroToFour = CustomOperations.randInt(new int[]{2,3}, 5);
        print("2x3 Random Ints between zero and four", twoByThreeRandIntZeroToFour);

        /* C. INSPECTING PROPERTIES */
        // 1. arr.size - Returns number of elements in arr
        int size = fourByFiveRandomZeroToOne.length();
        System.out.println("Array size: " + size);
        // 2. arr.shape - Returns dimensions of arr (rows, columns)
        int [] shape = fourByFiveRandomZeroToOne.shape();
        System.out.println("Array shape: " + Arrays.toString(shape));
        // 3. arr.dtype - Returns type of elements in arr
        String type = CustomOperations.type(fourByFiveRandomZeroToOne);
        System.out.println("Array type: " + type);
        // 4. arr.astype(dtype) - Convert arr elements to type dtype
        /* This can't be implemented as according to the documentation all ND4J arrays should have the same datatype
         * If you want to set it globally then use the following function
         * ------------------------------------------
         * For 0.4-rc3.8 and earlier:
         * ------------------------------------------
         * Nd4j.dtype = DataBuffer.Type.DOUBLE;
         * NDArrayFactory factory = Nd4j.factory();
         * factory.setDType(DataBuffer.Type.DOUBLE);
         * ------------------------------------------
         * For 0.4-rc3.9 and later:
         * ------------------------------------------
         * DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
         * ------------------------------------------
         */
        // 5. arr.tolist() - Convert arr to a Python list
        byte[] bytes = fourByFiveRandomZeroToOne.data().asBytes();
        System.out.println("Array byte: " + Arrays.toString(bytes));
        double[] doubles = fourByFiveRandomZeroToOne.data().asDouble();
        System.out.println("Array doubles: " + Arrays.toString(doubles));
        float[] floats = fourByFiveRandomZeroToOne.data().asFloat();
        System.out.println("Array floats: " + Arrays.toString(floats));
        int[] ints = fourByFiveRandomZeroToOne.data().asInt();
        System.out.println("Array ints: " + Arrays.toString(ints)); //and so on...
        // 6. np.info(np.eye) - View documentation for np.eye
        String arrayInfo = CustomOperations.arrayInfo(fourByFiveRandomZeroToOne);
        System.out.println("Array info: " + arrayInfo);

        /* D. COPYING/SORTING/RESHAPING */
        // 1. np.copy(arr) - Copies arr to new memory
        INDArray copy = fourByFiveRandomZeroToOne.dup();
        print("Copied array: ", copy);
        // 2. arr.view(dtype) - Creates view of arr elements with type dtype
        /* This can't be implemented as according to the documentation all ND4J arrays should have the same datatype.
         * So if we change an array's view it's going to be reflected on all the arrays
         */
        // 3. arr.sort() - Sorts arr
        INDArray sortedArray = Nd4j.sort(fourByFiveRandomZeroToOne, true);
        print("Ascended sorted array: ", sortedArray);
        // 4. arr.sort(axis=0) - Sorts specific axis of arr
        INDArray axisSortedArray = Nd4j.sort(fourByFiveRandomZeroToOne, 0, true);
        print("Ascended sorted array on zero axis: ", axisSortedArray);
        // 5. two_d_arr.flatten() - Flattens 2D array two_d_arr to 1D
        INDArray flattened = Nd4j.toFlattened(fourByFiveRandomZeroToOne);
        print("Flattened array", flattened);
        // 6. arr.T - Transposes arr (rows become columns and vice versa)
        INDArray transpose = fourByFiveRandomZeroToOne.transpose();
        print("Transposed array", transpose);
        // 7. arr.reshape(5,4) - Reshapes arr to 5 rows, 4 columns without changing data
        INDArray reshaped = fourByFiveRandomZeroToOne.reshape(5, 4);
        print("Reshaped array", reshaped);
        // 8. arr.resize((5,6)) - Changes arr shape to 5x6 and fills new values with 0
        INDArray resized = CustomOperations.resize(fourByFiveRandomZeroToOne, new int []{5,6});
        print("Resized array", resized);

        /* E. ADDING/REMOVING ELEMENTS */
        // 1. np.append(arr,values) - Appends values to end of arr
        INDArray appended = CustomOperations.append(reshaped, resized);
        print("Appended array", appended);
        // 2. np.insert(arr,2,values) - Inserts values into arr before index 2
        INDArray inserted = CustomOperations.insert(reshaped, 2, resized);
        print("Inserted array", inserted);
        // 3. np.delete(arr,3,axis=0) - Deletes row on index 3 of arr
        INDArray deletedIndex3Axis0 = CustomOperations.delete(0, resized, 3);
        print("Deleted array on index 3 axis 0", deletedIndex3Axis0);
        // 4. np.delete(arr,4,axis=1) - Deletes column on index 4 of arr
        INDArray deletedIndex4Axis1 = CustomOperations.delete(1, resized, 4);
        print("Deleted array on index 4 axis 1", deletedIndex4Axis1);

        /* F. COMBINING/SPLITTING */
        // 1. np.concatenate((arr1,arr2),axis=0) - Adds arr2 as rows to the end of arr1
        INDArray concatenatedAxisZero = Nd4j.concat(0, Nd4j.create(3, 2), Nd4j.create(5, 2));
        print("Concatenated arrays on dimension zero", concatenatedAxisZero);
        // 2. np.concatenate((arr1,arr2),axis=1) - Adds arr2 as columns to end of arr1
        INDArray concatenatedAxisOne = Nd4j.concat(1, Nd4j.create(3, 2), Nd4j.create(3, 5));
        print("Concatenated arrays on dimension 1", concatenatedAxisOne);
        // 3. np.split(arr,3) - Splits arr into 3 sub-arrays
        INDArray [] verticalSplit = CustomOperations.split(Nd4j.valueArrayOf(new int[] {9, 9}, 9),
            3);
        print("Vertical Split", verticalSplit);
        // 4. np.hsplit(arr,5) - Splits arr horizontally into 5 sub-arrays
        INDArray [] horizontalSplit = CustomOperations.hsplit(Nd4j.valueArrayOf(new int[]{10, 10}, 10),
            5);
        print("Horizontal Split", horizontalSplit);

        /* G. INDEXING/SLICING/SUBSETTING */
        // 1. arr[5] - Returns the element at index 5
        double oneValue = fourByFiveRandomZeroToOne.getDouble(5);
        System.out.println("Get one value from 1D array: " + oneValue);
        // 2. arr[2,4] - Returns the 2D array element on index [2][5]
        double oneValue2D = fourByFiveRandomZeroToOne.getDouble(2, 4);
        System.out.println("Get one value from 2D array: " + oneValue2D);
        // 3. arr[1]=4 - Assigns array element on index 1 the value 4
        fourByFiveRandomZeroToOne.putScalar(1, 4);
        print("Assigned value to array (1 => 4)", fourByFiveRandomZeroToOne);
        // 4. arr[1,3]=10 - Assigns array element on index [1][3] the value 10
        fourByFiveRandomZeroToOne.putScalar(new int[] {1,3}, 10);
        print("Assigned value to array (1x3 => 10)", fourByFiveRandomZeroToOne);
        // 5. arr[0:3] - Returns the elements at indices 0,1,2 (On a 2D array: returns rows 0,1,2)
        INDArray threeValuesArray = fourByFiveRandomZeroToOne.get(NDArrayIndex.interval(0, 3));
        print("Get interval from array ([0:3])", threeValuesArray);
        // 6. arr[0:3,4] - Returns the elements on rows 0,1,2 at column 4
        INDArray threeValuesArrayColumnFour = fourByFiveRandomZeroToOne.get(NDArrayIndex.interval(0, 3), NDArrayIndex.point(4));
        print("Get interval from array ([0:3,4])", threeValuesArrayColumnFour);
        // 7. arr[:2] - Returns the elements at indices 0,1 (On a 2D array: returns rows 0,1)
        INDArray threeValuesArrayAgain = fourByFiveRandomZeroToOne.get(NDArrayIndex.interval(0, 2));
        print("Get interval from array ([:2])", threeValuesArrayAgain);
        // 8. arr[:,1] - Returns the elements at index 1 on all rows
        INDArray allRowsIndexOne = fourByFiveRandomZeroToOne.get(NDArrayIndex.all(), NDArrayIndex.point(1));
        print("Get interval from array ([:,1])", allRowsIndexOne);
        //For the functions below, since there's no boolean type in ND4J so I'll work on 0.0s(false) and 1.0s(true)
        // 9. arr<5 - Returns an array with boolean values
        INDArray lessThan5 = CustomOperations.booleanOp(CustomOperations.randInt(new int[]{3, 3}, 10), Conditions.lessThan(5));
        print("Less than 5", lessThan5);
        // 10. (arr1<3) & (arr2>5) - Returns an array with boolean values
        INDArray lessThan3 = CustomOperations.booleanOp(CustomOperations.randInt(new int[]{3, 3}, 10),
            Conditions.lessThan(3));
        INDArray greaterThan5 = CustomOperations.booleanOp(CustomOperations.randInt(new int[]{3, 3}, 10),
            Conditions.greaterThan(5));
        INDArray compared = CustomOperations.compare(lessThan3, greaterThan5, new Predicate<Boolean[]>() {
            @Override
            public boolean test(Boolean[] booleans) {
                return booleans[0] & booleans[1];
            }
        });
        print("Compared", compared);
        // 11. ~arr - Inverts a boolean array
        INDArray inverted = CustomOperations.invert(lessThan5);
        print("Inverted", inverted);
        // 12. arr[arr<5] - Returns array elements smaller than 5
        INDArray lessThan5Elements = CustomOperations.find(CustomOperations.randInt(new int[]{3, 3}, 10),
            new Predicate<Double>() {
                @Override
                public boolean test(Double aDouble) {
                    return aDouble < 5;
                }
            });
        print("Less than 5 elements", lessThan5Elements);

        /* H. SCALAR MATH */
        // 1. np.add(arr,1) - Add 1 to each array element
        INDArray addOne = fourByFiveRandomZeroToOne.add(1);
        print("Add 1 to array", addOne);
        // 2. np.subtract(arr,2) - Subtract 2 from each array element
        INDArray subtractTwo = fourByFiveRandomZeroToOne.sub(2);
        print("Subtract 2 from array", subtractTwo);
        // 3. np.multiply(arr,3) - Multiply each array element by 3
        INDArray multiplyThree = fourByFiveRandomZeroToOne.mul(3);
        print("Multiply 3 to array", multiplyThree);
        // 4. np.divide(arr,4) - Divide each array element by 4 (returns np.nan for division by zero)
        INDArray divideFour = fourByFiveRandomZeroToOne.div(4);
        print("Divide array by 4", divideFour);
        // 5. np.power(arr,5) - Raise each array element to the 5th power
        INDArray pow = pow(fourByFiveRandomZeroToOne, 5);
        print("5th power of array", pow);

        /* I. VECTOR MATH */
        // 1. np.add(arr1,arr2) - Elementwise add arr2 to arr1
        INDArray secondArray = Nd4j.create(new int[]{4,5}).add(10);
        INDArray vectorAdd = fourByFiveRandomZeroToOne.add(secondArray);
        print("Vector add", vectorAdd);
        // 2. np.subtract(arr1,arr2) - Elementwise subtract arr2 from arr1
        INDArray vectorSubtract = fourByFiveRandomZeroToOne.sub(secondArray);
        print("Vector subtract", vectorSubtract);
        // 3. np.multiply(arr1,arr2) - Elementwise multiply arr1 by arr2
        INDArray vectorMultiply = fourByFiveRandomZeroToOne.mul(secondArray);
        print("Vector multiply", vectorMultiply);
        // 4. np.divide(arr1,arr2) - Elementwise divide arr1 by arr2
        INDArray vectorDivide = fourByFiveRandomZeroToOne.div(secondArray);
        print("Vector divide", vectorDivide);
        // 5. np.power(arr1,arr2) - Elementwise raise arr1 raised to the power of arr2
        INDArray power = pow(fourByFiveRandomZeroToOne, secondArray);
        print("Vector power", power);
        // 6. np.array_equal(arr1,arr2) - Returns True if the arrays have the same elements and shape
        boolean areArraysEquals1 = CustomOperations.Equal(fourByFiveRandomZeroToOne, threeByFourOnes);
        boolean areArraysEquals2 = CustomOperations.Equal(fourByFiveRandomZeroToOne, fourByFiveRandomZeroToOne);
        System.out.println("Are arrays equals: 1. " + areArraysEquals1 + ", 2. " + areArraysEquals2);
        // 7. np.sqrt(arr) - Square root of each element in the array
        INDArray sqrt = sqrt(fourByFiveRandomZeroToOne);
        print("Vector square root", sqrt);
        // 8. np.sin(arr) - Sine of each element in the array
        INDArray sin = sin(fourByFiveRandomZeroToOne);
        print("Vector sin", sin);
        // 9. np.log(arr) - Natural log of each element in the array
        INDArray log = log(fourByFiveRandomZeroToOne);
        print("Vector log", log);
        // 10. np.abs(arr) - Absolute value of each element in the array
        INDArray abs = abs(fourByFiveRandomZeroToOne);
        print("Vector abs", abs);
        // 11. np.ceil(arr) - Rounds up to the nearest int
        INDArray ceil = ceil(fourByFiveRandomZeroToOne);
        print("Vector ceil", ceil);
        // 12. np.floor(arr) - Rounds down to the nearest int
        INDArray floor = floor(fourByFiveRandomZeroToOne);
        print("Vector floor", floor);
        // 13. np.round(arr) - Rounds to the nearest int
        INDArray round = round(fourByFiveRandomZeroToOne);
        print("Vector round", round);

        /* J. STATISTICS */
        // 1. np.mean(arr,axis=0) - Returns mean along specific axis
        INDArray mean = Nd4j.mean(fourByFiveRandomZeroToOne, 0);
        print("Mean on dimension zero", mean);
        // 2. arr.sum() - Returns sum of arr
        Number sum = fourByFiveRandomZeroToOne.sumNumber();
        System.out.println("Sum: " + sum);
        // 3. arr.min() - Returns minimum value of arr
        Number min = fourByFiveRandomZeroToOne.minNumber();
        System.out.println("Min: " + min);
        // 4. arr.max(axis=0) - Returns maximum value of specific axis
        Number max = fourByFiveRandomZeroToOne.maxNumber();
        System.out.println("Max: " + max);
        // 5. np.var(arr) - Returns the variance of array
        INDArray var = Nd4j.var(fourByFiveRandomZeroToOne);
        print("Variance", var);
        // 6. np.std(arr,axis=1) - Returns the standard deviation of specific axis
        INDArray std = Nd4j.std(fourByFiveRandomZeroToOne, 1);
        print("Standard deviation", std);
        // 7. arr.corrcoef() - Returns correlation coefficient of array
        //todo: Returns correlation coefficient of array
    }

    private static void print(String tag, INDArray arr) {
        System.out.println("----------------");
        System.out.println(tag + ":\n" + arr.toString());
        System.out.println("----------------");
    }

    private static void print(String tag, INDArray [] arrays) {
        System.out.println("----------------");
        System.out.println(tag);
        for (INDArray array : arrays) {
            System.out.println("\n" + array);
        }
        System.out.println("----------------");
    }

    private static String makeResourcePath(String template) {
        return NumpyCheatSheat.class.getResource(template).getPath();
    }
}
