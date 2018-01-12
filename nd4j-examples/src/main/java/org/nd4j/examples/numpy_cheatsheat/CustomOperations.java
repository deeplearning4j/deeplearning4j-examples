package org.nd4j.examples.numpy_cheatsheat;

import com.google.common.base.Function;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;

import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.List;

/**
 * These are some functions which I couldn't find the the ND4J library so I implemented them myself.
 * You can see the usages in the file NumpyCheatSheat.java
 *
 * Following is the link to the cheatsheat I've implemented
 * https://www.dataquest.io/blog/images/cheat-sheets/numpy-cheat-sheet.pdf
 *
 * @author Shams Ul Azeem
 */
interface Predicate<T> {
    boolean test(T t);
}

class CustomOperations {
    static INDArray arange(double start, double end, double step) {
        int elements = (int) ((end - start) / step);
        System.out.println(elements);
        return Nd4j.linspace(start, start + elements * step,  elements + 1);
    }

    static INDArray randInt(int [] shape, int upper) {
        return Transforms.floor(Nd4j.rand(shape).mul(upper));
    }

    static String type(INDArray arr) {
        return arr.data().dataType().name();
    }

    static String arrayInfo(INDArray arr) {
        return arr.data().dataType().toString();
    }

    static INDArray resize(INDArray arr, int [] shape) {
        INDArray resized = Nd4j.create(shape);
        resized.get(NDArrayIndex.createCoveringShape(arr.shape())).assign(arr);
        return resized;
    }

    static boolean Equal(INDArray arr1, INDArray arr2) {
        return ArrayUtil.equals(arr1.data().asFloat(), arr2.data().asDouble());
    }

    static INDArray append(INDArray arr1, INDArray values) {
        return append(arr1, values, -1);
    }

    static INDArray append(INDArray arr1, INDArray values, int dimension) {
        if(dimension == -1) {
            return Nd4j.toFlattened(arr1, values);
        } else {
            return Nd4j.concat(dimension, arr1, values);
        }
    }

    static INDArray insert(INDArray arr1, int index, INDArray values) {
        return insert(arr1, index, values, -1);
    }

    static INDArray insert(INDArray arr1, int index, INDArray values, int dimension) {
        if(dimension == -1) {
            INDArray flat1 = Nd4j.toFlattened(arr1);
            INDArray flatValues = Nd4j.toFlattened(values);
            INDArray firstSlice = flat1.get(NDArrayIndex.interval(0, index));
            INDArray secondSlice = flat1.get(NDArrayIndex.interval(index, flat1.length()));
            return Nd4j.toFlattened(firstSlice, flatValues, secondSlice);
        } else {
            INDArray firstSlice = arr1.get(createIntervalOnDimension(dimension, false,
                0, index));
            INDArray secondSlice = arr1.get(createIntervalOnDimension(dimension, false,
                index, arr1.shape()[dimension]));
            return Nd4j.concat(dimension, firstSlice, values, secondSlice);
        }
    }

    static INDArray delete(INDArray arr1, int... interval) {
        return delete(-1, arr1, interval);
    }

    static INDArray delete(int dimension, INDArray arr1, int... interval) {
        int length = interval.length;
        int lastIntervalValue = interval[length - 1];

        if(dimension == -1) {
            INDArray array1 = arr1.get(NDArrayIndex.interval(0, interval[0]));
            if(lastIntervalValue == arr1.length() - 1) {
                return Nd4j.toFlattened(array1);
            } else {
                INDArray array2 = arr1.get(NDArrayIndex.interval(lastIntervalValue + 1,
                    arr1.length()));
                return Nd4j.toFlattened(array1, array2);
            }
        } else {
            INDArray array1 = arr1.get(createIntervalOnDimension(dimension, false, 0, interval[0]));
            if(lastIntervalValue == arr1.shape()[dimension] - 1) {
                return array1;
            } else {
                INDArray array2 = arr1.get(createIntervalOnDimension(dimension, false,
                    lastIntervalValue + 1,
                    arr1.shape()[dimension]));
                return Nd4j.concat(dimension, array1, array2);
            }
        }
    }

    static INDArray [] split(INDArray arr1, int numOfSplits) {
        return split(arr1, numOfSplits, -1);
    }

    static INDArray [] split(INDArray arr1, int numOfSplits, int dimension) {
        dimension = dimension == -1 ? 0 : dimension;
        INDArray [] splits = new INDArray[numOfSplits];
        int intervalLength = arr1.shape()[dimension] / numOfSplits;

        for (int i = 0; i < numOfSplits; i++) {
            splits[i] = arr1.get(createIntervalOnDimension(dimension,
                false,
                intervalLength * i, intervalLength * (i + 1)));
        }
        return splits;
    }

    static INDArray [] hsplit(INDArray arr1, int numOfSplits) {
        return split(arr1, numOfSplits, 1);
    }

    static INDArray booleanOp(INDArray arr, Condition condition) {
        INDArray dup = arr.dup();
        BooleanIndexing.applyWhere(dup, condition,
            new Function<Number, Number>() {
                @Override
                public Number apply(@Nullable Number number) {
                    return 1.0;
                }
            }, new Function<Number, Number>() {
                @Override
                public Number apply(@Nullable Number number) {
                    return 0.0;
                }
            });
        return dup;
    }

    static INDArray invert(INDArray arr1) {
        return booleanOp(arr1, Conditions.equals(0));
    }

    static INDArray compare(INDArray arr1, INDArray arr2, Predicate<Boolean []> predicate) {
        INDArray result = Nd4j.create(arr1.shape());

        for (int i = 0; i < arr1.length(); i++) {
            boolean answer = predicate.test(new Boolean[]{arr1.getDouble(i) == 1.0, arr2.getDouble(i) == 1.0});
            result.putScalar(i, answer ? 1.0 : 0.0);
        }
        return result;
    }

    static INDArray find(INDArray arr1, Predicate<Double> predicate) {
        List<Double> values = new ArrayList<>();

        for(int i = 0; i < arr1.length(); i++) {
            Double value = arr1.getDouble(i);
            if(predicate.test(value)) {
                values.add(value);
            }
        }

        INDArray result = Nd4j.create(new int[]{values.size()});
        for(int i = 0; i < values.size(); i++) {
            result.putScalar(i, values.get(i));
        }

        return result;
    }

    static INDArrayIndex [] createIntervalOnDimension(int dimension, boolean inclusive, int... interval) {
        INDArrayIndex [] indexInterval = new INDArrayIndex[dimension + 1];

        for(int i = 0; i <= dimension; i++) {
            indexInterval[i] = i != dimension ?
                NDArrayIndex.all() :
                NDArrayIndex.interval(interval[0], interval[1], inclusive);
        }

        return indexInterval;
    }
}
