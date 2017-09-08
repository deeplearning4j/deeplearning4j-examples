package org.nd4j.examples.numpy_cheatsheat;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;

/**
 * Created by shams on 8/12/2017.
 */
class CustomOperations {
    static INDArray genFromTxt(String fileName, String delimiter) {
        try(FileInputStream inputStream = new FileInputStream(fileName)) {
            String[] everything = IOUtils.toString(inputStream).split(delimiter);
            float[] floats = new float[everything.length];

            for (int i = 0; i < everything.length; i++) {
                floats[i] = Float.parseFloat(everything[i].trim());
            }
            return Nd4j.create(floats, new int[]{everything.length},'c');
        } catch(Exception e) {
            e.printStackTrace();
            return Nd4j.create(new float[]{}, new int[]{0},'c');
        }
    }

    static INDArray loadText(String fileName) {
        return genFromTxt(fileName, "\\r?\\n");
    }

    static void saveTxt(String fileName, INDArray arr, String delimiter) {
        try (PrintStream out = new PrintStream(new FileOutputStream(fileName))) {
            float [] floats = arr.data().asFloat();
            String [] textArray = new String[floats.length];
            for(int i = 0; i < floats.length; i++) {
                textArray[i] = String.valueOf(floats[i]);
                System.out.println(textArray[i]);
            }

            out.print(StringUtils.join(textArray, delimiter));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    static INDArray arange(int lower, int upper, int step) {
        INDArray arange = Nd4j.arange(lower, upper);
        return arange.get(NDArrayIndex.interval(0, step, arange.length()));
    }

    static INDArray full(int [] shape, Number value) {
        return Nd4j.zeros(shape).add(value);
    }

    static INDArray randInt(int [] shape, int upper) {
        return Transforms.floor(Nd4j.rand(shape).mul(upper));
    }

    static INDArray asType(INDArray arr, DataBuffer.Type type) {
        INDArray typeChanges = Nd4j.create(arr.shape(), type);
        typeChanges.assign(arr);
        return typeChanges;
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
}
