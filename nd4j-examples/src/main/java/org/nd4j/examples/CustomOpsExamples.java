package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * Custom ops are those defined in C++ (libnd4j) that haven't yet been mapped to have Java convenience methods,
 * as of the 1.0.0-beta release.
 * Eventually all ops will be mapped to have Java convenience methods.
 * In the mean time, they can be accessed using the "DynamicCustomOp" approach shown below
 *
 */
public class CustomOpsExamples {


    public static void main(String[] args){

        //First example: Reverse op. This op reverses the values along a specified dimension
        //c++ code: https://github.com/deeplearning4j/libnd4j/blob/master/include/ops/declarable/generic/transforms/reverse.cpp#L15
        INDArray input = Nd4j.linspace(1, 50, 50).reshape(5,10);
        INDArray output = Nd4j.create(input.shape());
        CustomOp op = DynamicCustomOp.builder("reverse")
            .addInputs(input)
            .addOutputs(output)
            .addIntegerArguments(0) //Reverse along dimension 0
            .build();
        Nd4j.getExecutioner().exec(op);

        System.out.println(input);
        System.out.println();
        System.out.println(output);

        System.out.println("-------------");

        //Another example: meshgrid
        //c++ code: https://github.com/deeplearning4j/libnd4j/blob/master/include/ops/declarable/generic/broadcastable/meshgrid.cpp
        INDArray input1 = Nd4j.linspace(0, 1, 4);
        INDArray input2 = Nd4j.linspace(0, 1, 5);
        INDArray output1 = Nd4j.create(5,4);
        INDArray output2 = Nd4j.create(5, 4);

        op = DynamicCustomOp.builder("meshgrid")
            .addInputs(input1, input2)
            .addOutputs(output1, output2)
            .build();
        Nd4j.getExecutioner().exec(op);

        System.out.println(output1 + "\n\n" + output2);

    }

}
