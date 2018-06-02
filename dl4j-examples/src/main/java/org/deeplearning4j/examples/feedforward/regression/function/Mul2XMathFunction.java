package org.deeplearning4j.examples.feedforward.regression.function;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Mul2XMathFunction implements MathFunction {

    @Override
    public INDArray getFunctionValues(INDArray x) {
        return x.dup().mul(2);
    }

    @Override
    public String getName() {
        return "Mul2XMathFunction";
    }

}
