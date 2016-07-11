package org.deeplearning4j.examples.feedforward.regression.function;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface MathFunction {

    INDArray getFunctionValues(INDArray x);

    String getName();
}
