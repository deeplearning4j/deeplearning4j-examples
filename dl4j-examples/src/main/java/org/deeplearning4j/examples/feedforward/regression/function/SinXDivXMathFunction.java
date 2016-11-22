package org.deeplearning4j.examples.feedforward.regression.function;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sin;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Calculate function value of sine of x divided by x.
 */
public class SinXDivXMathFunction implements MathFunction {

    @Override
    public INDArray getFunctionValues(final INDArray x) {
        return Nd4j.getExecutioner().execAndReturn(new Sin(x.dup())).div(x);
    }

    @Override
    public String getName() {
        return "SinXDivX";
    }
}
