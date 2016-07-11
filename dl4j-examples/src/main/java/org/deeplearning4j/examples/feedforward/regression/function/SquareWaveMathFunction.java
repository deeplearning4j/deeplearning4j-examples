package org.deeplearning4j.examples.feedforward.regression.function;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sign;
import org.nd4j.linalg.api.ops.impl.transforms.Sin;
import org.nd4j.linalg.factory.Nd4j;

public class SquareWaveMathFunction implements MathFunction {

    @Override
    public INDArray getFunctionValues(final INDArray x) {
        final INDArray sin = Nd4j.getExecutioner().execAndReturn(new Sin(x.dup()));
        return Nd4j.getExecutioner().execAndReturn(new Sign(sin));
    }

    @Override
    public String getName() {
        return "SquareWave";
    }
}
