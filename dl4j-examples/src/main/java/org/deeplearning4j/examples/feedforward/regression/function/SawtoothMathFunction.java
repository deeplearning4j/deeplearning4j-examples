package org.deeplearning4j.examples.feedforward.regression.function;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SawtoothMathFunction implements MathFunction {

    @Override
    public INDArray getFunctionValues(final INDArray x) {
        final double sawtoothPeriod = 4.0;
        final double[] xd2 = x.data().asDouble();
        final double[] yd2 = new double[xd2.length];
        for (int i = 0; i < xd2.length; i++) {
            yd2[i] = 2 * (xd2[i] / sawtoothPeriod - Math.floor(xd2[i] / sawtoothPeriod + 0.5));
        }
        return Nd4j.create(yd2, new int[]{xd2.length, 1});  //Column vector
    }

    @Override
    public String getName() {
        return "Sawtooth";
    }
}
