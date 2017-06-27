package org.nd4j.examples;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class Nd4j_FP {

    public static void main(String[] args) {
        Nd4j.getRandom().setSeed(12345);

        INDArray initial = Nd4j.rand(3, 5, -10.0, 10.0, Nd4j.getRandom());
        INDArray exp = initial.dup();

        double initialMean = initial.meanNumber().doubleValue();
        double initialDev =  initial.stdNumber().doubleValue();

        INDArray amax = initial.amax(1);
        log.info("Amax: {}", amax);

        INDArray logmax = Transforms.log(amax,2.0, true);

        log.info("LogMax: {}", logmax);

        INDArray scale = Transforms.pow(logmax.subi(14), 2);
        initial.muliColumnVector(scale.rdiv(1.0));

        Nd4j.getCompressor().setDefaultCompression("FLOAT16");
        INDArray compressed = Nd4j.getCompressor().compress(initial);
        INDArray decompressed = Nd4j.getCompressor().decompress(compressed);

        decompressed.muliColumnVector(scale);

        double restoredMean = decompressed.meanNumber().doubleValue();
        double restoredDev =  decompressed.stdNumber().doubleValue();

        log.info("Means: {} vs {}", initialMean, restoredMean);
        log.info("Std: {} vs {}", initialDev, restoredDev);
    }
}
