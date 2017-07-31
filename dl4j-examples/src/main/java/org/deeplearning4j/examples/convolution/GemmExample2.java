package org.deeplearning4j.examples.convolution;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

/**
 * Created by raver119 on 07.07.17.
 */
@Slf4j
public class GemmExample2 {
    private static int nOut = 1;
    private static int nIn = 4;
    private static int minibatch = 5;

    public static void main(String[] args) throws Exception {
        Nd4j.getRandom().setSeed(12345);

        INDArray input = Nd4j.rand(minibatch, nIn);

        INDArray z = Nd4j.create(new double[]{1.9383466, 1.578151, 2.330189, 0.6744037, 1.4629194});
        INDArray delta = Nd4j.create(new double[]{0.014565547, 0.15064883, 0.021107223, 0.51233447, 0.12016498}, new int[]{5, 1},'f');

        INDArray params = Nd4j.create(1000, 1, 'f');
        INDArray weightGrad = params.get(NDArrayIndex.interval(4,8)).reshape('f', 4, 1);//Nd4j.create(4, 1, 'f');
        INDArray biasGrad = Nd4j.create(1);

        log.info("input shape: {}", Arrays.toString(input.shapeInfoDataBuffer().asInt()));
        log.info("w shape: {}", Arrays.toString(weightGrad.shapeInfoDataBuffer().asInt()));
        log.info("delta shape: {}", Arrays.toString(delta.shapeInfoDataBuffer().asInt()));

        Nd4j.gemm(input, delta, weightGrad, true, false, 1.0, 0.0);

        //Nd4j.getExecutioner().commit();
        //log.info("WeightGrad: {}", weightGrad);
        //System.out.println("W: " + weightGrad);
        //Nd4j.getAffinityManager().ensureLocation(weightGrad, AffinityManager.Location.HOST);

        log.info("W: {}",weightGrad);


        delta.sum(biasGrad, 0);

        log.info("W: {}",weightGrad);
    }
}
