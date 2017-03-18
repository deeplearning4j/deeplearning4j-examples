package org.deeplearning4j.examples.unsupervised.deepbelief.utils;

import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

/**
 * Created by vlad on 3/18/2017.
 * @author Vladimir Sadilovski
 */
public class RBMUtils {
    /**
     * Performs exactly one step of Markov's chain starting from the states of hidden units
     * h(k-1) -> v(k) -> h(k)
     * @param h0 the hidden state
     * @param vBias biases of the visible units
     * @param hBias biases of the hidden units
     * @param W weights on connections between hidden and visible units
     * @return 2x2 matrix <<P(v=1|h),visible state sample>,<Q(h=1|v),hidden state sample>>
     */
    public static Pair<Pair<INDArray, INDArray>, Pair<INDArray, INDArray>> computePairState(INDArray h0, INDArray vBias, INDArray hBias, INDArray W) {
        /**
         * Generative pass
         * Generates visible states from given hidden states by performing Gibbs sampling
         * Use weight matrix W
         * r
         */
        Pair<INDArray, INDArray> vPair = oneGibbsPass(h0, vBias, W);
        INDArray v1 = vPair.getSecond();

        /**
         * Recognition pass
         * Infers hidden states from given visible states by performing Gibbs sampling
         * Uses transposed weight matrix W
         */
        Pair<INDArray, INDArray> hPair = oneGibbsPass(v1, hBias, W.transpose());
        return new Pair<>(vPair, hPair);
    }

    /**
     * Performs calculates probability and samples (Gibbs) v given the state of h or h given the sate of v
     * h(k-1) -> v(k) or
     * v(k) -> h(k)
     * @param srcState the states of the source units (visible or hidden)
     * @param trgBias the biases of the target units (hidden or visible)
     * @param W weights on connections between hidden and visible units
     * @return 2x2 matrix <P(target state=1|source state),target sample>
     */
    public static Pair<INDArray, INDArray> oneGibbsPass(INDArray srcState, INDArray trgBias, INDArray W) {
        INDArray trgProb = sigmoid(srcState.mmul(W).addiRowVector(trgBias));
        Distribution dist = Nd4j.getDistributions().createBinomial(1, trgProb);
        INDArray trgState = dist.sample(trgProb.shape());
        return new Pair<> (trgProb, trgState);
    }

    /**
     * Squared error between the data and the reconstruction
     * @param labels original data
     * @param output reconstruction
     * @param scale scale down factor
     * @return Squared error between the original data and the reconstruction
     */
    public static double sqError(INDArray labels, INDArray output, double scale) {
        INDArray scoreArr = output.rsub(labels);
        scoreArr = scoreArr.muli(scoreArr);
        return scoreArr.sumNumber().doubleValue()/scoreArr.size(0)/scale;
    }

    public static double score(INDArray hProb, INDArray vProb) {
        INDArray log = Transforms.log(hProb.transpose().mmul(vProb), false).negi();

        return log.sumNumber().doubleValue()/log.size(0);
    }

    public static double score1(INDArray data, INDArray recon) {
        INDArray output = recon.dup();//activationFn.getActivation(recon.dup(), true);

        // Clip output and data to be between Nd4j.EPS_THREsHOLD and 1, i.e. a valid non-zero probability
        output = Transforms.min(Transforms.max(output, Nd4j.EPS_THRESHOLD, false), 1, false);
        data = Transforms.min(Transforms.max(data, Nd4j.EPS_THRESHOLD, true), 1, false);

        INDArray logRatio = Transforms.log(output.rdivi(data), false);

        INDArray scoreArr = logRatio.muli(data);
        return scoreArr.sumNumber().doubleValue()/scoreArr.size(0);
    }
}
