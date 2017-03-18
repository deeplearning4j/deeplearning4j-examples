package org.deeplearning4j.examples.unsupervised.deepbelief;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.examples.unsupervised.deepbelief.utils.RBMUtils;
import org.deeplearning4j.examples.unsupervised.deepbelief.utils.RMBVisualizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by vlad on 3/18/2017.
 * @author Vladimir Sadilovski
 *
 * This is a reference implementation example for Restricted boltzmann machine learning
 * were all unis are binary, i.e. states can assume values 0 or 1
 *
 * references:
 *     A fast learning algorithm for deep belief nets", Geoffrey E. Hinton et al., 2006
 *          https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf
 *     A Practical Guide to Training Restricted Boltzmann Machines, Geoffrey E. Hinton et al., 2010
 *          http://www.csri.utoronto.ca/~hinton/absps/guideTR.pdf
 *     IPAM Summer School 2012 Tutorial on: Deep Learning
 *          http://helper.ipam.ucla.edu/publications/gss2012/gss2012_10596.pdf
 */
public class BinaryRBMRefExample {
    public static void main(String[] args) throws Exception {
        int hidSize = 1000;
        int visSize = 28*28;
//        int hidSize = 10;
//        int visSize = 6;
        double lRate = 0.1; // constant learning rate
        int numEpochs = 1;
        int numIters = 2000;
        int miniBatch = 10;
        int numSamples = 500;

        DataSetIterator trainIter = new MnistDataSetIterator(miniBatch,numSamples,true, true, true, 12345);
        DataSetIterator testIter = new MnistDataSetIterator(10,10,true, false, true, 12345);

        Distribution inpDist = Nd4j.getDistributions().createBinomial(1, 0.5);
        inpDist.reseedRandomGenerator(12345);

        BinaryRBM rbm = new BinaryRBM(visSize, hidSize, numIters, lRate);
        INDArray testSample = Nd4j.createUninitialized(visSize);
        if (testIter.hasNext()) {
            DataSet next = trainIter.next();
            testSample = next.getFeatures();
            RMBVisualizer.plot(testSample, rbm.output(testSample), "before training");
        }

        for (int epoch = 0; epoch < numEpochs; epoch++) {
            while (trainIter.hasNext()) {
                DataSet next = trainIter.next();
                rbm.train(next.getFeatures());
            }
            trainIter.reset();
        }

        RMBVisualizer.plot(testSample, rbm.output(testSample), "after training");
    }

    public static class BinaryRBM {
        INDArray hBias;
        INDArray vBias;
        INDArray W;
        double lRate;
        int numIters;
        int lIter = 0;

        private BinaryRBM(int visSize, int hidSize, int numIters, double lRate) {
            int[] shape = new int[] {hidSize, visSize};
            this.numIters = numIters;
            this.lRate = lRate;
            vBias = Nd4j.zeros(1, visSize);
            hBias = Nd4j.zeros(1, hidSize);
            W = Nd4j.linspace(0.001, 0.001, visSize * hidSize).reshape(shape);
//        W = Nd4j.zeros(hidSize, visSize).reshape(shape);
        }

        private INDArray output(INDArray sample) {
            INDArray v0 = sample.dup();
            INDArray vk = Nd4j.rand(sample.shape());

            /**
             * Start Markov's chain: infer initial hidden states by doing Gibbs sampling from data (v0)
             * Use matrix W transposed do do te inference
             */
            Pair<INDArray, INDArray> hPair = RBMUtils.oneGibbsPass(v0, hBias, W.transpose());
            INDArray h0 = hPair.getSecond();
            INDArray h0Prob = hPair.getFirst();

            /**
             * Run Markov's chain for k iterations to get vk states
             * When W is small k=1 is sufficient to reach equilibrium
             * With growth of W, k needs to be bigger
             * In this example k is always 1
             */
            int k = 1;

            INDArray hk = h0.dup();
            for (int j = 0; j < k; j++) {
                /**
                 * Perform full step h0 -> v1 -> h1
                 */
                Pair<Pair<INDArray, INDArray>, Pair<INDArray, INDArray>> pairState
                    = RBMUtils.computePairState(hk, vBias, hBias, W);
                INDArray v1Prob = pairState.getFirst().getFirst();
                vk = pairState.getFirst().getSecond();
                INDArray h1Prob = pairState.getSecond().getFirst();
                hk = pairState.getSecond().getSecond();
            }

            return vk;
        }

        private void train(INDArray sample) {
            int hidSize = hBias.size(1);
            int visSize = vBias.size(1);
            int sz = hidSize * visSize;
            INDArray v0 = sample.dup();
            INDArray vk = Nd4j.rand(sample.shape());

            int printIter = 100;
            for (int i = 0; i < numIters; i++) {

                /**
                 * Start Markov's chain: infer initial hidden states by doing Gibbs sampling from data (v0)
                 * Use matrix W transposed do do te inference
                 */
                Pair<INDArray, INDArray> hPair = RBMUtils.oneGibbsPass(v0, hBias, W.transpose());
                INDArray h0 = hPair.getSecond();
                INDArray h0Prob = hPair.getFirst();

                /**
                 * Run Markov's chain for k iterations
                 * When W is small, k=1 is sufficient to reach equilibrium
                 * With growth of W, k needs to be bigger. In this example k is always 1
                 */
                int k = 1;

                INDArray hk = h0.dup();
                for (int j = 0; j < k; j++) {
                    /**
                     * Perform full step h0 -> v1 -> h1
                     */
                    Pair<Pair<INDArray, INDArray>, Pair<INDArray, INDArray>> pairState
                        = RBMUtils.computePairState(hk, vBias, hBias, W);
                    INDArray v1Prob = pairState.getFirst().getFirst();
                    vk = pairState.getFirst().getSecond();
                    INDArray h1Prob = pairState.getSecond().getFirst();
                    hk = pairState.getSecond().getSecond();
                }

                if (printIter-- <= 0) {
                    System.out.println("iter " + lIter + ": score=" + RBMUtils.sqError(v0, vk, hidSize) + ", W=" + W.sumNumber().doubleValue() / sz);
                    printIter = 100;
                }

                /**
                 * Learning rules goes as the following
                 *
                 * ∆wij = e(<vihj>0 − <vihj>k)
                 * ∆vBi = e(vi0 - vik)
                 * ∆hBj = e(hj0 - hjk)
                 *
                 * where <·> denotes an average over the sampled states
                 * and "e" is the learning rate
                 */
                INDArray dW = v0.transpose().mmul(h0).sub(vk.transpose().mmul(hk)).muli(lRate).divi(hidSize);
                W.addi(dW.transpose());
                INDArray dvB = v0.sub(vk).sum(0).mul(lRate).div(v0.size(0));
                vBias.addi(dvB);
                INDArray dhB = h0.sub(hk).sum(0).mul(lRate).div(v0.size(0));
                hBias.addi(dhB);

                lIter++;
            }
        }
    }
}
