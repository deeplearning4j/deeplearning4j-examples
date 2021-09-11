/*******************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.wip.advanced.modelling.mixturedensitynetwork;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;

/**
 * This is an iterator which creates a parameterized mixture of gaussians
 * for the purpose of verifying the convergence of a mixture-density
 * loss function and its corresponding gradient.
 *
 * @author Jonathan Arney
 */
public class GaussianMixtureIterator implements DataSetIterator {

    private final int iterationsPerBatch = 32;
    private final int miniBatchSize = 1000;
    private final int numExamplesToFetch = iterationsPerBatch * miniBatchSize;
    private int examplesSoFar = 0;
    private final Random mRNG;
    private final int mMixturesPerLabel;

    public GaussianMixtureIterator(int nMixturesPerLabel) {
        mRNG = new Random();
        mMixturesPerLabel = nMixturesPerLabel;
    }

    @Override
    public DataSet next() {
        return next(miniBatchSize);
    }

    @Override
    public boolean hasNext() {
        return (examplesSoFar < numExamplesToFetch);
    }

    @Override
    public DataSet next(int num) {
        if (examplesSoFar + num > numExamplesToFetch) {
            throw new NoSuchElementException();
        }
        try {
            DataSet nextData = nextThrows(num);
            examplesSoFar += num;
            return nextData;
        }
        catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }

    public DataSet nextThrows(int num) throws Exception {

        INDArray input = Nd4j.zeros(num, inputColumns());
        INDArray output = Nd4j.zeros(num, totalOutcomes());

        for (int i = 0; i < num; i++) {
            double x = mRNG.nextDouble() - 0.5;

            // The effect of this is two two-dimensional gaussians
            // mixed 50/50 with one another.
            // The first gaussian is fixed with a mean of (-0.5, -0.5).
            // The second gaussian has a mean that varies from -0.25 to 0.25.
            // The variance of both is 0.01 (std-deviation 0.1)

            boolean mid = mRNG.nextBoolean();
            double meanFactor = mid ? -0.5 : 0.5*x;
            double sigma = mid ? 0.01 : 0.01;

            MultivariateNormalDistribution mnd = new MultivariateNormalDistribution(
                    new double[] {1*meanFactor, 1*meanFactor},
                    new double[][] {
                        {sigma, 0.0},
                        {0.0, sigma}
                    }
            );

            double[] samples = mnd.sample();

            input.putScalar(new int[]{i, 0}, x*10);
            output.putScalar(new int[]{i, 0}, samples[0]);
            output.putScalar(new int[]{i, 1}, samples[1]);
        }


        return new DataSet(input, output);
    }

    @Override
    public int inputColumns() {
        return 1;
    }

    @Override
    public int totalOutcomes() {
        return 2;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        examplesSoFar = 0;
    }

    @Override
    public int batch() {
        return 1;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public static void main(String[] args) {
        GaussianMixtureIterator it = new GaussianMixtureIterator(1);

        int j = 0;
        while (it.hasNext()) {
            if (j == 8) break;
            DataSet next = it.next();
            INDArray features = next.getFeatures();
            INDArray labels = next.getLabels();

            for (int i = 0; i < features.rows(); i++) {
                System.out.println("" + features.getDouble(i) + "\t" + labels.getDouble(i, 0));
            }
            j++;
        }
    }


}
