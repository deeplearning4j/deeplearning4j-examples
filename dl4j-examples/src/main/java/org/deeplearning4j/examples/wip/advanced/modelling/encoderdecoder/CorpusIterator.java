/* *****************************************************************************
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

package org.deeplearning4j.examples.wip.advanced.modelling.encoderdecoder;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@SuppressWarnings("serial")
public class CorpusIterator implements MultiDataSetIterator {

    /*
     * Motivation: I want to get asynchronous data iteration while not blocking on net.fit() until the end of epoch. I want to checkpoint
     * the network, show intermediate test results and some stats, it would be harder to achieve with listeners I think so this is how I
     * solved the problem. This way the learn process is asynchronous inside one macrobatch and synchronous across all the macrobatches.
     *
     * Macrobatch is a group of minibatches. The iterator is modified so that it reports the end of data when it exhausts a macrobatch. Then
     * it advances (manually) to the next macrobatch.
     */

    private List<List<Double>> corpus;
    private int batchSize;
    private int batchesPerMacrobatch;
    private int totalBatches;
    private int totalMacroBatches;
    private int currentBatch = 0;
    private int currentMacroBatch = 0;
    private int dictSize;
    private int rowSize;

    CorpusIterator(List<List<Double>> corpus, int batchSize, int batchesPerMacrobatch, int dictSize, int rowSize) {
        this.corpus = corpus;
        this.batchSize = batchSize;
        this.batchesPerMacrobatch = batchesPerMacrobatch;
        this.dictSize = dictSize;
        this.rowSize = rowSize;
        totalBatches = (int) Math.ceil((double) corpus.size() / batchSize);
        totalMacroBatches = (int) Math.ceil((double) totalBatches / batchesPerMacrobatch);
    }

    @Override
    public boolean hasNext() {
        return currentBatch < totalBatches && getMacroBatchByCurrentBatch() == currentMacroBatch;
    }

    private int getMacroBatchByCurrentBatch() {
        return currentBatch / batchesPerMacrobatch;
    }

    @Override
    public MultiDataSet next() {
        return next(batchSize);
    }

    @Override
    public MultiDataSet next(int num) {
        int i = currentBatch * batchSize;
        int currentBatchSize = Math.min(batchSize, corpus.size() - i - 1);
        INDArray input = Nd4j.zeros(currentBatchSize, 1, rowSize);
        INDArray prediction = Nd4j.zeros(currentBatchSize, dictSize, rowSize);
        INDArray decode = Nd4j.zeros(currentBatchSize, dictSize, rowSize);
        INDArray inputMask = Nd4j.zeros(currentBatchSize, rowSize);
        // this mask is also used for the decoder input, the length is the same
        INDArray predictionMask = Nd4j.zeros(currentBatchSize, rowSize);
        for (int j = 0; j < currentBatchSize; j++) {
            List<Double> rowIn = new ArrayList<>(corpus.get(i));
            Collections.reverse(rowIn);
            List<Double> rowPred = new ArrayList<>(corpus.get(i + 1));
            rowPred.add(1.0); // add <eos> token
            // replace the entire row in "input" using NDArrayIndex, it's faster than putScalar(); input is NOT made of one-hot vectors
            // because of the embedding layer that accepts token indexes directly
            input.put(new INDArrayIndex[] { NDArrayIndex.point(j), NDArrayIndex.point(0), NDArrayIndex.interval(0, rowIn.size()) },
                    Nd4j.create(ArrayUtils.toPrimitive(rowIn.toArray(new Double[0]))));
            inputMask.put(new INDArrayIndex[] { NDArrayIndex.point(j), NDArrayIndex.interval(0, rowIn.size()) }, Nd4j.ones(rowIn.size()));
            predictionMask.put(new INDArrayIndex[] { NDArrayIndex.point(j), NDArrayIndex.interval(0, rowPred.size()) },
                    Nd4j.ones(rowPred.size()));
            // prediction (output) and decode ARE one-hots though, I couldn't add an embedding layer on top of the decoder and I'm not sure
            // it's a good idea either
            double[][] predOneHot = new double[dictSize][rowPred.size()];
            double[][] decodeOneHot = new double[dictSize][rowPred.size()];
            decodeOneHot[2][0] = 1; // <go> token
            int predIdx = 0;
            for (Double pred : rowPred) {
                predOneHot[pred.intValue()][predIdx] = 1;
                if (predIdx < rowPred.size() - 1) { // put the same vals to decode with +1 offset except the last token that is <eos>
                    decodeOneHot[pred.intValue()][predIdx + 1] = 1;
                }
                ++predIdx;
            }
            prediction.put(new INDArrayIndex[] { NDArrayIndex.point(j), NDArrayIndex.interval(0, dictSize),
                    NDArrayIndex.interval(0, rowPred.size()) }, Nd4j.create(predOneHot));
            decode.put(new INDArrayIndex[] { NDArrayIndex.point(j), NDArrayIndex.interval(0, dictSize),
                    NDArrayIndex.interval(0, rowPred.size()) }, Nd4j.create(decodeOneHot));
            ++i;
        }
        ++currentBatch;
        return new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[] { input, decode }, new INDArray[] { prediction },
                new INDArray[] { inputMask, predictionMask }, new INDArray[] { predictionMask });
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public boolean resetSupported() {
        // we don't want this iterator to be reset on each macrobatch pseudo-epoch
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        // but we still can do it manually before the epoch starts
        currentBatch = 0;
        currentMacroBatch = 0;
    }

    public int batch() {
        return currentBatch;
    }

    int totalBatches() {
        return totalBatches;
    }

    void setCurrentBatch(int currentBatch) {
        this.currentBatch = currentBatch;
        currentMacroBatch = getMacroBatchByCurrentBatch();
    }

    boolean hasNextMacrobatch() {
        return getMacroBatchByCurrentBatch() < totalMacroBatches && currentMacroBatch < totalMacroBatches;
    }

    void nextMacroBatch() {
        ++currentMacroBatch;
    }

    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
    }
}
