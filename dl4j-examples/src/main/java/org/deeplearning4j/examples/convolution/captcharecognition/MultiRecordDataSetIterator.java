/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.convolution.captcharecognition;

import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * @author WangFeng
 */

public class MultiRecordDataSetIterator implements MultiDataSetIterator {
    private int batchSize = 0;
    private int batchNum = 0;
    private int numExample = 0;
    private MulRecordDataLoader load;
    private MultiDataSetPreProcessor preProcessor;

    public MultiRecordDataSetIterator(int batchSize, String dataSetType) {
        this(batchSize, null, dataSetType);
    }
    public MultiRecordDataSetIterator(int batchSize, ImageTransform imageTransform, String dataSetType) {
        this.batchSize = batchSize;
        load = new MulRecordDataLoader(imageTransform, dataSetType);
        numExample = load.totalExamples();
    }


    @Override
    public MultiDataSet next(int i) {
        batchNum += i;
        MultiDataSet mds = load.next(i);
        if (preProcessor != null) {
            preProcessor.preProcess(mds);
        }
        return mds;
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor multiDataSetPreProcessor) {
        this.preProcessor = multiDataSetPreProcessor;
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        batchNum = 0;
        load.reset();
    }

    @Override
    public boolean hasNext() {
        if(batchNum < numExample){
            return true;
        } else {
            return false;
        }
    }

    @Override
    public MultiDataSet next() {
        return next(batchSize);
    }
}
