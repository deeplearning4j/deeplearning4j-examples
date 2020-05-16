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

package org.deeplearning4j.datasets;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;
/**
 * DatasetIteratorFromHdfs implements the methods in the DataSetIterator interface,this class mainly provides iteration of the data set.
 * @author wangfeng
 */
public class DatasetIteratorFromHdfs  implements DataSetIterator {
    private int batchSize = 0;
    private int batchNum = 0;
    private int numExample = 0;
    private DatasetReaderFromHdfs load;
    private DataSetPreProcessor preProcessor;

    public DatasetIteratorFromHdfs() {
        load = new DatasetReaderFromHdfs();
    }
    public DatasetIteratorFromHdfs(int batchSize, boolean train) {
        this.batchSize = batchSize;
        load = new DatasetReaderFromHdfs(train);
        numExample = load.totalExamples();
    }

    @Override
    public DataSet next(int i) {
        batchNum += i;
        DataSet ds = load.next(i);
        if (preProcessor != null) {
            preProcessor.preProcess(ds);
        }
        return ds;
    }

    @Override
    public int inputColumns() {
        return 0;
    }

    @Override
    public int totalOutcomes() {
        return 0;
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
    public int batch() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return load.getLabels();
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
    public DataSet next() {
        return next(batchSize);
    }


}
