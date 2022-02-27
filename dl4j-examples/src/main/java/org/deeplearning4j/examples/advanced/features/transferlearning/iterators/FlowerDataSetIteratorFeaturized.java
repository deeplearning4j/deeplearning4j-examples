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

package org.deeplearning4j.examples.advanced.features.transferlearning.iterators;

import org.deeplearning4j.examples.advanced.features.transferlearning.editlastlayer.presave.FeaturizedPreSave;
import org.nd4j.linalg.dataset.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;

/**
 * Iterator for featurized data.
 * @author susaneraly on 3/10/17.
 */
public class FlowerDataSetIteratorFeaturized {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FlowerDataSetIteratorFeaturized.class);

    private static String featureExtractorLayer = FeaturizedPreSave.featurizeExtractionLayer;

    public static DataSetIterator trainIterator() throws  IOException {
        runFeaturize();
        DataSetIterator existingTrainingData = new ExistingMiniBatchDataSetIterator(new File("trainFolder"),"flowers-"+featureExtractorLayer+"-train-%d.bin");
        DataSetIterator asyncTrainIter = new AsyncDataSetIterator(existingTrainingData);
        return asyncTrainIter;
    }
    public static DataSetIterator testIterator() {
        DataSetIterator existingTestData = new ExistingMiniBatchDataSetIterator(new File("testFolder"),"flowers-"+featureExtractorLayer+"-test-%d.bin");
        DataSetIterator asyncTestIter = new AsyncDataSetIterator(existingTestData);
        return asyncTestIter;
    }

    private static void runFeaturize() throws IOException {
        File trainDir = new File("trainFolder","flowers-"+featureExtractorLayer+"-train-0.bin");
        if (!trainDir.isFile()) {
            log.info("\n\tFEATURIZED DATA NOT FOUND. \n\t\tRUNNING \"FeaturizedPreSave\" first to do presave of featurized data");
            FeaturizedPreSave.main(null);
        }
    }
}
