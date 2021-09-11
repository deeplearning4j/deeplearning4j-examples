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

package org.deeplearning4j.examples.advanced.features.transferlearning.editlastlayer.presave;

import org.deeplearning4j.examples.advanced.features.transferlearning.iterators.FlowerDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;

/**
 * The TransferLearningHelper class allows users to "featurize" a dataset at specific intermediate vertices/layers of a model
 * This example demonstrates how to presave these
 * Refer to the "FitFromFeaturized" example for how to fit a model with these featurized datasets
 * @author susaneraly on 2/28/17.
 */
public class FeaturizedPreSave {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FeaturizedPreSave.class);

    private static final int trainPerc = 80;
    protected static final int batchSize = 16;
    public static final String featurizeExtractionLayer = "fc2";

    public static void main(String [] args) throws IOException {

        //import org.deeplearning4j.transferlearning.vgg16 and print summary
        log.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        log.info(vgg16.summary());

        //use the TransferLearningHelper to freeze the specified vertices and below
        //NOTE: This is done in place! Pass in a cloned version of the model if you would prefer to not do this in place
        TransferLearningHelper transferLearningHelper = new TransferLearningHelper(vgg16, featurizeExtractionLayer);
        log.info(vgg16.summary());

        FlowerDataSetIterator.setup(batchSize,trainPerc);
        DataSetIterator trainIter = FlowerDataSetIterator.trainIterator();
        DataSetIterator testIter = FlowerDataSetIterator.testIterator();

        int trainDataSaved = 0;
        while(trainIter.hasNext()) {
            DataSet currentFeaturized = transferLearningHelper.featurize(trainIter.next());
            saveToDisk(currentFeaturized,trainDataSaved,true);
            trainDataSaved++;
        }

        int testDataSaved = 0;
        while(testIter.hasNext()) {
            DataSet currentFeaturized = transferLearningHelper.featurize(testIter.next());
            saveToDisk(currentFeaturized,testDataSaved,false);
            testDataSaved++;
        }

        log.info("Finished pre saving featurized test and train data");
    }

    private static void saveToDisk(DataSet currentFeaturized, int iterNum, boolean isTrain) {
        File fileFolder = isTrain ? new File("trainFolder"): new File("testFolder");
        if (iterNum == 0) {
            fileFolder.mkdirs();
        }
        String fileName = "flowers-" + featurizeExtractionLayer + "-";
        fileName += isTrain ? "train-" : "test-";
        fileName += iterNum + ".bin";
        currentFeaturized.save(new File(fileFolder,fileName));
        log.info("Saved " + (isTrain?"train ":"test ") + "dataset #"+ iterNum);
    }
}
