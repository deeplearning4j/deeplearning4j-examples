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

package org.deeplearning4j.examples.quickstart.features.presavingdatasets;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Pre saving the dataset is crucial.
 * Unlike with other frameworks that force you
 * to use 1 data format, deeplearning4j,
 * allows you to load arbitrary data, and also provides
 * tools such as datavec for pre processing a wide variety
 * of data from text, images, video, to log data.
 *
 * In this example, we pre show how to use a datasetiterator
 * to save pre save data.
 * In the other class {@link LoadPreSavedLenetMnistExample}
 * we then use the output to load data from the trainFolder
 * and testFolder.
 *
 * By pre saving the datasets, we save ALOT of time.
 * Anytime you end up trying to re do the processing every time
 * it ends up being a bottleneck.
 *
 * Pre saving the data allows you to have higher throughput during training.
 *
 * @author Adam Gibson
 */
public class PreSaveFirst {
    private static final Logger log = LoggerFactory.getLogger(LoadPreSavedLenetMnistExample.class);
    public static final String TRAIN_FOLDER = System.getProperty("user.home") + "/dl4j-examples-data/quickstart-presave/train";
    public static final String TEST_FOLDER = System.getProperty("user.home") + "/dl4j-examples-data/quickstart-presave/test";

    public static void main(String[] args) throws Exception {
        int batchSize = 64; // Test batch size


        /*
            Create an iterator using the batch size for one iteration
         */
        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);
        File trainFolder = new File(TRAIN_FOLDER);
        trainFolder.mkdirs();
        File testFolder = new File(TEST_FOLDER);
        testFolder.mkdirs();
        log.info("Saving train data to " + trainFolder.getAbsolutePath() +  " and test data to " + testFolder.getAbsolutePath());
        //Track the indexes of the files being saved.
        //These batch indexes are used for indexing which minibatch is being saved by the iterator.
        int trainDataSaved = 0;
        int testDataSaved = 0;
        while(mnistTrain.hasNext()) {
            //note that we use testDataSaved as an index in to which batch this is for the file
            mnistTrain.next().save(new File(trainFolder,"mnist-train-" + trainDataSaved + ".bin"));
                                                                              //^^^^^^^
                                                                              //******************
                                                                              //YOU NEED TO KNOW WHAT THIS IS.
                                                                              //This is the index for the file saved.
                                                                              //******************************************
            trainDataSaved++;
        }

        while(mnistTest.hasNext()) {
            //note that we use testDataSaved as an index in to which batch this is for the file
            mnistTest.next().save(new File(testFolder,"mnist-test-" + testDataSaved + ".bin"));
                                                                            //^^^^^^^
                                                                            //******************
                                                                            //YOU NEED TO KNOW WHAT THIS IS.
                                                                            //This is the index for the file saved.
                                                                            //******************************************
            testDataSaved++;
        }

        log.info("Finished pre saving test and train data");


    }

}
