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

package org.deeplearning4j.examples.advanced.features.transferlearning.finetuneonly;

import org.deeplearning4j.examples.advanced.features.transferlearning.iterators.FlowerDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;

/**
 * Important:
 * 1. Either run "EditAtBottleneckOthersFrozen" first or save a model named "MyComputationGraph.zip" based on org.deeplearning4j.transferlearning.vgg16 with block4_pool and below intact
 * 2. You will need a LOT of RAM, at the very least 16G. Set max JVM heap space accordingly
 *
 * Here we read in an already saved model based off on org.deeplearning4j.transferlearning.vgg16 from one of our earlier runs and "finetune"
 * Since we already have reasonable results with our saved off model we can be assured that there will not be any
 * large disruptive gradients flowing back to wreck the carefully trained weights in the lower layers in vgg.
 *
 * Finetuning like this is usually done with a low learning rate and a simple SGD optimizer
 * @author susaneraly on 3/6/17.
 */
@SuppressWarnings("DuplicatedCode")
public class FineTuneFromBlockFour {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FineTuneFromBlockFour.class);

    protected static final int numClasses = 5;
    protected static final long seed = 12345;

    private static final String featureExtractionLayer = "block4_pool";
    private static final int trainPerc = 80;
    private static final int batchSize = 16;

    public static void main(String [] args) throws IOException {

        //Import the saved model
        File locationToSave = new File("MyComputationGraph.zip");
        log.info("\n\nRestoring saved model...\n\n");
        ComputationGraph vgg16Transfer = ModelSerializer.restoreComputationGraph(locationToSave);

        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        //  For eg. We override the learning rate and updater
        //          But our optimization algorithm remains unchanged (already sgd)
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .updater(new Sgd(1e-5))
            .seed(seed)
            .build();
        ComputationGraph vgg16FineTune = new TransferLearning.GraphBuilder(vgg16Transfer)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor(featureExtractionLayer)
            .build();
        log.info(vgg16FineTune.summary());

        //Dataset iterators
        FlowerDataSetIterator.setup(batchSize,trainPerc);
        DataSetIterator trainIter = FlowerDataSetIterator.trainIterator();
        DataSetIterator testIter = FlowerDataSetIterator.testIterator();

        Evaluation eval;
        eval = vgg16FineTune.evaluate(testIter);
        log.info("Eval stats BEFORE fit.....");
        log.info(eval.stats() + "\n");
        testIter.reset();

        int iter = 0;
        while(trainIter.hasNext()) {
            vgg16FineTune.fit(trainIter.next());
            if (iter % 10 == 0) {
                log.info("Evaluate model at iter "+iter +" ....");
                eval = vgg16FineTune.evaluate(testIter);
                log.info(eval.stats());
                testIter.reset();
            }
            iter++;
        }

        log.info("Model build complete");

        //Save the model
        File locationToSaveFineTune = new File("MyComputationGraphFineTune.zip");
        boolean saveUpdater = false;
        ModelSerializer.writeModel(vgg16FineTune, locationToSaveFineTune, saveUpdater);
        log.info("Model saved");

    }
}
