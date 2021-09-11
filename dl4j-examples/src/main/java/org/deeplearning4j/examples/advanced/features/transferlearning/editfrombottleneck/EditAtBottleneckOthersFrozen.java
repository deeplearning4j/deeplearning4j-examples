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

package org.deeplearning4j.examples.advanced.features.transferlearning.editfrombottleneck;

import org.deeplearning4j.examples.advanced.features.transferlearning.iterators.FlowerDataSetIterator;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;

/**
 * @author susaneraly on 3/1/17.
 *
 * IMPORTANT:
 * 1. The forward pass on VGG16 is time consuming. Refer to "FeaturizedPreSave" and "FitFromFeaturized" for how to use presaved datasets
 * 2. RAM at the very least 16G, set JVM mx heap space accordingly
 *
 * We use the transfer learning API to construct a new model based of org.deeplearning4j.transferlearning.vgg16.
 * We keep block5_pool and below frozen
 *      and modify/add dense layers to form
 *          block5_pool -> flatten -> fc1 -> fc2 -> fc3 -> newpredictions (5 classes)
 *       from
 *          block5_pool -> flatten -> fc1 -> fc2 -> predictions (1000 classes)
 *
 * Note that we could presave the output out block5_pool like we do in FeaturizedPreSave + FitFromFeaturized
 * Refer to those two classes for more detail
 */
public class EditAtBottleneckOthersFrozen {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(EditAtBottleneckOthersFrozen.class);

    protected static final int numClasses = 5;

    protected static final long seed = 12345;
    private static final int trainPerc = 80;
    private static final int batchSize = 16;
    private static final String featureExtractionLayer = "block5_pool";

    public static void main(String [] args) throws Exception {

        //Import vgg
        //Note that the model imported does not have an output layer (check printed summary)
        //  nor any training related configs (model from keras was imported with only weights and json)
        log.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");

        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        log.info(vgg16.summary());

        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .activation(Activation.LEAKYRELU)
            .weightInit(WeightInit.RELU)
            .updater(new Nesterovs(5e-5))
            .dropOut(0.5)
            .seed(seed)
            .build();

        //Construct a new model with the intended architecture and print summary
        //  Note: This architecture is constructed with the primary intent of demonstrating use of the transfer learning API,
        //        secondary to what might give better results
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor(featureExtractionLayer) //"block5_pool" and below are frozen
            .nOutReplace("fc2",1024, WeightInit.XAVIER) //modify nOut of the "fc2" vertex
            .removeVertexAndConnections("predictions") //remove the final vertex and it's connections
            .addLayer("fc3",new DenseLayer.Builder().activation(Activation.TANH).nIn(1024).nOut(256).build(),"fc2") //add in a new dense layer
            .addLayer("newpredictions",new OutputLayer
                                        .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                        .activation(Activation.SOFTMAX)
                                        .nIn(256)
                                        .nOut(numClasses)
                                        .build(),"fc3") //add in a final output dense layer,
                                                        // note that learning related configurations applied on a new layer here will be honored
                                                        // In other words - these will override the finetune confs.
                                                        // For eg. activation function will be softmax not RELU
            .setOutputs("newpredictions") //since we removed the output vertex and it's connections we need to specify outputs for the graph
            .build();
        log.info(vgg16Transfer.summary());

        //Dataset iterators
        FlowerDataSetIterator.setup(batchSize,trainPerc);
        DataSetIterator trainIter = FlowerDataSetIterator.trainIterator();
        DataSetIterator testIter = FlowerDataSetIterator.testIterator();

        Evaluation eval;
        eval = vgg16Transfer.evaluate(testIter);
        log.info("Eval stats BEFORE fit.....");
        log.info(eval.stats() + "\n");
        testIter.reset();

        int iter = 0;
        while(trainIter.hasNext()) {
            vgg16Transfer.fit(trainIter.next());
            if (iter % 10 == 0) {
                log.info("Evaluate model at iter "+iter +" ....");
                eval = vgg16Transfer.evaluate(testIter);
                log.info(eval.stats());
                testIter.reset();
            }
            iter++;
        }
        log.info("Model build complete");

        //Save the model
        //Note that the saved model will not know which layers were frozen during training.
        //Frozen models always have to specified before training.
        // Models with frozen layers can be constructed in the following two ways:
        //      1. .setFeatureExtractor in the transfer learning API which will always a return a new model (as seen in this example)
        //      2. in place with the TransferLearningHelper constructor which will take a model, and a specific vertexname
        //              and freeze it and the vertices on the path from an input to it (as seen in the FeaturizePreSave class)
        //The saved model can be "fine-tuned" further as in the class "FitFromFeaturized"
        File locationToSave = new File("MyComputationGraph.zip");
        boolean saveUpdater = false;
        vgg16Transfer.save(locationToSave, saveUpdater);

        log.info("Model saved");
    }
}
