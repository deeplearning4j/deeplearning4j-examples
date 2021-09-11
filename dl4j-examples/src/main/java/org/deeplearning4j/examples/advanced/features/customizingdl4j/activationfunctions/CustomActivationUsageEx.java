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

package org.deeplearning4j.examples.advanced.features.customizingdl4j.activationfunctions;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

import static org.deeplearning4j.examples.quickstart.modeling.feedforward.regression.SumModel.getTrainingData;

/**
 * This is an example that illustrates how to instantiate and use a custom activation function.
 * The example is identical to the one in org.deeplearning4j.examples.feedforward.regression.RegressionSum
 * except for the custom activation function
 */
public class CustomActivationUsageEx {

    public static void main(String[] args) {

        DataSetIterator iterator = getTrainingData(100, new Random(1234));

        //Create the network
        int numInput = 2;
        int numOutputs = 1;
        int nHidden = 10;
        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(1234)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.001, 0.9))
                .list()
                //INSTANTIATING CUSTOM ACTIVATION FUNCTION here as follows
                //Refer to CustomActivation class for more details on implementation
                .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
                        .activation(new CustomActivationDefinition())
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(nHidden).nOut(numOutputs).build())
                .build()
        );
        net.init();
        net.setListeners(new ScoreIterationListener(100));
        net.fit(iterator, 10);
        System.out.println("Training complete");

    }
}
