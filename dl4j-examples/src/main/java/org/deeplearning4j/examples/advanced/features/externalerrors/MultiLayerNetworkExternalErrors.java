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

package org.deeplearning4j.examples.advanced.features.externalerrors;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.common.primitives.Pair;

/**
 * This example: shows how to train a MultiLayerNetwork where the errors come from an external source, instead
 * of using an Output layer and a labels array.
 * <p>
 * Possible use cases for this are reinforcement learning and testing/development of new algorithms.
 * <p>
 * For some uses cases, the following alternatives may be worth considering:
 * - Implement a custom loss function
 * - Implement a custom (output) layer
 * <p>
 * Both of these alternatives are available in DL4J
 *
 * @author Alex Black
 */
public class MultiLayerNetworkExternalErrors {

    public static void main(String[] args) {

        //Create the model
        int nIn = 4;
        int nOut = 3;
        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .updater(new Nadam())
            .list()
            .layer(new DenseLayer.Builder().nIn(nIn).nOut(3).build())
            .layer(new DenseLayer.Builder().nIn(3).nOut(3).build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        //Calculate gradient with respect to an external error
        int minibatch = 32;
        INDArray input = Nd4j.rand(minibatch, nIn);
        model.setInput(input);
        //Do forward pass, but don't clear the input activations in each layers - we need those set so we can calculate
        // gradients based on them
        model.feedForward(true, false);

        INDArray externalError = Nd4j.rand(minibatch, nOut);
        Pair<Gradient, INDArray> p = model.backpropGradient(externalError, null);  //Calculate backprop gradient based on error array

        //Update the gradient: apply learning rate, momentum, etc
        //This modifies the Gradient object in-place
        Gradient gradient = p.getFirst();
        int iteration = 0;
        int epoch = 0;
        model.getUpdater().update(model, gradient, iteration, epoch, minibatch, LayerWorkspaceMgr.noWorkspaces());

        //Get a row vector gradient array, and apply it to the parameters to update the model
        INDArray updateVector = gradient.gradient();
        model.params().subi(updateVector);
    }

}
