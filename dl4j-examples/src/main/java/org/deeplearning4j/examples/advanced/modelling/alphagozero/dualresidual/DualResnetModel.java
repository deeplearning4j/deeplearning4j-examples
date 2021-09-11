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

package org.deeplearning4j.examples.advanced.modelling.alphagozero.dualresidual;

import org.deeplearning4j.nn.graph.ComputationGraph;

/**
 * Define and load an AlphaGo Zero dual ResNet architecture
 * into DL4J.
 *
 * The dual residual architecture is the strongest
 * of the architectures tested by DeepMind for AlphaGo
 * Zero. It consists of an initial convolution layer block,
 * followed by a number (40 for the strongest, 20 as
 * baseline) of residual blocks. The network is topped
 * off by two "heads", one to predict policies and one
 * for value functions.
 *
 * @author Max Pumperla
 */
public class DualResnetModel {

    public static ComputationGraph getModel(int blocks, int numPlanes) {

        DL4JAlphaGoZeroBuilder builder = new DL4JAlphaGoZeroBuilder();
        String input = "in";

        builder.addInputs(input);
        String initBlock = "init";
        String convOut = builder.addConvBatchNormBlock(initBlock, input, numPlanes, true);
        String towerOut = builder.addResidualTower(blocks, convOut);
        String policyOut = builder.addPolicyHead(towerOut, true);
        String valueOut = builder.addValueHead(towerOut, true);
        builder.addOutputs(policyOut, valueOut);

        ComputationGraph model = new ComputationGraph(builder.buildAndReturn());
        model.init();

        return model;
    }
}
