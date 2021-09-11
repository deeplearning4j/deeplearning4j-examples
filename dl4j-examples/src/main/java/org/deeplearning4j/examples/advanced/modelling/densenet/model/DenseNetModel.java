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

package org.deeplearning4j.examples.advanced.modelling.densenet.model;

import org.deeplearning4j.nn.graph.ComputationGraph;

import java.util.List;

public class DenseNetModel {

    private static DenseNetModel instance;

    public static DenseNetModel getInstance() {
        if (instance == null) {
            instance = new DenseNetModel();
        }
        return instance;
    }

    public ComputationGraph buildNetwork(long seed, int channels, int numLabels, int width, int height) {

        DenseNetBuilder denseNetModel = new DenseNetBuilder(height, width, channels, seed, 12, true); //227x227x3

        int l1 = 6, l2 = 12, l3 = 24, l4 = 16;

        int nIn1 = l1 * denseNetModel.getGrowthRate() + 2 * denseNetModel.getGrowthRate();
        int nIn2 = l2 * denseNetModel.getGrowthRate() + nIn1 / 2;
        int nIn3 = l3 * denseNetModel.getGrowthRate() + nIn2 / 2;
        int nIn4 = l4 * denseNetModel.getGrowthRate() + nIn3 / 2;

        String init = denseNetModel.initLayer(5, 2, 1, channels);
        List<String> block1 = denseNetModel.buildDenseBlock("b1", l1, init);
        String trans1 = denseNetModel.addTransitionLayer("t1", nIn1, block1);
        List<String> block2 = denseNetModel.buildDenseBlock("b2", l2, trans1);
        String trans2 = denseNetModel.addTransitionLayer("t2", nIn2, block2);
        List<String> block3 = denseNetModel.buildDenseBlock("b3", l3, trans2);
        String trans3 = denseNetModel.addTransitionLayer("t3", nIn3, block3);
        List<String> block4 = denseNetModel.buildDenseBlock("b4", l4, trans3);
        denseNetModel.addOutputLayer(nIn4, numLabels, block4.toArray(new String[block4.size()]));

        return denseNetModel.getModel();
    }
}
