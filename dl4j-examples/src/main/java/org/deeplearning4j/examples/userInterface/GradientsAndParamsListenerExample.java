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

package org.deeplearning4j.examples.userInterface;

import org.deeplearning4j.examples.userInterface.util.GradientsAndParamsListener;
import org.deeplearning4j.examples.userInterface.util.UIExampleUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * An example of how to view params and gradients for a samples of neurons
 * in a running network using a TrainingListener and JavaFX 3D.
 * Params include weights and biases.
 *
 * You can navigate in 3d space by dragging the mouse or by the arrow keys.
 *
 * Red gradients are large positive. Blue gradients are large negative.
 * Large positive weights or biases cause a large radius.
 * Large negative weights or biases cause a small radius.
 *
 * The slider on the bottom of the window adjusts the mapping of gradients to colors.
 *
 * Note: if you're using Java 7 or earlier, you need to set the
 * environment variable JAVAFX_HOME to the directory of the JavaFX SDK.
 *
 * @author Donald A. Smith, Alex Black
 */
public class GradientsAndParamsListenerExample {

    public static void main(String[] args){

        //Get our network and training data
        MultiLayerNetwork net = UIExampleUtils.getMnistNetwork();
        DataSetIterator trainData = UIExampleUtils.getMnistData();

        System.out.println();
        for(Layer layer:net.getLayers()) {
            System.out.println(layer);
        }
        System.out.println();
        net.setListeners(new GradientsAndParamsListener(net,100));
        net.fit(trainData);

    }
}

