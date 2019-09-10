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

import org.deeplearning4j.examples.userInterface.util.ActivationsListener;
import org.deeplearning4j.examples.userInterface.util.UIExampleUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * An example of how to attach Deeplearning4j's training UI to a network and view activations.
 *
 * To change the UI port (usually not necessary) - set the org.deeplearning4j.ui.port system property
 * i.e., run the example and pass the following to the JVM, to use port 9001: -Dorg.deeplearning4j.ui.port=9001
 *
 * @author Donald A. Smith, Alex Black
 */
public class ActivationsListenerExample {

    public static void main(String[] args){

        //Get our network and training data
        MultiLayerNetwork net = UIExampleUtils.getMnistNetwork();
        DataSetIterator trainData = UIExampleUtils.getMnistData();

        System.out.println();
        for(Layer layer:net.getLayers()) {
            System.out.println(layer);
        }
        System.out.println();
        net.setListeners(new ActivationsListener(net,20));
        net.fit(trainData);

    }
}

