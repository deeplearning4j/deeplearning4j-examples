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

package org.deeplearning4j.examples.quickstart.features.userinterface;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.examples.quickstart.features.userinterface.util.UIExampleUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

/**
 * A simple example of how to attach Deeplearning4j's training UI to a network
 *
 * To change the UI port (usually not necessary) - set the org.deeplearning4j.ui.port system property
 * i.e., run the example and pass the following to the JVM, to use port 9001: -Dorg.deeplearning4j.ui.port=9001
 *
 * @author Alex Black
 */
public class BasicUIExample {

    public static void main(String[] args){
        //Get our network and training data
        MultiLayerNetwork net = UIExampleUtils.getMnistNetwork();
        DataSetIterator trainData = UIExampleUtils.getMnistData();

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        int listenerFrequency = 1;
        net.setListeners(new StatsListener(statsStorage, listenerFrequency));

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Start training:
        net.fit(trainData);

        //Finally: open your browser and go to http://localhost:9000/train
    }
}
