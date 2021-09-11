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
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

/**
 * A version of UIStorageExample showing how to saved network training data to a file, and then
 * reload it later, to display in the UI
 *
 * @author Alex Black
 */
public class UIStorageExample {

    public static void main(String[] args){

        //Run this example twice - once with collectStats = true, and then again with collectStats = false
        boolean collectStats = true;

        if(collectStats){
            //First run: Collect training stats from the network
            //Note that we don't have to actually plot it when we collect it - though we can do that too, if required

            MultiLayerNetwork net = UIExampleUtils.getMnistNetwork();
            DataSetIterator trainData = UIExampleUtils.getMnistData();

            StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
            net.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));

            net.fit(trainData);

            System.out.println("Done");
        } else {
            //Second run: Load the saved stats and visualize. Go to http://localhost:9000/train

            StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
            UIServer uiServer = UIServer.getInstance();
            uiServer.attach(statsStorage);
        }
    }
}
