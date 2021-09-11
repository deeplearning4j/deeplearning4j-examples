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

import org.deeplearning4j.core.storage.StatsStorageRouter;
import org.deeplearning4j.core.storage.impl.RemoteUIStatsStorageRouter;
import org.deeplearning4j.examples.quickstart.features.userinterface.util.UIExampleUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * A version of UIExample that shows how you can host the UI in a different JVM to the
 *
 * For the case of this example, both are done in the same JVM. See comments for what goes in each JVM in practice.
 *
 * NOTE: Don't use this unless you *actually* need the UI to be hosted in a separate JVM for training.
 *       For a single JVM, this approach will be slower than doing it the normal way
 *
 * To change the UI port (usually not necessary) - set the org.deeplearning4j.ui.port system property
 * i.e., run the example and pass the following to the JVM, to use port 9001: -Dorg.deeplearning4j.ui.port=9001
 *
 * @author Alex Black
 */
public class RemoteUIExample {

    public static void main(String[] args){

        //------------ In the first JVM: Start the UI server and enable remote listener support ------------
        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        uiServer.enableRemoteListener();        //Necessary: remote support is not enabled by default
        //uiServer.enableRemoteListener(new FileStatsStorage(new File("myFile.dl4j")), true);       //Alternative: persist them to disk


        //------------ In the second JVM: Perform training ------------

        //Get our network and training data
        MultiLayerNetwork net = UIExampleUtils.getMnistNetwork();
        DataSetIterator trainData = UIExampleUtils.getMnistData();

        //Create the remote stats storage router - this sends the results to the UI via HTTP, assuming the UI is at http://localhost:9000
        StatsStorageRouter remoteUIRouter = new RemoteUIStatsStorageRouter("http://localhost:9000");
        net.setListeners(new StatsListener(remoteUIRouter));

        //Start training:
        net.fit(trainData);

        //Finally: open your browser and go to http://localhost:9000/train
    }
}
