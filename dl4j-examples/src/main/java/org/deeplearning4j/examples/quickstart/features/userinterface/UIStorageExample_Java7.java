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
import org.deeplearning4j.ui.model.stats.J7StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

/**
 * A variant of the UI example showing the approach for Java 7 compatibility
 *
 * *** Notes ***
 * 1: If you don't specifically need Java 7, use the approach in the standard UIStorageExample as it should be faster
 * 2: The UI itself requires Java 8 (uses the Play framework as a backend). But you can store stats on one machine, copy
 *    the file to another (with Java 8) and visualize there
 * 3: J7FileStatsStorage and FileStatsStorage formats are NOT compatible. Save/load with the same one
 *    (J7FileStatsStorage works on Java 8 too, but FileStatsStorage does not work on Java 7)
 *
 * @author Alex Black
 */
public class UIStorageExample_Java7 {

    public static void main(String[] args){
        MultiLayerNetwork net = UIExampleUtils.getMnistNetwork();
        DataSetIterator trainData = UIExampleUtils.getMnistData();

        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats-j7.dl4j"));
        net.setListeners(new J7StatsListener(statsStorage), new ScoreIterationListener(10));
        UIServer.getInstance().attach(statsStorage);

        net.fit(trainData);

        System.out.println("Done");
    }
}
