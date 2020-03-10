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

package org.deeplearning4j.examples.rl4j;

import java.io.IOException;

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscrete;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteConv;
import org.deeplearning4j.rl4j.learning.configuration.A3CLearningConfiguration;
import org.deeplearning4j.rl4j.mdp.ale.ALEMDP;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdConv;
import org.deeplearning4j.rl4j.network.configuration.ActorCriticNetworkConfiguration;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.learning.config.Adam;

/**
 * @author saudet
 * <p>
 * Main example for A3C with The Arcade Learning Environment (ALE)
 */
public class A3CALE {

    public static HistoryProcessor.Configuration ALE_HP =
        new HistoryProcessor.Configuration(
            4,       //History length
            84,      //resize width
            110,     //resize height
            84,      //crop width
            84,      //crop height
            0,       //cropping x offset
            0,       //cropping y offset
            4        //skip mod (one frame is picked every x
        );

    public static A3CLearningConfiguration ALE_A3C =
        A3CLearningConfiguration.builder()
            .seed(123L)
            .maxEpochStep(10000)
            .maxStep(8000000)
            .numThreads(8)
            .nStep(32)
            .rewardFactor(0.1)
            .gamma(0.99)
            .build();


    public static final ActorCriticNetworkConfiguration ALE_NET_A3C =
        ActorCriticNetworkConfiguration.builder()
            .updater(new Adam(0.00025))
            .useLSTM(false)
            .build();


    public static void main(String[] args) throws IOException {

        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager(true);

        //setup the emulation environment through ALE, you will need a ROM file
        ALEMDP mdp = null;
        try {
            mdp = new ALEMDP("pong.bin");
        } catch (UnsatisfiedLinkError e) {
            System.out.println("To run this example, uncomment the \"ale-platform\" dependency in the pom.xml file.");
        }

        //setup the training
        A3CDiscreteConv<ALEMDP.GameScreen> a3c = new A3CDiscreteConv(mdp, ALE_NET_A3C, ALE_HP, ALE_A3C, manager);

        //start the training
        a3c.train();

        //save the model at the end
        a3c.getPolicy().save("ale-a3c.model");

        //close the ALE env
        mdp.close();
    }
}
