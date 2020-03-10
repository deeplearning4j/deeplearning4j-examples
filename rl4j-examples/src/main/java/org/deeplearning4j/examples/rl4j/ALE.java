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
import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteConv;
import org.deeplearning4j.rl4j.mdp.ale.ALEMDP;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdConv;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.learning.config.Adam;

/**
 * @author saudet
 * <p>
 * Main example for DQN with The Arcade Learning Environment (ALE)
 */
public class ALE {

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

    private static QLearningConfiguration ALE_QLEARNING_CONFIG = QLearningConfiguration.builder()
        .seed(123L)
        .maxEpochStep(10000)
        .maxStep(8000000)
        .expRepMaxSize(1000000)
        .batchSize(32)
        .targetDqnUpdateFreq(10000)
        .updateStart(500)
        .rewardFactor(0.1)
        .gamma(0.99)
        .errorClamp(100.0)
        .minEpsilon(0.1)
        .epsilonNbStep(100000)
        .doubleDQN(true)
        .build();

    public static DQNDenseNetworkConfiguration ALE_NET_CONFIG =
        DQNDenseNetworkConfiguration.builder()
            .updater(new Adam(0.00025))
            .build();

    public static void main(String[] args) throws IOException {

        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager(true);

        //setup the emulation environment through ALE, you will need a ROM file
        ALEMDP mdp = null;
        try {
            mdp = new ALEMDP("../../ALE/ROMS/Breakout.bin");
        } catch (UnsatisfiedLinkError e) {
            System.out.println("To run this example, uncomment the \"ale-platform\" dependency in the pom.xml file.");
        }
        //setup the training
        QLearningDiscreteConv<ALEMDP.GameScreen> dql = new QLearningDiscreteConv(mdp, ALE_NET_CONFIG, ALE_HP, ALE_QLEARNING_CONFIG, manager);

        //start the training
        dql.train();

        //save the model at the end
        dql.getPolicy().save("ale-dql.model");

        //close the ALE env
        mdp.close();
    }
}
