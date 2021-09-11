/* *****************************************************************************
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
package org.deeplearning4j.rl4j.examples.advanced.ale;

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteConv;
import org.deeplearning4j.rl4j.mdp.ale.ALEMDP;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdConv;

import java.io.IOException;

/**
 *
 * Main example for DQN with The Arcade Learning Environment (ALE)
 * This sample shows how to set up a simple ALE for training. This setup will take a long time to master the game.
 */
public class DQN_ALE {

    public static void main(String[] args) throws IOException {

        HistoryProcessor.Configuration ALE_HP = HistoryProcessor.Configuration.builder()
            .historyLength(4)
            .rescaledWidth(84)
            .rescaledHeight(110)
            .croppingWidth(84)
            .croppingHeight(84)
            .offsetX(0)
            .offsetY(0)
            .skipFrame(4)
            .build();

        QLearning.QLConfiguration ALE_QL = QLearning.QLConfiguration.builder()
            .seed(123)
            .maxEpochStep(1000)
            .maxStep(8000000)
            .expRepMaxSize(1000000)
            .batchSize(32)
            .targetDqnUpdateFreq(10000)
            .updateStart(500)
            .rewardFactor(0.1)
            .gamma(0.99)
            .errorClamp(100.0)
            .minEpsilon(0.1f)
            .epsilonNbStep(100000)
            .doubleDQN(true)
            .build();

        DQNFactoryStdConv.Configuration ALE_NET_QL =
                DQNFactoryStdConv.Configuration.builder()
            .learningRate(0.00025)
            .l2(0)
            .build();

        //setup the emulation environment through ALE, you will need a ROM file
        // set render to true to see the agent play (poorly). You can also see how slowly the data is generated and
        // understand why training would take a long time.
        ALEMDP mdp = new ALEMDP("pong.bin", true);

        //setup the training
        QLearningDiscreteConv<ALEMDP.GameScreen> dql = new QLearningDiscreteConv<ALEMDP.GameScreen>(mdp, ALE_NET_QL, ALE_HP, ALE_QL);

        dql.train(); //start the training
        dql.getPolicy().save("ale-dql.model"); //save the model at the end
        mdp.close();
    }
}
