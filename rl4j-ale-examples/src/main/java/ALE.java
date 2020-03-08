/* *****************************************************************************
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

import java.io.IOException;
import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteConv;
import org.deeplearning4j.rl4j.mdp.ale.ALEMDP;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdConv;

/**
 * @author saudet
 *
 * Main example for DQN with The Arcade Learning Environment (ALE)
 * This sample shows how to set up a simple ALE for training. This setup will take a long time to master the game.
 */
public class ALE {

    public static void main(String[] args) throws IOException {

        HistoryProcessor.Configuration ALE_HP = new HistoryProcessor.Configuration(
                4,       //History length
                84,      //resize width
                110,     //resize height
                84,      //crop width
                84,      //crop height
                0,       //cropping x offset
                0,       //cropping y offset
                4        //skip mod (one frame is picked every x
        );

        QLearning.QLConfiguration ALE_QL =
                new QLearning.QLConfiguration(
                        123,      //Random seed
                        10000,    //Max step By epoch
                        8000000,  //Max step
                        1000000,  //Max size of experience replay
                        32,       //size of batches
                        10000,    //target update (hard)
                        500,      //num step noop warmup
                        0.1,      //reward scaling
                        0.99,     //gamma
                        100.0,    //td-error clipping
                        0.1f,     //min epsilon
                        100000,   //num step for eps greedy anneal
                        true      //double-dqn
                );

        DQNFactoryStdConv.Configuration ALE_NET_QL =
                new DQNFactoryStdConv.Configuration(
                        0.00025, //learning rate
                        0.000,   //l2 regularization
                        null, null
                );

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
