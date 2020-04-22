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
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteConv;
import org.deeplearning4j.rl4j.mdp.ale.ALEMDP;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdConv;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author saudet
 * <p>
 * Main example for DQN with The Arcade Learning Environment (ALE)
 */
public class ALE {

    public static HistoryProcessor.Configuration ALE_HP =
        new HistoryProcessor.Configuration(
            4,       //History length
            152,      //resize width
            194,     //resize height
            152,      //crop width
            194,      //crop height
            8,       //cropping x offset
            32,       //cropping y offset
            4       //skip mod (one frame is picked every x
        );

    public static QLearning.QLConfiguration ALE_QL =
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

    public static DQNFactoryStdConv.Configuration ALE_NET_QL =
        new DQNFactoryStdConv.Configuration(
            0.00025, //learning rate
            0.000,   //l2 regularization
            null, null
        );

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
        QLearningDiscreteConv<ALEMDP.GameScreen> dql = new QLearningDiscreteConv(mdp, ALE_NET_QL, ALE_HP, ALE_QL, manager);

        //start the training
        dql.train();

        //save the model at the end
        dql.getPolicy().save("ale-dql.model");

        //close the ALE env
        mdp.close();
    }
}
