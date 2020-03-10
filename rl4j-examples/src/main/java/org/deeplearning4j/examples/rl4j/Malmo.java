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

import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.util.DataManager;

import org.deeplearning4j.malmo.MalmoBox;
import org.deeplearning4j.malmo.MalmoActionSpaceDiscrete;
import org.deeplearning4j.malmo.MalmoConnectionError;
import org.deeplearning4j.malmo.MalmoDescretePositionPolicy;
import org.deeplearning4j.malmo.MalmoEnv;
import org.deeplearning4j.malmo.MalmoObservationSpace;
import org.deeplearning4j.malmo.MalmoObservationSpacePosition;
import org.nd4j.linalg.learning.config.Adam;

import java.io.IOException;
import java.util.logging.Logger;

/**
 * Simple example for Malmo DQN w/ x,y,z position as input
 *
 * @author howard-abrams (howard.abrams@ca.com) on 1/12/17.
 */
public class Malmo {

    private static QLearningConfiguration MALMO_Q_LEARNING_CONFIG = QLearningConfiguration.builder()
        .maxStep(200000)
        .expRepMaxSize(200000)
        .targetDqnUpdateFreq(50)
        .minEpsilon(0.15)
        .rewardFactor(0.1)
        .epsilonNbStep(1000)
        .doubleDQN(true)
        .build();

    private static DQNDenseNetworkConfiguration MALMO_NET_CONFIG = DQNDenseNetworkConfiguration.builder()
        .updater(new Adam(0.001))
        .numHiddenNodes(50)
        .numLayers(3)
        .build();

    public static void main(String[] args) throws IOException {
        try {
            malmoCliffWalk();
            loadMalmoCliffWalk();
        } catch (MalmoConnectionError e) {
            System.out.println(
                "To run this example, download and start Project Malmo found at https://github.com/Microsoft/malmo.");
        }
    }

    private static MalmoEnv createMDP() {
        // Create action space comprised of just discrete north, south, east and west
        MalmoActionSpaceDiscrete actionSpace =
            new MalmoActionSpaceDiscrete("movenorth 1", "movesouth 1", "movewest 1", "moveeast 1");

        // Create a basic observation space that simply contains the x, y, z world position
        MalmoObservationSpace observationSpace = new MalmoObservationSpacePosition();

        // Create a simple policy that just ensures the agent has moved and there is a reward
        MalmoDescretePositionPolicy obsPolicy = new MalmoDescretePositionPolicy();

        // Create the MDP with the above arguments, and load a mission using an XML file
        return new MalmoEnv("cliff_walking_rl4j.xml", actionSpace, observationSpace, obsPolicy);
    }

    public static void malmoCliffWalk() throws MalmoConnectionError, IOException {
        // record the training data in rl4j-data in a new folder (save)
        DataManager manager = new DataManager(true);

        // Create the MDP complete with a Malmo mission
        MalmoEnv mdp = createMDP();

        // define the training
        QLearningDiscreteDense<MalmoBox> dql = new QLearningDiscreteDense<MalmoBox>(mdp, MALMO_NET_CONFIG, MALMO_Q_LEARNING_CONFIG, manager);

        // train
        dql.train();

        // get the final policy
        DQNPolicy<MalmoBox> pol = dql.getPolicy();

        // serialize and save (serialization showcase, but not required)
        pol.save("cliffwalk.policy");

        // close the mdp
        mdp.close();
    }

    // showcase serialization by using the trained agent on a new similar mdp
    public static void loadMalmoCliffWalk() throws MalmoConnectionError, IOException {
        // Create the MDP complete with a Malmo mission
        MalmoEnv mdp = createMDP();

        // load the previous agent
        DQNPolicy<MalmoBox> pol = DQNPolicy.load("cliffwalk.policy");

        // evaluate the agent 10 times
        double rewards = 0;
        for (int i = 0; i < 10; i++) {
            double reward = pol.play(mdp);
            rewards += reward;
            Logger.getAnonymousLogger().info("Reward: " + reward);
        }

        // Clean up
        mdp.close();

        // Print average reward over 10 runs
        Logger.getAnonymousLogger().info("average: " + rewards / 10);
    }
}
