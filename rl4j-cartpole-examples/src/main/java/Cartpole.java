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

import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.space.Box;
import org.nd4j.linalg.learning.config.Adam;

import java.util.logging.Logger;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/11/16.
 *
 * Main example for Cartpole DQN
 */
public class Cartpole
{
    private static QLearning.QLConfiguration CARTPOLE_QL =
            new QLearning.QLConfiguration(
                    123,    //Random seed
                    200,    //Max step By epoch
                    150000, //Max step
                    150000, //Max size of experience replay
                    32,     //size of batches
                    500,    //target update (hard)
                    10,     //num step noop warmup
                    0.01,   //reward scaling
                    0.99,   //gamma
                    1.0,    //td-error clipping
                    0.1f,   //min epsilon
                    1000,   //num step for eps greedy anneal
                    true    //double DQN
            );

    private static DQNFactoryStdDense.Configuration CARTPOLE_NET =
            DQNFactoryStdDense.Configuration.builder()
                    .l2(0.001).updater(new Adam(0.0005)).numHiddenNodes(16).numLayer(3).build();

    public static void main(String[] args) {
        DQNPolicy<Box>  pol = cartPole();
        loadCartpole(pol);
    }

    private static DQNPolicy<Box> cartPole() {
        //define the mdp from gym (name, render)
        GymEnv<Box, Integer, org.deeplearning4j.rl4j.space.DiscreteSpace> mdp = new GymEnv<Box, Integer, org.deeplearning4j.rl4j.space.DiscreteSpace>("CartPole-v0", false, false);
        QLearningDiscreteDense<Box> dql = new QLearningDiscreteDense<Box>(mdp, CARTPOLE_NET, CARTPOLE_QL);

        dql.train();
        mdp.close();

        return dql.getPolicy(); //get the final policy
    }

    private static void loadCartpole(DQNPolicy<Box> pol) {
        //use the trained agent on a new similar mdp (but render it this time)

        //define the mdp from gym (name, render)
        GymEnv<Box, Integer, org.deeplearning4j.rl4j.space.ActionSpace<Integer>> mdp2 = new GymEnv<Box, Integer, org.deeplearning4j.rl4j.space.ActionSpace<Integer>>("CartPole-v0", true, false);

        //evaluate the agent
        double rewards = 0;
        for (int i = 0; i < 1000; i++) {
            mdp2.reset();
            double reward = pol.play(mdp2);
            rewards += reward;
            Logger.getAnonymousLogger().info("Reward: " + reward);
        }

        Logger.getAnonymousLogger().info("average: " + rewards/1000);
    }
}
