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

import org.deeplearning4j.rl4j.learning.async.nstepq.discrete.AsyncNStepQLearningDiscreteDense;
import org.deeplearning4j.rl4j.learning.configuration.AsyncQLearningConfiguration;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.learning.config.Adam;

import java.io.IOException;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/18/16.
 * <p>
 * main example for Async NStep QLearning on cartpole
 */
public class AsyncNStepCartpole {

    public static AsyncQLearningConfiguration CARTPOLE_NSTEPQ_CONFIG =
        AsyncQLearningConfiguration.builder()
            .seed(123L)
            .maxEpochStep(200)
            .maxStep(300000)
            .numThreads(16)
            .nStep(5)
            .targetDqnUpdateFreq(100)
            .updateStart(10)
            .rewardFactor(0.01)
            .gamma(0.99)
            .errorClamp(100.0)
            .minEpsilon(0.1)
            .epsilonNbStep(9000)
            .build();

    private static DQNDenseNetworkConfiguration CARTPOLE_NET_CONFIG = DQNDenseNetworkConfiguration.builder()
        .l2(0.001)
        .updater(new Adam(0.0005))
        .numHiddenNodes(16)
        .numLayers(3)
        .build();


    public static void main(String[] args) throws IOException {
        cartPole();
    }


    public static void cartPole() throws IOException {

        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager(true);


        //define the mdp from gym (name, render)
        GymEnv mdp = null;
        try {
            mdp = new GymEnv("CartPole-v0", false, false);
        } catch (RuntimeException e) {
            System.out.print("To run this example, download and start the gym-http-api repo found at https://github.com/openai/gym-http-api.");
        }

        //define the training
        AsyncNStepQLearningDiscreteDense<Box> dql = new AsyncNStepQLearningDiscreteDense<Box>(mdp, CARTPOLE_NET_CONFIG, CARTPOLE_NSTEPQ_CONFIG, manager);

        //train
        dql.train();

        //get the final policy
        DQNPolicy<Box> pol = (DQNPolicy<Box>) dql.getPolicy();

        //serialize and save (serialization showcase, but not required)
        pol.save("/tmp/pol1");

        //close the mdp (close connection)
        mdp.close();

        //reload the policy, will be equal to "pol"
        DQNPolicy<Box> pol2 = DQNPolicy.load("/tmp/pol1");
    }
}
