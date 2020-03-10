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

import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.async.nstepq.discrete.AsyncNStepQLearningDiscrete;
import org.deeplearning4j.rl4j.learning.async.nstepq.discrete.AsyncNStepQLearningDiscreteDense;
import org.deeplearning4j.rl4j.learning.configuration.AsyncQLearningConfiguration;
import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.mdp.toy.HardDeteministicToy;
import org.deeplearning4j.rl4j.mdp.toy.SimpleToy;
import org.deeplearning4j.rl4j.mdp.toy.SimpleToyState;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.learning.config.Adam;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/11/16.
 * <p>
 * main example for toy DQN
 */
public class Toy {

    private static QLearningConfiguration TOY_QLEARNING_CONFIG =
        QLearningConfiguration.builder()
            .seed(123L)
            .maxEpochStep(100000)
            .maxStep(80000)
            .expRepMaxSize(10000)
            .batchSize(32)
            .targetDqnUpdateFreq(100)
            .rewardFactor(0.05)
            .gamma(0.99)
            .errorClamp(10.0)
            .minEpsilon(0.1)
            .epsilonNbStep(2000)
            .doubleDQN(true)
            .build();


    public static AsyncQLearningConfiguration TOY_ASYNC_QL =
        AsyncQLearningConfiguration.builder()
            .seed(123L)
            .maxEpochStep(100000)
            .maxStep(80000)
            .numThreads(8)
            .nStep(5)
            .targetDqnUpdateFreq(100)
            .rewardFactor(0.1)
            .gamma(0.99)
            .errorClamp(10.0)
            .minEpsilon(0.1)
            .epsilonNbStep(2000)
            .build();


    public static DQNDenseNetworkConfiguration TOY_NET =
        DQNDenseNetworkConfiguration.builder()
            .l2(0.01)
            .updater(new Adam(1e-2))
            .numLayers(3)
            .numHiddenNodes(16)
            .build();

    public static void main(String[] args) throws IOException {
        simpleToy();
        //toyAsyncNstep();

    }

    public static void simpleToy() throws IOException {

        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager();

        //define the mdp from toy (toy length)
        SimpleToy mdp = new SimpleToy(20);

        //define the training method
        Learning<SimpleToyState, Integer, DiscreteSpace, IDQN> dql = new QLearningDiscreteDense<SimpleToyState>(mdp, TOY_NET, TOY_QLEARNING_CONFIG, manager);

        //enable some logging for debug purposes on toy mdp
        mdp.setFetchable(dql);

        //start the training
        dql.train();

        //useless on toy but good practice!
        mdp.close();

    }

    public static void hardToy() throws IOException {

        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager();

        //define the mdp from toy (toy length)
        MDP mdp = new HardDeteministicToy();

        //define the training
        ILearning<SimpleToyState, Integer, DiscreteSpace> dql = new QLearningDiscreteDense(mdp, TOY_NET, TOY_QLEARNING_CONFIG, manager);

        //start the training
        dql.train();

        //useless on toy but good practice!
        mdp.close();


    }


    public static void toyAsyncNstep() throws IOException {

        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager();

        //define the mdp
        SimpleToy mdp = new SimpleToy(20);

        //define the training
        AsyncNStepQLearningDiscreteDense dql = new AsyncNStepQLearningDiscreteDense<SimpleToyState>(mdp, TOY_NET, TOY_ASYNC_QL, manager);

        //enable some logging for debug purposes on toy mdp
        mdp.setFetchable(dql);

        //start the training
        dql.train();

        //useless on toy but good practice!
        mdp.close();

    }

}
