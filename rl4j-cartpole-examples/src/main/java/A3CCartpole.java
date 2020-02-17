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

import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscrete;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteDense;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactorySeparateStdDense;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.learning.config.Adam;

import java.io.IOException;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/18/16.
 *
 * A3C on cartpole
 * This example shows the classes in rl4j that implement the article here: https://arxiv.org/abs/1602.01783
 * Asynchronous Methods for Deep Reinforcement Learning. Mnih et al.
 *
 */
public class A3CCartpole {

    public static void main(String[] args) throws IOException {
        A3CcartPole();
    }

    private static void A3CcartPole() throws IOException {

        //define the mdp from gym (name, render)
        String envUD = "CartPole-v1";
        GymEnv<Encodable, Integer, DiscreteSpace> mdp = new GymEnv<Encodable, Integer, DiscreteSpace>(envUD, false, false);

        A3CDiscrete.A3CConfiguration CARTPOLE_A3C =
                new A3CDiscrete.A3CConfiguration(
                        123,            //Random seed
                        200,    //Max step By epoch
                        500000,      //Max step
                        8,         //Number of threads
                        20,             //t_max
                        10,        //num step noop warmup
                        0.01,     //reward scaling
                        0.99,          //gamma
                        1.0         //td-error clipping
                );

        ActorCriticFactorySeparateStdDense.Configuration CARTPOLE_NET_A3C =  ActorCriticFactorySeparateStdDense.Configuration
                .builder().updater(new Adam(1e-2)).l2(0).numHiddenNodes(16).numLayer(3).build();

        //define the training
        A3CDiscreteDense<Encodable> a3c = new A3CDiscreteDense<Encodable>(mdp, CARTPOLE_NET_A3C, CARTPOLE_A3C);

        a3c.train(); //start the training
        mdp.close();

        ACPolicy<org.deeplearning4j.rl4j.space.Encodable> pol = a3c.getPolicy();

        pol.save("/tmp/val1/", "/tmp/pol1");

        //reload the policy, will be equal to "pol", but without the randomness
        ACPolicy<Box> pol2 = ACPolicy.load("/tmp/val1/", "/tmp/pol1");
        Cartpole.loadCartpole(pol2, envUD);
        System.out.println("sample finished.");
    }

}
