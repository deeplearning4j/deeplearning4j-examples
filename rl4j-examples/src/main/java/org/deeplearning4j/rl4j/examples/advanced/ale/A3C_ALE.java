/* *****************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 * Copyright (c) 2020 Konduit K.K.
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
package org.deeplearning4j.rl4j.examples.advanced.ale;

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscrete;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteConv;
import org.deeplearning4j.rl4j.mdp.ale.ALEMDP;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdConv;
import org.nd4j.linalg.learning.config.Adam;

import java.io.IOException;

/**
 *
 * Main example for A3C with The Arcade Learning Environment (ALE)
 *
 */
public class A3C_ALE {

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

        A3CDiscrete.A3CConfiguration ALE_A3C = A3CDiscrete.A3CConfiguration.builder()
            .seed(123)
            .maxEpochStep(10000)
            .maxStep(8000000)
            .numThread(8)
            .nstep(32)
            .updateStart(500)
            .rewardFactor(0.1)
            .gamma(0.99)
            .errorClamp(10.0)
            .build();

        final ActorCriticFactoryCompGraphStdConv.Configuration ALE_NET_A3C =
                ActorCriticFactoryCompGraphStdConv.Configuration.builder()
            .l2(0)
            .updater(new Adam(0.00025)) // Learning Rate with Adam Updater
            .build();

        //setup the emulation environment through ALE, you will need a ROM file
        ALEMDP mdp = new ALEMDP("pong.bin");

        //setup the training
        A3CDiscreteConv<ALEMDP.GameScreen> a3c = new A3CDiscreteConv<ALEMDP.GameScreen>(mdp, ALE_NET_A3C, ALE_HP, ALE_A3C);

        //start the training
        a3c.train();

        //save the model at the end
        a3c.getPolicy().save("ale-a3c.model");

        //close the ALE env
        mdp.close();
    }
}
