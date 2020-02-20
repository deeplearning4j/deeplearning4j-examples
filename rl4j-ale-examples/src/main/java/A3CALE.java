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

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscrete;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteConv;
import org.deeplearning4j.rl4j.mdp.ale.ALEMDP;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdConv;
import org.nd4j.linalg.learning.config.Adam;

import java.io.IOException;

/**
 * @author saudet
 *
 * Main example for A3C with The Arcade Learning Environment (ALE)
 *
 */
public class A3CALE {

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

        A3CDiscrete.A3CConfiguration ALE_A3C = new A3CDiscrete.A3CConfiguration(
                        123,            //Random seed
                        10000,          //Max step By epoch
                        8000000,        //Max step
                        8,              //Number of threads
                        32,             //t_max
                        500,            //num step noop warmup
                        0.1,            //reward scaling
                        0.99,           //gamma
                        10.0            //td-error clipping
                );

        final ActorCriticFactoryCompGraphStdConv.Configuration ALE_NET_A3C =
                new ActorCriticFactoryCompGraphStdConv.Configuration(
                        0.000,   //l2 regularization
                        new Adam(0.00025), //learning rate
                        null, false
                );



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
