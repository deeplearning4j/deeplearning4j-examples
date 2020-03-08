/* *****************************************************************************
 * Copyright (c) 2020 Konduit, Inc.
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
import org.deeplearning4j.rl4j.mdp.ale.ALEMDP;
import org.deeplearning4j.rl4j.policy.ACPolicy;

import java.io.IOException;

/**
 *  @author robaltena
 *
 *  This sample shows how to plat an ALE game with a trained model.
 */
public class PlayALE {
    public static void main(String[] args) throws IOException {
        ALEMDP mdp = new ALEMDP("pong.bin", true);

        //load the trained agent
        ACPolicy<ALEMDP.GameScreen> pol2 = ACPolicy.load("ale-a3c.model");

        //The training history processor used for data pre processing steps.
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

        pol2.play(mdp, ALE_HP);
        mdp.close();
    }
}
