/* *****************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

        pol2.play(mdp, ALE_HP);
        mdp.close();
    }
}
