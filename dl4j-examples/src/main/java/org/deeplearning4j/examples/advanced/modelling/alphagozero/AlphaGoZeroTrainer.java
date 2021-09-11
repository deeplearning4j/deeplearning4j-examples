/*******************************************************************************
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

package org.deeplearning4j.examples.advanced.modelling.alphagozero;

import org.deeplearning4j.examples.advanced.modelling.alphagozero.dualresidual.DualResnetModel;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Train AlphaGo Zero model on dummy data. To run a full AGZ system with DL4J, check out
 *
 * https://github.com/maxpumperla/ScalphaGoZero
 *
 * for a complete example. The input to the network has a number of "planes" the size of the
 * board, i.e. in most cases 19x19. One such plane could be "the number of liberties each
 * black stone on the board has". In the AGZ paper a total of 11 planes are used, while in
 * the previous AlphaGo version there have been 48 (resp. 49).
 *
 * The output of the policy head of the network produces move probabilities and emits one
 * probability for each move on the board, including passing (i.e. 19x19 + 1 = 362 in total).
 * The value head produces winning probabilities for the current position.
 *
 * @author Max Pumperla
 */
public class AlphaGoZeroTrainer {

    private static final Logger log = LoggerFactory.getLogger(AlphaGoZeroTrainer.class);

    public static void main(String[] args) {

        int miniBatchSize = 32;
        int boardSize = 19;

        int numResidualBlocks = 20;
        int numFeaturePlanes = 11;

        log.info("Initializing AGZ model");
        ComputationGraph model = DualResnetModel.getModel(numResidualBlocks, numFeaturePlanes);

        log.info("Create dummy data");
        INDArray input = Nd4j.create(miniBatchSize,numFeaturePlanes, boardSize, boardSize);

        // move prediction has one value for each point on the board (19x19) plus one for passing.
        INDArray policyOutput = Nd4j.create(miniBatchSize, boardSize * boardSize + 1);

        // the value network spits out a value between 0 and 1 to assess how good the current board situation is.
        INDArray valueOutput = Nd4j.create(miniBatchSize, 1);

        log.info("Train AGZ model");
        model.fit(new INDArray[] {input}, new INDArray[] {policyOutput, valueOutput});
    }
}
