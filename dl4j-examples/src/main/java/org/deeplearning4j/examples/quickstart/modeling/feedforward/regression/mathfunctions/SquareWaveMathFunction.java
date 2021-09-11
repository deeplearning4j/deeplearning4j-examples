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

package org.deeplearning4j.examples.quickstart.modeling.feedforward.regression.mathfunctions;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Sign(x) or Sign of a real number, x, is -1 if x is negative, 0 if x is zero and 1 if x is positive.
 *
 * Calculate function value of Sign of Sine of x, which can be -1, 0 or 1.
 * The three possible outputs of Sign(sin) will form a line that resembles "squares" in the graph.
 */
public class SquareWaveMathFunction implements MathFunction {

    @Override
    public INDArray getFunctionValues(final INDArray x) {
        INDArray sin = Transforms.sin(x, true);
        return Transforms.sign(sin, false);
    }

    @Override
    public String getName() {
        return "SquareWave";
    }
}
