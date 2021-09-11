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
import org.nd4j.linalg.factory.Nd4j;

/**
 * Calculate function value of Triangle Wave, which is an absolute value of Sawtooth Wave.
 *
 */
public class TriangleWaveMathFunction implements MathFunction {

    @Override
    public INDArray getFunctionValues(final INDArray x) {
        final double period = 6.0;
        final double[] xd = x.data().asDouble();
        final double[] yd = new double[xd.length];
        for (int i = 0; i < xd.length; i++) {
            yd[i] = Math.abs(2 * (xd[i] / period - Math.floor(xd[i] / period + 0.5)));
        }
        return Nd4j.create(yd, xd.length, 1);  //Column vector
    }

    @Override
    public String getName() {
        return "TriangleWave";
    }
}
