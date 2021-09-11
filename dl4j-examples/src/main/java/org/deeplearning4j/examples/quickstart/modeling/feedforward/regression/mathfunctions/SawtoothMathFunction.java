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
 * This example transformes an inputted array-presumably an array representing particle
 * movement- such that the inputted values represent specific intervals at which a general sawtooth
 * wave function should be used to calculate the output
 *
 * @author Unknown
 * Documentation added by ERRM
 */
public class SawtoothMathFunction implements MathFunction {

    @Override
    public INDArray getFunctionValues(final INDArray x) {
        final double sawtoothPeriod = 4.0;
        //the input data is the intervals at which the wave is being calculated
        final double[] xd2 = x.data().asDouble();
        final double[] yd2 = new double[xd2.length];
        for (int i = 0; i < xd2.length; i++) {  //Using the sawtooth wave function, find the values at the given intervals
            yd2[i] = 2 * (xd2[i] / sawtoothPeriod - Math.floor(xd2[i] / sawtoothPeriod + 0.5));
        }
        return Nd4j.create(yd2, xd2.length, 1);  //Column vector
    }

    @Override
    public String getName() {
        return "Sawtooth";
    }
}
