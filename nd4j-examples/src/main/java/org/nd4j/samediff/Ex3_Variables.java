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

package org.nd4j.samediff;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.TruncatedNormalDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.DistributionInitScheme;

/**
 * This example: shows some more ways to create variables
 *
 * @author Alex Black
 */
public class Ex3_Variables {

    public static void main(String[] args) {

        SameDiff sd = SameDiff.create();

        //There are multiple ways to create the initial variables in your graph
        // First, you can create them directly, with the specified array:
        INDArray arr = Nd4j.arange(10);
        SDVariable var1 = sd.var("var1", arr);

        /*
        Or create a variable by specifying a shape and an initializer.
        The idea with an initializer is that is specifies how the initial value should be created.
        This initialization is performed once only, when the array is first required.

        -- Available Initialization Schemes --
        ContantInitScheme
        DistributionInitScheme
        IdentityInitScheme
        LecunInitScheme
        OneInitScheme
        ReluInitScheme
        ReluUniformInitScheme
        SigmoidInitScheme
        UniformInitScheme
        VarScalingNormalFanAvgInitScheme
        VarScalingNormalFanInInitScheme
        VarScalingNormalFanOutInitScheme
        VarScalingNormalUniformFanInInitScheme
        VarScalingNormalUniformFanOutInitScheme
        VarScalingUniformFanAvgInitScheme
        XavierFanInInitScheme
        XavierInitScheme
        XavierUniformInitScheme
        ZeroInitScheme
         */
        long[] shape = new long[]{3,4};
        WeightInitScheme initScheme = new DistributionInitScheme(Nd4j.order(), new TruncatedNormalDistribution(0, 1));
        SDVariable var2 = sd.var("var2", initScheme, DataType.FLOAT, shape);

        //Note that the array will be allocated using the specified distribution (truncated normal), when we try to get the array:
        INDArray var2Array = var2.getArr();
        System.out.println("var2 array values:\n" + var2Array);

        //Alternatively, we can simply specify an shape. This will default to a zero initialization for the array (if required)
        // or you can set the array directly
        SDVariable var3 = sd.var("var3", 3,4);

        INDArray values = Nd4j.ones(3,4);
        var3.setArray(values);


        //Note also that there are a number of functions that can be used to create variables.
        //However, unlike the WeightInitScheme of the variables earlier, the random values here will be re-generated
        // on every forward pass
        SDVariable scalar = sd.scalar("scalar", 0.5);
        SDVariable zero = sd.zero("zero", new long[]{3,4});
        SDVariable zeroToNine = sd.linspace("zeroToNine", DataType.FLOAT, 0, 9, 10);
        SDVariable randomUniform = sd.random().uniform(-1, 1, 3,4);      //-1 to 1, shape [3,4]
        SDVariable randomBernoulli = sd.random().bernoulli(0.5, 3,4);          //Random Bernoulli: 0 or 1 with probability 0.5
    }

}
