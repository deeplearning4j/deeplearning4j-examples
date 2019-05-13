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

package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.*;


/**
 * --- Nd4j Example 9: Functions ---
 *
 * In this example, we'll see how apply some mathematical functions to a matrix
 *
 * Created by cvn on 9/7/14.
 */

public class Nd4jEx9_Functions {

    public static void main(String[] args) {

        INDArray nd = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, new int[]{2, 6});
        INDArray nd2 = Nd4j.create(new float[]{15,16,17,18,19,20,21,22,23,24,25,26,27,28}, new int[]{2, 7});
        INDArray ndv; // a placeholder variable to print out and leave the original data unchanged

        //this normalizes data and helps activate artificial neurons in deep-learning nets and assigns it to var ndv
        ndv = sigmoid(nd);
        System.out.println(ndv);

        //this gives you absolute value
        ndv = abs(nd);
        System.out.println(ndv);

        //a hyperbolic function to transform data much like sigmoid.
        ndv = tanh(nd);
        System.out.println(ndv);

        // ndv = hardTanh(nd);
        // System.out.println(ndv);

        //exponentiation
        ndv = exp(nd);
        System.out.println(ndv);

        //square root
        ndv = sqrt(nd);
        System.out.println(ndv);
    }
}
