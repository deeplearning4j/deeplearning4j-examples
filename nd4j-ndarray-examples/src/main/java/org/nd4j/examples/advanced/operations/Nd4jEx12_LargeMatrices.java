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

package org.nd4j.examples.advanced.operations;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * --- Nd4j Example 13: Large Matrix ---
 *
 * In this example, we'll see operations with a large matrix
 *
 * @author Adam Gibson
 */
public class Nd4jEx12_LargeMatrices {

    public static void main(String[] args) {
        INDArray n = Nd4j.linspace(1,10000000,10000000).reshape(1, 10000000);
        System.out.println("MMUL" + n.mmul(n.transpose()));
    }
}
