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

package org.nd4j.examples.advanced.operations;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * --- Nd4j Example 11: Axpy ---
 * *
 * @author Adam Gibson
 */
public class Nd4jEx11_Axpy {

    public static void main(String[] args) {

        INDArray arr = Nd4j.create(DataType.DOUBLE, 300);
        double numTimes = 10000000;
        double total = 0;

        for(int i = 0; i < numTimes; i++) {
            long start = System.nanoTime();
            Nd4j.getBlasWrapper().axpy(new Integer(1), arr,arr);
            long after = System.nanoTime();
            long add = Math.abs(after - start);
            System.out.println("Took " + add);
            total += Math.abs(after - start);
        }
        System.out.println("Avg time " + (total / numTimes));
    }
}
