/* *****************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

package org.nd4j.examples.quickstart;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 * --- Nd4j Example 6: Boolean Indexing ---
 *
 * In this example, we'll see ways to use boolean indexing to perform some simple conditional element-wise operations
 *
 * @author Alex Black
 */
public class Nd4jEx6_BooleanIndexing {

    public static void main(String[] args){

        long nRows = 3;
        long nCols = 5;
        long rngSeed = 12345;
        //Generate random numbers between -1 and +1
        Nd4j.getRandom().setSeed(rngSeed);
        INDArray random = Nd4j.rand(DataType.FLOAT, nRows, nCols).muli(2).subi(1);

        System.out.println("Array values:");
        System.out.println(random);

        //For example, we can conditionally replace values less than 0.0 with 0.0:
        INDArray randomCopy = random.dup();
        BooleanIndexing.replaceWhere(randomCopy, 0.0, Conditions.lessThan(0.0));
        System.out.println("After conditionally replacing negative values:\n" + randomCopy);

        //Or conditionally replace NaN values:
        INDArray hasNaNs = Nd4j.create(new double[]{1.0,1.0,Double.NaN,1.0});
        BooleanIndexing.replaceWhere(hasNaNs,0.0, Conditions.isNan());
        System.out.println("hasNaNs after replacing NaNs with 0.0:\n" + hasNaNs);

        //Or we can conditionally copy values from one array to another:
        randomCopy = random.dup();
        INDArray tens = Nd4j.valueArrayOf(nRows, nCols, 10.0);
        BooleanIndexing.replaceWhere(randomCopy, tens, Conditions.lessThan(0.0));
        System.out.println("Conditionally copying values from array 'tens', if original value is less than 0.0\n" + randomCopy);


        //One simple task is to count the number of values that match the condition
        MatchCondition op = new MatchCondition(random, Conditions.greaterThan(0.0));
        int countGreaterThanZero = Nd4j.getExecutioner().exec(op).getInt(0);
        System.out.println("Number of values matching condition 'greater than 0': " + countGreaterThanZero);
    }

}
