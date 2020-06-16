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

package org.deeplearning4j.datapipelineexamples.loading;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 *
 * @author Robert Altena
 */
public class Ex06_KFoldIteratorFromDataSet {

    public static void main(String[] args) {

        INDArray x = Nd4j.create(new float[][]{{1,2},{2,3},{3,4},{4,5}});
        INDArray y = Nd4j.create(new float[][]{{1},{2},{3},{4}});
        DataSet ds = new DataSet(x,y);

        System.out.println("Full dataset: ");
        System.out.println(ds);

        KFoldIterator kIter = new KFoldIterator(2, ds);
        while (kIter.hasNext()){
            DataSet now = kIter.next();
            DataSet test = kIter.testFold();
            System.out.println();
            System.out.println("Train: ");
            System.out.println(now);
            System.out.println("Test: ");
            System.out.println(test);

        }
    }
}
