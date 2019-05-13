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

package org.deeplearning4j.examples.recurrent.processlottery;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;


/**
 * a->b b->c c->d ....
 * @author WangFeng
 */
public class LotteryCharacterSequenceDataSetReader extends BaseDataSetReader {

    public LotteryCharacterSequenceDataSetReader(File file) {
        filePath = file.toPath();
        doInitialize();
    }

    public DataSet next(int num) {
        //Though ‘c’ order arrays will also work, performance will be reduced due to the need to copy the arrays to ‘f’ order first, for certain operations.
        INDArray features = Nd4j.create(new int[]{num, 10, 16}, 'f');
        INDArray labels = Nd4j.create(new int[]{num, 10, 16}, 'f');


        INDArray featuresMask = null;
        INDArray labelsMask = null;
        for (int i =0; i < num && iter.hasNext(); i ++) {
            String featureStr = iter.next();
            currentCursor ++;
            featureStr = featureStr.replaceAll(",", "");
            String[] featureAry = featureStr.split("");
            for (int j = 0; j < featureAry.length - 1; j ++) {
                int feature = Integer.parseInt(featureAry[j]);
                int label = Integer.parseInt(featureAry[j + 1]);
                features.putScalar(new int[]{i, feature, j}, 1.0);
                labels.putScalar(new int[]{i, label, j}, 1.0);
            }
        }
        return new DataSet(features, labels, featuresMask, labelsMask);
    }

}

