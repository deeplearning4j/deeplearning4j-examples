/*******************************************************************************
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

package org.deeplearning4j.examples.advanced.modelling.sequenceprediction;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * lottery numbers -> lottery numbers
 * @author WangFeng
 */

public class LotteryCombinationDataSetReader extends BaseDataSetReader  {


    public LotteryCombinationDataSetReader(File file) {
        filePath = file.toPath();
        doInitialize();
    }

    public DataSet next(int num) {
        int lotteryLength = 5;
        String currentValStr = "";
        INDArray features = Nd4j.create(new int[]{num-1, lotteryLength, 10});
        INDArray labels = Nd4j.create(new int[]{num-1, lotteryLength, 10});
        for (int i =0; i < num - 1  && iter.hasNext(); i ++) {
            if ("".equals(currentValStr)) {
                currentValStr = iter.next();
                currentCursor ++;
            }
            String featureStr = currentValStr;

            featureStr = featureStr.split(",")[2];
            String[] featureAry = featureStr.split("");
            for (int j = 0; j < lotteryLength; j ++) {
                int l = Integer.parseInt(featureAry[j]);
                features.putScalar(new int[]{i, j, l}, 1);
            }

            String labelStr = iter.next();
            currentCursor ++;
            currentValStr = labelStr;
            labelStr = labelStr.split(",")[2];
            String[] labelAry = labelStr.split("");
            for (int j = 0; j < lotteryLength; j ++) {
                int l = Integer.parseInt(labelAry[j]);
                labels.putScalar(new int[]{i, j, l}, 1);
            }

        }
        return new DataSet(features, labels);
    }

    //based on the lottery rule,here will the openning lottery date and term switch to the long integer
    //if anyone need extend this model, maybe you can use the method
    private String decorateRecordData(String line) {
        if (line == null || line.isEmpty()) {
            return null;
        }
        String[] strArr = line.split(",");
        if (strArr.length >= 2) {
            //translation all time,
            String openDateStr = strArr[0].substring(0, strArr[0].length() - 3);
            openDateStr = openDateStr.substring(0, 4) + "-" + openDateStr.substring(4, 6) + "-" + openDateStr.substring(6, 8);
            String issueNumStr = strArr[0].substring(strArr[0].length() - 3);
            int issueNum = Integer.parseInt(issueNumStr);
            int minutes;
            int hours;
            if (issueNum >= 24 && issueNum < 96) {
                int temp = (issueNum - 24) * 10;
                minutes = temp % 60;
                hours = temp / 60;
                hours += 10;
            } else if (issueNum >= 96 && issueNum <= 120) {
                int temp = (issueNum - 96) * 5;
                minutes = temp % 60;
                hours = temp / 60;
                hours += 22;
            } else {
                int temp = issueNum * 5;
                minutes = temp % 60;
                hours = temp / 60;
            }
            openDateStr = openDateStr + " " + hours + ":" + (minutes == 0 ? "00" : minutes) + ":00";
            long openDateStrNum = 0;
            try {
                SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");//yyyy-MM-dd HH:mm:ss
                Date midDate = formatter.parse(openDateStr);
                openDateStrNum = midDate.getTime();

            } catch (Exception e) {
                throw  new RuntimeException("the decorateRecordData function shows exception!", e.getCause());
            }
            String lotteryValue = strArr[1];
            lotteryValue = lotteryValue.replace("", ",");
            line = openDateStrNum + lotteryValue;
        }
        return line;
    }


}
