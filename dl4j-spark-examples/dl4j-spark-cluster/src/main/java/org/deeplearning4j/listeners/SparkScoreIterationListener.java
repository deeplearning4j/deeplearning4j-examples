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

package org.deeplearning4j.listeners;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.ipc.RemoteException;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.utils.CommonUtils;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *  In distributed training, there is a synchronization problem in the monitoring training process log. In order to solve this problem, this SparkScoreIterationListener extends BaseTrainingListener class
 *  @author wangfeng
 */

public class SparkScoreIterationListener extends BaseTrainingListener {

    private int printIterations = 10;
    private static final Logger log = LoggerFactory.getLogger(ScoreIterationListener.class);
    private String pathStr;
    public SparkScoreIterationListener() {
    }

    public SparkScoreIterationListener(int printIterations, String pathStr) {
        this.printIterations = printIterations;
        this.pathStr = pathStr;
        FileSystem nfs = CommonUtils.openHdfsConnect();

        try {
            Path path = new Path(pathStr);
            if (!nfs.exists(path)) {
                nfs.createNewFile(path);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            CommonUtils.closeHdfsConnect(nfs);
        }
    }
    @Override
    public void iterationDone(Model model, int iteration, int epoch) {

        if (printIterations <= 0) {
            printIterations = 1;
        }
        String newScore = "";
        if (iteration % printIterations == 0) {
            double score = model.score();
            newScore += "Score at iteration {" + iteration + "} is {" + score + "}";
            log.info(newScore);
        }
        FileSystem nfs = null;
        try {
            nfs = CommonUtils.openHdfsConnect();
            Path path = new Path(pathStr);
            //although using append function isn't best ways, but currently it still solve the score log existing or not
            FSDataOutputStream out = nfs.append(path);//. .create(path);
            out.write(newScore.getBytes());
            out.write("\n".getBytes());
            out.hsync();
            out.close();
            CommonUtils.closeHdfsConnect(nfs);
        } catch (RemoteException e) {
            if (nfs != null) {
                CommonUtils.closeHdfsConnect(nfs);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }


}
