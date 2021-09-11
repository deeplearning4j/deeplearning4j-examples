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

package org.deeplearning4j.distributedtrainingexamples.patent.utils.evaluation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This is a runnable used to asynchronously and periodically save a copy of the specified parameters vector,
 * while Spark training is proceeding.
 * This is used to record the parameters so we can later evaluate "covergence (accuracy) vs. time"
 *
 * @author Alex Black
 */
public class ConvergenceRunnable implements Runnable {
    private static final Logger log = LoggerFactory.getLogger(ConvergenceRunnable.class);

    private final File saveRootDir;
    private final AtomicInteger currentSubset;
    private final AtomicBoolean isRunning;
    private final long saveEverySec;
    private final INDArray params;
    private final Queue<ToEval> toEval = new ConcurrentLinkedQueue<>();

    private long totalTrainMs;
    private int saveCount = 0;

    public ConvergenceRunnable(File saveRootDir, AtomicInteger currentSubset, AtomicBoolean isRunning, long saveEverySec, INDArray params) {
        this.saveRootDir = saveRootDir;
        this.currentSubset = currentSubset;
        this.isRunning = isRunning;
        this.saveEverySec = saveEverySec;
        this.params = params;
        /*
        Note re: params here: this should be the same one that is passed to the SparkComputationGraph and hence SharedTrainingMaster
        This is in turn passed to SilentTrainingDriver, which updates the array IN-PLACE as messages come in.
        Consequently, the array should always be up to date - on the master - with the latest update messages
         */
    }


    @Override
    public void run() {
        long lastLoopTime = System.currentTimeMillis();
        long lastSaveTime = lastLoopTime;
        while (true) {    //Daemon thread
            long now = System.currentTimeMillis();

            if (isRunning.get()) {
                long delta = now - lastLoopTime;
                totalTrainMs += delta;

                if (now - lastSaveTime >= 1000 * saveEverySec) {
                    int c = saveCount;
                    File f = new File(saveRootDir, "netParams_subset" + currentSubset + "_" + (saveCount++) + "_train" + totalTrainMs + "ms_" + now + ".bin");
                    try {
                        Nd4j.getExecutioner().commit(); //Required for CUDA thread safety
                        Nd4j.saveBinary(params, f);
                        log.info("Saved parameters to file: {}", f.getAbsolutePath());
                    } catch (IOException e) {
                        log.error("Error saving parameters to file: {}", f.getAbsolutePath(), e);
                    }
                    toEval.add(new ToEval(f, c, totalTrainMs));
                    lastSaveTime = now;
                }
            }
            lastLoopTime = now;

            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                log.error("***** Hit interrupted exception in convergence runnable - exiting *****");
                break;
            }
        }
    }

    public Queue<ToEval> getToEval(){
        return toEval;
    }

    public static Queue<ToEval> startConvergenceThread(File baseParamSaveDir, AtomicInteger currentSubset, AtomicBoolean isTraining, long saveEverySec, INDArray params){
        ConvergenceRunnable cr = new ConvergenceRunnable(baseParamSaveDir, currentSubset, isTraining, saveEverySec, params);
        Thread t = new Thread(cr);
        t.setDaemon(true);
        t.start();
        return cr.getToEval();
    }

}
