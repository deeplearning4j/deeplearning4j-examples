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

package org.deeplearning4j.examples.utils;

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author: Ousmane A. Dia
 */
public class LoggingEarlyStoppingListener implements EarlyStoppingListener<ComputationGraph> {

    private static Logger log = LoggerFactory.getLogger(LoggingEarlyStoppingListener.class);
    private int onStartCallCount = 0;
    private int onEpochCallCount = 0;
    private int onCompletionCallCount = 0;

    @Override
    public void onStart(EarlyStoppingConfiguration<ComputationGraph> esConfig, ComputationGraph net) {
        log.info("EarlyStopping: onStart called");
        onStartCallCount++;
    }

    @Override
    public void onEpoch(int epochNum, double score, EarlyStoppingConfiguration<ComputationGraph> esConfig,
                    ComputationGraph net) {
        log.info("EarlyStopping: onEpoch called (epochNum={}, score={}}", epochNum, score);
        onEpochCallCount++;
    }

    @Override
    public void onCompletion(EarlyStoppingResult<ComputationGraph> esResult) {
        log.info("EarlyStopping: onCompletion called (result: {})", esResult);
        onCompletionCallCount++;
    }

    public int getOnCompletionCallCount() {
        return onCompletionCallCount;
    }

    public int getOnStartCallCount() {
        return onStartCallCount;
    }

    public int getOnEpochCallCount() {
        return onEpochCallCount;
    }

}
