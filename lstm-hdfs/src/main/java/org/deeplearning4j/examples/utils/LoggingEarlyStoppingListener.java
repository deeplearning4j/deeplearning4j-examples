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
