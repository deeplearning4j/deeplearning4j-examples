package org.deeplearning4j.examples.convolution;

import lombok.extern.slf4j.Slf4j;

/**
 * Created by raver119 on 08.07.17.
 */
@Slf4j
public class LR {

    public static void main(String[] args) throws Exception {
        int numIterations = 1;
        int numEpochs = 2;
        double learningRate = 0.025;
        double minLearningRate = 0.0001;
        long totalWordsCount = 1000;
        long wordsCounter = 1700;

        double alpha = Math.max(minLearningRate, learningRate * (1 - (1.0 * wordsCounter / ((double) totalWordsCount) / (numIterations * numEpochs))));
        log.info("New alpha: {}", alpha);
    }
}
