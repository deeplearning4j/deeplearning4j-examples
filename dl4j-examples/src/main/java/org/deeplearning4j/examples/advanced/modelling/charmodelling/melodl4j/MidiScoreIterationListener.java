package org.deeplearning4j.examples.advanced.modelling.charmodelling.melodl4j;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;

import java.io.Serializable;
import java.text.NumberFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class MidiScoreIterationListener extends BaseTrainingListener implements Serializable {
    private int printIterations = 10;
    private long lastTimeInMls = System.currentTimeMillis();
    private int lastIteration = 0;

    private static final NumberFormat numberFormat = NumberFormat.getNumberInstance();
    static {
        numberFormat.setMinimumFractionDigits(1);
        numberFormat.setMaximumFractionDigits(1);
    }
    public MidiScoreIterationListener(int printIterations) {
        this.printIterations = printIterations;
    }

    public MidiScoreIterationListener() {
    }
    public void iterationDone(Model model, int iteration, int epoch) {
        if (this.printIterations <= 0) {
            this.printIterations = 1;
        }
        if (iteration % this.printIterations == 0) {
            double score = model.score();
            long now = System.currentTimeMillis();
            double seconds = 0.001*(now-lastTimeInMls)/(1+iteration-lastIteration);
            System.out.println(numberFormat.format(seconds) + " seconds per iteration: score at iteration " +  iteration + " is " + score);
            lastTimeInMls = now;
            lastIteration = iteration;
        }
    }
}