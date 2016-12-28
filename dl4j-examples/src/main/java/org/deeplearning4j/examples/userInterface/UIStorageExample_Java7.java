package org.deeplearning4j.examples.userInterface;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.examples.userInterface.util.UIExampleUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.J7StatsListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.sqlite.J7FileStatsStorage;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

/**
 * A variant of the UI example showing the approach for Java 7 compatibility
 *
 * *** Notes ***
 * 1: If you don't specifically need Java 7, use the approach in the standard UIStorageExample as it should be faster
 * 2: The UI itself requires Java 8 (uses the Play framework as a backend). But you can store stats on one machine, copy
 *    the file to another (with Java 8) and visualize there
 * 3: J7FileStatsStorage and FileStatsStorage formats are NOT compatible. Save/load with the same one
 *    (J7FileStatsStorage works on Java 8 too, but FileStatsStorage does not work on Java 7)
 *
 * @author Alex Black
 */
public class UIStorageExample_Java7 {

    public static void main(String[] args){

        //Run this example twice - once with collectStats = true, and then again with collectStats = false
        boolean collectStats = true;

        File statsFile = new File("UIStorageExampleStats_Java7.dl4j");

        //First run: Collect training stats from the network
        //Note that we don't have to actually plot it when we collect it - though we can do that too, if required

        MultiLayerNetwork net = UIExampleUtils.getMnistNetwork();
        DataSetIterator trainData = UIExampleUtils.getMnistData();

        StatsStorage statsStorage = new J7FileStatsStorage(statsFile);                                      //Note the J7
        net.setListeners(new J7StatsListener(statsStorage), new ScoreIterationListener(10));

        net.fit(trainData);

        System.out.println("Done");
    }
}
