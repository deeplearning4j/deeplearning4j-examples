package org.deeplearning4j.examples.userInterface;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.examples.userInterface.util.UIExampleUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

/**
 * A version of UIStorageExample showing how to saved network training data to a file, and then
 * reload it later, to display in in the UI
 *
 * @author Alex Black
 */
public class UIStorageExample {

    public static void main(String[] args){

        //Run this example twice - once with collectStats = true, and then again with collectStats = false
        boolean collectStats = true;

        File statsFile = new File("UIStorageExampleStats.dl4j");

        if(collectStats){
            //First run: Collect training stats from the network
            //Note that we don't have to actually plot it when we collect it - though we can do that too, if required

            MultiLayerNetwork net = UIExampleUtils.getMnistNetwork();
            DataSetIterator trainData = UIExampleUtils.getMnistData();

            StatsStorage statsStorage = new FileStatsStorage(statsFile);
            net.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));

            net.fit(trainData);

            System.out.println("Done");
        } else {
            //Second run: Load the saved stats and visualize. Go to http://localhost:9000/train

            StatsStorage statsStorage = new FileStatsStorage(statsFile);    //If file already exists: load the data from it
            UIServer uiServer = UIServer.getInstance();
            uiServer.attach(statsStorage);
        }
    }
}
