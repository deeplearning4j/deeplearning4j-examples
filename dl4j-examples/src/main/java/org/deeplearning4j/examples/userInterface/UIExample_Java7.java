package org.deeplearning4j.examples.userInterface;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.examples.userInterface.util.UIExampleUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.J7StatsListener;
import org.deeplearning4j.ui.storage.sqlite.J7FileStatsStorage;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

/**
 * A variant of the UI example showing the approach for Java 7 compatibility
 *
 * *** Note: If you don't specifically need Java 7, use the approach in the standard UIExample as it should be faster ***
 *
 * @author Alex Black
 */
public class UIExample_Java7 {

    public static void main(String[] args){

        //Get our network and training data
        MultiLayerNetwork net = UIExampleUtils.getMnistNetwork();
        DataSetIterator trainData = UIExampleUtils.getMnistData();

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new J7FileStatsStorage(new File("J7StatsExample.dl4j"));             //InMemoryStatsStorage is also Java 7 compatible
        net.setListeners(new J7StatsListener(statsStorage));

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Start training:
        net.fit(trainData);

        //Finally: open your browser and go to http://localhost:9000/train
    }
}
