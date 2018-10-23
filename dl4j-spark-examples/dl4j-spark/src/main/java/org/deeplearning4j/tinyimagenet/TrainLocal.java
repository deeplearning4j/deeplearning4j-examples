package org.deeplearning4j.tinyimagenet;

import com.beust.jcommander.Parameter;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.patent.utils.JCommanderUtils;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.charset.StandardCharsets;

/**
 * This is a local (single-machine) version of the Tiny ImageNet image classifier from TrainSpark.
 * See that example for details.
 *
 * Note that this local (single machine) version does not require the preprocessing scripts to be run
 *
 * @author Alex Black
 */
public class TrainLocal {
    public static Logger log = LoggerFactory.getLogger(TrainLocal.class);

    @Parameter(names = {"--numEpochs"}, description = "Number of epochs for training")
    private int numEpochs = 10;

    @Parameter(names = {"--saveDir"}, description = "If set, the directory to save the trained network")
    private String saveDir;

    public static void main(String[] args) throws Exception {
        new TrainLocal().entryPoint(args);
    }

    public void entryPoint(String[] args) throws Exception {
        JCommanderUtils.parseArgs(this, args);

        //Create the data pipeline
        int batchSize = 32;
        DataSetIterator iter = new TinyImageNetDataSetIterator(batchSize);
        iter.setPreProcessor(new ImagePreProcessingScaler());   //Scale 0-255 valued pixels to 0-1 range

        //Create the network
        ComputationGraph net = TrainSpark.getNetwork();
        net.setListeners(new PerformanceListener(50, true));

        //Reduce auto GC frequency for better performance
        Nd4j.getMemoryManager().setAutoGcWindow(10000);

        //Fit the network
        net.fit(iter, numEpochs);
        log.info("Training complete. Starting evaluation.");

        //Evaluate the network on test set data
        DataSetIterator test = new TinyImageNetDataSetIterator(batchSize, DataSetType.TEST);
        test.setPreProcessor(new ImagePreProcessingScaler());   //Scale 0-255 valued pixels to 0-1 range
        Evaluation e = new Evaluation(TinyImageNetDataSetIterator.getLabels(false), 5); //Set up for top 5 accuracy
        net.doEvaluation(test, e);

        log.info("Evaluation complete");
        log.info(e.stats());

        if(saveDir != null && !saveDir.isEmpty()){
            File sd = new File(saveDir);
            if(!sd.exists())
                sd.mkdirs();

            log.info("Saving network and evaluation stats to directory: {}", saveDir);
            net.save(new File(saveDir, "trainedNet.bin"));
            FileUtils.writeStringToFile(new File(saveDir, "evaulation.txt"), e.stats(), StandardCharsets.UTF_8);
        }

        log.info("----- Examples Complete -----");
    }
}
