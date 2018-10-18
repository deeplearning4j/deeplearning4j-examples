package org.deeplearning4j.tinyimagenet;

import com.beust.jcommander.Parameter;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.patent.utils.JCommanderUtils;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.charset.StandardCharsets;

public class TrainLocal {
    public static Logger log = LoggerFactory.getLogger(TrainLocal.class);

    @Parameter(names = {"--numEpochs"}, description = "Number of epochs for training")
    private int numEpochs = 1;

    @Parameter(names = {"--ui"}, description = "Whether to use UI or not", arity = 1)
    private boolean ui;

    @Parameter(names = {"--saveDir"}, description = "If set, the directory to save the trained network")
    private String saveDir;

    public static void main(String[] args) throws Exception {
        new TrainLocal().entryPoint(args);
    }

    public void entryPoint(String[] args) throws Exception {
        JCommanderUtils.parseArgs(this, args);


        int batchSize = 32;
        DataSetIterator iter = new TinyImageNetDataSetIterator(batchSize);
        iter.setPreProcessor(new ImagePreProcessingScaler());   //Scale 0-255 valued pixels to 0-1 range

        ComputationGraph net = TrainTinyImageNetSpark.getNetwork();

        if(ui){
//            StatsStorage ss = new FileStatsStorage
//            net.setListeners(new );
        } else {
            net.setListeners(new ScoreIterationListener(10));
        }

        net.fit(iter, numEpochs);
        log.info("Training complete. Starting evaluation.");

        DataSetIterator test = new TinyImageNetDataSetIterator(batchSize, DataSetType.TRAIN);
        Evaluation e = net.evaluate(test);

        log.info("Evaluation complete");
        log.info(e.stats());

        if(saveDir != null && !saveDir.isEmpty()){
            log.info("Saving network and evaluation stats to directory: {}", saveDir);
            net.save(new File(saveDir, "trainedNet.bin"));
            FileUtils.writeStringToFile(new File(saveDir, "evaulation.txt"), e.stats(), StandardCharsets.UTF_8);
        }

        log.info("----- Examples Complete -----");
    }
}
