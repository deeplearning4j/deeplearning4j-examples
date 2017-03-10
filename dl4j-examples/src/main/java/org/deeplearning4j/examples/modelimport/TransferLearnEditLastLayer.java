package org.deeplearning4j.examples.modelimport;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModelHelper;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

/**
 * Created by susaneraly on 3/9/17.
 */
@Slf4j
public class TransferLearnEditLastLayer {

    protected static final int numClasses = 5;
    protected static final long seed = 12345;

    private static final int trainPerc = 80;
    private static final int batchSize = 15;
    //private static final String dataDir = "/home/seraly/flower_photos";
    private static final String dataDir = "/Users/susaneraly/flower_photos";

    public static void main(String [] args) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {

        TrainedModelHelper modelImportHelper = new TrainedModelHelper(TrainedModels.VGG16);
        ComputationGraph vgg16 = modelImportHelper.loadModel();
        log.info(vgg16.summary());

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .learningRate(5e-5)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            //.regularization(true).l2(0.001)
            //.dropOut(0.5)
            .seed(seed)
            .build();

        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
                                        .fineTuneConfiguration(fineTuneConf)
                                        .setFeatureExtractor("fc2")
                                        .removeVertexKeepConnections("predictions")
                                        .addLayer("predictions",new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                                                                .nIn(4096)
                                                                                .nOut(numClasses)
                                                                                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,0.002*(2.0/(4096+10))))
                                                                                .activation(Activation.SOFTMAX)
                                                                                .build(),"fc2")
                                        .build();

        log.info(vgg16Transfer.summary());

        FlowerDataSetIterator.setup(dataDir,batchSize,trainPerc);
        DataSetIterator trainIter = FlowerDataSetIterator.trainIterator();
        DataSetIterator testIter = FlowerDataSetIterator.testIterator();

        Evaluation eval;
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        int listenerFrequency = 1;
        vgg16Transfer.setListeners(new StatsListener(statsStorage, listenerFrequency));
        uiServer.attach(statsStorage);

        int iter = 0;
        while(trainIter.hasNext()) {
            vgg16Transfer.fit(trainIter.next());
            if (iter % 10 == 0 && iter !=0) {
                log.info("Evaluate model at iter "+iter +" ....");
                eval = vgg16Transfer.evaluate(testIter);
                log.info(eval.stats());
                testIter.reset();
            }
            iter++;
        }
    }
}
