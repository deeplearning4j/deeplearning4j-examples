package org.deeplearning4j.examples.transferlearning.vgg16;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Created by susaneraly on 3/6/17.
 */
@Slf4j
public class FineTuneFromBlockFour {
    protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    protected static final int numClasses = 5;

    protected static final int batchSize = 25;
    protected static final long seed = 12345;
    public static final Random rng = new Random(seed);

    protected static int height = 224;
    protected static int width = 224;
    protected static int channels = 3;
    protected static final int nEpochs = 50;

    public static void main(String [] args) throws IOException {

        //load trained model
        File locationToSave = new File("MyComputationGraph.zip");
        ComputationGraph vgg16Transfer = ModelSerializer.restoreComputationGraph(locationToSave);

        ComputationGraph vgg16FineTune = new TransferLearning.GraphBuilder(vgg16Transfer)
            .fineTuneConfiguration(new FineTuneConfiguration.Builder()
                .learningRate(1e-5)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                //.regularization(true).l2(0.001)
                .seed(seed)
                .build())
            .setFeatureExtractor("block4_pool")
            .build();
        log.info(vgg16FineTune.summary());

        /*
            Set up dataset with the train and test split
            Set up the training dataset iterator
         */
        File parentDir = new File("/home/seraly/flower_photos");
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];
        //set up iterators
        ImageRecordReader recordReaderTest = new ImageRecordReader(height,width,channels,labelMaker);
        recordReaderTest.initialize(testData);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReaderTest, batchSize, 1, numClasses);
        testIter.setPreProcessor(TrainedModels.VGG16.getPreProcessor());
        ImageRecordReader recordReaderTrain = new ImageRecordReader(height,width,channels,labelMaker);
        recordReaderTrain.initialize(trainData);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, numClasses);
        trainIter.setPreProcessor(TrainedModels.VGG16.getPreProcessor());

        Evaluation eval;
        //vgg16FineTune.setListeners = (new ScoreIterationListener(1));
        UIServer uiServer = UIServer.getInstance();
        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new InMemoryStatsStorage();             //Alternative: new FileStatsStorage(File) - see UIStorageExample
        int listenerFrequency = 1;
        vgg16FineTune.setListeners(new StatsListener(statsStorage, listenerFrequency));
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        int iter = 0;
        while(trainIter.hasNext()) {
            vgg16FineTune.fit(trainIter.next());
            if (iter % 10 == 0 && iter !=0) {
                log.info("Evaluate model at iter "+iter +" ....");
                eval = new Evaluation(numClasses);
                while (testIter.hasNext()) {
                    DataSet ds = testIter.next();
                    INDArray output = vgg16FineTune.output(false, ds.getFeatures())[0];
                    eval.eval(ds.getLabels(), output);
                }
                log.info(eval.stats());
                testIter.reset();
            }
            iter++;
        }
        //Save the model
        File locationToSaveFineTune = new File("MyComputationGraphFineTune.zip");       //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = false;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(vgg16Transfer, locationToSaveFineTune, saveUpdater);

    }
}
