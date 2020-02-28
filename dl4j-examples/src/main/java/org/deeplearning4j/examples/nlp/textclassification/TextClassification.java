package org.deeplearning4j.examples.nlp.textclassification;


import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.iterator.BertIterator;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.examples.utilities.DataUtilities;
import org.nd4j.linalg.schedule.ExponentialSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.SigmoidSchedule;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.api.Updater;


import java.io.File;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class TextClassification {


    /**
     * Data URL for downloading
     */
    public static final String DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
    /**
     * Location to save and extract the training/testing data
     */
    public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/");

    public static BertIterator getBertDataSetIterator(boolean isTraining, BertWordPieceTokenizerFactory t) {

        String path = FilenameUtils.concat(DATA_PATH, (isTraining ? "aclImdb/train/" : "aclImdb/test/"));
        String positiveBaseDir = FilenameUtils.concat(path, "pos");
        String negativeBaseDir = FilenameUtils.concat(path, "neg");
        Random rng = new Random(42);

        File filePositive = new File(positiveBaseDir);
        File fileNegative = new File(negativeBaseDir);

        Map<String, List<File>> reviewFilesMap = new HashMap<>();
        reviewFilesMap.put("Positive", Arrays.asList(Objects.requireNonNull(filePositive.listFiles())));
        reviewFilesMap.put("Negative", Arrays.asList(Objects.requireNonNull(fileNegative.listFiles())));


        BertIterator b = BertIterator.builder()
            .tokenizer(t)
            .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 256)
            .minibatchSize(32)
            .sentenceProvider(new FileLabeledSentenceProvider(reviewFilesMap, rng))
            .featureArrays(BertIterator.FeatureArrays.INDICES_MASK)
            .vocabMap(t.getVocab())
            .task(BertIterator.Task.SEQ_CLASSIFICATION)
            .build();


        return b;
    }


    public static void main(String[] args) throws Exception {


        //Download and extract data
        downloadData();


        final int seed = 0;     //Seed for reproducibility
        String pathToVocab = "/home/jenkins/uncased_L-12_H-768_A-12/vocab.txt";
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(new File(pathToVocab), true, true, StandardCharsets.UTF_8);


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .updater(new Adam(1e-3))
            .l2(1e-6)
            .weightInit(WeightInit.XAVIER)
            .list()
            .setInputType(InputType.recurrent(1))
            .layer(0, new EmbeddingSequenceLayer.Builder().weightInit(new NormalDistribution(0, 1)).l2(0).hasBias(true).nIn(t.getVocab().size()).nOut(128).build())
            .layer(new Bidirectional(new LSTM.Builder().nOut(256).activation(Activation.TANH).build()))
            .layer(new Bidirectional(new LSTM.Builder().nOut(256).activation(Activation.TANH).build()))
            .layer(new GlobalPoolingLayer(PoolingType.MAX))
            .layer(new OutputLayer.Builder().nOut(2).activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
            .build();

        BertIterator train = getBertDataSetIterator(true, t);
        BertIterator test = getBertDataSetIterator(false, t);

        MultiDataSetPreProcessor mdsPreprocessor = new MultiDataSetPreProcessor() {
            @Override
            public void preProcess(MultiDataSet multiDataSet) {
                multiDataSet.setFeaturesMaskArray(0, multiDataSet.getFeaturesMaskArray(0).castTo(DataType.FLOAT));
            }
        };


        train.setPreProcessor(mdsPreprocessor);
        test.setPreProcessor(mdsPreprocessor);


        MultiLayerNetwork net = new MultiLayerNetwork(conf);


        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats-" + System.currentTimeMillis() + ".dl4j"));
        int listenerFrequency = 20;
        net.setListeners(new StatsListener(statsStorage, listenerFrequency), new ScoreIterationListener(50));
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);


        for (int i = 1; i <= 10; i++) {

            net.fit(train);

            Evaluation eval = net.doEvaluation(test, new Evaluation[]{new Evaluation()})[0];
            System.out.println(eval.stats());
        }

        System.out.print(net.summary());

    }

    public static void downloadData() throws Exception {
        //Create directory if required
        File directory = new File(DATA_PATH);
        if (!directory.exists()) directory.mkdir();

        //Download file:
        String archizePath = DATA_PATH + "aclImdb_v1.tar.gz";
        File archiveFile = new File(archizePath);
        String extractedPath = DATA_PATH + "aclImdb";
        File extractedFile = new File(extractedPath);

        if (!archiveFile.exists()) {
            System.out.println("Starting data download (80MB)...");
            FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile);
            System.out.println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath());
            //Extract tar.gz file to output directory
            DataUtilities.extractTarGz(archizePath, DATA_PATH);
        } else {
            //Assume if archive (.tar.gz) exists, then data has already been extracted
            System.out.println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath());
            if (!extractedFile.exists()) {
                //Extract tar.gz file to output directory
                DataUtilities.extractTarGz(archizePath, DATA_PATH);
            } else {
                System.out.println("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
            }
        }
    }


}

