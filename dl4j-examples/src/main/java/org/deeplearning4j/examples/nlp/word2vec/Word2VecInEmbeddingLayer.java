package org.deeplearning4j.examples.nlp.word2vec;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.examples.utilities.DataUtilities;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URL;
import java.util.Arrays;
import java.util.List;

public class Word2VecInEmbeddingLayer {

    private static Logger log = LoggerFactory.getLogger(Word2VecInEmbeddingLayer.class);

    public static final String DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";

    public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/");

    public static final String WORD2VEC_PATH = FilenameUtils.concat(DATA_PATH, "wordvectors.txt");

    public static final int embeddingSize = 100;

    public static WordVectors wordVectors = null;

    public static void main(String[] args) throws Exception {
        downloadData();

        File word2vecFile = new File(WORD2VEC_PATH);
        if (!word2vecFile.exists()){
            trainingWord2Vec();
        }

        wordVectors = WordVectorSerializer.readWord2VecModel(WORD2VEC_PATH);

        VocabCache vocab =  wordVectors.vocab();

        ImdbSentimentIterator trainIter = new ImdbSentimentIterator(DATA_PATH, vocab, 50, 150, true);
        ImdbSentimentIterator testIter = new ImdbSentimentIterator(DATA_PATH, vocab, 50, 150, false);

        INDArray table = Nd4j.zeros(vocab.numWords(), wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length);
        vocab.words().stream().forEach(x -> table.putRow(vocab.indexOf((String)x)+1, Nd4j.create(wordVectors.getWordVector((String)x))));

        // define networks layers
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(123)
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Adam(0.0001))
            .l2(0.0001)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            .gradientNormalizationThreshold(1.0)
            .graphBuilder()
            .addInputs("input")
            .setInputTypes(InputType.recurrent(150))
            .addLayer("embeddingEncoder",
                new FrozenLayer.Builder().layer(
                    new EmbeddingLayer.Builder()
                        .nIn(vocab.numWords())
                        .nOut(embeddingSize)
                        .activation(Activation.IDENTITY)
                        .biasInit(0.0)
                        .build()
                ).build(),
                "input")
            .addLayer("lstm",
                new LSTM.Builder()
                    .weightInitRecurrent(WeightInit.XAVIER)
                    .nIn(embeddingSize)
                    .nOut(300)
                    .activation(Activation.TANH)
                    .build(),
                "embeddingEncoder")
            .addVertex("last", new LastTimeStepVertex("input"), "lstm")
            .addLayer("dense1",
                new DenseLayer.Builder()
                    .nIn(300)
                    .nOut(100)
                    .activation(Activation.LEAKYRELU)
                    .build(),
                "last")
            .addLayer("bn1", new BatchNormalization.Builder().build(), "dense1")
            .addLayer("output",
                new OutputLayer.Builder()
                    .nIn(100)
                    .nOut(2)
                    .activation(Activation.SOFTMAX)
                    .lossFunction(LossFunctions.LossFunction.MCXENT)
                    .build(),
                "bn1")
            .setOutputs("output")
            .build();

        ComputationGraph model = new ComputationGraph(conf);
        model.init();
        model.getLayer("embeddingEncoder").setParam("W", table);

        log.info(model.summary());

        // define score and evaluation listener for training
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage),
            new ScoreIterationListener(10)
        );

        System.out.println("Starting training...");
        model.fit(trainIter, 2);

        System.out.println("Evaluating...");
        Evaluation eval = model.evaluate(testIter);
        System.out.println(eval.stats());
    }

    public static void trainingWord2Vec(){

        File dataDir = new File(DATA_PATH+"/aclImdb/train/");
        SentenceIterator iter = new FileSentenceIterator(dataDir);
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
            .minWordFrequency(7)
            .epochs(3)
            .layerSize(embeddingSize)
            .seed(42)
            .windowSize(5)
            .iterate(iter)
            .tokenizerFactory(t)
            .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Training Finished");

        // saved for future use.
        WordVectorSerializer.writeWord2VecModel(vec, WORD2VEC_PATH);
        log.info("Model Saved");
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

        List<String> fileToDelete = Arrays.asList("labeledBow.feat", "unsupBow.feat", "urls_pos.txt",
            "urls_neg.txt", "urls_unsup.txt");

        fileToDelete.forEach(f -> new File(extractedPath + "/train/" +  f).delete());
    }
}
