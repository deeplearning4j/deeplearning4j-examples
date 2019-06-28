package org.deeplearning4j.examples.nlp.bert;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.examples.recurrent.word2vecsentiment.Word2VecSentimentRNN;
import org.deeplearning4j.iterator.BertIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.deeplearning4j.ui.api.UIServer;
import org.nd4j.autodiff.listeners.checkpoint.CheckpointListener;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.listeners.impl.UIListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.imports.tensorflow.TFImportOverride;
import org.nd4j.imports.tensorflow.TFOpImportFilter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.schedule.FixedSchedule;
import org.nd4j.linalg.schedule.RampSchedule;
import org.nd4j.resources.Downloader;
import org.nd4j.util.ArchiveUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.TimeUnit;


//As per BertSentimentExample but only the output layer is trained
public class BertFrozenSentimentExample {
    public static Logger log = LoggerFactory.getLogger(BertFrozenSentimentExample.class);

    public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/");

    public static void main(String[] args) throws Exception {

//        UIServer.getInstance();
//        Thread.sleep(100000);

//        File rootDir = new File("/home/skymind/bert_test");
        File rootDir = new File("C:/Temp/Bert_Frozen/");
        if(!rootDir.exists()){
            rootDir.mkdir();
        }


        String url = "https://deeplearning4jblob.blob.core.windows.net/testresources/bert_mrpc_frozen_v1.zip";
        File saveDir = new File(rootDir, "bert_mrpc_frozen_v1");
        saveDir.mkdirs();

        File localFile = new File(saveDir, "bert_mrpc_frozen_v1.zip");
        String md5 = "7cef8bbe62e701212472f77a0361f443";

        if(localFile.exists() && !Downloader.checkMD5OfFile(md5, localFile)) {
            log.info("Deleting local file: does not match MD5. {}", localFile.getAbsolutePath());
            localFile.delete();
        }

        if (!localFile.exists()) {
            log.info("Starting resource download from: {} to {}", url, localFile.getAbsolutePath());
            Downloader.download("BERT MRPC", new URL(url), localFile, md5, 3);
        }

        //Extract
        File f = new File(saveDir, "bert_mrpc_frozen.pb");
        if(!f.exists() || !Downloader.checkMD5OfFile("93d82bca887625632578df37ea3d3ca5", f)){
            if(f.exists()) {
                f.delete();
            }
            ArchiveUtils.zipExtractSingleFile(localFile, f, "bert_mrpc_frozen.pb");
        }

        /*
        Important node: This BERT model uses a FIXED (hardcoded) minibatch size, not dynamic as most models use
         */
        int minibatchSize = 4;

        /*
         * Define: Op import overrides. This is used to skip the IteratorGetNext node and instead crate some placeholders
         */
        Map<String, TFImportOverride> m = new HashMap<>();
        m.put("IteratorGetNext", (inputs, controlDepInputs, nodeDef, initWith, attributesForNode, graph) -> {
            //Return 3 placeholders called "IteratorGetNext:0", "IteratorGetNext:1", "IteratorGetNext:3" instead of the training iterator
            return Arrays.asList(
                initWith.placeHolder("IteratorGetNext", DataType.INT, minibatchSize, 128),
                initWith.placeHolder("IteratorGetNext:1", DataType.INT, minibatchSize, 128),
                initWith.placeHolder("IteratorGetNext:4", DataType.INT, minibatchSize, 128)
            );
        });

        //Skip the "IteratorV2" op - we don't want or need this
        TFOpImportFilter filter = (nodeDef, initWith, attributesForNode, graph) -> { return "IteratorV2".equals(nodeDef.getName()); };

        SameDiff sd = TFGraphMapper.getInstance().importGraph(f, m, filter);

        sd.renameVariable("IteratorGetNext", "tokenIdxs");
        sd.renameVariable("IteratorGetNext:1", "mask");
        sd.renameVariable("IteratorGetNext:4", "sentenceIdx");  //only ever 0, but needed by this model...


        Set<String> floatConstants = new HashSet<>(Arrays.asList(
            "bert/encoder/ones"
        ));

        //For training, convert _output layer only_ weights and biases from constants to variables:
        sd.getVariable("output_weights").convertToVariable();
        sd.getVariable("output_bias").convertToVariable();
        sd.getVariable("bert/pooler/dense/kernel").convertToVariable();
        sd.getVariable("bert/pooler/dense/bias").convertToVariable();

        //For training, we'll need to add a label placeholder for one-hot labels:
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, 4, 2);
        SDVariable softmax = sd.getVariable("loss/Softmax");
        sd.loss().logLoss("loss", label, softmax);

        //Also randomize the output layer weights, zero the bias:
        sd.getVariable("output_weights").getArr().assign(Nd4j.randn(DataType.FLOAT, 2, 768).muli(Math.sqrt(2.0 / (2.0 + 768.0))));  //Xavier init
        sd.getVariable("output_bias").getArr().assign(0);


        //Next: create training pipeline...
        MultiDataSetIterator iterTrain = getDataSetIterator(rootDir, true, 4, 128, new Random(12345));
        MultiDataSetIterator iterTest = getDataSetIterator(rootDir, false, 4, 128, new Random(12345));

        //Set up training configuration...

        sd.setTrainingConfig(TrainingConfig.builder()
            .updater(new Adam(new RampSchedule(new FixedSchedule(5e-4), 50)))
            .dataSetFeatureMapping("tokenIdxs", "sentenceIdx")
            .dataSetFeatureMaskMapping("mask")
            .dataSetLabelMapping("label")
            .build());

        File dir = new File(rootDir, "lr_5e-4_2layer_cls_prefix");
        dir.mkdirs();
        File uiFile = new File(dir, "UIData.bin");
        File checkpointDir = new File(dir, "checkpoints");
        sd.setListeners(new ScoreListener(10, true, true),
            new CheckpointListener.Builder(checkpointDir)
                .saveEvery(30, TimeUnit.MINUTES)
                .saveEveryNEpochs(1)
                .keepLastAndEvery(3,3)
                .saveUpdaterState(false)    //Until 2GB limit is fixed
                .build(),
            new UIListener.Builder(uiFile)
                .learningRate(10)
                .plotLosses(1)
                .trainAccuracy("loss/Softmax", 0)
                .updateRatios(20)
                .build()
        );

//        Evaluation evalBefore = new Evaluation();
//        sd.evaluate(iterTest, "loss/Softmax", 0, evalBefore);
//
//        log.info("Evaluation, before:");
//        log.info(evalBefore.stats());

        if(!iterTrain.hasNext()){
            throw new RuntimeException("No data");
        }

        sd.fit(iterTrain, 10);

//        for(int i=0; i<10; i++ ) {
//            sd.fit(iterTrain, 1);
//            Evaluation e = new Evaluation();
//            sd.evaluate(iterTest, "loss/Softmax", 0, e);
//            log.info("Evaluation, end epoch {}:", i);
//            log.info(e.stats());
//        }
    }


    private static MultiDataSetIterator getDataSetIterator(File rootDir, boolean isTraining, int minibatchSize,
                                                           int maxSentenceLength, Random rng ) throws Exception {

        Word2VecSentimentRNN.downloadData();
        String path = FilenameUtils.concat(DATA_PATH, (isTraining ? "aclImdb/train/" : "aclImdb/test/"));
        String positiveBaseDir = FilenameUtils.concat(path, "pos");
        String negativeBaseDir = FilenameUtils.concat(path, "neg");

        File filePositive = new File(positiveBaseDir);
        File fileNegative = new File(negativeBaseDir);

        Map<String,List<File>> reviewFilesMap = new HashMap<>();
        reviewFilesMap.put("Positive", Arrays.asList(filePositive.listFiles()));
        reviewFilesMap.put("Negative", Arrays.asList(fileNegative.listFiles()));

        //FOR DEBUGGING: SUBSET
//        reviewFilesMap.put("Positive", Arrays.asList(filePositive.listFiles()).subList(0, 100));
//        reviewFilesMap.put("Negative", Arrays.asList(fileNegative.listFiles()).subList(0, 100));


        LabeledSentenceProvider sentenceProvider = new FileLabeledSentenceProvider(reviewFilesMap, rng);


        //Need BERT WordPiece tokens...
        File wordPieceTokens = new File(rootDir, "uncased_L-12_H-768_A-12/vocab.txt");
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(wordPieceTokens, true, true, StandardCharsets.UTF_8);

        BertIterator b = BertIterator.builder()
            .tokenizer(t)
            .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, maxSentenceLength)
            .minibatchSize(minibatchSize)
            .padMinibatches(true)
            .sentenceProvider(sentenceProvider)
            .featureArrays(BertIterator.FeatureArrays.INDICES_MASK_SEGMENTID)
            .vocabMap(t.getVocab())
            .task(BertIterator.Task.SEQ_CLASSIFICATION)
            .prependToken("[CLS]")
            .build();

        return b;
    }


}
