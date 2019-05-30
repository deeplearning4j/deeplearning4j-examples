package org.deeplearning4j.examples.nlp.bert;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.examples.recurrent.word2vecsentiment.Word2VecSentimentRNN;
import org.deeplearning4j.iterator.BertIterator;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
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
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
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


public class BertSentimentExample {
    public static Logger log = LoggerFactory.getLogger(BertSentimentExample.class);

    public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/");

    public static void main(String[] args) throws Exception {


        String url = "https://deeplearning4jblob.blob.core.windows.net/testresources/bert_mrpc_frozen_v1.zip";
        File saveDir = new File("C:/Temp/BERT_Example/", "bert_mrpc_frozen_v1");
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

        //For training, convert weights and biases from constants to variables:
        for(SDVariable v : sd.variables()){
            if(v.isConstant() && v.dataType().isFPType() && !v.getArr().isScalar() && !floatConstants.contains(v.getVarName())){    //Skip scalars - trainable params
                log.info("Converting to variable: {} - dtype: {} - shape: {}", v.getVarName(), v.dataType(), Arrays.toString(v.getArr().shape()));
                v.convertToVariable();
            }
        }

        //For training, we'll need to add a label placeholder for one-hot labels:
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, 4, 2);
        SDVariable softmax = sd.getVariable("loss/Softmax");
        sd.loss().logLoss("loss", label, softmax);

        //Also randomize the output layer weights, zero the bias:
        sd.getVariable("output_weights").getArr().assign(Nd4j.randn(DataType.FLOAT, 2, 768).muli(Math.sqrt(2.0 / (2.0 + 768.0))));  //Xavier init
        sd.getVariable("output_bias").getArr().assign(0);


        //Next: create training pipeline...
        MultiDataSetIterator iterTrain = getDataSetIterator(true, 4, 128, new Random(12345));
        MultiDataSetIterator iterTest = getDataSetIterator(false, 4, 128, new Random(12345));

        //Set up training configuration...

        sd.setTrainingConfig(TrainingConfig.builder()
                .updater(new Adam(new RampSchedule(new FixedSchedule(1e-3), 50)))
//                .dataSetFeatureMapping("IteratorGetNext", "IteratorGetNext:4")
//                .dataSetFeatureMaskMapping("IteratorGetNext:1")
                .dataSetFeatureMapping("tokenIdxs", "sentenceIdx")
                .dataSetFeatureMaskMapping("mask")
                .dataSetLabelMapping("label")
                .build());

        File uiFile = new File("C:\\Temp\\BERT_Example\\UIData.bin");
        sd.setListeners(new ScoreListener(10, true, true),
            new UIListener.Builder(uiFile)
            .learningRate(1)
            .plotLosses(1)
            .trainAccuracy("loss/Softmax", 0)
            .updateRatios(1)
            .build()
        );

//        Evaluation evalBefore = new Evaluation();
//        sd.evaluate(iterTest, "loss/Softmax", 0, evalBefore);
//
//        log.info("Evaluation, before:");
//        log.info(evalBefore.stats());

        for(int i=0; i<10; i++ ) {
            sd.fit(iterTrain, 1);
            Evaluation e = new Evaluation();
            sd.evaluate(iterTest, "loss/Softmax", 0, e);
            log.info("Evaluation, end epoch {}:", i);
            log.info(e.stats());
        }
    }


    private static MultiDataSetIterator getDataSetIterator(boolean isTraining, int minibatchSize,
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
        File wordPieceTokens = new File("C:/Temp/BERT_Example/uncased_L-12_H-768_A-12/vocab.txt");
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
            .build();

        return b;
    }



}
