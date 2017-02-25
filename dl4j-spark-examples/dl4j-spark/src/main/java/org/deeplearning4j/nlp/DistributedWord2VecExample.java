package org.deeplearning4j.nlp;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.spark.models.sequencevectors.export.impl.HdfsModelExporter;
import org.deeplearning4j.spark.models.sequencevectors.export.impl.VocabCacheExporter;
import org.deeplearning4j.spark.models.sequencevectors.learning.elements.SparkCBOW;
import org.deeplearning4j.spark.models.sequencevectors.learning.elements.SparkSkipGram;
import org.deeplearning4j.spark.models.word2vec.SparkWord2Vec;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;

import java.io.File;
import java.util.Arrays;

/**
 * This example shows how to build Word2Vec model with distributed p2p ParameterServer.
 *
 * PLEASE NOTE: If you're using this example via spark-submit, you'll need to feed corpus via hdfs/s3/etc
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class DistributedWord2VecExample {

    @Parameter(names = {"-l","--layer"}, description = "Word2Vec layer size")
    protected int layerSize = 100;

    @Parameter(names = {"-s", "--shards"}, description = "Number of ParameterServer Shards")
    protected int numShards = 2;

    @Parameter(names = {"-t","--text"}, description = "HDFS path to training corpus")
    protected String corpusTextFile = "dl4j-examples/src/main/resources/raw_sentences.txt";

    @Parameter(names = {"-x"}, description = "Launch locally (NOT RECOMMENDED!)", arity = 1)
    protected boolean useSparkLocal = true;


    public void entryPoint(String[] args) {
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try { Thread.sleep(500); } catch (Exception e2) { }
            throw e;
        }

        // TODO: don't forget to delete this block
        log.info("Working Directory: {}", System.getProperty("user.dir"));
        if (!corpusTextFile.startsWith("hdfs")) {
            File file = new File(corpusTextFile);
            if (!file.exists())
                throw new RuntimeException("File not exists: [" + file.getAbsolutePath() + "]");
        }



        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("DL4j Spark Word2Vec + ParameterServer example");
        sparkConf.set("spark.kryo.registrationRequired","true");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        JavaRDD<String> corpus = sc.textFile(corpusTextFile);

        long lines = corpus.count();
        log.info("Total number of text lines: {}", lines);

        VoidConfiguration paramServerConfig = VoidConfiguration.builder()
            //.networkMask("172.16.0.0/12")
            //.shardAddresses(Arrays.asList("172.31.8.139:48381"))
            .numberOfShards(2)
            .ttl(4)
            .build();

        // tokenizer & preprocessor that'll be used during training
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());


        // Do NOT use this exporter impl if your corpus/model is huge, and can't fit into driver's memory
        VocabCacheExporter exporter = new VocabCacheExporter();



        SparkWord2Vec word2Vec = new SparkWord2Vec.Builder(paramServerConfig)
            .setTokenizerFactory(t)
            .setLearningAlgorithm(new SparkCBOW())
            .setModelExporter(exporter)
            .layerSize(113)
            .epochs(1)
            .workers(1)
            .useHierarchicSoftmax(false)
            .negativeSampling(10)
            .build();

        word2Vec.fitSentences(corpus);

        Word2Vec w2v = exporter.getWord2Vec();

        /*
            Just checking out what we have now. In ideal world it should be something like this:
            nearest words to 'day': [week, night, game, year, former, season, director, office, university, time]
         */
        log.info("VectorLength: {}", w2v.getWordVectorMatrix("day").length());
        log.info("day/night: {}", Transforms.cosineSim(w2v.getWordVectorMatrix("day"), w2v.getWordVectorMatrix("night")));
        log.info("one/two: {}", Transforms.cosineSim(w2v.getWordVectorMatrix("one"), w2v.getWordVectorMatrix("two")));
        log.info("nearest words to 'one': {}", w2v.wordsNearest("one",10));
        log.info("nearest words to 'day': {}", w2v.wordsNearest("day",10));
    }

    public static void main(String[] args) throws Exception {
        new DistributedWord2VecExample().entryPoint(args);
    }
}
