package org.deeplearning4j.nlp;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.spark.models.sequencevectors.export.impl.HdfsModelExporter;
import org.deeplearning4j.spark.models.sequencevectors.export.impl.VocabCacheExporter;
import org.deeplearning4j.spark.models.sequencevectors.learning.elements.SparkSkipGram;
import org.deeplearning4j.spark.models.word2vec.SparkWord2Vec;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;

import java.io.File;
import java.util.Arrays;

/**
 * This example shows how to build Word2Vec model with distributed p2p ParameterServer.
 *
 * PLEASE NOTE: This example is NOT meant to be run on localhost, consider spark-submit ONLY
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
            .networkMask("172.16.0.0/12")
            .shardAddresses(Arrays.asList("172.31.8.139:48381"))
            .ttl(4)
            .build();

        SparkWord2Vec word2Vec = new SparkWord2Vec.Builder(paramServerConfig)
            .setTokenizerFactory(new DefaultTokenizerFactory())
            .setLearningAlgorithm(new SparkSkipGram())
            .setModelExporter(new HdfsModelExporter<>("mymodel.txt"))
            .workers(48)
            .build();

        word2Vec.fitSentences(corpus);
    }

    public static void main(String[] args) throws Exception {
        new DistributedWord2VecExample().entryPoint(args);
    }
}
