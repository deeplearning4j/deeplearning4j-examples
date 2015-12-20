package org.deeplearning4j.examples.glove;

import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.io.File;
import java.util.Collection;

/**
 * @author raver119@gmail.com
 */
public class GloVeExample {

    private static final Logger log = LoggerFactory.getLogger(GloVeExample.class);

    public static void main(String[] args) throws Exception {
        File inputFile = new ClassPathResource("raw_sentences.txt").getFile();

        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());

        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Glove glove = new Glove.Builder()
                .iterate(iter)
                .tokenizerFactory(t)
                .alpha(0.75)
                .learningRate(0.05)
                .epochs(25)
                .xMax(100)
                .shuffle(true)
                .symmetric(true)
                .build();

        glove.fit();

        double simD = glove.similarity("day", "night");
        log.info("Day/night similarity: " + simD);

        Collection<String> words = glove.wordsNearest("day", 10);
        log.info("Nearest words to 'day': " + words);
    }
}
