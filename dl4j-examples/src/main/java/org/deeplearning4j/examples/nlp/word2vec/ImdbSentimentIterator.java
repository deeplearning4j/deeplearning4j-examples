package org.deeplearning4j.examples.nlp.word2vec;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

public class ImdbSentimentIterator implements DataSetIterator {
    private final int batchSize;
    private final int truncateLength;

    private int cursor = 0;
    private final File[] positiveFiles;
    private final File[] negativeFiles;

    private final TokenizerFactory tokenizerFactory;
    private final VocabCache vocab;

    ImdbSentimentIterator(String dataDirectory, VocabCache vocab, int batchSize, int truncateLength, boolean train){
        this.batchSize = batchSize;
        this.vocab = vocab;

        File p = new File(FilenameUtils.concat(dataDirectory, "aclImdb/" + (train ? "train" : "test") + "/pos/") + "/");
        File n = new File(FilenameUtils.concat(dataDirectory, "aclImdb/" + (train ? "train" : "test") + "/neg/") + "/");
        positiveFiles = p.listFiles();
        negativeFiles = n.listFiles();

        this.truncateLength = truncateLength;

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }

    @Override
    public DataSet next(int i) {
        if (cursor >= positiveFiles.length + negativeFiles.length) throw new NoSuchElementException();
        try{
            return nextDataSet(i);
        }catch(IOException e){
            throw new RuntimeException(e);
        }
    }

    private DataSet nextDataSet(int num) throws IOException {
        //First: load reviews to String. Alternate positive and negative reviews
        List<String> reviews = new ArrayList<>(num);
        boolean[] positive = new boolean[num];
        for( int i=0; i<num && cursor<totalExamples(); i++ ){
            if(cursor % 2 == 0){
                //Load positive review
                int posReviewNumber = cursor / 2;
                String review = FileUtils.readFileToString(positiveFiles[posReviewNumber], (Charset)null);
                reviews.add(review);
                positive[i] = true;
            } else {
                //Load negative review
                int negReviewNumber = cursor / 2;
                String review = FileUtils.readFileToString(negativeFiles[negReviewNumber], (Charset)null);
                reviews.add(review);
                positive[i] = false;
            }
            cursor++;
        }

        //Second: tokenize reviews and filter out unknown words
        List<List<Integer>> allTokensIndex = new ArrayList<>(reviews.size());
        for(String s : reviews){
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            List<Integer> tokensIndex = new ArrayList<>();
            for(String t : tokens ){
                if(vocab.hasToken(t)){
                    tokensIndex.add(vocab.indexOf(t)+1);
                }else{
                    tokensIndex.add(0);
                }
            }
            allTokensIndex.add(tokensIndex);
        }

        //Create data for training
        INDArray features = Nd4j.create(reviews.size(), 1, this.truncateLength);
        INDArray labels = Nd4j.create(reviews.size(), 2);    //Two labels: positive or negative

        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(reviews.size(), this.truncateLength);

        for( int i=0; i<reviews.size(); i++ ){
            List<Integer> tokensIndex = allTokensIndex.get(i);

            int seqLength = Math.min(tokensIndex.size(), this.truncateLength);

            int startSeqIndex = this.truncateLength - seqLength;

            // Assign token index into feature array
            features.put(
                new INDArrayIndex[] {
                    NDArrayIndex.point(i), NDArrayIndex.point(0), NDArrayIndex.interval(startSeqIndex, this.truncateLength)
                },
                Nd4j.create(tokensIndex.subList(0, seqLength)));

            // Assign "1" to each position where a feature is present
            featuresMask.get(NDArrayIndex.point(i), NDArrayIndex.interval(startSeqIndex, this.truncateLength)).assign(1);

            int idx = (positive[i] ? 0 : 1);
            labels.putScalar(new int[]{i,idx},1.0);   //Set label: [0,1] for negative, [1,0] for positive
        }

        return new DataSet(features,labels,featuresMask,null);
    }

    private int totalExamples() {
        return positiveFiles.length + negativeFiles.length;
    }

    @Override
    public int inputColumns() {
        return 0;
    }

    @Override
    public int totalOutcomes() {
        return 2;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return Arrays.asList("positive","negative");
    }

    @Override
    public boolean hasNext() {
        return cursor < totalExamples();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }
}
