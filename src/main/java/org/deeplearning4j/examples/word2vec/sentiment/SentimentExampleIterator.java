package org.deeplearning4j.examples.word2vec.sentiment;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

/** This is a DataSetIterator that is specialized for the IMDB review dataset used in the Word2VecSentimentRNN example
 * It takes either the train or test set data from this data set, plus a WordVectors object (typically the Google News
 * 300 pretrained vectors from https://code.google.com/p/word2vec/) and generates training data sets.<br>
 * Inputs/features: variable-length time series, where each word (with unknown words removed) is represented by
 * its Word2Vec vector representation.<br>
 * Labels/target: a single class (negative or positive), predicted at the final time step (word) of each review
 *
 * @author Alex Black
 */
public class SentimentExampleIterator implements DataSetIterator {
    private final WordVectors wordVectors;
    private final int batchSize;
    private final int vectorSize;
    private final int truncateLength;

    private int cursor = 0;
    private final File[] positiveFiles;
    private final File[] negativeFiles;
    private final TokenizerFactory tokenizerFactory;

    /**
     * @param dataDirectory the directory of the IMDB review data set
     * @param wordVectors WordVectors object
     * @param batchSize Size of each minibatch for training
     * @param truncateLength If reviews exceed
     * @param train If true: return the training data. If false: return the testing data.
     */
    public SentimentExampleIterator(String dataDirectory, WordVectors wordVectors, int batchSize, int truncateLength, boolean train) throws IOException {
        this.batchSize = batchSize;
        this.vectorSize = wordVectors.lookupTable().layerSize();

        File p = new File(FilenameUtils.concat(dataDirectory, "aclImdb/" + (train ? "train" : "test") + "/pos/") + "/");
        File n = new File(FilenameUtils.concat(dataDirectory, "aclImdb/" + (train ? "train" : "test") + "/neg/") + "/");
        positiveFiles = p.listFiles();
        negativeFiles = n.listFiles();

        this.wordVectors = wordVectors;
        this.truncateLength = truncateLength;

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }


    @Override
    public DataSet next(int num) {
        if (cursor >= positiveFiles.length + negativeFiles.length) throw new NoSuchElementException();
        try{
            return nextDataSet(num);
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
                String review = FileUtils.readFileToString(positiveFiles[posReviewNumber]);
                reviews.add(review);
                positive[i] = true;
            } else {
                //Load negative review
                int negReviewNumber = cursor / 2;
                String review = FileUtils.readFileToString(negativeFiles[negReviewNumber]);
                reviews.add(review);
                positive[i] = false;
            }
            cursor++;
        }

        //Second: tokenize reviews and filter out unknown words
        List<List<String>> allTokens = new ArrayList<>(reviews.size());
        int maxLength = 0;
        for(String s : reviews){
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            List<String> tokensFiltered = new ArrayList<>();
            for(String t : tokens ){
                if(wordVectors.hasWord(t)) tokensFiltered.add(t);
            }
            allTokens.add(tokensFiltered);
            maxLength = Math.max(maxLength,tokensFiltered.size());
        }

        //If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
        if(maxLength > truncateLength) maxLength = truncateLength;

        //Create data for training
        //Here: we have reviews.size() examples of varying lengths
        INDArray features = Nd4j.create(reviews.size(), vectorSize, maxLength);
        INDArray labels = Nd4j.create(reviews.size(), 2, maxLength);    //Two labels: positive or negative
        //Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(reviews.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(reviews.size(), maxLength);

        int[] temp = new int[2];
        for( int i=0; i<reviews.size(); i++ ){
            List<String> tokens = allTokens.get(i);
            temp[0] = i;
            //Get word vectors for each word in review, and put them in the training data
            for( int j=0; j<tokens.size() && j<maxLength; j++ ){
                String token = tokens.get(j);
                INDArray vector = wordVectors.getWordVectorMatrix(token);
                features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);

                temp[1] = j;
                featuresMask.putScalar(temp, 1.0);  //Word is present (not padding) for this example + time step -> 1.0 in features mask
            }

            int idx = (positive[i] ? 0 : 1);
            int lastIdx = Math.min(tokens.size(),maxLength);
            labels.putScalar(new int[]{i,idx,lastIdx-1},1.0);   //Set label: [0,1] for negative, [1,0] for positive
            labelsMask.putScalar(new int[]{i,lastIdx-1},1.0);   //Specify that an output exists at the final time step for this example
        }

        return new DataSet(features,labels,featuresMask,labelsMask);
    }

    @Override
    public int totalExamples() {
        return positiveFiles.length + negativeFiles.length;
    }

    @Override
    public int inputColumns() {
        return vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return 2;
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
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return Arrays.asList("positive","negative");
    }

    @Override
    public boolean hasNext() {
        return cursor < numExamples();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {

    }

    /** Convenience method for loading review to String */
    public String loadReviewToString(int index) throws IOException{
        File f;
        if(index%2 == 0) f = positiveFiles[index/2];
        else f = negativeFiles[index/2];
        return FileUtils.readFileToString(f);
    }

    /** Convenience method to get label for review */
    public boolean isPositiveReview(int index){
        return index%2 == 0;
    }
}
