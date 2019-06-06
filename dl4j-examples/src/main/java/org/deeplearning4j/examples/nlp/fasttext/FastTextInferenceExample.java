package org.deeplearning4j.examples.nlp.fasttext;

import org.deeplearning4j.examples.convolution.sentenceclassification.CnnSentenceClassificationExample;
import org.deeplearning4j.examples.recurrent.word2vecsentiment.Word2VecSentimentRNN;
import org.deeplearning4j.models.fasttext.FastText;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * @author Alex Black
 */
public class FastTextInferenceExample {

    public static void main(String[] args) throws Exception {

        /*
        Deeplearning4j supports FastText
        FastText trained models are available here: https://fasttext.cc/docs/en/pretrained-vectors.html
        Note that until further notice, only the .bin files are supported.
        The .vec files cannot be loaded

        Instructions:
        1. Download the English Wiki FastText file from here: https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
        2. Extract the cc.en.300.bin file
        3. Set the FastText file path below to where it was extracted

        The FastText vectors object can then be used as a drop-in replacement in places such as the following:
        * CnnSentenceClassificationExample (via CnnSentenceDataSetIterator)
        * Word2VecSentimentRNN (via the SentimentExampleIterator)

         */

        File fastTextFile = new File("C:/Temp/FastText/cc.en.300.bin");

        if(!fastTextFile.exists() || !fastTextFile.isFile())
            throw new RuntimeException();

        System.out.println("Loading FastText file...");
        FastText fastText = new FastText(fastTextFile);

        System.out.println("Printing word vectors:");
        List<String> words = Arrays.asList("some", "test", "words", "here");
        for(String s : words){
            INDArray arr = fastText.getWordVectorMatrix(s);
            System.out.println("Word: \"" + s + "\"");
            System.out.println(arr);
        }


        System.out.println("Downloading iterator data...");
        Word2VecSentimentRNN.downloadData();

        System.out.println("Creating Iterator...");
        DataSetIterator iter = CnnSentenceClassificationExample.getDataSetIterator(true, fastText, 4, 32, new Random(12345));

        System.out.println("Printing first dataset...");
        DataSet ds = iter.next();
        System.out.println(Arrays.toString(ds.getFeatures().shape()));
        System.out.println(ds.getFeatures());

    }

}
