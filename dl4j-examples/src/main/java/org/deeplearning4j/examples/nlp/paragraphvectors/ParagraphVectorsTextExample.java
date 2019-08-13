/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.nlp.paragraphvectors;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.resources.Downloader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;

/**
 * This is example code for dl4j ParagraphVectors implementation. In this example we build distributed representation of all sentences present in training corpus.
 * However, you still use it for training on labelled documents, using sets of LabelledDocument and LabelAwareIterator implementation.
 * <p>
 * *************************************************************************************************
 * PLEASE NOTE: THIS EXAMPLE REQUIRES DL4J/ND4J VERSIONS >= rc3.8 TO COMPILE SUCCESSFULLY
 * *************************************************************************************************
 *
 * @author raver119@gmail.com
 */
public class ParagraphVectorsTextExample {

    private static final Logger log = LoggerFactory.getLogger(ParagraphVectorsTextExample.class);

    public static final String DATA_LOCAL_PATH;

    static {
        final String DATA_URL = "https://deeplearning4jblob.blob.core.windows.net/dl4j-examples/dl4j-examples/nlp.zip";
        final String MD5 = "1ac7cd7ca08f13402f0e3b83e20c0512";
        final int DOWNLOAD_RETRIES = 10;
        final String DOWNLOAD_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "nlp.zip");
        final String EXTRACT_DIR = FilenameUtils.concat(System.getProperty("user.home"), "dl4j-examples-data/dl4j-examples");
        DATA_LOCAL_PATH = FilenameUtils.concat(EXTRACT_DIR, "nlp");
        if (!new File(DATA_LOCAL_PATH).exists()) {
            try {
                System.out.println("_______________________________________________________________________");
                System.out.println("Downloading data (91MB) and extracting to \n\t" + DATA_LOCAL_PATH);
                System.out.println("_______________________________________________________________________");
                Downloader.downloadAndExtract("files",
                    new URL(DATA_URL),
                    new File(DOWNLOAD_PATH),
                    new File(EXTRACT_DIR),
                    MD5,
                    DOWNLOAD_RETRIES);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            System.out.println("_______________________________________________________________________");
            System.out.println("Example data present in \n\t" + DATA_LOCAL_PATH);
            System.out.println("_______________________________________________________________________");
        }
    }

    public static void main(String[] args) throws Exception {
        File file = new File(DATA_LOCAL_PATH, "raw_sentences.txt");
        SentenceIterator iter = new BasicLineIterator(file);

        AbstractCache<VocabWord> cache = new AbstractCache<>();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        /*
             if you don't have LabelAwareIterator handy, you can use synchronized labels generator
              it will be used to label each document/sequence/line with it's own label.

              But if you have LabelAwareIterator ready, you can can provide it, for your in-house labels
        */
        LabelsSource source = new LabelsSource("DOC_");

        ParagraphVectors vec = new ParagraphVectors.Builder()
            .minWordFrequency(1)
            .iterations(5)
            .epochs(1)
            .layerSize(100)
            .learningRate(0.025)
            .labelsSource(source)
            .windowSize(5)
            .iterate(iter)
            .trainWordVectors(false)
            .vocabCache(cache)
            .tokenizerFactory(t)
            .sampling(0)
            .build();

        vec.fit();

        /*
            In training corpus we have few lines that contain pretty close words invloved.
            These sentences should be pretty close to each other in vector space

            line 3721: This is my way .
            line 6348: This is my case .
            line 9836: This is my house .
            line 12493: This is my world .
            line 16393: This is my work .

            this is special sentence, that has nothing common with previous sentences
            line 9853: We now have one .

            Note that docs are indexed from 0
         */

        double similarity1 = vec.similarity("DOC_9835", "DOC_12492");
        log.info("9836/12493 ('This is my house .'/'This is my world .') similarity: " + similarity1);

        double similarity2 = vec.similarity("DOC_3720", "DOC_16392");
        log.info("3721/16393 ('This is my way .'/'This is my work .') similarity: " + similarity2);

        double similarity3 = vec.similarity("DOC_6347", "DOC_3720");
        log.info("6348/3721 ('This is my case .'/'This is my way .') similarity: " + similarity3);

        // likelihood in this case should be significantly lower
        double similarityX = vec.similarity("DOC_3720", "DOC_9852");
        log.info("3721/9853 ('This is my way .'/'We now have one .') similarity: " + similarityX +
            "(should be significantly lower)");
    }
}
