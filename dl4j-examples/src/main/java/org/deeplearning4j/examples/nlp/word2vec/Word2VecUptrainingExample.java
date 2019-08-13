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

package org.deeplearning4j.examples.nlp.word2vec;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
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
import java.util.Collection;

/**
 * This is simple example for model weights update after initial vocab building.
 * If you have built your w2v model, and some time later you've decided that it can be
 * additionally trained over new corpus, here's an example how to do it.
 * <p>
 * PLEASE NOTE: At this moment, no new words will be added to vocabulary/model.
 * Only weights update process will be issued. It's often called "frozen vocab training".
 *
 * @author raver119@gmail.com
 */
public class Word2VecUptrainingExample {

    private static Logger log = LoggerFactory.getLogger(Word2VecUptrainingExample.class);
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
        /*
                Initial model training phase
         */
        String filePath = new File(DATA_LOCAL_PATH, "raw_sentences.txt").getAbsolutePath();

        log.info("Load & Vectorize Sentences....");
        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(filePath);
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        // manual creation of VocabCache and WeightLookupTable usually isn't necessary
        // but in this case we'll need them
        VocabCache<VocabWord> cache = new AbstractCache<>();
        WeightLookupTable<VocabWord> table = new InMemoryLookupTable.Builder<VocabWord>()
            .vectorLength(100)
            .useAdaGrad(false)
            .cache(cache).build();

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
            .minWordFrequency(5)
            .iterations(1)
            .epochs(1)
            .layerSize(100)
            .seed(42)
            .windowSize(5)
            .iterate(iter)
            .tokenizerFactory(t)
            .lookupTable(table)
            .vocabCache(cache)
            .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();


        Collection<String> lst = vec.wordsNearest("day", 10);
        log.info("Closest words to 'day' on 1st run: " + lst);

        /*
            at this moment we're supposed to have model built, and it can be saved for future use.
         */
        WordVectorSerializer.writeWord2VecModel(vec, "pathToSaveModel.txt");

        /*
            Let's assume that some time passed, and now we have new corpus to be used to weights update.
            Instead of building new model over joint corpus, we can use weights update mode.
         */
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("pathToSaveModel.txt");

        /*
            PLEASE NOTE: after model is restored, it's still required to set SentenceIterator and TokenizerFactory, if you're going to train this model
         */
        SentenceIterator iterator = new BasicLineIterator(filePath);
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        word2Vec.setTokenizerFactory(tokenizerFactory);
        word2Vec.setSentenceIterator(iterator);


        log.info("Word2vec uptraining...");

        word2Vec.fit();

        lst = word2Vec.wordsNearestSum("day", 10);
        log.info("Closest words to 'day' on 2nd run: " + lst);

        /*
            Model can be saved for future use now
         */
    }
}
