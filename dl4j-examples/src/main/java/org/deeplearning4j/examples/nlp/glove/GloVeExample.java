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

package org.deeplearning4j.examples.nlp.glove;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.models.glove.Glove;
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
 * @author raver119@gmail.com
 */
public class GloVeExample {

    private static final Logger log = LoggerFactory.getLogger(GloVeExample.class);
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
        File inputFile = new File(DATA_LOCAL_PATH, "raw_sentences.txt");

        // creating SentenceIterator wrapping our training corpus
        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());

        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Glove glove = new Glove.Builder()
            .iterate(iter)
            .tokenizerFactory(t)


            .alpha(0.75)
            .learningRate(0.1)

            // number of epochs for training
            .epochs(25)

            // cutoff for weighting function
            .xMax(100)

            // training is done in batches taken from training corpus
            .batchSize(1000)

            // if set to true, batches will be shuffled before training
            .shuffle(true)

            // if set to true word pairs will be built in both directions, LTR and RTL
            .symmetric(true)
            .build();

        glove.fit();

        double simD = glove.similarity("day", "night");
        log.info("Day/night similarity: " + simD);

        Collection<String> words = glove.wordsNearest("day", 10);
        log.info("Nearest words to 'day': " + words);

        System.exit(0);
    }
}
