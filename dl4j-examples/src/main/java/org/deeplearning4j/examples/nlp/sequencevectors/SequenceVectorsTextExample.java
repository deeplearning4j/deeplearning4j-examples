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

package org.deeplearning4j.examples.nlp.sequencevectors;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
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
 * This is example of abstract sequence of data is learned using SequenceVectors. In this example, we use text sentences as Sequences, and VocabWords as SequenceElements.
 * So, this example is  a simple demonstration how one can learn distributed representation of data sequences.
 * <p>
 * For training on different data, you can extend base class SequenceElement, and feed model with your Iterable. Aslo, please note, in this case model persistence should be handled on your side.
 * <p>
 * *************************************************************************************************
 * PLEASE NOTE: THIS EXAMPLE REQUIRES DL4J/ND4J VERSIONS >= rc3.8 TO COMPILE SUCCESSFULLY
 * *************************************************************************************************
 *
 * @author raver119@gmail.com
 */
public class SequenceVectorsTextExample {

    protected static final Logger logger = LoggerFactory.getLogger(SequenceVectorsTextExample.class);
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

        File file = new File(DATA_LOCAL_PATH,"raw_sentences.txt");

        AbstractCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>().build();

        /*
            First we build line iterator
         */
        BasicLineIterator underlyingIterator = new BasicLineIterator(file);


        /*
            Now we need the way to convert lines into Sequences of VocabWords.
            In this example that's SentenceTransformer
         */
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        SentenceTransformer transformer = new SentenceTransformer.Builder()
            .iterator(underlyingIterator)
            .tokenizerFactory(t)
            .build();


        /*
            And we pack that transformer into AbstractSequenceIterator
         */
        AbstractSequenceIterator<VocabWord> sequenceIterator =
            new AbstractSequenceIterator.Builder<>(transformer).build();


        /*
            Now we should build vocabulary out of sequence iterator.
            We can skip this phase, and just set AbstractVectors.resetModel(TRUE), and vocabulary will be mastered internally
        */
        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
            .addSource(sequenceIterator, 5)
            .setTargetVocabCache(vocabCache)
            .build();

        constructor.buildJointVocabulary(false, true);

        /*
            Time to build WeightLookupTable instance for our new model
        */

        WeightLookupTable<VocabWord> lookupTable = new InMemoryLookupTable.Builder<VocabWord>()
            .vectorLength(150)
            .useAdaGrad(false)
            .cache(vocabCache)
            .build();

         /*
             reset model is viable only if you're setting AbstractVectors.resetModel() to false
             if set to True - it will be called internally
        */
        lookupTable.resetWeights(true);

        /*
            Now we can build AbstractVectors model, that suits our needs
         */
        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration())
            // minimum number of occurencies for each element in training corpus. All elements below this value will be ignored
            // Please note: this value has effect only if resetModel() set to TRUE, for internal model building. Otherwise it'll be ignored, and actual vocabulary content will be used
            .minWordFrequency(5)

            // WeightLookupTable
            .lookupTable(lookupTable)

            // abstract iterator that covers training corpus
            .iterate(sequenceIterator)

            // vocabulary built prior to modelling
            .vocabCache(vocabCache)

            // batchSize is the number of sequences being processed by 1 thread at once
            // this value actually matters if you have iterations > 1
            .batchSize(250)

            // number of iterations over batch
            .iterations(1)

            // number of iterations over whole training corpus
            .epochs(1)

            // if set to true, vocabulary will be built from scratches internally
            // otherwise externally provided vocab will be used
            .resetModel(false)


                /*
                    These two methods define our training goals. At least one goal should be set to TRUE.
                 */
            .trainElementsRepresentation(true)
            .trainSequencesRepresentation(false)

                /*
                    Specifies elements learning algorithms. SkipGram, for example.
                 */
            .elementsLearningAlgorithm(new SkipGram<VocabWord>())

            .build();

        /*
            Now, after all options are set, we just call fit()
         */
        vectors.fit();

        /*
            As soon as fit() exits, model considered built, and we can test it.
            Please note: all similarity context is handled via SequenceElement's labels, so if you're using AbstractVectors to build models for complex
            objects/relations please take care of Labels uniqueness and meaning for yourself.
         */
        double sim = vectors.similarity("day", "night");
        logger.info("Day/night similarity: " + sim);

    }
}
