/* *****************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.advanced.modelling.textclassification.customcorpusword2vec;

import org.deeplearning4j.examples.utils.DownloaderUtility;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**-
 * This program generates a word-vector from news items stored in resources folder.
 * News File is located in ~/dl4j-examples-data/dl4j-examples/NewsData/
 * Word vector file : ~/dl4j-examples-data/dl4j-examples/NewsData/
 * Word vectors generated in this program are used in Training RNN to categorise news headlines.
 * <b></b>KIT Solutions Pvt. Ltd. (www.kitsol.com)</b>
 */
public class PrepareWordVector {

    private static Logger log = LoggerFactory.getLogger(PrepareWordVector.class);

    public static String dataLocalPath;

    public static void main(String[] args) throws Exception {

        dataLocalPath = DownloaderUtility.NEWSDATA.Download();
        // Gets Path to Text file
        String filePath = new File(dataLocalPath, "RawNewsToGenerateWordVector.txt").getAbsolutePath();

        log.info("Load & Vectorize Sentences....");
        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(filePath);
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();

        //CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
        //So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
        //Additionally it forces lower case for all tokens.
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
            .minWordFrequency(2)
            .iterations(5)
            .layerSize(100)
            .seed(42)
            .windowSize(20)
            .iterate(iter)
            .tokenizerFactory(t)
            .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Writing word vectors to text file....");

        // Write word vectors to file
        //noinspection unchecked
        WordVectorSerializer.writeWordVectors(vec.lookupTable(), new File(dataLocalPath, "NewsWordVector.txt").getAbsolutePath());
    }
}
