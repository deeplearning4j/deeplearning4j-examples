/*******************************************************************************
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

package org.deeplearning4j.examples.advanced.modelling.embeddingsfromcorpus.paragraphvectors.tools;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Simple utility class that builds centroid vector for LabelledDocument
 * based on previously trained ParagraphVectors model
 *
 * @author raver119@gmail.com
 */
public class MeansBuilder {
    private VocabCache<VocabWord> vocabCache;
    private InMemoryLookupTable<VocabWord> lookupTable;
    private TokenizerFactory tokenizerFactory;

    public MeansBuilder(InMemoryLookupTable<VocabWord> lookupTable, TokenizerFactory tokenizerFactory) {
        this.lookupTable = lookupTable;
        this.vocabCache = lookupTable.getVocab();
        this.tokenizerFactory = tokenizerFactory;
    }

    /**
     * This method returns centroid (mean vector) for document.
     *
     * @param document
     * @return
     */
    public INDArray documentAsVector(LabelledDocument document) {
        List<String> documentAsTokens = tokenizerFactory.create(document.getContent()).getTokens();
        AtomicInteger cnt = new AtomicInteger(0);
        for (String word: documentAsTokens) {
            if (vocabCache.containsWord(word)) cnt.incrementAndGet();
        }
        INDArray allWords = Nd4j.create(cnt.get(), lookupTable.layerSize());

        cnt.set(0);
        for (String word: documentAsTokens) {
            if (vocabCache.containsWord(word))
                allWords.putRow(cnt.getAndIncrement(), lookupTable.vector(word));
        }

        INDArray mean = allWords.mean(0);

        return mean;
    }
}
