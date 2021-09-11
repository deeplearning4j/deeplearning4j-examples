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

package org.deeplearning4j.examples.advanced.modelling.embeddingsfromcorpus.paragraphvectors;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.examples.utils.DownloaderUtility;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * This is example code for dl4j ParagraphVectors inference use implementation.
 * In this example we load previously built model, and pass raw sentences, probably never seen before, to get their vector representation.
 * <p>
 * <p>
 * *************************************************************************************************
 * PLEASE NOTE: THIS EXAMPLE REQUIRES DL4J/ND4J VERSIONS >= 0.6.0 TO COMPILE SUCCESSFULLY
 * *************************************************************************************************
 *
 * @author raver119@gmail.com
 */
public class ParagraphVectorsInferenceExample {

    private static final Logger log = LoggerFactory.getLogger(ParagraphVectorsInferenceExample.class);

    public static String dataLocalPath;

    public static void main(String[] args) throws Exception {
        dataLocalPath = DownloaderUtility.NLPDATA.Download();
        File resource = new File(dataLocalPath, "paravec/simple.pv");
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        // we load externally originated model
        ParagraphVectors vectors = WordVectorSerializer.readParagraphVectors(resource);
        vectors.setTokenizerFactory(t);
        vectors.getConfiguration().setIterations(1); // please note, we set iterations to 1 here, just to speedup inference

        /*
        // here's alternative way of doing this, word2vec model can be used directly
        // PLEASE NOTE: you can't use Google-like model here, since it doesn't have any Huffman tree information shipped.

        ParagraphVectors vectors = new ParagraphVectors.Builder()
            .useExistingWordVectors(word2vec)
            .build();
        */
        // we have to define tokenizer here, because restored model has no idea about it


        INDArray inferredVectorA = vectors.inferVector("This is my world .");
        INDArray inferredVectorA2 = vectors.inferVector("This is my world .");
        INDArray inferredVectorB = vectors.inferVector("This is my way .");

        // high similarity expected here, since in underlying corpus words WAY and WORLD have really close context
        log.info("Cosine similarity A/B: {}", Transforms.cosineSim(inferredVectorA, inferredVectorB));

        // equality expected here, since inference is happening for the same sentences
        log.info("Cosine similarity A/A2: {}", Transforms.cosineSim(inferredVectorA, inferredVectorA2));
    }
}
