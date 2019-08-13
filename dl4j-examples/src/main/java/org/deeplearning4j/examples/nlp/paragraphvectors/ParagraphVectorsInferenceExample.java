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
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.resources.Downloader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;

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
        File resource = new File(DATA_LOCAL_PATH, "paravec/simple.pv");
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
