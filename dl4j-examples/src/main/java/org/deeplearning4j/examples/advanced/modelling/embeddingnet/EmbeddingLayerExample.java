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

package org.deeplearning4j.examples.advanced.modelling.embeddingnet;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.EmbeddingSequenceLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Embedding Layer Examples: EmbeddingLayer and EmbeddingSequenceLayer.
 *
 * Embedding layers map discrete integer indices to dense continuous vectors.
 * They are fundamental building blocks for NLP and recommendation systems.
 *
 * DL4J provides two embedding layer types:
 *
 * 1. EmbeddingLayer — Single index lookup per sample.
 *    Input: [batchSize, 1] of integer indices
 *    Output: [batchSize, embeddingDim]
 *    Use case: categorical feature encoding, entity embeddings
 *
 * 2. EmbeddingSequenceLayer — Sequence of index lookups per sample.
 *    Input: [batchSize, sequenceLength] of integer indices
 *    Output: [batchSize, embeddingDim, sequenceLength] (NCW format)
 *        or: [batchSize, sequenceLength, embeddingDim] (NWC format)
 *    Use case: word embeddings for sequences (NLP), can feed into RNNs or 1D CNNs
 *
 * EmbeddingSequenceLayer parameters:
 * - nIn: vocabulary size (number of distinct indices)
 * - nOut: embedding dimension
 * - inputLength: expected sequence length (or use inferInputLength=true)
 * - inferInputLength: automatically determine sequence length from input
 * - outputFormat: NCW (channels first, default) or NWC (channels last)
 * - hasBias: whether to include a bias term (default false)
 *
 * Both layers can be initialized with pretrained embeddings (e.g., GloVe, Word2Vec)
 * using WeightInit.DISTRIBUTION or by directly setting the weight matrix.
 */
public class EmbeddingLayerExample {
    private static final Logger log = LoggerFactory.getLogger(EmbeddingLayerExample.class);

    public static void main(String[] args) throws Exception {
        int vocabSize = 10000;   // Number of distinct words/tokens
        int embeddingDim = 128;  // Embedding vector dimension
        int seqLength = 50;      // Max sequence length
        int numClasses = 5;      // Classification categories
        int batchSize = 32;
        int seed = 123;

        // =====================================================================
        // Example 1: EmbeddingSequenceLayer → LSTM for text classification
        // =====================================================================
        log.info("=== Example 1: EmbeddingSequenceLayer + LSTM ===");

        MultiLayerConfiguration embLstmConf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                // EmbeddingSequenceLayer looks up embeddings for each token in the sequence.
                // Input: [batchSize, seqLength] of integer indices (0 to vocabSize-1)
                // Output: [batchSize, embeddingDim, seqLength] in NCW format
                .layer(new EmbeddingSequenceLayer.Builder()
                        .nIn(vocabSize)
                        .nOut(embeddingDim)
                        .inferInputLength(true)  // Determine sequence length from input
                        .outputDataFormat(RNNFormat.NCW)
                        .build())
                // LSTM processes the embedding sequence
                .layer(new LSTM.Builder()
                        .nOut(64)
                        .activation(Activation.TANH)
                        .build())
                // Take last time step output for classification
                .layer(new LastTimeStep(new LSTM.Builder()
                        .nOut(64)
                        .activation(Activation.TANH)
                        .build()))
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.feedForward(seqLength))
                .build();

        MultiLayerNetwork embLstmModel = new MultiLayerNetwork(embLstmConf);
        embLstmModel.init();
        log.info("EmbeddingSequence + LSTM model parameters: {}", embLstmModel.numParams());

        // Create a dummy input batch of token indices
        // In practice, these would come from a tokenizer
        INDArray tokenInput = Nd4j.rand(batchSize, seqLength).muli(vocabSize).castTo(org.nd4j.linalg.api.buffer.DataType.INT);
        INDArray output = embLstmModel.output(tokenInput);
        log.info("Output shape: {}", java.util.Arrays.toString(output.shape()));

        // =====================================================================
        // Example 2: EmbeddingSequenceLayer → 1D Conv for text classification
        // =====================================================================
        log.info("=== Example 2: EmbeddingSequenceLayer + Conv1D ===");

        MultiLayerConfiguration embConvConf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                .layer(new EmbeddingSequenceLayer.Builder()
                        .nIn(vocabSize)
                        .nOut(embeddingDim)
                        .inferInputLength(true)
                        .outputDataFormat(RNNFormat.NCW)  // [batch, channels, length] for Conv1D
                        .build())
                // Conv1D acts as n-gram feature detector
                .layer(new Convolution1DLayer.Builder()
                        .nOut(64)
                        .kernelSize(3)
                        .stride(1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.MAX)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.feedForward(seqLength))
                .build();

        MultiLayerNetwork embConvModel = new MultiLayerNetwork(embConvConf);
        embConvModel.init();
        log.info("EmbeddingSequence + Conv1D model parameters: {}", embConvModel.numParams());

        // =====================================================================
        // Example 3: Single EmbeddingLayer for categorical features
        // =====================================================================
        log.info("=== Example 3: EmbeddingLayer for categorical features ===");

        int numCategories = 100;  // Number of unique categories
        int catEmbedDim = 16;     // Embedding size for the category

        MultiLayerConfiguration catEmbConf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(1e-3))
                .list()
                // EmbeddingLayer: single index per sample
                // Input: [batchSize, 1] integer index
                // Output: [batchSize, catEmbedDim]
                .layer(new EmbeddingLayer.Builder()
                        .nIn(numCategories)
                        .nOut(catEmbedDim)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork catEmbModel = new MultiLayerNetwork(catEmbConf);
        catEmbModel.init();
        log.info("Category embedding model parameters: {}", catEmbModel.numParams());

        log.info("**************** Embedding Layer Example finished ********************");
    }
}
