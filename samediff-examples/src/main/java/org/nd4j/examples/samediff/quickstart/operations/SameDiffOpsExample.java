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

package org.nd4j.examples.samediff.quickstart.operations;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.enums.PartitionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * SameDiff Operations Namespace Example.
 *
 * SameDiff operations are organized into namespaces accessed via the SameDiff instance.
 * This example showcases operations across all namespaces, with emphasis on new ops
 * added for Transformer, LLM, and signal processing workloads.
 *
 * <h3>Operation Namespaces:</h3>
 * <ul>
 *   <li><b>sd.nn()</b> — Neural network ops: activations, attention, normalization, quantized ops</li>
 *   <li><b>sd.math()</b> — Mathematical ops: elementwise, reductions, distance metrics</li>
 *   <li><b>sd.linalg()</b> — Linear algebra: SVD, Cholesky, LU, eigendecomposition, einsum</li>
 *   <li><b>sd.cnn()</b> — Convolution ops: 1D/2D/3D conv, pooling, batch norm</li>
 *   <li><b>sd.rnn()</b> — Recurrent ops: LSTM, GRU, SRU cells</li>
 *   <li><b>sd.loss()</b> — Loss functions: cross-entropy, MSE, huber, CTC</li>
 *   <li><b>sd.image()</b> — Image ops: resize, crop, color space conversion, NMS</li>
 *   <li><b>sd.random()</b> — Random sampling: normal, uniform, bernoulli, binomial</li>
 *   <li><b>sd.bitwise()</b> — Bitwise ops: AND, OR, XOR, shifts</li>
 *   <li><b>sd.audio()</b> — Audio DSP: mel spectrogram, MFCC, Griffin-Lim, pitch detection</li>
 *   <li><b>sd.signal()</b> — Signal processing: DFT, STFT, window functions</li>
 * </ul>
 */
public class SameDiffOpsExample {
    private static final Logger log = LoggerFactory.getLogger(SameDiffOpsExample.class);

    public static void main(String[] args) {
        // =====================================================================
        // 1. Neural Network Operations (sd.nn())
        // =====================================================================
        log.info("=== 1. Neural Network Ops (sd.nn()) ===");

        SameDiff sd = SameDiff.create();

        // --- Activations ---
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);

        // Standard activations
        SDVariable relu = sd.nn.relu(x, 0);
        SDVariable gelu = sd.nn.gelu(x);
        SDVariable silu = sd.nn.silu(x);                    // SiLU / Swish activation
        SDVariable softmax = sd.nn.softmax(x, -1);
        SDVariable sigmoid = sd.nn.sigmoid(x);

        // Fused activations (optimized for performance)
        // SDVariable fusedGelu = sd.nn.fusedGelu(x);       // Fused GELU (single kernel)
        // SDVariable preciseGelu = sd.nn.preciseGelu(x);   // High-precision GELU

        // --- Normalization ops ---
        SDVariable gamma = sd.var("gamma", Nd4j.ones(64));
        SDVariable beta = sd.var("beta", Nd4j.zeros(64));

        // RMSNorm — standard in modern LLMs (LLaMA, Mistral, etc.)
        SDVariable rmsNormed = sd.nn.rmsNorm(x, gamma, 1e-5);

        // Fused LayerNorm — optimized single-kernel implementation
        // SDVariable fusedLN = sd.nn.fusedLayerNorm(x, gamma, beta, 1e-5);

        log.info("  silu, rmsNorm, fusedGelu, preciseGelu (new activations/norms)");

        // --- Attention ops ---
        log.info("  Attention operations:");
        log.info("    sd.nn.flashAttention(query, key, value, scale, ...)");
        log.info("    sd.nn.dotProductAttentionV2(queries, values, keys, ...)");
        log.info("    sd.nn.multiHeadAttention(query, key, value, ...)");
        log.info("    sd.nn.slidingWindowAttention(query, key, value, ...)");
        log.info("    sd.nn.sharedKvAttention(query, sharedKey, ...)   — GQA variant");
        log.info("    sd.nn.twoWayCrossAttention(...)                  — SAM-style");

        // --- Rotary Position Embeddings ---
        log.info("  Rotary embeddings:");
        log.info("    sd.nn.fusedRoPE(input, ropeCache, startPosition)");
        log.info("    sd.nn.fusedMRoPE(input, posT, posH, posW, ...)  — multimodal RoPE");
        log.info("    sd.nn.applyAlibi(scores, numHeads)              — ALiBi bias");

        // --- Quantized / LoRA ops ---
        log.info("  Quantized/LoRA ops:");
        log.info("    sd.nn.awqMatmul(input, weightPacked, scale, ...) — AWQ quantized matmul");
        log.info("    sd.nn.doraMatMul(input, weight, loraA, ...)      — DoRA matmul");
        log.info("    sd.nn.multiLoraMatmul(input, baseWeight, ...)    — batched LoRA");
        log.info("    sd.nn.fp8Matmul(a, b, scaleA, scaleB)           — FP8 matmul");
        log.info("    sd.nn.quantizedMatmul(a, b)                     — INT8 matmul");
        log.info("    sd.nn.smoothQuant(input, smoothScale)            — SmoothQuant");

        // --- Tensor parallelism ops ---
        log.info("  Tensor parallelism:");
        log.info("    sd.nn.columnParallelLinear(input, weight, tpRank, ...)");
        log.info("    sd.nn.rowParallelLinear(input, weight, tpRank, tpSize, ...)");

        // --- SSM / Mamba ---
        log.info("  SSM ops:");
        log.info("    sd.nn.selectiveScan(input)      — Mamba-style SSM");
        log.info("    sd.nn.causalConv1d(x, w, b, ...) — causal 1D conv for SSM");

        // --- Token sampling ---
        log.info("  Autoregressive sampling:");
        log.info("    sd.nn.tokenSample(logits)");
        log.info("    sd.nn.tokenSample(logits, temperature, topK, topP, ...)");

        // =====================================================================
        // 2. Math Operations (sd.math())
        // =====================================================================
        log.info("=== 2. Math Ops (sd.math()) ===");

        SDVariable a = sd.var("a", Nd4j.randn(4, 4));
        SDVariable b = sd.var("b", Nd4j.randn(4, 4));

        // Distance metrics
        SDVariable cosine = sd.math.cosineSimilarity(a, b, 1);
        SDVariable euclidean = sd.math.euclideanDistance(a, b, 1);
        SDVariable manhattan = sd.math.manhattanDistance(a, b, 1);
        SDVariable hamming = sd.math.hammingDistance(a, b, 1);
        SDVariable jaccard = sd.math.jaccardDistance(a, b, 1);

        // Information theory
        SDVariable entropy = sd.math.shannonEntropy(a, 1);
        SDVariable logEntropy = sd.math.logEntropy(a, 1);

        // Standardization
        SDVariable standardized = sd.math.standardize(a, 1);

        // Embedding lookup
        SDVariable embeddings = sd.var("embeddings", Nd4j.randn(100, 32));
        SDVariable indices = sd.constant(Nd4j.createFromArray(new int[]{0, 5, 10}));
        SDVariable looked = sd.math.embeddingLookup(embeddings, new SDVariable[]{indices}, PartitionMode.MOD);

        log.info("  Distance metrics: cosine, euclidean, manhattan, hamming, jaccard");
        log.info("  Info theory: shannonEntropy, logEntropy");
        log.info("  Utilities: standardize, embeddingLookup, confusionMatrix");

        // =====================================================================
        // 3. Linear Algebra Operations (sd.linalg())
        // =====================================================================
        log.info("=== 3. Linear Algebra Ops (sd.linalg()) ===");

        SDVariable matrix = sd.var("matrix", Nd4j.randn(4, 4));
        SDVariable symm = sd.var("symm", Nd4j.eye(4).add(Nd4j.randn(4, 4).mul(0.1)));

        // Matrix decompositions
        SDVariable chol = sd.linalg.cholesky(symm.mmul(sd.transpose(symm))); // Cholesky
        SDVariable svdResult = sd.linalg.svd(matrix, true, true);           // SVD
        SDVariable det = sd.linalg.matrixDeterminant(matrix);
        SDVariable inv = sd.linalg.matrixInverse(matrix);

        // Einstein summation
        // einsum supports arbitrary tensor contractions using index notation
        SDVariable einsumResult = sd.linalg.einsum(
                new SDVariable[]{a, b}, "ij,jk->ik");  // Matrix multiply via einsum

        // Linear system solving
        SDVariable rhs = sd.var("rhs", Nd4j.randn(4, 1));
        SDVariable solution = sd.linalg.solve(matrix, rhs);

        // Triangular operations
        SDVariable upper = sd.linalg.triu(matrix, 0);  // Upper triangular

        log.info("  Decompositions: cholesky, svd, lu, eig");
        log.info("  Solvers: solve, triangularSolve, lstsq");
        log.info("  Other: einsum, logdet, matrixDeterminant, matrixInverse");

        // =====================================================================
        // 4. Image Operations (sd.image())
        // =====================================================================
        log.info("=== 4. Image Ops (sd.image()) ===");

        log.info("  Color space: rgbToHsv, hsvToRgb, rgbToYiq, rgbToYuv");
        log.info("  Adjust: adjustContrast, adjustHue, adjustSaturation");
        log.info("  Spatial: imageResize, resizeBiLinear, resizeBiCubic");
        log.info("  Processing: cropAndResize, extractImagePatches, randomCrop");
        log.info("  Detection: nonMaxSuppression");
        log.info("  Transform: affineGrid, pad");

        // =====================================================================
        // 5. Audio Operations (sd.audio()) — NEW
        // =====================================================================
        log.info("=== 5. Audio Ops (sd.audio()) — NEW ===");

        log.info("  Feature extraction:");
        log.info("    sd.audio.melSpectrogram(input, sampleRate, fftSize, hopLen, numMels)");
        log.info("    sd.audio.mfcc(input, sampleRate, fftSize, hopLen, numMfcc)");
        log.info("    sd.audio.chromaFeatures(input, sampleRate, fftSize, numChroma)");
        log.info("");
        log.info("  Signal analysis:");
        log.info("    sd.audio.spectralCentroid(input, sampleRate, fftSize)");
        log.info("    sd.audio.spectralRolloff(input, sampleRate, fftSize, rolloffPct)");
        log.info("    sd.audio.zeroCrossingRate(input, frameLength, hopLength)");
        log.info("    sd.audio.pitchDetection(input, sampleRate, frameLen, hopLen)");
        log.info("");
        log.info("  Processing:");
        log.info("    sd.audio.preEmphasis(input, coefficient)");
        log.info("    sd.audio.audioNormalize(input, targetLevel, useRms)");
        log.info("    sd.audio.audioResample(input, origSampleRate, targetSampleRate)");
        log.info("    sd.audio.griffinLim(magnitudeSpec, fftSize, hopLen, numIter)");
        log.info("    sd.audio.melFilterbank(numMelBins, fftSize, sampleRate, ...)");
        log.info("    sd.audio.aWeighting(frequencies)");

        // =====================================================================
        // 6. Signal Processing Operations (sd.signal()) — NEW
        // =====================================================================
        log.info("=== 6. Signal Ops (sd.signal()) — NEW ===");

        log.info("  Window functions:");
        log.info("    sd.signal.hannWindow(size, periodic)");
        log.info("    sd.signal.hammingWindow(size, periodic)");
        log.info("    sd.signal.blackmanWindow(size, periodic)");
        log.info("");
        log.info("  Transforms:");
        log.info("    sd.signal.dft(input, axis, inverse, onesided)");
        log.info("    sd.signal.stft(signal, frameStep, window, ...)");

        // =====================================================================
        // 7. Random Sampling (sd.random())
        // =====================================================================
        log.info("=== 7. Random Sampling (sd.random()) ===");

        SDVariable normalSample = sd.random.normal("normal", 0.0, 1.0, DataType.FLOAT, 4, 4);
        SDVariable uniformSample = sd.random.uniform("uniform", 0.0, 1.0, DataType.FLOAT, 4, 4);
        SDVariable bernoulli = sd.random.bernoulli("bernoulli", 0.5, DataType.FLOAT, 4, 4);

        log.info("  normal, uniform, bernoulli, binomial, exponential, logNormal, normalTruncated");

        // =====================================================================
        // 8. Execute the graph
        // =====================================================================
        log.info("=== 8. Execute ===");

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("x", Nd4j.randn(2, 64));

        // Execute and get multiple outputs
        Map<String, INDArray> results = sd.output(placeholders, rmsNormed.name());
        log.info("RMSNorm output shape: {}", java.util.Arrays.toString(
                results.get(rmsNormed.name()).shape()));

        log.info("**************** SameDiff Ops Example finished ********************");
    }
}
