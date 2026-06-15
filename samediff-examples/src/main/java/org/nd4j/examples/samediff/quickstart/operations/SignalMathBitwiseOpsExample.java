/*
 *
 * This program and the accompanying materials are made available under the
 *  terms of the Apache License, Version 2.0 which is available at
 *  https://www.apache.org/licenses/LICENSE-2.0.
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  License for the specific language governing permissions and limitations
 *  under the License.
 *
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package org.nd4j.examples.samediff.quickstart.operations;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * SameDiff Signal, Math, Bitwise, and Random Operations - Complete API Example
 *
 * This example covers three op namespaces:
 *
 * sd.signal() - Digital Signal Processing:
 *   - dft          - Discrete Fourier Transform
 *   - stft         - Short-Time Fourier Transform
 *   - stftSimple   - Simplified STFT
 *   - hannWindow   - Hann window function
 *   - hammingWindow - Hamming window function
 *   - blackmanWindow - Blackman window function
 *
 * sd.math() - Advanced Math Operations (non-NumPy highlights):
 *   - Distance metrics: cosine, euclidean, manhattan, hamming, jaccard
 *   - Entropy: entropy, logEntropy, shannonEntropy
 *   - embeddingLookup
 *   - confusionMatrix
 *   - moments, normalizeMoments
 *   - mergeAdd, mergeAvg, mergeMax
 *   - logSumExp, standardize
 *   - firstIndex, lastIndex (condition-based)
 *
 * sd.bitwise() - Bitwise Operations:
 *   - and, or, xor
 *   - leftShift, rightShift
 *   - leftShiftCyclic, rightShiftCyclic
 *   - bitRotl, bitRotr
 *   - bitsHammingDistance
 *
 * sd.random() - Random Number Generation:
 *   - normal, normalTruncated, uniform
 *   - bernoulli, binomial, exponential, logNormal
 */
public class SignalMathBitwiseOpsExample {

    public static void main(String[] args) {

        // ================================================================
        // PART 1: SIGNAL PROCESSING OPS (sd.signal())
        // ================================================================
        System.out.println("========================================");
        System.out.println("PART 1: Signal Processing Operations");
        System.out.println("========================================");

        // ============================================================
        // 1.1 WINDOW FUNCTIONS
        // ============================================================
        System.out.println("\n=== Window Functions ===");
        {
            SameDiff sd = SameDiff.create();

            SDVariable windowSize = sd.constant("size", Nd4j.scalar(DataType.INT, 256));

            // Hann window: raised cosine, good general-purpose window
            SDVariable hann = sd.signal().hannWindow("hann", windowSize, true);
            SDVariable hannNonPeriodic = sd.signal().hannWindow("hannNP", windowSize, false);

            // Hamming window: raised cosine with non-zero endpoints
            SDVariable hamming = sd.signal().hammingWindow("hamming", windowSize, true);

            // Blackman window: three-term cosine, better sidelobe suppression
            SDVariable blackman = sd.signal().blackmanWindow("blackman", windowSize, true);

            // Default (periodic=true)
            SDVariable hannDefault = sd.signal().hannWindow("hannDef", windowSize);
            SDVariable hammingDefault = sd.signal().hammingWindow("hammingDef", windowSize);
            SDVariable blackmanDefault = sd.signal().blackmanWindow("blackmanDef", windowSize);

            Map<String, INDArray> result = sd.output(Collections.emptyMap(),
                    "hann", "hamming", "blackman");
            System.out.println("  Hann window shape:     " + result.get("hann").shapeInfoToString());
            System.out.println("  Hamming window shape:  " + result.get("hamming").shapeInfoToString());
            System.out.println("  Blackman window shape: " + result.get("blackman").shapeInfoToString());
            System.out.println("  Periodic=true for FFT-based analysis");
            System.out.println("  Periodic=false for symmetric (filter design)");
        }

        // ============================================================
        // 1.2 DFT - Discrete Fourier Transform
        // ============================================================
        System.out.println("\n=== Discrete Fourier Transform ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable signal = sd.placeHolder("signal", DataType.FLOAT, -1, 256);

            // Full parameters: axis, inverse, onesided
            SDVariable fft = sd.signal().dft("fft", signal, -2, false, false);

            // One-sided (real FFT, returns only positive frequencies)
            SDVariable rfft = sd.signal().dft("rfft", signal, -2, false, true);

            // Inverse DFT
            SDVariable ifft = sd.signal().dft("ifft", signal, -2, true, false);

            // Default: axis=-2, inverse=false, onesided=false
            SDVariable fftDefault = sd.signal().dft("fftDefault", signal);

            INDArray signalData = Nd4j.randn(DataType.FLOAT, 1, 256);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("signal", signalData), "fft", "rfft");
            System.out.println("  Full FFT shape:    " + result.get("fft").shapeInfoToString());
            System.out.println("  One-sided FFT shape: " + result.get("rfft").shapeInfoToString());
        }

        // ============================================================
        // 1.3 STFT - Short-Time Fourier Transform
        // ============================================================
        System.out.println("\n=== Short-Time Fourier Transform ===");
        {
            SameDiff sd = SameDiff.create();

            int signalLen = 16000; // 1 second at 16kHz
            SDVariable signal = sd.placeHolder("signal", DataType.FLOAT, -1, signalLen);

            // Create a window
            SDVariable windowSize = sd.constant("wsize", Nd4j.scalar(DataType.INT, 512));
            SDVariable window = sd.signal().hannWindow("window", windowSize, true);

            // Frame step (hop length)
            SDVariable frameStep = sd.constant("hop", Nd4j.scalar(DataType.INT, 256));

            // Frame length
            SDVariable frameLength = sd.constant("flen", Nd4j.scalar(DataType.INT, 512));

            // Full STFT: signal, frameStep, window, frameLength, onesided
            SDVariable stft = sd.signal().stft("stft", signal, frameStep, window, frameLength, true);

            // Without onesided (defaults to false)
            SDVariable stftFull = sd.signal().stft("stftFull", signal, frameStep, window, frameLength);

            // Simplified STFT (auto-generates window)
            SDVariable stftSimple = sd.signal().stftSimple("stftSimple", signal, frameStep, true);
            SDVariable stftSimpleDefault = sd.signal().stftSimple("stftSimpleDef", signal, frameStep);

            INDArray signalData = Nd4j.randn(DataType.FLOAT, 1, signalLen);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("signal", signalData), "stft");
            System.out.println("  STFT output shape: " + result.get("stft").shapeInfoToString());
            System.out.println("  Window: Hann(512), Hop: 256, One-sided: true");
        }

        // ================================================================
        // PART 2: ADVANCED MATH OPS (sd.math())
        // ================================================================
        System.out.println("\n========================================");
        System.out.println("PART 2: Advanced Math Operations");
        System.out.println("========================================");

        // ============================================================
        // 2.1 DISTANCE METRICS
        // ============================================================
        System.out.println("\n=== Distance Metrics ===");
        {
            SameDiff sd = SameDiff.create();

            SDVariable x = sd.placeHolder("x", DataType.DOUBLE, -1, 128);
            SDVariable y = sd.placeHolder("y", DataType.DOUBLE, -1, 128);

            // Cosine similarity: dot(x,y) / (||x|| * ||y||)
            SDVariable cosSim = sd.math().cosineSimilarity("cosSim", x, y, 1);
            // Cosine distance: 1 - cosSim
            SDVariable cosDist = sd.math().cosineDistance("cosDist", x, y, 1);

            // Euclidean distance: sqrt(sum((x-y)^2))
            SDVariable eucDist = sd.math().euclideanDistance("eucDist", x, y, 1);

            // Manhattan distance: sum(|x-y|)
            SDVariable manDist = sd.math().manhattanDistance("manDist", x, y, 1);

            // Hamming distance: count(x != y)
            SDVariable hamDist = sd.math().hammingDistance("hamDist", x, y, 1);

            // Jaccard distance
            SDVariable jacDist = sd.math().jaccardDistance("jacDist", x, y, 1);

            INDArray xData = Nd4j.randn(DataType.DOUBLE, 4, 128);
            INDArray yData = Nd4j.randn(DataType.DOUBLE, 4, 128);
            HashMap<String, INDArray> ph = new HashMap<>();
            ph.put("x", xData);
            ph.put("y", yData);

            Map<String, INDArray> result = sd.output(ph,
                    "cosSim", "cosDist", "eucDist", "manDist");
            System.out.println("  Cosine similarity:  " + result.get("cosSim"));
            System.out.println("  Cosine distance:    " + result.get("cosDist"));
            System.out.println("  Euclidean distance: " + result.get("eucDist"));
            System.out.println("  Manhattan distance: " + result.get("manDist"));
        }

        // ============================================================
        // 2.2 ENTROPY
        // ============================================================
        System.out.println("\n=== Entropy Operations ===");
        {
            SameDiff sd = SameDiff.create();

            // Probability distribution (must sum to 1 along reduction axis)
            SDVariable probs = sd.placeHolder("probs", DataType.DOUBLE, -1, 10);

            // Shannon entropy: -sum(p * log(p))
            SDVariable entropy = sd.math().entropy("entropy", probs, 1);

            // Log entropy
            SDVariable logEntropy = sd.math().logEntropy("logEntropy", probs, 1);

            // Shannon entropy (with keepDims variant)
            SDVariable shannonEntropy = sd.math().shannonEntropy("shannonEntropy", probs, true, 1);

            // Create a simple probability distribution
            INDArray probData = Nd4j.rand(DataType.DOUBLE, 3, 10);
            // Normalize to sum to 1
            INDArray sums = probData.sum(1);
            for (int i = 0; i < 3; i++) {
                probData.getRow(i).divi(sums.getDouble(i));
            }

            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("probs", probData), "entropy", "shannonEntropy");
            System.out.println("  Entropy: " + result.get("entropy"));
            System.out.println("  Shannon entropy (keepDims): " + result.get("shannonEntropy"));
        }

        // ============================================================
        // 2.3 EMBEDDING LOOKUP
        // ============================================================
        System.out.println("\n=== Embedding Lookup ===");
        {
            SameDiff sd = SameDiff.create();

            // Embedding table: [vocabSize, embeddingDim]
            int vocabSize = 1000, embDim = 64;
            SDVariable embeddings = sd.var("embeddings",
                    Nd4j.randn(DataType.FLOAT, vocabSize, embDim).mul(0.02));

            // Lookup indices
            SDVariable indices = sd.placeHolder("indices", DataType.INT, -1, -1);

            // embeddingLookup(table, indices, partitionMode)
            SDVariable looked = sd.math().embeddingLookup("embedded",
                    embeddings, new SDVariable[]{indices},
                    org.nd4j.enums.PartitionMode.MOD);

            INDArray idxData = Nd4j.createFromArray(new int[][]{{5, 10, 15}, {20, 25, 30}});
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("indices", idxData), "embedded");
            System.out.println("  Embedding lookup shape: " + result.get("embedded").shapeInfoToString());
            System.out.println("  Input indices: [2, 3] -> Output: [2, 3, 64]");
        }

        // ============================================================
        // 2.4 CONFUSION MATRIX
        // ============================================================
        System.out.println("\n=== Confusion Matrix ===");
        {
            SameDiff sd = SameDiff.create();

            SDVariable labels = sd.placeHolder("labels", DataType.INT, -1);
            SDVariable predictions = sd.placeHolder("preds", DataType.INT, -1);

            // confusionMatrix(labels, predictions, numClasses)
            SDVariable cm = sd.math().confusionMatrix("cm", labels, predictions, 3);

            INDArray labelsData = Nd4j.createFromArray(0, 1, 2, 0, 1, 2, 0, 1, 2);
            INDArray predsData = Nd4j.createFromArray(0, 1, 2, 0, 2, 1, 1, 1, 2);
            HashMap<String, INDArray> ph = new HashMap<>();
            ph.put("labels", labelsData);
            ph.put("preds", predsData);

            Map<String, INDArray> result = sd.output(ph, "cm");
            System.out.println("  3-class confusion matrix:");
            System.out.println("  " + result.get("cm"));
        }

        // ============================================================
        // 2.5 MOMENTS AND STATISTICS
        // ============================================================
        System.out.println("\n=== Moments ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.DOUBLE, -1, 100);

            // moments: returns [mean, variance]
            SDVariable[] moments = sd.math().moments(new String[]{"mean", "var"}, x,
                    new long[]{1}, true);

            // Log-sum-exp: log(sum(exp(x)))
            SDVariable lse = sd.math().logSumExp("lse", x, 1);

            // Standardize: (x - mean) / std
            SDVariable standardized = sd.math().standardize("standardized", x, 1);

            // Zero fraction: fraction of elements that are zero
            SDVariable zf = sd.math().zeroFraction("zeroFrac", x);

            INDArray xData = Nd4j.randn(DataType.DOUBLE, 4, 100);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("x", xData),
                    "mean", "var", "lse", "standardized", "zeroFrac");
            System.out.println("  Mean: " + result.get("mean"));
            System.out.println("  Variance: " + result.get("var"));
            System.out.println("  Log-sum-exp: " + result.get("lse"));
            System.out.println("  Zero fraction: " + result.get("zeroFrac"));
        }

        // ============================================================
        // 2.6 MERGE OPERATIONS
        // ============================================================
        System.out.println("\n=== Merge Operations ===");
        {
            SameDiff sd = SameDiff.create();

            SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 10);
            SDVariable b = sd.placeHolder("b", DataType.FLOAT, -1, 10);
            SDVariable c = sd.placeHolder("c", DataType.FLOAT, -1, 10);

            // Element-wise merge operations across multiple inputs
            SDVariable mergeAdd = sd.math().mergeAdd("mergeAdd", a, b, c);
            SDVariable mergeAvg = sd.math().mergeAvg("mergeAvg", a, b, c);
            SDVariable mergeMax = sd.math().mergeMax("mergeMax", a, b, c);

            INDArray aData = Nd4j.createFromArray(new float[][]{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}});
            INDArray bData = Nd4j.createFromArray(new float[][]{{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}});
            INDArray cData = Nd4j.createFromArray(new float[][]{{5, 5, 5, 5, 5, 5, 5, 5, 5, 5}});
            HashMap<String, INDArray> ph = new HashMap<>();
            ph.put("a", aData);
            ph.put("b", bData);
            ph.put("c", cData);

            Map<String, INDArray> result = sd.output(ph, "mergeAdd", "mergeAvg", "mergeMax");
            System.out.println("  mergeAdd: " + result.get("mergeAdd"));
            System.out.println("  mergeAvg: " + result.get("mergeAvg"));
            System.out.println("  mergeMax: " + result.get("mergeMax"));
        }

        // ============================================================
        // 2.7 CLIP AND NORM OPERATIONS
        // ============================================================
        System.out.println("\n=== Clipping and Norms ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 100);

            // Clip by value
            SDVariable clipVal = sd.math().clipByValue("clipVal", x, -1.0, 1.0);

            // Clip by norm: rescale if norm > clipValue
            SDVariable clipNorm = sd.math().clipByNorm("clipNorm", x, 5.0, 1);

            // Clip by average norm
            SDVariable clipAvgNorm = sd.math().clipByAvgNorm("clipAvgNorm", x, 1.0, 1);

            INDArray xData = Nd4j.randn(DataType.FLOAT, 2, 100).mul(5);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("x", xData), "clipVal", "clipNorm");
            System.out.println("  Clip by value [-1,1] max: " + result.get("clipVal").amaxNumber());
            System.out.println("  Clip by norm (5.0): " + result.get("clipNorm").norm2Number());
        }

        // ============================================================
        // 2.8 SPECIAL FUNCTIONS
        // ============================================================
        System.out.println("\n=== Special Functions ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.DOUBLE, -1);

            // Error function and complement
            SDVariable erfResult = sd.math().erf("erf", x);
            SDVariable erfcResult = sd.math().erfc("erfc", x);

            // Non-standard activations
            SDVariable rationalTanh = sd.math().rationalTanh("rTanh", x);
            SDVariable rectifiedTanh = sd.math().rectifiedTanh("rectTanh", x);

            // Absolute sum reduction
            SDVariable asum = sd.math().asum("asum", x, true, 0);

            INDArray xData = Nd4j.createFromArray(-2.0, -1.0, 0.0, 1.0, 2.0);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("x", xData),
                    "erf", "erfc", "rTanh", "rectTanh");
            System.out.println("  erf:           " + result.get("erf"));
            System.out.println("  erfc:          " + result.get("erfc"));
            System.out.println("  rationalTanh:  " + result.get("rTanh"));
            System.out.println("  rectifiedTanh: " + result.get("rectTanh"));
        }

        // ================================================================
        // PART 3: BITWISE OPERATIONS (sd.bitwise())
        // ================================================================
        System.out.println("\n========================================");
        System.out.println("PART 3: Bitwise Operations");
        System.out.println("========================================");
        {
            SameDiff sd = SameDiff.create();

            SDVariable x = sd.placeHolder("x", DataType.INT, -1);
            SDVariable y = sd.placeHolder("y", DataType.INT, -1);

            // Basic bitwise ops
            SDVariable andResult = sd.bitwise().and("and", x, y);
            SDVariable orResult = sd.bitwise().or("or", x, y);
            SDVariable xorResult = sd.bitwise().xor("xor", x, y);

            // Shift operations
            SDVariable shiftAmount = sd.placeHolder("shift", DataType.INT, -1);
            SDVariable leftShift = sd.bitwise().leftShift("lshift", x, shiftAmount);
            SDVariable rightShift = sd.bitwise().rightShift("rshift", x, shiftAmount);

            // Cyclic (rotating) shifts
            SDVariable rotl = sd.bitwise().leftShiftCyclic("rotl", x, shiftAmount);
            SDVariable rotr = sd.bitwise().rightShiftCyclic("rotr", x, shiftAmount);

            // Bit rotate operations
            SDVariable bitRotl = sd.bitwise().bitRotl("bitRotl", x, shiftAmount);
            SDVariable bitRotr = sd.bitwise().bitRotr("bitRotr", x, shiftAmount);

            // Hamming distance (count differing bits)
            SDVariable hammDist = sd.bitwise().bitsHammingDistance("hammDist", x, y);

            INDArray xData = Nd4j.createFromArray(0b1010, 0b1100, 0b1111, 0b0001);
            INDArray yData = Nd4j.createFromArray(0b0101, 0b1010, 0b0000, 0b1110);
            INDArray shiftData = Nd4j.createFromArray(1, 2, 3, 4);

            HashMap<String, INDArray> ph = new HashMap<>();
            ph.put("x", xData);
            ph.put("y", yData);
            ph.put("shift", shiftData);

            Map<String, INDArray> result = sd.output(ph,
                    "and", "or", "xor", "lshift", "rshift", "hammDist");
            System.out.println("  x:         " + xData);
            System.out.println("  y:         " + yData);
            System.out.println("  x AND y:   " + result.get("and"));
            System.out.println("  x OR y:    " + result.get("or"));
            System.out.println("  x XOR y:   " + result.get("xor"));
            System.out.println("  x << shift: " + result.get("lshift"));
            System.out.println("  x >> shift: " + result.get("rshift"));
            System.out.println("  Hamming distance: " + result.get("hammDist"));
        }

        // ================================================================
        // PART 4: RANDOM NUMBER GENERATION (sd.random())
        // ================================================================
        System.out.println("\n========================================");
        System.out.println("PART 4: Random Number Generation");
        System.out.println("========================================");
        {
            SameDiff sd = SameDiff.create();

            // Normal distribution: N(mean=0, std=1)
            SDVariable normal = sd.random().normal("normal", 0.0, 1.0, DataType.FLOAT, 2, 5);

            // Truncated normal: values > 2*std are resampled
            SDVariable truncNormal = sd.random().normalTruncated("truncNormal",
                    0.0, 1.0, DataType.FLOAT, 2, 5);

            // Uniform distribution: U(min=0, max=1)
            SDVariable uniform = sd.random().uniform("uniform", 0.0, 1.0, DataType.FLOAT, 2, 5);

            // Bernoulli: binary outcomes with probability p
            SDVariable bernoulli = sd.random().bernoulli("bernoulli", 0.5, DataType.FLOAT, 2, 5);

            // Binomial: number of successes in n trials
            SDVariable binomial = sd.random().binomial("binomial", 10, 0.3, DataType.FLOAT, 2, 5);

            // Exponential: with rate parameter lambda
            SDVariable exponential = sd.random().exponential("exponential", 1.0, DataType.FLOAT, 2, 5);

            // Log-normal: exp(N(mean, std))
            SDVariable logNormal = sd.random().logNormal("logNormal", 0.0, 0.5, DataType.FLOAT, 2, 5);

            Map<String, INDArray> result = sd.output(Collections.emptyMap(),
                    "normal", "truncNormal", "uniform", "bernoulli", "binomial",
                    "exponential", "logNormal");
            System.out.println("  Normal(0,1):        " + result.get("normal"));
            System.out.println("  TruncNormal(0,1):   " + result.get("truncNormal"));
            System.out.println("  Uniform(0,1):       " + result.get("uniform"));
            System.out.println("  Bernoulli(0.5):     " + result.get("bernoulli"));
            System.out.println("  Binomial(10,0.3):   " + result.get("binomial"));
            System.out.println("  Exponential(1.0):   " + result.get("exponential"));
            System.out.println("  LogNormal(0,0.5):   " + result.get("logNormal"));
        }

        System.out.println("\nAll signal, math, bitwise, and random operations demonstrated successfully.");
    }
}
