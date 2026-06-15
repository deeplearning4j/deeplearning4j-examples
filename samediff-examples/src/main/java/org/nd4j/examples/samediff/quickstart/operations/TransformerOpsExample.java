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
 * SameDiff Transformer / LLM Operations (sd.nn()) - Complete API Example
 *
 * The SDNN namespace includes modern transformer and LLM-specific operations
 * that enable building and running large language models natively in SameDiff.
 *
 * Operations covered:
 *
 *   Attention:
 *     - flashAttention          - Flash Attention with GQA support
 *     - slidingWindowAttention  - Sliding window (Mistral-style)
 *     - sharedKvAttention       - Shared/grouped KV attention
 *
 *   Positional Encoding:
 *     - fusedRoPE               - Rotary Position Embeddings
 *     - fusedMRoPE              - Multi-dimensional RoPE (Qwen2-VL style)
 *
 *   Normalization & Activations:
 *     - rmsNorm                 - Root Mean Square normalization
 *     - silu                    - SiLU / Swish activation
 *     - fusedGelu               - Fused GELU activation
 *
 *   Sequence Modeling:
 *     - causalConv1d            - Causal 1D convolution (Mamba SSM)
 *     - selectiveScan           - Selective scan (Mamba SSM)
 *     - tokenSample             - Token sampling with temperature/top-k/top-p
 *
 *   Quantized & Efficient Operations:
 *     - fp8Matmul               - FP8 matrix multiplication with scaling
 *     - quantizedMatmul         - Quantized matrix multiplication
 *     - awqMatmul               - AWQ quantized matmul
 *     - smoothQuant             - SmoothQuant activation scaling
 *     - doraMatMul              - DoRA (Weight-Decomposed LoRA) matmul
 *
 *   Tensor Parallelism:
 *     - columnParallelLinear    - Column-parallel linear layer
 *     - rowParallelLinear       - Row-parallel linear layer
 */
public class TransformerOpsExample {

    public static void main(String[] args) {

        // ============================================================
        // 1. FLASH ATTENTION - Efficient multi-head attention with GQA
        // ============================================================
        System.out.println("=== Flash Attention ===");
        {
            SameDiff sd = SameDiff.create();

            int batch = 2, seqLen = 128, headDim = 64;
            int numHeads = 8, numKvHeads = 2; // GQA: 8 query heads, 2 KV heads (4:1 ratio)

            SDVariable query = sd.placeHolder("query", DataType.FLOAT, batch, seqLen, numHeads, headDim);
            SDVariable key = sd.placeHolder("key", DataType.FLOAT, batch, seqLen, numKvHeads, headDim);
            SDVariable value = sd.placeHolder("value", DataType.FLOAT, batch, seqLen, numKvHeads, headDim);

            // flashAttention(query, key, value, scale, isCausal, numHeads, numKvHeads)
            double scale = 1.0 / Math.sqrt(headDim);
            SDVariable attnOut = sd.nn().flashAttention("flashAttn", query, key, value,
                    scale, true, numHeads, numKvHeads);

            INDArray q = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);
            INDArray k = Nd4j.randn(DataType.FLOAT, batch, seqLen, numKvHeads, headDim);
            INDArray v = Nd4j.randn(DataType.FLOAT, batch, seqLen, numKvHeads, headDim);

            HashMap<String, INDArray> ph = new HashMap<>();
            ph.put("query", q);
            ph.put("key", k);
            ph.put("value", v);

            Map<String, INDArray> result = sd.output(ph, "flashAttn");
            System.out.println("  Flash Attention output: " + result.get("flashAttn").shapeInfoToString());
            System.out.println("  GQA ratio: " + numHeads + " query heads / " + numKvHeads + " KV heads");
            System.out.println("  Causal masking: enabled (autoregressive)");
        }

        // ============================================================
        // 2. SLIDING WINDOW ATTENTION - Local attention (Mistral-style)
        // ============================================================
        System.out.println("\n=== Sliding Window Attention ===");
        {
            SameDiff sd = SameDiff.create();

            int batch = 2, seqLen = 256, headDim = 64;
            int numHeads = 8, numKvHeads = 2, windowSize = 128;

            SDVariable query = sd.placeHolder("query", DataType.FLOAT, batch, seqLen, numHeads, headDim);
            SDVariable key = sd.placeHolder("key", DataType.FLOAT, batch, seqLen, numKvHeads, headDim);
            SDVariable value = sd.placeHolder("value", DataType.FLOAT, batch, seqLen, numKvHeads, headDim);

            double scale = 1.0 / Math.sqrt(headDim);
            SDVariable slidingAttn = sd.nn().slidingWindowAttention("slidingAttn",
                    query, key, value, windowSize, numHeads, numKvHeads, scale);

            System.out.println("  Sliding window size: " + windowSize);
            System.out.println("  Each token attends to its " + windowSize + " nearest neighbors");
            System.out.println("  Memory: O(n*w) vs O(n^2) for full attention");
        }

        // ============================================================
        // 3. SHARED KV ATTENTION - Grouped/shared key-value attention
        // ============================================================
        System.out.println("\n=== Shared KV Attention ===");
        {
            SameDiff sd = SameDiff.create();

            int batch = 2, seqLen = 128, headDim = 64;
            int numHeads = 8, numKvHeads = 2;

            SDVariable query = sd.placeHolder("q", DataType.FLOAT, batch, seqLen, numHeads, headDim);
            SDVariable sharedKey = sd.placeHolder("k", DataType.FLOAT, batch, seqLen, numKvHeads, headDim);
            SDVariable sharedValue = sd.placeHolder("v", DataType.FLOAT, batch, seqLen, numKvHeads, headDim);

            // Minimal: no mask
            SDVariable out1 = sd.nn().sharedKvAttention("simple", query, sharedKey, sharedValue,
                    numHeads, numKvHeads);

            // With mask and causal + sliding window
            SDVariable mask = sd.placeHolder("mask", DataType.FLOAT, batch, 1, seqLen, seqLen);
            SDVariable out2 = sd.nn().sharedKvAttention("masked", query, sharedKey, sharedValue,
                    mask, numHeads, numKvHeads, 1, 0, 0.0);

            System.out.println("  Shared KV: " + numKvHeads + " KV heads shared across " + numHeads + " query heads");
            System.out.println("  Variants: minimal (no mask), masked, causal+sliding, full config");
        }

        // ============================================================
        // 4. FUSED ROPE - Rotary Position Embeddings
        // ============================================================
        System.out.println("\n=== Fused RoPE (Rotary Position Embeddings) ===");
        {
            SameDiff sd = SameDiff.create();

            int batch = 2, seqLen = 128, numHeads = 8, headDim = 64;
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, numHeads, headDim);

            // Variant 1: Precomputed RoPE cache with start position
            // ropeCache shape: [maxSeqLen, headDim/2, 2] (cos/sin pairs)
            SDVariable ropeCache = sd.placeHolder("ropeCache", DataType.FLOAT, 2048, headDim / 2, 2);
            SDVariable rope1 = sd.nn().fusedRoPE("rope_cached", input, ropeCache, 0);

            // Variant 2: Dynamic position (for KV cache decoding)
            // ropeType: 0=standard, 1=NeoX-style interleaved
            SDVariable posOffset = sd.placeHolder("posOffset", DataType.INT, batch);
            SDVariable rope2 = sd.nn().fusedRoPE("rope_dynamic", input, posOffset,
                    0,         // ropeType: 0=standard
                    10000.0,   // freqBase (theta)
                    1.0,       // freqScale
                    headDim);  // rotaryDims

            System.out.println("  RoPE variant 1: precomputed cache (training/prefill)");
            System.out.println("  RoPE variant 2: dynamic position (KV cache decoding)");
            System.out.println("  freqBase=10000 (standard), adjust for long-context models");
        }

        // ============================================================
        // 5. FUSED MROPE - Multi-dimensional RoPE (Qwen2-VL style)
        // ============================================================
        System.out.println("\n=== Fused MRoPE (Multi-dimensional RoPE) ===");
        {
            SameDiff sd = SameDiff.create();

            int batch = 1, seqLen = 64, numHeads = 8, headDim = 64;
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, numHeads, headDim);

            // Position IDs for temporal, height, width dimensions
            SDVariable posT = sd.placeHolder("posT", DataType.INT, batch, seqLen);
            SDVariable posH = sd.placeHolder("posH", DataType.INT, batch, seqLen);
            SDVariable posW = sd.placeHolder("posW", DataType.INT, batch, seqLen);

            // sectionT + sectionH + sectionW should sum to headDim
            SDVariable mrope = sd.nn().fusedMRoPE("mrope", input, posT, posH, posW,
                    22,        // sectionT: dims for temporal
                    21,        // sectionH: dims for height
                    21,        // sectionW: dims for width
                    false,     // interleaved
                    10000.0);  // freqBase

            System.out.println("  MRoPE: separate position encoding for T, H, W dimensions");
            System.out.println("  Used in vision-language models (Qwen2-VL)");
            System.out.println("  Section split: T=22, H=21, W=21 (sum=64=headDim)");
        }

        // ============================================================
        // 6. RMS NORM - Root Mean Square Layer Normalization
        // ============================================================
        System.out.println("\n=== RMS Normalization ===");
        {
            SameDiff sd = SameDiff.create();

            int batch = 2, seqLen = 128, hiddenDim = 512;
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, hiddenDim);
            SDVariable gamma = sd.var("gamma", Nd4j.ones(DataType.FLOAT, hiddenDim));

            // Full: input, gamma, epsilon
            SDVariable rms1 = sd.nn().rmsNorm("rms_full", input, gamma, 1e-5);

            // Without epsilon (defaults to 1e-5)
            SDVariable rms2 = sd.nn().rmsNorm("rms_default_eps", input, gamma);

            // Without gamma (no learnable scale)
            SDVariable rms3 = sd.nn().rmsNorm("rms_no_gamma", input, 1e-6);

            // Minimal: input only
            SDVariable rms4 = sd.nn().rmsNorm("rms_minimal", input);

            INDArray inputData = Nd4j.randn(DataType.FLOAT, batch, seqLen, hiddenDim);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", inputData), "rms_full");
            System.out.println("  RMSNorm output: " + result.get("rms_full").shapeInfoToString());
            System.out.println("  Formula: x * gamma / sqrt(mean(x^2) + eps)");
            System.out.println("  Used in: LLaMA, Mistral, Gemma (replaces LayerNorm)");
        }

        // ============================================================
        // 7. SILU and FUSED GELU - Modern activations
        // ============================================================
        System.out.println("\n=== SiLU and Fused GELU ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 512);

            // SiLU = x * sigmoid(x), also called Swish
            SDVariable siluOut = sd.nn().silu("silu", input);

            // Fused GELU (fast approximation)
            SDVariable geluOut = sd.nn().fusedGelu("gelu", input);

            INDArray inputData = Nd4j.randn(DataType.FLOAT, 2, 512);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", inputData), "silu", "gelu");
            System.out.println("  SiLU output: " + result.get("silu").shapeInfoToString());
            System.out.println("  GELU output: " + result.get("gelu").shapeInfoToString());
            System.out.println("  SiLU: used in LLaMA/Mistral FFN (gate * silu(up))");
            System.out.println("  GELU: used in GPT/BERT/Gemma FFN layers");
        }

        // ============================================================
        // 8. CAUSAL CONV1D - Mamba SSM causal convolution
        // ============================================================
        System.out.println("\n=== Causal Conv1D (Mamba SSM) ===");
        {
            SameDiff sd = SameDiff.create();

            int batch = 2, seqLen = 128, dModel = 256, kernelSize = 4;
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, batch, seqLen, dModel);
            SDVariable weight = sd.placeHolder("weight", DataType.FLOAT, dModel, 1, kernelSize);
            SDVariable bias = sd.placeHolder("bias", DataType.FLOAT, dModel);

            // Minimal: x + weight
            SDVariable[] conv1 = sd.nn().causalConv1d(new String[]{"conv_out", "conv_state"}, x, weight);

            // With bias
            SDVariable[] conv2 = sd.nn().causalConv1d(new String[]{"conv_bias_out", "conv_bias_state"},
                    x, weight, bias);

            // Full: with conv state (for incremental decoding)
            SDVariable convStateIn = sd.placeHolder("convState", DataType.FLOAT, batch, dModel, kernelSize - 1);
            SDVariable[] conv3 = sd.nn().causalConv1d(
                    new String[]{"conv_full_out", "conv_full_state"},
                    x, weight, bias, convStateIn,
                    1,   // activation: 0=none, 1=silu
                    0);  // wFormat

            System.out.println("  Causal Conv1D returns [output, updatedConvState]");
            System.out.println("  Used in Mamba SSM for sequence-local processing");
            System.out.println("  Conv state enables incremental token-by-token decoding");
        }

        // ============================================================
        // 9. SELECTIVE SCAN - Mamba SSM core operation
        // ============================================================
        System.out.println("\n=== Selective Scan (Mamba SSM) ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 2, 128, 256);

            SDVariable scanOut = sd.nn().selectiveScan("ssm_out", input);

            System.out.println("  Selective scan: core of Mamba state-space model");
            System.out.println("  Replaces attention with O(n) linear recurrence");
            System.out.println("  Input-dependent state transitions for selectivity");
        }

        // ============================================================
        // 10. TOKEN SAMPLE - LLM token sampling
        // ============================================================
        System.out.println("\n=== Token Sampling ===");
        {
            SameDiff sd = SameDiff.create();

            // Logits from LLM output: [batch, vocabSize]
            int vocabSize = 32000;
            SDVariable logits = sd.placeHolder("logits", DataType.FLOAT, -1, vocabSize);

            // Greedy (default): argmax
            SDVariable greedy = sd.nn().tokenSample("greedy", logits);

            // With temperature, top-k, top-p
            SDVariable sampled = sd.nn().tokenSample("sampled", logits,
                    0.8,    // temperature (< 1.0 = more deterministic)
                    40,     // top-k (keep top 40 tokens)
                    0.95,   // top-p / nucleus (keep tokens summing to 95% probability)
                    42L);   // random seed

            INDArray logitData = Nd4j.randn(DataType.FLOAT, 1, vocabSize);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("logits", logitData), "greedy", "sampled");
            System.out.println("  Greedy token: " + result.get("greedy"));
            System.out.println("  Sampled token (temp=0.8, k=40, p=0.95): " + result.get("sampled"));
        }

        // ============================================================
        // 11. FP8 MATMUL - FP8 precision matrix multiply
        // ============================================================
        System.out.println("\n=== FP8 Matrix Multiplication ===");
        {
            SameDiff sd = SameDiff.create();

            SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 256);
            SDVariable b = sd.placeHolder("b", DataType.FLOAT, 256, 512);

            // Per-tensor scaling factors for FP8 quantization
            SDVariable scaleA = sd.constant("scaleA", Nd4j.scalar(DataType.FLOAT, 1.0));
            SDVariable scaleB = sd.constant("scaleB", Nd4j.scalar(DataType.FLOAT, 1.0));

            SDVariable fp8Result = sd.nn().fp8Matmul("fp8mm", a, b, scaleA, scaleB);

            System.out.println("  FP8 matmul with per-tensor scaling");
            System.out.println("  E4M3 for forward pass (higher precision)");
            System.out.println("  E5M2 for backward pass (wider range for gradients)");
            System.out.println("  ~2x throughput over FP16 on supported hardware");
        }

        // ============================================================
        // 12. QUANTIZED MATMUL
        // ============================================================
        System.out.println("\n=== Quantized Matrix Multiplication ===");
        {
            SameDiff sd = SameDiff.create();

            SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 256);
            SDVariable b = sd.placeHolder("b", DataType.FLOAT, 256, 512);

            SDVariable qResult = sd.nn().quantizedMatmul("qmm", a, b);

            System.out.println("  Quantized matmul: automatic quantization for inference");
        }

        // ============================================================
        // 13. AWQ MATMUL - Activation-Aware Weight Quantization
        // ============================================================
        System.out.println("\n=== AWQ Matmul ===");
        {
            SameDiff sd = SameDiff.create();

            int inFeatures = 256, outFeatures = 512, groupSize = 128;
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, inFeatures);

            // AWQ uses packed INT4 weights with per-group FP16 scales
            SDVariable weightPacked = sd.placeHolder("weightPacked", DataType.INT,
                    inFeatures, outFeatures / 8); // INT4 packed
            SDVariable weightScale = sd.placeHolder("weightScale", DataType.FLOAT,
                    inFeatures / groupSize, outFeatures);

            SDVariable awqOut = sd.nn().awqMatmul("awq", input, weightPacked, weightScale, groupSize);

            System.out.println("  AWQ: INT4 weights with activation-aware calibration");
            System.out.println("  Group size: " + groupSize + " (scale granularity)");
            System.out.println("  ~4x weight compression vs FP16");
        }

        // ============================================================
        // 14. SMOOTH QUANT - Activation smoothing for quantization
        // ============================================================
        System.out.println("\n=== SmoothQuant ===");
        {
            SameDiff sd = SameDiff.create();

            int hiddenDim = 512;
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, hiddenDim);

            // Pre-computed per-channel smooth scale
            SDVariable smoothScale = sd.placeHolder("smoothScale", DataType.FLOAT, hiddenDim);

            SDVariable smoothed = sd.nn().smoothQuant("smoothed", input, smoothScale);

            System.out.println("  SmoothQuant: shifts quantization difficulty from activations to weights");
            System.out.println("  output = input * diag(smoothScale)");
            System.out.println("  Enables INT8 quantization of transformer activations");
        }

        // ============================================================
        // 15. DORA MATMUL - Weight-Decomposed Low-Rank Adaptation
        // ============================================================
        System.out.println("\n=== DoRA MatMul ===");
        {
            SameDiff sd = SameDiff.create();

            int inDim = 256, outDim = 512, rank = 16;
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, inDim);
            SDVariable weight = sd.placeHolder("W", DataType.FLOAT, inDim, outDim);     // frozen
            SDVariable loraA = sd.placeHolder("loraA", DataType.FLOAT, inDim, rank);     // trainable
            SDVariable loraB = sd.placeHolder("loraB", DataType.FLOAT, rank, outDim);    // trainable
            SDVariable magnitude = sd.placeHolder("mag", DataType.FLOAT, outDim);        // trainable

            // DoRA = magnitude * normalize(W + scaling * loraB @ loraA)
            SDVariable doraOut = sd.nn().doraMatMul("dora", input, weight, loraA, loraB,
                    magnitude, 1.0 / rank);

            System.out.println("  DoRA: Weight-Decomposed LoRA");
            System.out.println("  Decomposes weight update into magnitude and direction");
            System.out.println("  Better convergence than standard LoRA with same rank");
        }

        // ============================================================
        // 16. COLUMN/ROW PARALLEL LINEAR - Tensor parallelism
        // ============================================================
        System.out.println("\n=== Tensor Parallelism Ops ===");
        {
            SameDiff sd = SameDiff.create();

            int inDim = 512, outDim = 2048;
            int tpSize = 4, tpRank = 0; // 4-way tensor parallel, this is rank 0

            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, inDim);

            // Column parallel: split output columns across TP ranks
            SDVariable colWeight = sd.placeHolder("colW", DataType.FLOAT, inDim, outDim / tpSize);
            SDVariable colOut = sd.nn().columnParallelLinear("colParallel", input, colWeight,
                    tpRank, tpSize, true);

            // Row parallel: split input rows across TP ranks
            SDVariable rowWeight = sd.placeHolder("rowW", DataType.FLOAT, outDim / tpSize, inDim);
            SDVariable rowInput = sd.placeHolder("rowInput", DataType.FLOAT, -1, outDim / tpSize);
            SDVariable rowOut = sd.nn().rowParallelLinear("rowParallel", rowInput, rowWeight,
                    tpRank, tpSize, true);

            System.out.println("  Column parallel: each rank has W[:, rank*cols:(rank+1)*cols]");
            System.out.println("  Row parallel: each rank has W[rank*rows:(rank+1)*rows, :]");
            System.out.println("  TP size: " + tpSize + ", TP rank: " + tpRank);
            System.out.println("  Typical pattern: ColParallel -> activation -> RowParallel");
        }

        // ============================================================
        // PIPELINE: Transformer block with modern ops
        // ============================================================
        System.out.println("\n=== Complete Transformer Block ===");
        {
            SameDiff sd = SameDiff.create();

            int batch = 1, seqLen = 64, hiddenDim = 256, numHeads = 4, numKvHeads = 2, headDim = 64;

            SDVariable x = sd.placeHolder("x", DataType.FLOAT, batch, seqLen, hiddenDim);
            SDVariable attnGamma = sd.var("attnGamma", Nd4j.ones(DataType.FLOAT, hiddenDim));
            SDVariable ffnGamma = sd.var("ffnGamma", Nd4j.ones(DataType.FLOAT, hiddenDim));

            // Pre-attention RMSNorm
            SDVariable normed = sd.nn().rmsNorm("preAttnNorm", x, attnGamma, 1e-5);

            // Project Q, K, V (simplified with matmul)
            SDVariable wq = sd.var("Wq", Nd4j.randn(DataType.FLOAT, hiddenDim, numHeads * headDim).mul(0.02));
            SDVariable wk = sd.var("Wk", Nd4j.randn(DataType.FLOAT, hiddenDim, numKvHeads * headDim).mul(0.02));
            SDVariable wv = sd.var("Wv", Nd4j.randn(DataType.FLOAT, hiddenDim, numKvHeads * headDim).mul(0.02));

            // Q: [batch, seqLen, numHeads*headDim] -> [batch, seqLen, numHeads, headDim]
            SDVariable q = sd.linalg().mmul("Q", normed.reshape(batch * seqLen, hiddenDim), wq)
                    .reshape(batch, seqLen, numHeads, headDim);
            SDVariable k = sd.linalg().mmul("K", normed.reshape(batch * seqLen, hiddenDim), wk)
                    .reshape(batch, seqLen, numKvHeads, headDim);
            SDVariable v = sd.linalg().mmul("V", normed.reshape(batch * seqLen, hiddenDim), wv)
                    .reshape(batch, seqLen, numKvHeads, headDim);

            // Flash Attention with GQA
            double scale = 1.0 / Math.sqrt(headDim);
            SDVariable attnOut = sd.nn().flashAttention("attn", q, k, v,
                    scale, true, numHeads, numKvHeads);

            // Project output and residual connection
            SDVariable wo = sd.var("Wo", Nd4j.randn(DataType.FLOAT, numHeads * headDim, hiddenDim).mul(0.02));
            SDVariable projected = sd.linalg().mmul("attnProj",
                    attnOut.reshape(batch * seqLen, numHeads * headDim), wo)
                    .reshape(batch, seqLen, hiddenDim);
            SDVariable residual1 = x.add("residual1", projected);

            // Pre-FFN RMSNorm
            SDVariable ffnNormed = sd.nn().rmsNorm("preFFNNorm", residual1, ffnGamma, 1e-5);

            // SwiGLU FFN: gate * silu(up)
            int ffnDim = hiddenDim * 4;
            SDVariable wGate = sd.var("Wgate", Nd4j.randn(DataType.FLOAT, hiddenDim, ffnDim).mul(0.02));
            SDVariable wUp = sd.var("Wup", Nd4j.randn(DataType.FLOAT, hiddenDim, ffnDim).mul(0.02));
            SDVariable wDown = sd.var("Wdown", Nd4j.randn(DataType.FLOAT, ffnDim, hiddenDim).mul(0.02));

            SDVariable flatFFN = ffnNormed.reshape(batch * seqLen, hiddenDim);
            SDVariable gate = sd.nn().silu("gate", sd.linalg().mmul("gateProj", flatFFN, wGate));
            SDVariable up = sd.linalg().mmul("upProj", flatFFN, wUp);
            SDVariable ffnOut = sd.linalg().mmul("downProj", gate.mul(up), wDown)
                    .reshape(batch, seqLen, hiddenDim);
            SDVariable output = residual1.add("output", ffnOut);

            INDArray inputData = Nd4j.randn(DataType.FLOAT, batch, seqLen, hiddenDim);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("x", inputData), "output");
            System.out.println("  Transformer block output: " + result.get("output").shapeInfoToString());
            System.out.println("  Architecture: RMSNorm -> GQA Flash Attention -> RMSNorm -> SwiGLU FFN");
            System.out.println("  This is the core block used in LLaMA/Mistral/Gemma models");
        }

        System.out.println("\nAll transformer operations demonstrated successfully.");
    }
}
