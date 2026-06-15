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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Advanced Transformer Operations in SameDiff (sd.nn() namespace)
 *
 * This example covers the advanced transformer primitives used by modern LLMs,
 * complementing the basic operations shown in TransformerOpsExample.java.
 *
 * Operations covered:
 *
 *   Attention variants:
 *     - Flash Attention (with causal mask and GQA grouping)
 *     - Grouped Query Attention (GQA) — separate from Flash Attention path
 *
 *   Positional encoding:
 *     - RoPE (Rotary Position Embeddings) - standard and dynamic variants
 *
 *   Normalization:
 *     - RMS Norm (replaces LayerNorm in LLaMA/Mistral/Gemma)
 *
 *   Activations:
 *     - SwiGLU (SiLU gate * linear up-projection — LLaMA-style FFN)
 *
 *   KV Cache:
 *     - KV cache update for efficient autoregressive inference
 *
 * Shape conventions used throughout:
 *   [batch, seqLen, numHeads, headDim]  - for Q/K/V tensors
 *   [batch, seqLen, hiddenDim]          - for hidden states
 *   [batch, seqLen]                     - for position indices
 *
 * Context for each operation:
 *   These are the building blocks of the LLaMA / Mistral / Gemma / Phi
 *   transformer architecture family.
 */
public class TransformerOpsAdvancedExample {

    public static void main(String[] args) {

        // ============================================================
        // 1. FLASH ATTENTION with causal mask and GQA
        // ============================================================
        System.out.println("=== 1. Flash Attention (Causal + GQA) ===");
        {
            // Flash Attention (Dao et al., 2022) computes exact attention in O(n)
            // memory (vs O(n^2) for naive attention) by tiling Q/K/V in SRAM and
            // never materializing the full attention matrix.
            //
            // Grouped Query Attention (GQA): use fewer KV heads than Q heads.
            //   numKvHeads = numHeads / groupSize
            //   Each KV head is shared by (numHeads / numKvHeads) query heads.
            //   Used in: LLaMA2-70B (GQA 8:1), Mistral (GQA 8:4), Falcon.
            //
            // Causal mask: token i can only attend to tokens 0..i (autoregressive).

            SameDiff sd = SameDiff.create();

            int batch = 2, seqLen = 256, headDim = 64;
            int numHeads = 8;    // total query heads
            int numKvHeads = 2;  // KV heads (GQA ratio = 4:1)

            SDVariable query = sd.placeHolder("q", DataType.FLOAT, batch, seqLen, numHeads, headDim);
            SDVariable key   = sd.placeHolder("k", DataType.FLOAT, batch, seqLen, numKvHeads, headDim);
            SDVariable value = sd.placeHolder("v", DataType.FLOAT, batch, seqLen, numKvHeads, headDim);

            // scale = 1/sqrt(headDim) is the standard QK scaling factor
            // that prevents dot products from growing too large in high dimensions
            double scale = 1.0 / Math.sqrt(headDim);

            SDVariable attnOut = sd.nn().flashAttention(
                    "flashAttn",   // output variable name
                    query,
                    key,
                    value,
                    scale,
                    true,          // isCausal: apply causal masking (autoregressive)
                    numHeads,
                    numKvHeads);

            // Execute
            INDArray qData = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim).mul(0.1);
            INDArray kData = Nd4j.randn(DataType.FLOAT, batch, seqLen, numKvHeads, headDim).mul(0.1);
            INDArray vData = Nd4j.randn(DataType.FLOAT, batch, seqLen, numKvHeads, headDim).mul(0.1);

            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put("q", qData);
            inputs.put("k", kData);
            inputs.put("v", vData);

            Map<String, INDArray> result = sd.output(inputs, "flashAttn");
            System.out.println("  Input Q shape:   " + Arrays.toString(qData.shape()));
            System.out.println("  Input K shape:   " + Arrays.toString(kData.shape()));
            System.out.println("  Output shape:    " + result.get("flashAttn").shapeInfoToString());
            System.out.println("  GQA ratio:       " + numHeads + "Q : " + numKvHeads + "KV (" + (numHeads/numKvHeads) + ":1)");
            System.out.println("  Causal mask:     enabled (each token only sees past tokens)");
            System.out.println("  Memory:          O(seq) vs O(seq^2) for naive attention");
        }

        // ============================================================
        // 2. GROUPED QUERY ATTENTION (GQA) - alternate path
        // ============================================================
        System.out.println("\n=== 2. Grouped Query Attention (GQA) ===");
        {
            // groupedQueryAttention is the general GQA kernel.
            // Unlike flashAttention, this may be used when FP8 or quantized weights
            // are involved, or when the Flash Attention kernel is not available.
            //
            // GQA reduces KV cache memory proportionally to the GQA ratio.
            // A 4:1 GQA on 8 heads means the KV cache is 4x smaller.

            SameDiff sd = SameDiff.create();

            int batch = 1, seqLen = 128, headDim = 64;
            int numHeads = 8, numKvHeads = 4;  // 2:1 GQA ratio

            SDVariable query = sd.placeHolder("gqa_q", DataType.FLOAT, batch, seqLen, numHeads, headDim);
            SDVariable key   = sd.placeHolder("gqa_k", DataType.FLOAT, batch, seqLen, numKvHeads, headDim);
            SDVariable value = sd.placeHolder("gqa_v", DataType.FLOAT, batch, seqLen, numKvHeads, headDim);

            SDVariable gqaOut = sd.nn().groupedQueryAttention(
                    "gqaOut",
                    query,
                    key,
                    value,
                    numKvHeads);   // number of KV heads (query heads inferred from Q shape)

            System.out.println("  groupedQueryAttention: general GQA kernel");
            System.out.println("  numKvHeads: " + numKvHeads + " (each KV head serves "
                    + (numHeads / numKvHeads) + " query heads)");
            System.out.println("  KV cache size vs MHA: " + numKvHeads + "/" + numHeads
                    + " = " + (100 * numKvHeads / numHeads) + "% of MHA size");
        }

        // ============================================================
        // 3. ROPE - Rotary Position Embeddings
        // ============================================================
        System.out.println("\n=== 3. RoPE (Rotary Position Embeddings) ===");
        {
            // RoPE (Su et al., 2021) encodes position by rotating Q and K vectors
            // in the complex plane. Unlike absolute position embeddings, RoPE:
            //   - Is applied after Q/K projection (not to input tokens)
            //   - Produces attention scores that depend only on relative position
            //   - Supports longer sequences than training length (with adjustments)
            //   - Is used in: LLaMA, Mistral, Falcon, GPT-NeoX, PaLM, Gemma
            //
            // Standard usage: precompute cos/sin caches once, apply to Q and K
            // before the attention op.

            SameDiff sd = SameDiff.create();

            int batch = 2, seqLen = 64, numHeads = 8, headDim = 64;

            // Q and K tensors that will have RoPE applied
            SDVariable query = sd.placeHolder("rope_q", DataType.FLOAT, batch, seqLen, numHeads, headDim);
            SDVariable key   = sd.placeHolder("rope_k", DataType.FLOAT, batch, seqLen, numHeads, headDim);

            // Precomputed RoPE cache: shape [maxSeqLen, headDim/2, 2]
            //   [:, :, 0] = cos values
            //   [:, :, 1] = sin values
            // Typically computed once at model load time and reused.
            int maxSeqLen = 2048;
            SDVariable ropeCache = sd.placeHolder("ropeCache", DataType.FLOAT, maxSeqLen, headDim / 2, 2);

            // Apply RoPE to Q and K (start position = 0 for full prefill)
            SDVariable rQ = sd.nn().rope("rope_q_out", query, ropeCache, 0);
            SDVariable rK = sd.nn().rope("rope_k_out", key,   ropeCache, 0);

            INDArray qData  = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim).mul(0.1);
            INDArray kData  = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim).mul(0.1);
            INDArray cache  = Nd4j.randn(DataType.FLOAT, maxSeqLen, headDim / 2, 2);

            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put("rope_q", qData);
            inputs.put("rope_k", kData);
            inputs.put("ropeCache", cache);

            Map<String, INDArray> result = sd.output(inputs, "rope_q_out", "rope_k_out");
            System.out.println("  Input Q shape:        " + Arrays.toString(qData.shape()));
            System.out.println("  RoPE cache shape:     " + Arrays.toString(cache.shape()));
            System.out.println("  Output Q (rotated):   " + result.get("rope_q_out").shapeInfoToString());
            System.out.println("  Output K (rotated):   " + result.get("rope_k_out").shapeInfoToString());
            System.out.println("  RoPE: position encoded via rotation in headDim/2 complex planes");
            System.out.println("  freqBase=10000 standard; YaRN/LongRoPE use modified freqBase for long context");
        }

        // ============================================================
        // 4. RMS NORM
        // ============================================================
        System.out.println("\n=== 4. RMS Normalization ===");
        {
            // RMSNorm (Zhang & Sennrich, 2019) is a simplified LayerNorm that
            // omits the mean subtraction step:
            //   LayerNorm: y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
            //   RMSNorm:   y = x / sqrt(mean(x^2) + eps) * gamma
            //
            // The mean subtraction is empirically unnecessary for transformer training
            // and RMSNorm is ~15% faster than LayerNorm.
            //
            // Used in: LLaMA, LLaMA2, Mistral, Gemma, Phi-2, DeepSeek
            // Note: gamma (scale) is learned; there is no learned bias in RMSNorm.

            SameDiff sd = SameDiff.create();

            int batch = 2, seqLen = 64, hiddenDim = 256;

            SDVariable input  = sd.placeHolder("rms_in", DataType.FLOAT, batch, seqLen, hiddenDim);
            SDVariable gamma  = sd.var("rms_gamma", Nd4j.ones(DataType.FLOAT, hiddenDim));

            // Full form: input, gamma (learnable scale), epsilon (numerical stability)
            SDVariable normed = sd.nn().rmsNorm("rms_out", input, gamma, 1e-5);

            INDArray inputData = Nd4j.randn(DataType.FLOAT, batch, seqLen, hiddenDim);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("rms_in", inputData), "rms_out");

            System.out.println("  Input shape:    " + Arrays.toString(inputData.shape()));
            System.out.println("  Output shape:   " + result.get("rms_out").shapeInfoToString());
            System.out.println("  Formula:        y = x * gamma / sqrt(mean(x^2) + 1e-5)");
            System.out.println("  Pre-norm vs Post-norm:");
            System.out.println("    Pre-norm (LLaMA):  RMSNorm applied BEFORE attention/FFN (more stable)");
            System.out.println("    Post-norm (BERT):  LayerNorm applied AFTER attention/FFN");
        }

        // ============================================================
        // 5. SWIGLU (SiLU gate + linear up-projection)
        // ============================================================
        System.out.println("\n=== 5. SwiGLU FFN ===");
        {
            // SwiGLU (Shazeer, 2020) is the gated linear unit activation used in
            // modern LLM feed-forward networks:
            //   gate  = W_gate * x
            //   up    = W_up   * x
            //   y     = W_down * (silu(gate) * up)
            //
            // Compared to the original FFN (W2 * relu(W1 * x)):
            //   - SwiGLU has 3 projection matrices instead of 2
            //   - The hidden dimension is typically reduced (e.g., 8/3 * d_model)
            //     so total parameters stay the same
            //   - Significantly better performance on language modeling tasks
            //
            // Used in: LLaMA, LLaMA2, Mistral, PaLM, Gemma, Phi-2

            SameDiff sd = SameDiff.create();

            int batch = 2, seqLen = 32, dModel = 256;
            // Note: ffnDim = int(8/3 * dModel) rounded to multiple of 64
            int ffnDim = 672; // ~8/3 * 256

            SDVariable x = sd.placeHolder("ffn_in", DataType.FLOAT, batch * seqLen, dModel);

            // Learnable projection matrices
            SDVariable wGate = sd.var("W_gate", Nd4j.randn(DataType.FLOAT, dModel, ffnDim).mul(0.02));
            SDVariable wUp   = sd.var("W_up",   Nd4j.randn(DataType.FLOAT, dModel, ffnDim).mul(0.02));
            SDVariable wDown = sd.var("W_down",  Nd4j.randn(DataType.FLOAT, ffnDim, dModel).mul(0.02));

            // SwiGLU: gate * silu(up), then project down
            //   silu(x) = x * sigmoid(x)  (also called Swish)
            //   The gate branch controls which features are passed through.
            SDVariable gate = sd.linalg().mmul("gate_proj", x, wGate);
            SDVariable up   = sd.linalg().mmul("up_proj",   x, wUp);
            SDVariable activated = sd.nn().silu("silu_gate", gate);   // silu on gate
            SDVariable gated = activated.mul("swiglu_hidden", up);    // element-wise gate
            SDVariable output = sd.linalg().mmul("ffn_out", gated, wDown);

            INDArray inputData = Nd4j.randn(DataType.FLOAT, batch * seqLen, dModel);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("ffn_in", inputData), "ffn_out");

            System.out.println("  Input shape:     " + Arrays.toString(inputData.shape()));
            System.out.println("  FFN hidden dim:  " + ffnDim + "  (~8/3 * dModel)");
            System.out.println("  Output shape:    " + result.get("ffn_out").shapeInfoToString());
            System.out.println("  SwiGLU formula:  W_down * (silu(W_gate * x) * (W_up * x))");
            System.out.println("  Parameters:      3 matrices (gate + up + down) vs 2 in ReLU FFN");
        }

        // ============================================================
        // 6. KV CACHE UPDATE
        // ============================================================
        System.out.println("\n=== 6. KV Cache Update ===");
        {
            // KV caching enables efficient autoregressive generation.
            // During prefill: compute K and V for the entire input sequence.
            // During decode:  compute K and V for the single new token,
            //                 then append to the cache and attend over all cached K/V.
            //
            // Without KV cache: every decode step re-computes K/V for ALL prior tokens.
            //   Cost per step: O(seqLen) attention computations
            //
            // With KV cache: only compute K/V for the new token, read cache for rest.
            //   Cost per step: O(1) new K/V, O(cachedLen) attention
            //
            // The kvCacheUpdate op updates the cache tensor at the current position.
            //   cache:     [batch, maxSeqLen, numKvHeads, headDim]
            //   newKeys:   [batch, 1,         numKvHeads, headDim]  (one new token)
            //   newValues: [batch, 1,         numKvHeads, headDim]
            //   position:  [batch]  (current decode position)

            SameDiff sd = SameDiff.create();

            int batch = 2, maxSeqLen = 512, numKvHeads = 4, headDim = 64;

            // Pre-allocated cache tensors (full sequence length capacity)
            SDVariable kCache = sd.var("kCache", Nd4j.zeros(DataType.FLOAT, batch, maxSeqLen, numKvHeads, headDim));
            SDVariable vCache = sd.var("vCache", Nd4j.zeros(DataType.FLOAT, batch, maxSeqLen, numKvHeads, headDim));

            // New K/V for a single decode step (one new token)
            SDVariable newKey = sd.placeHolder("newKey", DataType.FLOAT, batch, 1, numKvHeads, headDim);
            SDVariable newVal = sd.placeHolder("newVal", DataType.FLOAT, batch, 1, numKvHeads, headDim);

            // Current decode position (e.g., token 42 in the sequence)
            SDVariable position = sd.placeHolder("position", DataType.INT, batch);

            // kvCacheUpdate writes newKey/newValue into cache at `position`
            // and returns the updated cache (or the input unchanged, depending on impl).
            SDVariable[] updatedKv = sd.nn().kvCacheUpdate(
                    new String[]{"kCacheUpdated", "vCacheUpdated"},
                    kCache,
                    vCache,
                    newKey,
                    newVal,
                    position);

            System.out.println("  Cache shape:     " + Arrays.toString(new long[]{batch, maxSeqLen, numKvHeads, headDim}));
            System.out.println("  New key shape:   " + Arrays.toString(new long[]{batch, 1, numKvHeads, headDim}));
            System.out.println("  KV cache workflow:");
            System.out.println("    Prefill: compute K/V for all input tokens, fill cache");
            System.out.println("    Decode:  compute K/V for 1 token, update cache at position");
            System.out.println("             attend over cache[0:position+1] (causal)");
            System.out.println("  Memory:  maxSeqLen x numKvHeads x headDim x 2 (K+V) x precision");
            System.out.println("  Example: 512 tokens, 4 KV heads, 64 head_dim, FP16:");
            long kvBytes = 512L * 4 * 64 * 2 * 2;
            System.out.println("    KV cache = " + kvBytes / 1024 + " KB per batch element");
        }

        // ============================================================
        // 7. COMPLETE MODERN TRANSFORMER BLOCK
        // ============================================================
        System.out.println("\n=== 7. Complete Modern Transformer Block (LLaMA-style) ===");
        {
            // Assembles a complete decoder block using the ops from above:
            //   Pre-Attention RMSNorm
            //   -> Q, K, V projections
            //   -> RoPE on Q and K
            //   -> Flash Attention (GQA, causal)
            //   -> Output projection + residual
            //   Pre-FFN RMSNorm
            //   -> SwiGLU FFN
            //   -> Residual

            SameDiff sd = SameDiff.create();

            int batch = 1, seqLen = 32, dModel = 128, numHeads = 4, numKvHeads = 2, headDim = 32;
            int ffnDim = 336; // ~8/3 * dModel

            // Hidden state input
            SDVariable x = sd.placeHolder("hidden", DataType.FLOAT, batch, seqLen, dModel);

            // Learnable norms
            SDVariable attnGamma = sd.var("attnGamma", Nd4j.ones(DataType.FLOAT, dModel));
            SDVariable ffnGamma  = sd.var("ffnGamma",  Nd4j.ones(DataType.FLOAT, dModel));

            // Learnable weight matrices
            SDVariable wQ    = sd.var("Wq", Nd4j.randn(DataType.FLOAT, dModel, numHeads * headDim).mul(0.02));
            SDVariable wK    = sd.var("Wk", Nd4j.randn(DataType.FLOAT, dModel, numKvHeads * headDim).mul(0.02));
            SDVariable wV    = sd.var("Wv", Nd4j.randn(DataType.FLOAT, dModel, numKvHeads * headDim).mul(0.02));
            SDVariable wO    = sd.var("Wo", Nd4j.randn(DataType.FLOAT, numHeads * headDim, dModel).mul(0.02));
            SDVariable wGate = sd.var("Wgate", Nd4j.randn(DataType.FLOAT, dModel, ffnDim).mul(0.02));
            SDVariable wUp   = sd.var("Wup",   Nd4j.randn(DataType.FLOAT, dModel, ffnDim).mul(0.02));
            SDVariable wDown = sd.var("Wdown",  Nd4j.randn(DataType.FLOAT, ffnDim, dModel).mul(0.02));

            // RoPE cache
            SDVariable ropeCache = sd.placeHolder("ropeCache", DataType.FLOAT, seqLen, headDim / 2, 2);

            // ---- Attention sub-block ----
            SDVariable xNorm = sd.nn().rmsNorm("attnNorm", x, attnGamma, 1e-5);
            SDVariable flat  = xNorm.reshape(batch * seqLen, dModel);

            SDVariable q = sd.linalg().mmul("Qproj", flat, wQ).reshape(batch, seqLen, numHeads, headDim);
            SDVariable k = sd.linalg().mmul("Kproj", flat, wK).reshape(batch, seqLen, numKvHeads, headDim);
            SDVariable v = sd.linalg().mmul("Vproj", flat, wV).reshape(batch, seqLen, numKvHeads, headDim);

            SDVariable qR = sd.nn().rope("qRoPE", q, ropeCache, 0);
            SDVariable kR = sd.nn().rope("kRoPE", k, ropeCache, 0);

            double scale = 1.0 / Math.sqrt(headDim);
            SDVariable attn = sd.nn().flashAttention("attn", qR, kR, v, scale, true, numHeads, numKvHeads);

            SDVariable attnOut = sd.linalg().mmul("attnOut",
                    attn.reshape(batch * seqLen, numHeads * headDim), wO)
                    .reshape(batch, seqLen, dModel);
            SDVariable res1 = x.add("res1", attnOut);

            // ---- FFN sub-block ----
            SDVariable ffnNorm = sd.nn().rmsNorm("ffnNorm", res1, ffnGamma, 1e-5);
            SDVariable flatFFN = ffnNorm.reshape(batch * seqLen, dModel);

            SDVariable gate = sd.nn().silu("gateAct", sd.linalg().mmul("gateProj", flatFFN, wGate));
            SDVariable up   = sd.linalg().mmul("upProj",   flatFFN, wUp);
            SDVariable ffn  = sd.linalg().mmul("ffnDown", gate.mul(up), wDown)
                    .reshape(batch, seqLen, dModel);
            SDVariable output = res1.add("blockOut", ffn);

            // Execute
            INDArray hiddenData = Nd4j.randn(DataType.FLOAT, batch, seqLen, dModel);
            INDArray ropeCacheData = Nd4j.randn(DataType.FLOAT, seqLen, headDim / 2, 2);
            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put("hidden", hiddenData);
            inputs.put("ropeCache", ropeCacheData);

            Map<String, INDArray> result = sd.output(inputs, "blockOut");
            System.out.println("  Block output shape: " + result.get("blockOut").shapeInfoToString());
            System.out.println("  Architecture:");
            System.out.println("    x -> RMSNorm -> Q,K,V proj -> RoPE(Q,K) -> FlashAttn(GQA) -> O proj -> +x");
            System.out.println("    -> RMSNorm -> SwiGLU FFN -> +residual");
            System.out.println("  This is the exact block used in LLaMA2, Mistral, Gemma, Phi-2");
        }

        System.out.println("\nAll advanced transformer operations demonstrated successfully.");
    }
}
