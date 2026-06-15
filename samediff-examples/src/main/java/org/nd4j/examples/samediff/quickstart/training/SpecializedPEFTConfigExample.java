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

package org.nd4j.examples.samediff.quickstart.training;

import org.nd4j.autodiff.samediff.config.*;
import org.nd4j.autodiff.samediff.peft.LoraAdapterCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * Specialized PEFT Method Configuration Examples.
 *
 * This example covers PEFT methods beyond basic LoRA, including:
 *
 * 1. LoftQ    - LoRA-Fine-Tuning-aware Quantization initialization
 * 2. LoHa     - Low-rank Hadamard product adaptation
 * 3. LoKr     - Low-rank Kronecker product adaptation
 * 4. VeRA     - Vector-based Random Matrix Adaptation (shared random matrices)
 * 5. DyLoRA   - Dynamic LoRA with variable rank during training
 * 6. LoraAdapterCache - Two-tier (GPU/host) adapter hot-swap cache
 *
 * These methods are useful in specialized scenarios:
 *   - LoftQ: best initialization for quantized models (reduces quantization error)
 *   - LoHa/LoKr: structured alternatives to LoRA (used in image generation models)
 *   - VeRA: minimal parameters (only scaling vectors; random matrices are shared/frozen)
 *   - DyLoRA: rank-adaptive training without manual rank search
 *   - LoraAdapterCache: production serving with sub-millisecond adapter hot-swap
 */
public class SpecializedPEFTConfigExample {
    private static final Logger log = LoggerFactory.getLogger(SpecializedPEFTConfigExample.class);

    public static void main(String[] args) {

        // =====================================================================
        // 1. LoftQ — LoRA-Fine-Tuning-aware Quantization
        // =====================================================================
        log.info("=== 1. LoftQ Config ===");

        // LoftQ initializes LoRA matrices A and B using SVD of the quantization residual:
        //   Step 1: Quantize W → Q (4-bit NF4)
        //   Step 2: Compute residual R = W - Q
        //   Step 3: SVD of R: R ≈ U @ diag(S) @ Vt  (top-r components)
        //   Step 4: B = U[:, :r] @ diag(sqrt(S[:r])),  A = diag(sqrt(S[:r])) @ Vt[:r, :]
        // At runtime, LoftQ behaves identically to standard LoRA (getPeftType() returns LORA).
        LoftQConfig loftqConfig = LoftQConfig.builder()
                .r(16)                                              // LoRA rank
                .loraAlpha(32)                                      // Scaling factor
                .loraDropout(0.05)
                .numIterations(1)                                   // Alternating quantization+SVD iterations
                .quantType("nf4")                                   // "nf4" (recommended) or "fp4"
                .quantBits(4)                                       // 4 or 8 bits
                .blockSize(64)                                      // Quantization block size (elements per scale)
                .targetModules(Arrays.asList("q_proj", "v_proj"))
                .taskType(TaskType.CAUSAL_LM)
                .build();

        loftqConfig.validate();
        log.info("  LoftQ:");
        log.info("    Summary: {}", loftqConfig.getSummary());
        log.info("    PEFT type at runtime: {} (same as LoRA)", loftqConfig.getPeftType());
        log.info("    Init method: {} (triggers LoftQ initialization)", loftqConfig.getInitLoraWeights());
        log.info("    Quant type: {}-bit {}", loftqConfig.getQuantBits(), loftqConfig.getQuantType());
        log.info("    Block size: {} (elements per quantization scale)", loftqConfig.getBlockSize());
        log.info("    Iterations: {} (more = lower approximation error)", loftqConfig.getNumIterations());

        // Factory shortcuts
        LoftQConfig loftq4bit = LoftQConfig.default4Bit(16);    // NF4, rank=16
        LoftQConfig loftq8bit = LoftQConfig.default8Bit(16);    // FP4, 8-bit, rank=16
        LoftQConfig loftqMultiIter = LoftQConfig.withIterations(
                16, 5,                                            // 5 alternating iterations
                Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj")
        );
        log.info("  4-bit defaults: {}", loftq4bit.getSummary());
        log.info("  Multi-iteration (5 iters): quantBits={}, iters={}",
                loftqMultiIter.getQuantBits(), loftqMultiIter.getNumIterations());

        // =====================================================================
        // 2. LoHa — Low-rank Hadamard Product Adaptation
        // =====================================================================
        log.info("=== 2. LoHa Config ===");

        // LoHa uses Hadamard products of low-rank matrices instead of matrix multiplication.
        // W_delta = (A1 * A2) x (B1 * B2)  where x is Hadamard product
        // Common in image generation models (e.g., Stable Diffusion LoHa).
        // More expressive than standard LoRA at the same rank.
        LohaConfig lohaConfig = LohaConfig.builder()
                .dim(8)                         // Rank / dimension (equivalent to r in LoRA)
                .alpha(1.0)                     // Scaling factor
                .dropout(0.0)
                .useTucker(false)               // Use Tucker decomposition variant
                .initMethod("kaiming")          // "kaiming" or "zeros"
                .targetModules(Arrays.asList("attn.to_q", "attn.to_k", "attn.to_v"))
                .taskType(TaskType.IMAGE_CLS)
                .build();

        lohaConfig.validate();
        log.info("  LoHa:");
        log.info("    PEFT type: {}", lohaConfig.getPeftType());
        log.info("    Dim: {}", lohaConfig.getDim());
        log.info("    Effective max rank: {}", lohaConfig.getEffectiveMaxRank());
        log.info("    Trainable params (est.): {}",
                lohaConfig.calculateTrainableParameters(1_000_000_000L));

        // Factory shortcut
        LohaConfig lohaDefault = LohaConfig.defaultConfig(
                Arrays.asList("attn.to_q", "attn.to_k", "attn.to_v")
        );
        log.info("  LoHa default: dim={}", lohaDefault.getDim());

        // =====================================================================
        // 3. LoKr — Low-rank Kronecker Product Adaptation
        // =====================================================================
        log.info("=== 3. LoKr Config ===");

        // LoKr decomposes the weight update using Kronecker products.
        // W_delta = A kron B  where A and B are low-rank factors.
        // Also popular for image generation model fine-tuning.
        // More parameter-efficient than LoHa for large weight matrices.
        LokrConfig lokrConfig = LokrConfig.builder()
                .dim(8)                         // Rank
                .factor(-1)                     // Kronecker factor (-1 = auto-infer from weight shape)
                .alpha(1.0)
                .dropout(0.0)
                .decomposeKronecker(false)       // Further decompose Kronecker factors
                .fullMatrix(false)               // Use full matrix for small-dim layers
                .targetModules(Arrays.asList("attn.to_q", "attn.to_k", "attn.to_v"))
                .taskType(TaskType.IMAGE_CLS)
                .build();

        lokrConfig.validate();
        log.info("  LoKr:");
        log.info("    PEFT type: {}", lokrConfig.getPeftType());
        log.info("    Dim: {}", lokrConfig.getDim());
        log.info("    Factor: {} (auto)", lokrConfig.getFactor());
        log.info("    Trainable params (est.): {}",
                lokrConfig.calculateTrainableParameters(1_000_000_000L));

        // Factory shortcut
        LokrConfig lokrDefault = LokrConfig.defaultConfig(
                Arrays.asList("attn.to_q", "attn.to_k", "attn.to_v")
        );
        log.info("  LoKr default: dim={}", lokrDefault.getDim());

        // =====================================================================
        // 4. VeRA — Vector-based Random Matrix Adaptation
        // =====================================================================
        log.info("=== 4. VeRA Config ===");

        // VeRA uses a single pair of shared random matrices (frozen, no parameters)
        // plus per-layer learned scaling vectors b and d.
        // W_delta = diag(b) @ A @ diag(d) @ B
        // where A and B are shared random matrices (not trained), b and d are learned.
        // Extreme parameter efficiency: only 2 vectors per layer vs 2 matrices in LoRA.
        // Requires r >> typical LoRA rank to be effective (r=256 common).
        VeraConfig veraConfig = VeraConfig.builder()
                .r(256)                         // Shared random matrix rank (much larger than LoRA rank)
                .sharedSeed(42L)                // Seed for random A and B matrices (same across layers)
                .lambdaScaling(1.0)             // Global scaling factor for the update
                .targetModules(Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj"))
                .taskType(TaskType.CAUSAL_LM)
                .build();

        veraConfig.validate();
        log.info("  VeRA:");
        log.info("    PEFT type: {}", veraConfig.getPeftType());
        log.info("    Shared rank: {}", veraConfig.getR());
        log.info("    Shared seed: {} (same matrices across all layers)", veraConfig.getSharedSeed());
        log.info("    Trainable params (est.): {}",
                veraConfig.calculateTrainableParameters(1_000_000_000L));
        log.info("    Note: VeRA trains only scaling vectors b,d — matrices A,B are frozen random");

        // =====================================================================
        // 5. DyLoRA — Dynamic LoRA with variable rank
        // =====================================================================
        log.info("=== 5. DyLoRA Config ===");

        // DyLoRA trains multiple ranks simultaneously by randomly sampling a rank
        // from [minRank, r] each forward pass. This trains a nested set of adapters
        // that can be pruned to any rank between minRank and r at inference time
        // without re-training.
        DyLoraConfig dyLoraConfig = DyLoraConfig.builder()
                .r(32)                          // Maximum rank
                .loraAlpha(64)
                .loraDropout(0.05)
                .minRank(1)                     // Minimum rank to train (DyLoRA trains all ranks in [1, r])
                .targetModules(Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj"))
                .taskType(TaskType.CAUSAL_LM)
                .build();

        dyLoraConfig.validate();
        log.info("  DyLoRA:");
        log.info("    PEFT type: {}", dyLoraConfig.getPeftType());
        log.info("    Max rank: {}", dyLoraConfig.getR());
        log.info("    Min rank: {}", dyLoraConfig.getMinRank());
        log.info("    Training: simultaneously trains all ranks in [{}, {}]",
                dyLoraConfig.getMinRank(), dyLoraConfig.getR());
        log.info("    Benefit: choose any rank at inference without retraining");

        // =====================================================================
        // 6. AdapterConfig (Bottleneck Adapters)
        // =====================================================================
        log.info("=== 6. Bottleneck Adapter Config ===");

        // Classic adapter architecture: small bottleneck MLP inserted after each
        // transformer sub-layer (attention and/or FFN).
        // Structure: residual + LayerNorm + Linear(hidden→adapter) + act + Linear(adapter→hidden)
        AdapterConfig adapterConfig = AdapterConfig.builder()
                .adapterSize(64)                    // Bottleneck dimension
                .adapterActivation("relu")          // Activation: "relu", "gelu", "silu"
                .adapterDropout(0.1)                // Dropout in adapter
                .adapterScaling(1.0)                // Output scaling factor
                .adapterAfterAttention(true)        // Insert adapter after attention
                .adapterAfterFeedforward(true)      // Insert adapter after FFN
                .hiddenSize(4096)                   // Transformer hidden size (for param count)
                .numLayers(32)                      // Number of transformer layers
                .taskType(TaskType.CAUSAL_LM)
                .build();

        adapterConfig.validate();
        log.info("  Adapter:");
        log.info("    PEFT type: {}", adapterConfig.getPeftType());
        log.info("    Bottleneck size: {}", adapterConfig.getAdapterSize());
        log.info("    After attention: {}, after FFN: {}",
                adapterConfig.isAdapterAfterAttention(), adapterConfig.isAdapterAfterFeedforward());
        log.info("    Trainable params (est.): {}",
                adapterConfig.calculateTrainableParameters(1_000_000_000L));

        // Factory shortcut
        AdapterConfig adapterDefault = AdapterConfig.defaultConfig(4096, 32);  // hiddenSize=4096, 32 layers
        log.info("  Adapter default (4096 hidden, 32 layers): size={}", adapterDefault.getAdapterSize());

        // =====================================================================
        // 7. LoraAdapterCache — Production Serving Cache
        // =====================================================================
        log.info("=== 7. LoRA Adapter Cache (Production Serving) ===");

        // Two-tier cache for sub-millisecond adapter hot-swap during inference.
        //   GPU tier  (hot):  D2D memcpy, typically <1ms
        //   Host tier (warm): H2D transfer, typically 2-5ms
        //   Disk (cold):      load + H2D, typically 50-200ms
        //
        // Memory budget example (7B model, rank-16, 4 targets, 32 layers):
        //   Per adapter: ~32MB in FP16
        //   10 hot adapters: ~320MB GPU VRAM
        //   50 warm adapters: ~1.6GB pinned host RAM

        // Default settings: 10 GPU adapters, 50 host adapters
        try (LoraAdapterCache defaultCache = new LoraAdapterCache()) {
            log.info("  Default cache: maxGPU={}, maxHost={}",
                    defaultCache.getMaxGpuAdapters(), defaultCache.getMaxHostAdapters());
        }

        // Custom capacity for memory-constrained environments
        try (LoraAdapterCache customCache = new LoraAdapterCache(
                3,    // maxGpuAdapters: keep 3 hot (smallest common GPU budget)
                20    // maxHostAdapters: keep 20 warm (pinned host RAM)
        )) {
            log.info("  Custom cache: maxGPU={}, maxHost={}",
                    customCache.getMaxGpuAdapters(), customCache.getMaxHostAdapters());
            log.info("  GPU adapters: {}/{}", customCache.getGpuAdapterCount(), customCache.getMaxGpuAdapters());
            log.info("  Host adapters: {}/{}", customCache.getHostAdapterCount(), customCache.getMaxHostAdapters());
            log.info("  Active adapter: {}", customCache.getActiveAdapterName());

            log.info("  Cache API reference:");
            log.info("    loadAdapter(name, dir, variableNames) - load from numpy files");
            log.info("    registerAdapter(name, weights, onGpu)  - register pre-loaded weights");
            log.info("    applyAdapter(name, model)              - hot-swap into SameDiff model");
            log.info("    removeAdapter(model, loraVarNames)     - restore base model (zero B matrices)");
            log.info("    isHot(name) / isWarm(name)             - check cache tier");
            log.info("    isCached(name)                         - check if present at any tier");
            log.info("    getCachedAdapterNames()                - list all cached adapters");
            log.info("    getStats()                             - performance statistics");
            log.info("    close()                                - release all cached arrays (AutoCloseable)");

            log.info("  Stats: {}", customCache.getStats());
        }

        // =====================================================================
        // PEFT Method Comparison
        // =====================================================================
        log.info("=== PEFT Method Comparison ===");
        log.info("  +----------+------------+----------+----------------------------------+");
        log.info("  | Method   | Params     | Approx.  | Best Use Case                    |");
        log.info("  +----------+------------+----------+----------------------------------+");
        log.info("  | LoRA     | ~0.1-1%    | Med      | General NLP fine-tuning          |");
        log.info("  | QLoRA    | ~0.1-1%    | Low      | Memory-constrained fine-tuning   |");
        log.info("  | LoftQ    | ~0.1-1%    | Low      | Quantized models (best init)     |");
        log.info("  | DoRA     | ~0.1-1%    | Med      | When LoRA underfits              |");
        log.info("  | AdaLoRA  | ~0.1-1%    | Med      | Adaptive rank allocation         |");
        log.info("  | DyLoRA   | ~0.1-1%    | Med      | Rank search without retraining   |");
        log.info("  | LoHa     | ~0.1-0.5%  | High     | Image generation (SD/SDXL)       |");
        log.info("  | LoKr     | ~0.05-0.5% | High     | Image generation, large weights  |");
        log.info("  | VeRA     | ~0.001%    | Very Low | Extreme param efficiency          |");
        log.info("  | IA3      | ~0.01%     | Very Low | Few-shot minimal adaptation      |");
        log.info("  | Prefix   | ~0.1%      | Low      | Multi-task (one prefix per task) |");
        log.info("  | Prompt   | ~0.01%     | Very Low | Simple task conditioning         |");
        log.info("  | Adapters | ~0.5-2%    | Med      | Classic PEFT (NLP tasks)         |");
        log.info("  +----------+------------+----------+----------------------------------+");

        log.info("**************** Specialized PEFT Config Example finished ********************");
    }
}
