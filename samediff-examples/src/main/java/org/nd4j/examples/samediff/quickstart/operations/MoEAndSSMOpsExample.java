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
import java.util.HashMap;
import java.util.Map;

/**
 * Mixture of Experts (MoE) and State Space Model (SSM) Operations in SameDiff
 *
 * This example covers two architectural families used by the latest generation
 * of large language models:
 *
 * Mixture of Experts (MoE):
 *   Instead of passing every token through a single dense FFN, MoE routes each
 *   token to a small subset (top-k) of N specialized "expert" FFNs.
 *   Benefits: parameter count scales without proportional compute increase.
 *   Models: Mixtral 8x7B, DeepSeek MoE, GPT-4 (rumored), Grok-1, Qwen MoE.
 *
 *   Key concepts:
 *     - Gate (router): a small network that assigns each token a probability
 *       distribution over experts
 *     - Top-K selection: only the k highest-probability experts process each token
 *     - Load balancing: auxiliary loss encourages even expert utilization
 *     - Expert capacity: limits tokens per expert to prevent overflow
 *
 * State Space Models (SSM):
 *   SSMs represent sequences using a latent state that is updated recurrently.
 *   Unlike attention (O(n^2) compute, O(n) memory), SSMs are:
 *     - O(n * d) during training (parallel scan)
 *     - O(1) per token during inference (constant-size state)
 *   Models: Mamba, Mamba2, RWKV, Hawk, Griffin.
 *
 *   Selective SSM (Mamba): The key innovation is input-dependent state transitions:
 *     - B, C, delta are functions of the input (not fixed like classical SSMs)
 *     - This allows the model to selectively retain or forget information per step
 *
 * Operations covered:
 *   MoE:
 *     - mixtureOfExperts: full sparse MoE FFN layer
 *     - moeGate:          gating / routing network only
 *   SSM:
 *     - mamba2SSM:        Mamba2 structured state-space layer
 *     - selectiveScan:    core selective scan recurrence
 */
public class MoEAndSSMOpsExample {

    public static void main(String[] args) {

        // ============================================================
        // 1. MOE GATE (Router)
        // ============================================================
        System.out.println("=== 1. MoE Gate (Router) ===");
        {
            // The MoE gate is a simple linear layer that maps each token's hidden state
            // to a probability (or logit) distribution over N experts.
            //
            // Top-K selection:
            //   For each token, only the top-k experts with highest gate scores are used.
            //   Typical: top-2 (Mixtral), top-1 (Switch Transformer), top-8 (DeepSeek).
            //
            // Gate outputs:
            //   [0]: routerLogits   - raw scores [batch*seqLen, numExperts]
            //   [1]: expertIndices  - top-k expert indices [batch*seqLen, topK]
            //   [2]: expertWeights  - softmax-normalized weights [batch*seqLen, topK]

            SameDiff sd = SameDiff.create();

            int batch = 2, seqLen = 16, dModel = 128, numExperts = 8, topK = 2;
            int batchSeq = batch * seqLen;  // tokens are flattened for routing

            SDVariable input      = sd.placeHolder("gate_in",  DataType.FLOAT, batchSeq, dModel);
            SDVariable gateWeight = sd.placeHolder("gate_w",   DataType.FLOAT, dModel, numExperts);

            // moeGate: linear projection -> top-k selection -> softmax weighting
            SDVariable[] gateOut = sd.nn().moeGate(
                    new String[]{"routerLogits", "expertIndices", "expertWeights"},
                    input,
                    gateWeight,
                    numExperts,
                    topK);

            System.out.println("  Input shape:           [" + batchSeq + ", " + dModel + "] (batch*seq tokens)");
            System.out.println("  Gate weight shape:     [" + dModel + ", " + numExperts + "]");
            System.out.println("  numExperts:            " + numExperts);
            System.out.println("  topK:                  " + topK);
            System.out.println("  Output[0] routerLogits:   [" + batchSeq + ", " + numExperts + "] - raw scores");
            System.out.println("  Output[1] expertIndices:  [" + batchSeq + ", " + topK + "] - which experts");
            System.out.println("  Output[2] expertWeights:  [" + batchSeq + ", " + topK + "] - softmax weights");
            System.out.println("  Each token routes to " + topK + " of " + numExperts + " experts.");
            System.out.println("  Active params per token: " + topK + "/" + numExperts + " x expert_size");
        }

        // ============================================================
        // 2. MIXTURE OF EXPERTS (full sparse FFN layer)
        // ============================================================
        System.out.println("\n=== 2. Mixture of Experts (Full Sparse FFN) ===");
        {
            // mixtureOfExperts combines the gate and expert execution into one op:
            //   1. Gate: route each token to top-k experts
            //   2. Execute: run each token through its assigned expert(s)
            //   3. Aggregate: weighted sum of expert outputs using gate scores
            //
            // Each expert is a small FFN (same structure as a dense FFN, but smaller).
            // With N experts and top-k routing, compute cost per token is:
            //   k/N * (cost of one expert)  [vs a single dense expert]
            //
            // Expert weights layout: [numExperts, dModel, expertHiddenDim]
            //   Each expert is a matrix mapping dModel -> expertHiddenDim.
            //   A second set of down-projection weights maps expertHiddenDim -> dModel.

            SameDiff sd = SameDiff.create();

            int batch = 2, seqLen = 16, dModel = 128;
            int numExperts = 8, topK = 2, expertHiddenDim = 256;
            int batchSeq = batch * seqLen;

            SDVariable input         = sd.placeHolder("moe_in",   DataType.FLOAT, batchSeq, dModel);
            SDVariable gateWeights   = sd.placeHolder("moe_gate", DataType.FLOAT, dModel, numExperts);
            // Expert weights: stacked [up-proj, down-proj] for all experts
            SDVariable expertWeights = sd.placeHolder("moe_exp",  DataType.FLOAT,
                    numExperts, dModel, expertHiddenDim);

            SDVariable moeOut = sd.nn().mixtureOfExperts(
                    "moeOut",
                    input,
                    gateWeights,
                    expertWeights,
                    numExperts,
                    topK);

            System.out.println("  Input shape:            [" + batchSeq + ", " + dModel + "]");
            System.out.println("  Gate weights:           [" + dModel + ", " + numExperts + "]");
            System.out.println("  Expert weights:         [" + numExperts + ", " + dModel + ", " + expertHiddenDim + "]");
            System.out.println("  numExperts:             " + numExperts);
            System.out.println("  topK:                   " + topK);
            System.out.println("  Output shape:           [" + batchSeq + ", " + dModel + "]");
            System.out.println();
            System.out.println("  Compute characteristics:");
            System.out.println("    Active experts per token: " + topK + " / " + numExperts);
            System.out.println("    Compute per token:        " + topK + "/" + numExperts
                    + " x dense FFN cost");
            System.out.println("    Parameter efficiency:     " + numExperts + "x more params, ~"
                    + topK + "x compute");
            System.out.println();
            System.out.println("  MoE models:");
            System.out.println("    Mixtral 8x7B:  8 experts, top-2, 46.7B total / 12.9B active");
            System.out.println("    DeepSeek MoE:  fine-grained experts + shared experts");
            System.out.println("    Switch-Base:   N experts, top-1 routing (simplest)");
        }

        // ============================================================
        // 3. SELECTIVE SCAN (Mamba core operation)
        // ============================================================
        System.out.println("\n=== 3. Selective Scan (Mamba SSM Core) ===");
        {
            // The selective scan is the core of the Mamba state-space model.
            //
            // Classical SSM (fixed parameters A, B, C, delta):
            //   h_t = A * h_{t-1} + B * x_t
            //   y_t = C * h_t
            //
            // Selective SSM (Mamba): A, B, C, delta depend on input x_t:
            //   delta_t = softplus(linear(x_t))   [time step, learned per input]
            //   B_t     = linear(x_t)             [input gate, learned per input]
            //   C_t     = linear(x_t)             [output gate, learned per input]
            //   A_bar_t = exp(-delta_t * A)        [discretized A matrix]
            //   B_bar_t = delta_t * B_t            [discretized B matrix]
            //   h_t     = A_bar_t * h_{t-1} + B_bar_t * x_t
            //   y_t     = C_t * h_t
            //
            // During training: parallel scan over sequence dimension (O(n log n))
            // During inference: single recurrence step (O(1) per token)
            //
            // selectiveScan is the low-level operation. mamba2SSM (section 4)
            // is the full Mamba2 block that includes convolution, projection, etc.

            SameDiff sd = SameDiff.create();

            int batch = 2, seqLen = 64, dState = 128;
            // Input: processed hidden states after causal conv1d
            SDVariable input = sd.placeHolder("ssm_in", DataType.FLOAT, batch, seqLen, dState);

            SDVariable ssm = sd.nn().selectiveScan("ssm_out", input);

            INDArray inputData = Nd4j.randn(DataType.FLOAT, batch, seqLen, dState).mul(0.1);
            Map<String, INDArray> result = sd.output(
                    java.util.Collections.singletonMap("ssm_in", inputData), "ssm_out");

            System.out.println("  Input shape:   " + Arrays.toString(inputData.shape()));
            System.out.println("  Output shape:  " + result.get("ssm_out").shapeInfoToString());
            System.out.println("  Complexity:    O(n) compute, O(d_state) recurrent memory");
            System.out.println("  vs Attention:  O(n^2) compute, O(n) KV cache memory");
            System.out.println("  Trade-off:     SSM uses less memory but may miss long-range deps");
            System.out.println("                 at very long context vs attention");
        }

        // ============================================================
        // 4. MAMBA2 SSM BLOCK
        // ============================================================
        System.out.println("\n=== 4. Mamba2 SSM Block ===");
        {
            // Mamba2 (Dao & Gu, 2024) reformulates Mamba as a structured matrix
            // multiplication (SSD: Structured State Space Duality), allowing more
            // efficient hardware utilization (tensor cores) while keeping the
            // selectivity of Mamba1.
            //
            // Mamba2 SSM block inputs:
            //   input:  [batch, seqLen, dModel] - input sequence
            //   A:      [numHeads] or [dState]   - diagonal state matrix (log-scale)
            //   B:      [batch, seqLen, dState]  - input projection (selective)
            //   C:      [batch, seqLen, dState]  - output projection (selective)
            //   D:      [numHeads] or [dModel]   - skip connection scalar
            //   deltaT: [batch, seqLen, numHeads] - time step (selective)
            //
            // These correspond to the parameterization in the Mamba2 paper where:
            //   - A controls how much state to retain (recurrence weight)
            //   - B gates how much new input x enters the state
            //   - C gates how much state to read into the output
            //   - delta discretizes the continuous-time SSM

            SameDiff sd = SameDiff.create();

            int batch = 2, seqLen = 64, dModel = 128, dState = 64, numHeads = 8;

            SDVariable input  = sd.placeHolder("m2_in",    DataType.FLOAT, batch, seqLen, dModel);
            SDVariable A      = sd.placeHolder("m2_A",     DataType.FLOAT, numHeads);        // per-head state decay
            SDVariable B      = sd.placeHolder("m2_B",     DataType.FLOAT, batch, seqLen, dState);  // input gate
            SDVariable C      = sd.placeHolder("m2_C",     DataType.FLOAT, batch, seqLen, dState);  // output gate
            SDVariable D      = sd.placeHolder("m2_D",     DataType.FLOAT, numHeads);        // skip connection
            SDVariable deltaT = sd.placeHolder("m2_delta", DataType.FLOAT, batch, seqLen, numHeads); // time step

            SDVariable mamba2Out = sd.nn().mamba2SSM(
                    "mamba2Out",
                    input,
                    A,
                    B,
                    C,
                    D,
                    deltaT);

            // Execute with random inputs
            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put("m2_in",    Nd4j.randn(DataType.FLOAT, batch, seqLen, dModel).mul(0.1));
            inputs.put("m2_A",     Nd4j.randn(DataType.FLOAT, numHeads).mul(-0.1));  // negative => stable decay
            inputs.put("m2_B",     Nd4j.randn(DataType.FLOAT, batch, seqLen, dState).mul(0.1));
            inputs.put("m2_C",     Nd4j.randn(DataType.FLOAT, batch, seqLen, dState).mul(0.1));
            inputs.put("m2_D",     Nd4j.ones( DataType.FLOAT, numHeads));
            inputs.put("m2_delta", Nd4j.rand( DataType.FLOAT, batch, seqLen, numHeads).add(0.001)); // positive

            Map<String, INDArray> result = sd.output(inputs, "mamba2Out");

            System.out.println("  Input:   " + Arrays.toString(inputs.get("m2_in").shape()));
            System.out.println("  A (decay): " + Arrays.toString(inputs.get("m2_A").shape())
                    + "  per-head state decay (log-domain)");
            System.out.println("  B (in gate): " + Arrays.toString(inputs.get("m2_B").shape())
                    + "  selective input gating");
            System.out.println("  C (out gate): " + Arrays.toString(inputs.get("m2_C").shape())
                    + "  selective output gating");
            System.out.println("  D (skip): " + Arrays.toString(inputs.get("m2_D").shape())
                    + "  direct skip connection weight");
            System.out.println("  deltaT: " + Arrays.toString(inputs.get("m2_delta").shape())
                    + "  per-token time step");
            System.out.println("  Output:  " + result.get("mamba2Out").shapeInfoToString());
            System.out.println();
            System.out.println("  Mamba2 advantages over Mamba1:");
            System.out.println("    - Structured State Space Duality (SSD) formulation");
            System.out.println("    - Better GPU utilization (maps to matrix multiply)");
            System.out.println("    - Multi-head SSM (like multi-head attention)");
            System.out.println("    - Competitive with Transformers on language tasks");
        }

        // ============================================================
        // 5. COMBINED: MoE + SSM (Jamba/Zamba style)
        // ============================================================
        System.out.println("\n=== 5. Combined MoE + SSM Architecture ===");
        {
            // Recent architectures (Jamba, Zamba, MambaMoE) combine SSM layers
            // with MoE layers for long-context efficiency + parameter scaling.
            //
            // Typical pattern:
            //   - Attention/SSM layer every K blocks for sequence mixing
            //   - MoE FFN layers for parameter scaling
            //   - Both operations benefit from sparse computation
            //
            // This section demonstrates the composition pattern.

            System.out.println("  Hybrid SSM + MoE architecture pattern:");
            System.out.println();
            System.out.println("  for each block:");
            System.out.println("    if block_idx % ssm_freq == 0:");
            System.out.println("      x = RMSNorm(x)");
            System.out.println("      x = mamba2SSM(x, A, B, C, D, delta) + x  // SSM layer");
            System.out.println("    else:");
            System.out.println("      x = RMSNorm(x)");
            System.out.println("      x = flashAttention(Q,K,V) + x              // Attention layer");
            System.out.println("    x = RMSNorm(x)");
            System.out.println("    x = mixtureOfExperts(x, gateW, expertW) + x  // MoE FFN layer");
            System.out.println();
            System.out.println("  Models using this pattern:");
            System.out.println("    Jamba (AI21):       SSM + Attention + MoE");
            System.out.println("    Zamba (Zyphra):     SSM layers + shared attention + MoE");
            System.out.println("    Hawk/Griffin (DeepMind): SSM-based with MoE option");

            // Build a simplified single SSM + MoE block in SameDiff
            SameDiff sd = SameDiff.create();

            int batch = 1, seqLen = 32, dModel = 64, dState = 32, numHeads = 4;
            int numExperts = 4, topK = 1, expertHiddenDim = 128;

            SDVariable x = sd.placeHolder("hybrid_in", DataType.FLOAT, batch, seqLen, dModel);
            SDVariable normGamma1 = sd.var("g1", Nd4j.ones(DataType.FLOAT, dModel));
            SDVariable normGamma2 = sd.var("g2", Nd4j.ones(DataType.FLOAT, dModel));

            // SSM sub-block
            SDVariable A      = sd.var("A",     Nd4j.randn(DataType.FLOAT, numHeads).mul(-0.1));
            SDVariable B      = sd.placeHolder("B", DataType.FLOAT, batch, seqLen, dState);
            SDVariable C      = sd.placeHolder("C", DataType.FLOAT, batch, seqLen, dState);
            SDVariable D      = sd.var("D",     Nd4j.ones(DataType.FLOAT, numHeads));
            SDVariable delta  = sd.placeHolder("delta", DataType.FLOAT, batch, seqLen, numHeads);

            SDVariable norm1 = sd.nn().rmsNorm("norm1", x, normGamma1, 1e-5);
            SDVariable ssmOut = sd.nn().mamba2SSM("ssmOut", norm1, A, B, C, D, delta);
            SDVariable res1 = x.add("res1", ssmOut);

            // MoE sub-block
            SDVariable norm2      = sd.nn().rmsNorm("norm2", res1, normGamma2, 1e-5);
            SDVariable flatNorm2  = norm2.reshape(batch * seqLen, dModel);
            SDVariable gateWeight = sd.var("gateW", Nd4j.randn(DataType.FLOAT, dModel, numExperts).mul(0.02));
            SDVariable expertW    = sd.var("expertW", Nd4j.randn(DataType.FLOAT, numExperts, dModel, expertHiddenDim).mul(0.02));
            SDVariable moeOut     = sd.nn().mixtureOfExperts("moeOut", flatNorm2, gateWeight, expertW, numExperts, topK);
            SDVariable output     = res1.add("hybridOut", moeOut.reshape(batch, seqLen, dModel));

            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put("hybrid_in", Nd4j.randn(DataType.FLOAT, batch, seqLen, dModel).mul(0.1));
            inputs.put("B",         Nd4j.randn(DataType.FLOAT, batch, seqLen, dState).mul(0.1));
            inputs.put("C",         Nd4j.randn(DataType.FLOAT, batch, seqLen, dState).mul(0.1));
            inputs.put("delta",     Nd4j.rand( DataType.FLOAT, batch, seqLen, numHeads).add(0.001));

            Map<String, INDArray> result = sd.output(inputs, "hybridOut");
            System.out.println("\n  Hybrid block output: " + result.get("hybridOut").shapeInfoToString());
            System.out.println("  Block: RMSNorm -> Mamba2SSM -> +x -> RMSNorm -> MoE -> +x");
        }

        System.out.println("\nAll MoE and SSM operations demonstrated successfully.");
    }
}
