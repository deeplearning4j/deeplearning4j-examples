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
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMActivations;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDataFormat;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDirectionMode;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMLayerConfig;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.GRUWeights;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMLayerWeights;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.SRUWeights;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

/**
 * SameDiff RNN Operations (sd.rnn namespace) - Complete API Reference
 *
 * This example covers all recurrent neural network operations:
 *
 *   1. lstmLayer - Full LSTM sequence processing (most common)
 *   2. gruCell - GRU single timestep
 *   3. gru - GRU full sequence (simplified API)
 *   4. sru - Simple Recurrent Unit (full sequence)
 *   5. sruCell - SRU single timestep
 *
 * Key config classes:
 *   - LSTMLayerConfig: data format, direction, activations, return options
 *   - LSTMLayerWeights: input weights, recurrent weights, peephole, bias
 *   - GRUWeights: reset/update gate weights, cell weights, biases
 *   - SRUWeights: weights, biases
 *   - LSTMDataFormat: TNS, NTS, NST, T2NS
 *   - LSTMDirectionMode: FWD, BWD, BIDIR_SUM, BIDIR_CONCAT, BIDIR_EXTRA_DIM
 *   - LSTMActivations: TANH, RELU, SIGMOID, AFFINE, LEAKY_RELU, etc.
 */
public class RNNOpsExample {

    public static void main(String[] args) {

        int batch = 2;
        int seqLen = 10;
        int inSize = 8;
        int numUnits = 16;

        // ============================================================
        // 1. LSTM LAYER - Full Sequence Processing
        // ============================================================
        System.out.println("=== LSTM Layer ===");
        {
            SameDiff sd = SameDiff.create();

            // Input: [batch, timesteps, features] (NTS format)
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, inSize);

            // Weights: input-to-hidden [inSize, 4*numUnits]
            SDVariable weights = sd.var("lstm_w",
                    Nd4j.randn(DataType.FLOAT, inSize, 4 * numUnits).muli(0.1));

            // Recurrent weights: hidden-to-hidden [numUnits, 4*numUnits]
            SDVariable rWeights = sd.var("lstm_rw",
                    Nd4j.randn(DataType.FLOAT, numUnits, 4 * numUnits).muli(0.1));

            // Bias: [4*numUnits]
            SDVariable bias = sd.var("lstm_b", Nd4j.zeros(DataType.FLOAT, 4 * numUnits));

            LSTMLayerWeights lstmWeights = LSTMLayerWeights.builder()
                    .weights(weights)
                    .rWeights(rWeights)
                    .bias(bias)
                    .build();

            LSTMLayerConfig config = LSTMLayerConfig.builder()
                    .lstmdataformat(LSTMDataFormat.NTS)     // [batch, timesteps, features]
                    .directionMode(LSTMDirectionMode.FWD)   // forward only
                    .gateAct(LSTMActivations.SIGMOID)       // input/forget/output gates
                    .cellAct(LSTMActivations.TANH)          // cell state activation
                    .outAct(LSTMActivations.TANH)           // output activation
                    .retFullSequence(true)                   // return all timesteps
                    .retLastH(true)                          // return last hidden state
                    .retLastC(true)                          // return last cell state
                    .cellClip(0)                             // no clipping (0 = off)
                    .build();

            // lstmLayer returns SDVariable[] with up to 3 outputs:
            //   [0] = full sequence output (if retFullSequence=true)
            //   [1] = last hidden state (if retLastH=true)
            //   [2] = last cell state (if retLastC=true)
            SDVariable[] lstmOutputs = sd.rnn().lstmLayer(
                    input,      // sequence input
                    null,       // cLast (initial cell state, null = zeros)
                    null,       // yLast (initial hidden state, null = zeros)
                    null,       // maxTSLength (sequence lengths, null = use full)
                    lstmWeights,
                    config);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("input", Nd4j.randn(DataType.FLOAT, batch, seqLen, inSize));

            Map<String, INDArray> results = sd.output(ph, lstmOutputs[0].name(),
                    lstmOutputs[1].name(), lstmOutputs[2].name());

            System.out.println("  Input shape:          [" + batch + ", " + seqLen + ", " + inSize + "] (NTS)");
            System.out.println("  Full sequence output: " + java.util.Arrays.toString(results.get(lstmOutputs[0].name()).shape()));
            System.out.println("  Last hidden state:    " + java.util.Arrays.toString(results.get(lstmOutputs[1].name()).shape()));
            System.out.println("  Last cell state:      " + java.util.Arrays.toString(results.get(lstmOutputs[2].name()).shape()));
        }

        // ============================================================
        // 2. BIDIRECTIONAL LSTM
        // ============================================================
        System.out.println("\n=== Bidirectional LSTM ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, inSize);

            // Bidirectional uses 2x weights (forward + backward)
            SDVariable weights = sd.var("bilstm_w",
                    Nd4j.randn(DataType.FLOAT, 2, inSize, 4 * numUnits).muli(0.1));
            SDVariable rWeights = sd.var("bilstm_rw",
                    Nd4j.randn(DataType.FLOAT, 2, numUnits, 4 * numUnits).muli(0.1));
            SDVariable bias = sd.var("bilstm_b",
                    Nd4j.zeros(DataType.FLOAT, 2, 4 * numUnits));

            LSTMLayerWeights biWeights = LSTMLayerWeights.builder()
                    .weights(weights)
                    .rWeights(rWeights)
                    .bias(bias)
                    .build();

            // BIDIR_CONCAT: concatenates forward and backward outputs
            // Output features = 2 * numUnits
            LSTMLayerConfig biConfig = LSTMLayerConfig.builder()
                    .lstmdataformat(LSTMDataFormat.NTS)
                    .directionMode(LSTMDirectionMode.BIDIR_CONCAT) // or BIDIR_SUM, BIDIR_EXTRA_DIM
                    .gateAct(LSTMActivations.SIGMOID)
                    .cellAct(LSTMActivations.TANH)
                    .outAct(LSTMActivations.TANH)
                    .retFullSequence(true)
                    .retLastH(true)
                    .retLastC(false)
                    .build();

            SDVariable[] biOutputs = sd.rnn().lstmLayer(input, null, null, null, biWeights, biConfig);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("input", Nd4j.randn(DataType.FLOAT, batch, seqLen, inSize));

            Map<String, INDArray> results = sd.output(ph, biOutputs[0].name(), biOutputs[1].name());
            System.out.println("  Bidirectional mode: BIDIR_CONCAT");
            System.out.println("  Full sequence: " + java.util.Arrays.toString(results.get(biOutputs[0].name()).shape()));
            System.out.println("  Last hidden:   " + java.util.Arrays.toString(results.get(biOutputs[1].name()).shape()));
            System.out.println("  (Feature dim doubled: fwd + bwd concatenated)");
        }

        // ============================================================
        // 3. GRU CELL (Single Timestep)
        // ============================================================
        System.out.println("\n=== GRU Cell ===");
        {
            SameDiff sd = SameDiff.create();

            SDVariable x = sd.placeHolder("x", DataType.FLOAT, batch, inSize);
            SDVariable hLast = sd.placeHolder("hLast", DataType.FLOAT, batch, numUnits);

            // GRU weights
            // Reset/Update gate weights: [inSize + numUnits, 2*numUnits]
            SDVariable ruWeight = sd.var("gru_ru_w",
                    Nd4j.randn(DataType.FLOAT, inSize + numUnits, 2 * numUnits).muli(0.1));
            // Cell gate weights: [inSize + numUnits, numUnits]
            SDVariable cWeight = sd.var("gru_c_w",
                    Nd4j.randn(DataType.FLOAT, inSize + numUnits, numUnits).muli(0.1));
            // Biases (optional)
            SDVariable ruBias = sd.var("gru_ru_b", Nd4j.zeros(DataType.FLOAT, 2 * numUnits));
            SDVariable cBias = sd.var("gru_c_b", Nd4j.zeros(DataType.FLOAT, numUnits));

            GRUWeights gruWeights = GRUWeights.builder()
                    .ruWeight(ruWeight)
                    .cWeight(cWeight)
                    .ruBias(ruBias)
                    .cBias(cBias)
                    .build();

            // GRU cell returns: [r, u, c, h]
            //   r = reset gate, u = update gate, c = candidate, h = new hidden state
            SDVariable[] gruOutputs = sd.rnn().gruCell(x, hLast, gruWeights);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("x", Nd4j.randn(DataType.FLOAT, batch, inSize));
            ph.put("hLast", Nd4j.zeros(DataType.FLOAT, batch, numUnits));

            // The 4th output (index 3) is the new hidden state
            Map<String, INDArray> results = sd.output(ph, gruOutputs[3].name());
            System.out.println("  GRU cell output (new h): " +
                    java.util.Arrays.toString(results.get(gruOutputs[3].name()).shape()));
            System.out.println("  Returns: [resetGate, updateGate, candidate, newHidden]");
        }

        // ============================================================
        // 4. GRU (Full Sequence, Simplified API)
        // ============================================================
        System.out.println("\n=== GRU Full Sequence ===");
        {
            SameDiff sd = SameDiff.create();

            // Input: [seqLen, batch, inSize] (time-major)
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, seqLen, batch, inSize);
            SDVariable hLast = sd.placeHolder("hLast", DataType.FLOAT, batch, numUnits);

            // Combined weight matrices
            SDVariable Wx = sd.var("gru_Wx",
                    Nd4j.randn(DataType.FLOAT, inSize, 3 * numUnits).muli(0.1));
            SDVariable Wh = sd.var("gru_Wh",
                    Nd4j.randn(DataType.FLOAT, numUnits, 3 * numUnits).muli(0.1));
            SDVariable biases = sd.var("gru_biases",
                    Nd4j.zeros(DataType.FLOAT, 2, 3 * numUnits));

            // gru processes full sequence, returns last hidden state
            SDVariable gruOut = sd.rnn().gru("gru_out", x, hLast, Wx, Wh, biases);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("x", Nd4j.randn(DataType.FLOAT, seqLen, batch, inSize));
            ph.put("hLast", Nd4j.zeros(DataType.FLOAT, batch, numUnits));

            INDArray result = sd.output(ph, "gru_out").get("gru_out");
            System.out.println("  GRU full sequence output: " + java.util.Arrays.toString(result.shape()));
            System.out.println("  Input is time-major: [seqLen, batch, features]");
        }

        // ============================================================
        // 5. SRU (Simple Recurrent Unit)
        // ============================================================
        System.out.println("\n=== SRU (Simple Recurrent Unit) ===");
        {
            SameDiff sd = SameDiff.create();

            // SRU input: [seqLen, batch, inSize]
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, seqLen, batch, inSize);
            SDVariable initialC = sd.placeHolder("initialC", DataType.FLOAT, batch, inSize);

            // SRU weights: [inSize, 3*inSize]
            SDVariable weights = sd.var("sru_w",
                    Nd4j.randn(DataType.FLOAT, inSize, 3 * inSize).muli(0.1));
            // SRU bias: [2*inSize]
            SDVariable bias = sd.var("sru_b", Nd4j.zeros(DataType.FLOAT, 2 * inSize));

            SRUWeights sruWeights = SRUWeights.builder()
                    .weights(weights)
                    .bias(bias)
                    .build();

            SDVariable sruOut = sd.rnn().sru("sru_out", x, initialC, sruWeights);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("x", Nd4j.randn(DataType.FLOAT, seqLen, batch, inSize));
            ph.put("initialC", Nd4j.zeros(DataType.FLOAT, batch, inSize));

            INDArray result = sd.output(ph, "sru_out").get("sru_out");
            System.out.println("  SRU output: " + java.util.Arrays.toString(result.shape()));
            System.out.println("  SRU is faster than LSTM: no matrix-matrix multiply for hidden state");
            System.out.println("  Uses highway-style gating instead");
        }

        // ============================================================
        // 6. LSTM DATA FORMAT OPTIONS
        // ============================================================
        System.out.println("\n=== LSTM Data Formats ===");
        {
            System.out.println("  Available formats:");
            System.out.println("    TNS = [timeLength, numExamples, inOutSize] (time-major)");
            System.out.println("    NTS = [numExamples, timeLength, inOutSize] (batch-first, TF default)");
            System.out.println("    NST = [numExamples, inOutSize, timeLength]");
            System.out.println("    T2NS = [timeLength, 2, numExamples, inOutSize] (bidirectional/ONNX)");

            System.out.println("\n  Direction modes:");
            System.out.println("    FWD  = forward only");
            System.out.println("    BWD  = backward only");
            System.out.println("    BIDIR_SUM    = bidirectional, sum fwd+bwd");
            System.out.println("    BIDIR_CONCAT = bidirectional, concatenate fwd+bwd (doubles features)");
            System.out.println("    BIDIR_EXTRA_DIM = bidirectional, extra dimension for fwd/bwd");

            System.out.println("\n  Activation options:");
            System.out.println("    TANH, RELU, SIGMOID, AFFINE, LEAKY_RELU, THRESHHOLD_RELU,");
            System.out.println("    SCALED_TANH, HARD_SIGMOID, ELU, SOFTSIGN, SOFTPLUS");
        }

        // ============================================================
        // 7. STACKED LSTM (Multi-layer)
        // ============================================================
        System.out.println("\n=== Stacked LSTM (2 layers) ===");
        {
            SameDiff sd = SameDiff.create();
            int hidden1 = 32, hidden2 = 16;

            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, inSize);

            // Layer 1: input -> hidden1
            SDVariable w1 = sd.var("lstm1_w", Nd4j.randn(DataType.FLOAT, inSize, 4 * hidden1).muli(0.1));
            SDVariable rw1 = sd.var("lstm1_rw", Nd4j.randn(DataType.FLOAT, hidden1, 4 * hidden1).muli(0.1));
            SDVariable b1 = sd.var("lstm1_b", Nd4j.zeros(DataType.FLOAT, 4 * hidden1));

            LSTMLayerConfig seqConfig = LSTMLayerConfig.builder()
                    .lstmdataformat(LSTMDataFormat.NTS)
                    .directionMode(LSTMDirectionMode.FWD)
                    .gateAct(LSTMActivations.SIGMOID)
                    .cellAct(LSTMActivations.TANH)
                    .outAct(LSTMActivations.TANH)
                    .retFullSequence(true)
                    .retLastH(false)
                    .retLastC(false)
                    .build();

            SDVariable[] layer1Out = sd.rnn().lstmLayer(input, null, null, null,
                    LSTMLayerWeights.builder().weights(w1).rWeights(rw1).bias(b1).build(),
                    seqConfig);

            // Layer 2: hidden1 -> hidden2
            SDVariable w2 = sd.var("lstm2_w", Nd4j.randn(DataType.FLOAT, hidden1, 4 * hidden2).muli(0.1));
            SDVariable rw2 = sd.var("lstm2_rw", Nd4j.randn(DataType.FLOAT, hidden2, 4 * hidden2).muli(0.1));
            SDVariable b2 = sd.var("lstm2_b", Nd4j.zeros(DataType.FLOAT, 4 * hidden2));

            LSTMLayerConfig lastConfig = LSTMLayerConfig.builder()
                    .lstmdataformat(LSTMDataFormat.NTS)
                    .directionMode(LSTMDirectionMode.FWD)
                    .gateAct(LSTMActivations.SIGMOID)
                    .cellAct(LSTMActivations.TANH)
                    .outAct(LSTMActivations.TANH)
                    .retFullSequence(false)
                    .retLastH(true) // only need last hidden state for classification
                    .retLastC(false)
                    .build();

            SDVariable[] layer2Out = sd.rnn().lstmLayer(layer1Out[0], null, null, null,
                    LSTMLayerWeights.builder().weights(w2).rWeights(rw2).bias(b2).build(),
                    lastConfig);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("input", Nd4j.randn(DataType.FLOAT, batch, seqLen, inSize));

            INDArray lastHidden = sd.output(ph, layer2Out[0].name()).get(layer2Out[0].name());
            System.out.println("  Layer 1: LSTM(" + inSize + " -> " + hidden1 + ") full sequence");
            System.out.println("  Layer 2: LSTM(" + hidden1 + " -> " + hidden2 + ") last hidden only");
            System.out.println("  Final output: " + java.util.Arrays.toString(lastHidden.shape()));
            System.out.println("  (Ready for classification head)");
        }

        System.out.println("\nAll RNN operations demonstrated successfully.");
    }
}
