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

package org.nd4j.examples.samediff.gnn;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

/**
 * Graph Convolutional Network (GCN) node classification with the {@code sd.gnn()} API.
 *
 * <p>We build a small two-community graph and classify each node into its community. The graph is
 * supplied as a symmetric-normalised CSR adjacency (see {@link GnnExampleGraphs}); a two-layer GCN
 * ({@code sd.gnn().gcnConv}) smooths node features over the graph so that, even with noisy
 * per-node features, the community structure is recovered.
 *
 * <p>Because the graph topology is fixed, the CSR arrays and node features are {@code sd.constant}s
 * and only the layer weights are trained — here with a plain manual SGD loop so the mechanics are
 * fully visible.
 */
public class Ex1_GcnNodeClassification {

    public static void main(String[] args) {
        Nd4j.getRandom().setSeed(12345);

        // ---- Graph: two communities of 5 nodes (0-4 and 5-9), each densely linked + one bridge ----
        int n = 10;
        int[][] edges = {
                {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}, {0, 2},   // community 0
                {5, 6}, {6, 7}, {7, 8}, {8, 9}, {9, 5}, {5, 7},   // community 1
                {4, 5}                                            // bridge
        };
        GnnExampleGraphs.Csr g = GnnExampleGraphs.normalizedCsr(n, edges);

        int[] labels = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
        int numClasses = 2;
        int fIn = 2;

        // Noisy per-node features: community-correlated signal + noise (features alone are only
        // weakly separable; the graph convolution does the rest).
        INDArray features = Nd4j.randn(DataType.DOUBLE, n, fIn).muli(0.6);
        for (int i = 0; i < n; i++) {
            features.putScalar(i, labels[i], features.getDouble(i, labels[i]) + 1.0);
        }

        // ---- Build the GCN graph ----
        SameDiff sd = SameDiff.create();
        SDVariable colIdx = sd.constant("colIdx", g.colIdxArr());
        SDVariable rowPtr = sd.constant("rowPtr", g.rowPtrArr());
        SDVariable aNorm  = sd.constant("aNorm",  g.valuesArr());
        SDVariable X      = sd.constant("X",      features);
        SDVariable yOneHot = sd.constant("y",     GnnExampleGraphs.oneHot(labels, numClasses));

        int hidden = 8;
        SDVariable w0 = sd.var("w0", Nd4j.randn(DataType.DOUBLE, fIn, hidden).muli(0.5));
        SDVariable b0 = sd.var("b0", Nd4j.zeros(DataType.DOUBLE, hidden));
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.DOUBLE, hidden, numClasses).muli(0.5));
        SDVariable b1 = sd.var("b1", Nd4j.zeros(DataType.DOUBLE, numClasses));

        // Two GCN layers: ReLU on the hidden layer, raw logits out.
        SDVariable h   = sd.gnn().gcnConv(X, w0, b0, aNorm, colIdx, rowPtr, n, n, true);   // [n, hidden]
        SDVariable logits = sd.gnn().gcnConv(h, w1, b1, aNorm, colIdx, rowPtr, n, n, false); // [n, 2]
        SDVariable softmax = sd.nn().softmax("softmax", logits);

        // Cross-entropy loss (manual, so the example has no hidden machinery).
        SDVariable ce = sd.math().neg(yOneHot.mul(sd.math().log(softmax.add(1e-7))));
        sd.mean("loss", ce);
        sd.setLossVariables("loss");

        // ---- Train: manual SGD on the layer weights only ----
        String[] trainable = {"w0", "b0", "w1", "b1"};
        double lr = 0.1;
        Map<String, INDArray> noPlaceholders = new HashMap<>();
        System.out.println("Training a 2-layer GCN on a 10-node, 2-community graph...");
        for (int epoch = 0; epoch <= 200; epoch++) {
            Map<String, INDArray> grads = sd.calculateGradients(noPlaceholders, trainable);
            for (String name : trainable) {
                sd.getVariable(name).getArr().subi(grads.get(name).mul(lr));
            }
            if (epoch % 40 == 0) {
                double loss = sd.output(noPlaceholders, "loss").get("loss").getDouble(0);
                System.out.printf("  epoch %3d   loss = %.4f%n", epoch, loss);
            }
        }

        // ---- Evaluate ----
        INDArray pred = sd.output(noPlaceholders, "softmax").get("softmax").argMax(1);
        int correct = 0;
        for (int i = 0; i < n; i++) {
            if (pred.getInt(i) == labels[i]) correct++;
        }
        System.out.printf("%nNode-classification accuracy: %d/%d%n", correct, n);
        System.out.println("Predicted communities: " + pred);
    }
}
