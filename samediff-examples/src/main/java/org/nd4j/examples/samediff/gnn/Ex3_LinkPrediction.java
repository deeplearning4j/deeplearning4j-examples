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
 * Link prediction with a Variational Graph Auto-Encoder (VGAE, Kipf & Welling 2016).
 *
 * <p>A GCN encoder produces a mean and log-variance per node; the reparameterisation trick
 * ({@code sd.gnn().vgaeReparam}) samples a latent {@code Z}; the inner-product decoder
 * ({@code sd.gnn().innerProductDecoder}) reconstructs the adjacency as {@code sigmoid(Z·Zᵀ)}. The
 * loss is reconstruction (binary cross-entropy vs. the true adjacency) plus the KL regulariser
 * ({@code sd.gnn().vgaeKlLoss}). A fixed noise sample keeps the run reproducible.
 */
public class Ex3_LinkPrediction {

    public static void main(String[] args) {
        Nd4j.getRandom().setSeed(12345);

        int n = 10;
        int[][] edges = {
                {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}, {0, 2},   // community 0
                {5, 6}, {6, 7}, {7, 8}, {8, 9}, {9, 5}, {5, 7},   // community 1
                {4, 5}
        };
        GnnExampleGraphs.Csr g = GnnExampleGraphs.normalizedCsr(n, edges);

        // True adjacency (with self-loops) — the reconstruction target.
        INDArray adj = Nd4j.zeros(DataType.DOUBLE, n, n);
        for (int i = 0; i < n; i++) adj.putScalar(i, i, 1.0);
        for (int[] e : edges) { adj.putScalar(e[0], e[1], 1.0); adj.putScalar(e[1], e[0], 1.0); }

        int fIn = 8, hidden = 16, latent = 8;
        INDArray features = Nd4j.randn(DataType.DOUBLE, n, fIn);

        SameDiff sd = SameDiff.create();
        SDVariable colIdx = sd.constant("colIdx", g.colIdxArr());
        SDVariable rowPtr = sd.constant("rowPtr", g.rowPtrArr());
        SDVariable aNorm  = sd.constant("aNorm",  g.valuesArr());
        SDVariable X      = sd.constant("X",      features);
        SDVariable A      = sd.constant("A",      adj);
        SDVariable noise  = sd.constant("noise",  Nd4j.randn(DataType.DOUBLE, n, latent)); // fixed sample

        // Encoder: shared GCN -> two GCN heads for mu and log-variance.
        SDVariable w0  = sd.var("w0",  Nd4j.randn(DataType.DOUBLE, fIn, hidden).muli(0.3));
        SDVariable b0  = sd.var("b0",  Nd4j.zeros(DataType.DOUBLE, hidden));
        SDVariable wMu = sd.var("wMu", Nd4j.randn(DataType.DOUBLE, hidden, latent).muli(0.3));
        SDVariable bMu = sd.var("bMu", Nd4j.zeros(DataType.DOUBLE, latent));
        SDVariable wLv = sd.var("wLv", Nd4j.randn(DataType.DOUBLE, hidden, latent).muli(0.3));
        SDVariable bLv = sd.var("bLv", Nd4j.zeros(DataType.DOUBLE, latent));

        SDVariable h      = sd.gnn().gcnConv(X, w0, b0, aNorm, colIdx, rowPtr, n, n, true);     // [n, hidden]
        SDVariable mu     = sd.gnn().gcnConv(h, wMu, bMu, aNorm, colIdx, rowPtr, n, n, false);  // [n, latent]
        SDVariable logvar = sd.gnn().gcnConv(h, wLv, bLv, aNorm, colIdx, rowPtr, n, n, false);  // [n, latent]

        SDVariable z      = sd.gnn().vgaeReparam(mu, logvar, noise);                            // [n, latent]
        SDVariable logits = sd.gnn().innerProductDecoder(z);                                    // [n, n]
        SDVariable p      = sd.nn().sigmoid("recon", logits);                                   // edge probabilities

        // Reconstruction BCE + KL regulariser (KL scaled by 1/n, the usual VGAE weighting).
        SDVariable bce = sd.math().neg(A.mul(sd.math().log(p.add(1e-7)))
                .add(A.rsub(1.0).mul(sd.math().log(p.rsub(1.0).add(1e-7)))));
        SDVariable recon = sd.mean(bce);
        SDVariable kl    = sd.gnn().vgaeKlLoss(mu, logvar);
        recon.add("loss", kl.mul(1.0 / n));
        sd.setLossVariables("loss");

        String[] trainable = {"w0", "b0", "wMu", "bMu", "wLv", "bLv"};
        double lr = 0.02;
        Map<String, INDArray> empty = new HashMap<>();
        System.out.println("Training a VGAE to reconstruct the graph adjacency...");
        for (int epoch = 0; epoch <= 200; epoch++) {
            Map<String, INDArray> grads = sd.calculateGradients(empty, trainable);
            for (String name : trainable) {
                sd.getVariable(name).getArr().subi(grads.get(name).mul(lr));
            }
            if (epoch % 40 == 0) {
                double loss = sd.output(empty, "loss").get("loss").getDouble(0);
                System.out.printf("  epoch %3d   loss = %.4f%n", epoch, loss);
            }
        }

        // Reconstruction accuracy: how many of the n*n entries are predicted correctly (>0.5 == edge).
        INDArray probs = sd.output(empty, "recon").get("recon");
        int correct = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int predEdge = probs.getDouble(i, j) > 0.5 ? 1 : 0;
                if (predEdge == (int) adj.getDouble(i, j)) correct++;
            }
        }
        System.out.printf("%nAdjacency reconstruction accuracy: %d/%d entries%n", correct, n * n);
    }
}
