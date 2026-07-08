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
import org.nd4j.linalg.api.ops.impl.graph.GraphPooling;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

/**
 * Graph-level classification with a GIN convolution + a global pooling readout.
 *
 * <p>A batch of small graphs (some triangles, some paths) is packed into a single block-diagonal
 * CSR; a {@code graphIds} vector maps each node to its graph. {@code sd.gnn().ginConv} produces
 * node embeddings and {@link GraphPooling#globalMeanPool} reduces them to one vector per graph,
 * which a linear classifier maps to a label. Features are uniform (all ones) on purpose, so the
 * model must learn from <em>structure</em> alone — exactly what GIN is designed for.
 */
public class Ex2_GraphClassification {

    public static void main(String[] args) {
        Nd4j.getRandom().setSeed(12345);

        // 4 graphs of 3 nodes each: triangles (label 1) and paths (label 0), packed block-diagonally.
        int n = 12;
        int[][] edges = {
                {0, 1}, {1, 2}, {2, 0},   // graph 0: triangle
                {3, 4}, {4, 5},           // graph 1: path
                {6, 7}, {7, 8}, {8, 6},   // graph 2: triangle
                {9, 10}, {10, 11}         // graph 3: path
        };
        GnnExampleGraphs.RawCsr g = GnnExampleGraphs.rawCsr(n, edges);

        int[] graphIds = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
        int[] graphLabels = {1, 0, 1, 0};     // triangle = 1, path = 0
        int numGraphs = 4;
        int numClasses = 2;
        int fIn = 4;

        INDArray features = Nd4j.ones(DataType.DOUBLE, n, fIn);   // uniform: force structure learning

        SameDiff sd = SameDiff.create();
        SDVariable colIdx  = sd.constant("colIdx", g.colIdxArr());
        SDVariable rowPtr  = sd.constant("rowPtr", g.rowPtrArr());
        SDVariable gids    = sd.constant("graphIds", Nd4j.createFromArray(graphIds));
        SDVariable X       = sd.constant("X", features);
        SDVariable yOneHot = sd.constant("y", GnnExampleGraphs.oneHot(graphLabels, numClasses));

        // GIN: a 2-layer MLP over (1+eps)*X + sum(neighbours).
        int hidden = 8, embed = 8;
        SDVariable w1  = sd.var("w1",  Nd4j.randn(DataType.DOUBLE, fIn, hidden).muli(0.5));
        SDVariable b1  = sd.var("b1",  Nd4j.zeros(DataType.DOUBLE, hidden));
        SDVariable w2  = sd.var("w2",  Nd4j.randn(DataType.DOUBLE, hidden, embed).muli(0.5));
        SDVariable b2  = sd.var("b2",  Nd4j.zeros(DataType.DOUBLE, embed));
        SDVariable eps = sd.var("eps", Nd4j.scalar(DataType.DOUBLE, 0.0));
        SDVariable wCls = sd.var("wCls", Nd4j.randn(DataType.DOUBLE, embed, numClasses).muli(0.5));
        SDVariable bCls = sd.var("bCls", Nd4j.zeros(DataType.DOUBLE, numClasses));

        SDVariable nodeEmb  = sd.gnn().ginConv(X, w1, b1, w2, b2, eps, colIdx, rowPtr, n, n);   // [12, embed]
        SDVariable graphEmb = GraphPooling.globalMeanPool(sd, "readout", nodeEmb, gids, numGraphs); // [4, embed]
        SDVariable logits   = sd.mmul(graphEmb, wCls).add(bCls);                                // [4, 2]
        SDVariable softmax  = sd.nn().softmax("softmax", logits);

        SDVariable ce = sd.math().neg(yOneHot.mul(sd.math().log(softmax.add(1e-7))));
        sd.mean("loss", ce);
        sd.setLossVariables("loss");

        String[] trainable = {"w1", "b1", "w2", "b2", "eps", "wCls", "bCls"};
        double lr = 0.05;
        Map<String, INDArray> empty = new HashMap<>();
        System.out.println("Training GIN graph classifier (triangles vs paths) on 4 graphs...");
        for (int epoch = 0; epoch <= 300; epoch++) {
            Map<String, INDArray> grads = sd.calculateGradients(empty, trainable);
            for (String name : trainable) {
                sd.getVariable(name).getArr().subi(grads.get(name).mul(lr));
            }
            if (epoch % 60 == 0) {
                double loss = sd.output(empty, "loss").get("loss").getDouble(0);
                System.out.printf("  epoch %3d   loss = %.4f%n", epoch, loss);
            }
        }

        INDArray pred = sd.output(empty, "softmax").get("softmax").argMax(1);
        int correct = 0;
        for (int gI = 0; gI < numGraphs; gI++) {
            if (pred.getInt(gI) == graphLabels[gI]) correct++;
        }
        System.out.printf("%nGraph-classification accuracy: %d/%d  (predictions=%s, labels=%s)%n",
                correct, numGraphs, pred, java.util.Arrays.toString(graphLabels));
    }
}
