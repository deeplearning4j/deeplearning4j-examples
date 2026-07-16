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
import org.nd4j.linalg.api.ops.impl.graph.GraphSampler;
import org.nd4j.linalg.api.ops.impl.graph.GraphSampler.SampledSubgraph;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Mini-batch GraphSAGE training pattern via {@link GraphSampler} — the "systems" piece for graphs
 * too large for full-batch execution.
 *
 * <p>{@code GraphSampler.sampleMiniBatches} expands each batch of seed nodes into a multi-hop
 * sampled sub-graph (re-indexed to dense local indices); we gather the batch's node features, run
 * {@code sd.gnn().sageMean} on the sampled adjacency, and read the seed-node embeddings off the
 * first {@code numSeeds} rows. The same weights are reused across batches — in a real training
 * loop you would also accumulate gradients and update them here.
 */
public class Ex4_NeighborSampling {

    public static void main(String[] args) {
        Nd4j.getRandom().setSeed(12345);

        // 20-node graph: two communities of 10, densely linked within, sparse across.
        int n = 20;
        java.util.List<int[]> edgeList = new java.util.ArrayList<>();
        for (int c = 0; c < 2; c++) {
            int base = c * 10;
            for (int i = 0; i < 10; i++) {
                edgeList.add(new int[]{base + i, base + (i + 1) % 10});   // ring
                edgeList.add(new int[]{base + i, base + (i + 2) % 10});   // + chords
            }
        }
        edgeList.add(new int[]{9, 10});   // single bridge between communities
        GnnExampleGraphs.RawCsr g = GnnExampleGraphs.rawCsr(n, edgeList.toArray(new int[0][]));

        int fIn = 4, h = 4;
        INDArray features = Nd4j.randn(DataType.DOUBLE, n, fIn);
        // A single shared GraphSAGE weight [2F, H] reused across all mini-batches.
        INDArray sageW = Nd4j.randn(DataType.DOUBLE, 2 * fIn, h).muli(0.5);

        int[] allSeeds = new int[n];
        for (int i = 0; i < n; i++) allSeeds[i] = i;

        int batchSize = 5;
        int[] fanouts = {3, 3};   // 2-hop neighbourhood, up to 3 neighbours per node per hop
        List<SampledSubgraph> batches =
                GraphSampler.sampleMiniBatches(g.rowPtr, g.colIdx, allSeeds, batchSize, fanouts, 42L);

        System.out.printf("Full graph: %d nodes, %d directed edges%n", n, g.nnz());
        System.out.printf("Sampling %d seeds into mini-batches of %d, fanouts=%s%n%n",
                n, batchSize, java.util.Arrays.toString(fanouts));

        int b = 0;
        for (SampledSubgraph sg : batches) {
            // Gather this batch's node features (local index order; seeds occupy the first rows).
            INDArray subFeat = features.getRows(sg.nodeIds);

            SameDiff sd = SameDiff.create();
            SDVariable X      = sd.constant("X", subFeat);
            SDVariable colIdx = sd.constant("colIdx", sg.colIdxArr());
            SDVariable rowPtr = sd.constant("rowPtr", sg.rowPtrArr());
            SDVariable W      = sd.constant("W", sageW);

            SDVariable emb = sd.gnn().sageMean(X, W, null, colIdx, rowPtr, sg.numNodes(), sg.numNodes());
            INDArray seedEmb = emb.eval().get(
                    org.nd4j.linalg.indexing.NDArrayIndex.interval(0, sg.numSeeds),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());

            System.out.printf("batch %d: %d seeds -> sampled sub-graph of %d nodes / %d edges; "
                            + "seed embeddings shape %s%n",
                    b++, sg.numSeeds, sg.numNodes(), sg.numEdges(),
                    java.util.Arrays.toString(seedEmb.shape()));
        }
        System.out.println("\nEach seed's embedding is computed from only its sampled neighbourhood — "
                + "so memory/compute scale with the batch, not the whole graph.");
    }
}
