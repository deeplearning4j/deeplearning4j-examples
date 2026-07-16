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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

/**
 * Shared graph helpers for the {@code sd.gnn().*} examples.
 *
 * <p>The GNN layers consume the graph as a CSR triple ({@code rowPtr}, {@code colIdx},
 * {@code values}). For GCN-style layers the values are the symmetric-normalised adjacency
 * {@code D^{-1/2} (A + I) D^{-1/2}} (self-loops included), which this helper builds from a plain
 * undirected edge list so the examples can focus on the model rather than graph bookkeeping.
 */
public final class GnnExampleGraphs {

    private GnnExampleGraphs() { }

    /** A graph in CSR form (source/neighbour indices per row), with normalised edge weights. */
    public static final class Csr {
        public final int n;            // number of nodes
        public final int[] rowPtr;     // [n + 1]
        public final int[] colIdx;     // [nnz] neighbour (source) node per edge, incl. self-loops
        public final double[] values;  // [nnz] symmetric-normalised adjacency values

        Csr(int n, int[] rowPtr, int[] colIdx, double[] values) {
            this.n = n;
            this.rowPtr = rowPtr;
            this.colIdx = colIdx;
            this.values = values;
        }

        public int nnz()            { return colIdx.length; }
        public INDArray rowPtrArr() { return Nd4j.createFromArray(rowPtr); }   // INT32
        public INDArray colIdxArr() { return Nd4j.createFromArray(colIdx); }   // INT32
        public INDArray valuesArr() { return Nd4j.createFromArray(values); }   // DOUBLE
    }

    /**
     * Build the symmetric-normalised CSR {@code D^{-1/2} (A + I) D^{-1/2}} from undirected edges.
     *
     * @param n              number of nodes
     * @param undirectedEdges pairs {@code {u, v}} (each added in both directions); self-loops are
     *                       added automatically
     */
    public static Csr normalizedCsr(int n, int[][] undirectedEdges) {
        List<TreeSet<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            adj.add(new TreeSet<>());
            adj.get(i).add(i);                       // self-loop (the "+ I")
        }
        for (int[] e : undirectedEdges) {
            adj.get(e[0]).add(e[1]);
            adj.get(e[1]).add(e[0]);                 // undirected
        }

        int[] deg = new int[n];
        int[] rowPtr = new int[n + 1];
        for (int i = 0; i < n; i++) {
            deg[i] = adj.get(i).size();              // degree of (A + I)
            rowPtr[i + 1] = rowPtr[i] + deg[i];
        }

        int nnz = rowPtr[n];
        int[] colIdx = new int[nnz];
        double[] values = new double[nnz];
        int k = 0;
        for (int i = 0; i < n; i++) {
            for (int j : adj.get(i)) {
                colIdx[k] = j;
                values[k] = 1.0 / Math.sqrt((double) deg[i] * (double) deg[j]);
                k++;
            }
        }
        return new Csr(n, rowPtr, colIdx, values);
    }

    /** One-hot encode integer class labels into a [n, numClasses] DOUBLE matrix. */
    public static INDArray oneHot(int[] labels, int numClasses) {
        INDArray out = Nd4j.zeros(org.nd4j.linalg.api.buffer.DataType.DOUBLE, labels.length, numClasses);
        for (int i = 0; i < labels.length; i++) {
            out.putScalar(i, labels[i], 1.0);
        }
        return out;
    }

    /** A raw (un-normalised, no self-loop) CSR adjacency — for aggregation layers like GIN/SAGE/PNA. */
    public static final class RawCsr {
        public final int n;
        public final int[] rowPtr, colIdx;
        RawCsr(int n, int[] rowPtr, int[] colIdx) { this.n = n; this.rowPtr = rowPtr; this.colIdx = colIdx; }
        public int nnz()            { return colIdx.length; }
        public INDArray rowPtrArr() { return Nd4j.createFromArray(rowPtr); }
        public INDArray colIdxArr() { return Nd4j.createFromArray(colIdx); }
    }

    /** Build a raw undirected adjacency CSR (no self-loops, no normalisation). */
    public static RawCsr rawCsr(int n, int[][] undirectedEdges) {
        List<TreeSet<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) adj.add(new TreeSet<>());
        for (int[] e : undirectedEdges) {
            adj.get(e[0]).add(e[1]);
            adj.get(e[1]).add(e[0]);
        }
        int[] rowPtr = new int[n + 1];
        for (int i = 0; i < n; i++) rowPtr[i + 1] = rowPtr[i] + adj.get(i).size();
        int[] colIdx = new int[rowPtr[n]];
        int k = 0;
        for (int i = 0; i < n; i++) {
            for (int j : adj.get(i)) colIdx[k++] = j;
        }
        return new RawCsr(n, rowPtr, colIdx);
    }
}
