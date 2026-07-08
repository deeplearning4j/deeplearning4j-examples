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

import java.util.Arrays;

/**
 * A gallery of the {@code sd.gnn().*} convolution layers — each run forward once on the same small
 * graph, printing its output shape. Use it as a quick reference for the call signatures and as a
 * smoke test that every layer executes on the active backend.
 *
 * <p>GCN-family layers (GCN, ChebConv, GCNII) take the symmetric-normalised CSR; aggregation
 * layers (GATv2, GraphSAGE, GIN, PNA) take the raw adjacency CSR.
 */
public class Ex5_GnnLayerGallery {

    public static void main(String[] args) {
        Nd4j.getRandom().setSeed(12345);

        int n = 5, fIn = 3, h = 4;
        int[][] edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}, {0, 2}};
        GnnExampleGraphs.Csr norm = GnnExampleGraphs.normalizedCsr(n, edges);
        GnnExampleGraphs.RawCsr raw = GnnExampleGraphs.rawCsr(n, edges);

        // Per-edge destination index (rowIdx) for attention layers: edge k in row i -> rowIdx[k]=i.
        int[] rowIdx = new int[raw.nnz()];
        for (int i = 0; i < n; i++) {
            for (int k = raw.rowPtr[i]; k < raw.rowPtr[i + 1]; k++) rowIdx[k] = i;
        }

        SameDiff sd = SameDiff.create();
        SDVariable X       = sd.var("X", Nd4j.randn(DataType.DOUBLE, n, fIn));
        SDVariable colN    = sd.constant("colN", raw.colIdxArr());
        SDVariable ptrN    = sd.constant("ptrN", raw.rowPtrArr());
        SDVariable colNorm = sd.constant("colNorm", norm.colIdxArr());
        SDVariable ptrNorm = sd.constant("ptrNorm", norm.rowPtrArr());
        SDVariable aVals   = sd.constant("aVals", norm.valuesArr());
        SDVariable ridx    = sd.constant("ridx", Nd4j.createFromArray(rowIdx));

        System.out.println("Each GNN layer forward on a 5-node graph (F=3 -> H=4):\n");

        // GCN
        SDVariable gcn = sd.gnn().gcnConv(X, wv(sd, "gcnW", fIn, h), null, aVals, colNorm, ptrNorm, n, n, true);
        print("GCN (gcnConv)", gcn);

        // GATv2 (dynamic attention)
        SDVariable gatv2 = sd.gnn().gatV2ConvHead(X, wv(sd, "gatW", fIn, h), wv(sd, "gatAtt", h, 1),
                colN, ptrN, ridx, raw.nnz(), n, 0.2);
        print("GATv2 (gatV2ConvHead)", gatv2);

        // GraphSAGE-mean (weight is [2F, H] because it concatenates self + aggregated neighbours)
        SDVariable sage = sd.gnn().sageMean(X, wv(sd, "sageW", 2 * fIn, h), null, colN, ptrN, n, n);
        print("GraphSAGE (sageMean)", sage);

        // GIN
        SDVariable gin = sd.gnn().ginConv(X, wv(sd, "ginW1", fIn, h), bv(sd, "ginB1", h),
                wv(sd, "ginW2", h, h), bv(sd, "ginB2", h), sd.var("ginEps", Nd4j.scalar(DataType.DOUBLE, 0.0)),
                colN, ptrN, n, n);
        print("GIN (ginConv)", gin);

        // ChebConv (K=3 Chebyshev terms; here the normalised adjacency stands in for the scaled Laplacian)
        SDVariable cheb = sd.gnn().chebConv(X,
                new SDVariable[]{wv(sd, "ch0", fIn, h), wv(sd, "ch1", fIn, h), wv(sd, "ch2", fIn, h)},
                aVals, colNorm, ptrNorm, n, n);
        print("ChebConv (chebConv)", cheb);

        // PNA (weight is [4F, H] for the 4 aggregators: mean/max/min/std)
        SDVariable pna = sd.gnn().pnaConv(X, wv(sd, "pnaW", 4 * fIn, h), null, colN, ptrN, n, n, true);
        print("PNA (pnaConv)", pna);

        // GCNII (deep GCN; keeps the feature dimension, so W is [F, F])
        SDVariable gcnii = sd.gnn().gcniiConv(X, X, wv(sd, "gcniiW", fIn, fIn),
                aVals, colNorm, ptrNorm, n, n, 0.1, 0.5, true);
        print("GCNII (gcniiConv)", gcnii);

        System.out.println("\nAll layers executed successfully on backend: " + Nd4j.getBackend().getClass().getSimpleName());
    }

    private static SDVariable wv(SameDiff sd, String name, long rows, long cols) {
        return sd.var(name, Nd4j.randn(DataType.DOUBLE, rows, cols).muli(0.5));
    }

    private static SDVariable bv(SameDiff sd, String name, long len) {
        return sd.var(name, Nd4j.zeros(DataType.DOUBLE, len));
    }

    private static void print(String label, SDVariable out) {
        INDArray r = out.eval();
        System.out.printf("  %-24s -> shape %s%n", label, Arrays.toString(r.shape()));
    }
}
