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
import org.nd4j.linalg.api.ops.impl.graph.KgeEvaluation;
import org.nd4j.linalg.api.ops.impl.graph.KgeTripleSampler;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Knowledge-graph completion (link prediction) with a DistMult embedding model and the full
 * KGE training/eval stack: {@code sd.graph().distMult} scoring, {@link KgeTripleSampler} negative
 * sampling, {@code sd.graph().marginRankingLoss}, and {@link KgeEvaluation} MRR / Hits@K ranking.
 *
 * <p>Entity and relation embedding tables are trained so that observed triples
 * {@code (head, relation, tail)} score higher than corrupted ones. At eval time, for each test
 * triple we score every candidate tail and report where the true tail ranks.
 */
public class Ex6_KnowledgeGraphCompletion {

    public static void main(String[] args) {
        Nd4j.getRandom().setSeed(12345);

        // Tiny knowledge graph: 8 entities, 2 relations (0 = "knows", 1 = "worksWith").
        int numEntities = 8, numRelations = 2, dim = 16;
        int[][] triples = {
                {0, 0, 1}, {1, 0, 2}, {2, 0, 3}, {3, 0, 0},   // "knows" ring in group A {0,1,2,3}
                {4, 0, 5}, {5, 0, 6}, {6, 0, 7}, {7, 0, 4},   // "knows" ring in group B {4,5,6,7}
                {0, 1, 4}, {1, 1, 5}, {2, 1, 6}, {3, 1, 7},   // "worksWith" pairs A<->B
                {4, 1, 0}, {5, 1, 1}, {6, 1, 2}, {7, 1, 3}
        };
        Set<Long> known = KgeTripleSampler.knownSet(triples, numEntities, numRelations);

        // ---- Model: entity + relation embedding tables, DistMult scoring with a margin loss ----
        SameDiff sd = SameDiff.create();
        SDVariable entityEmb = sd.var("entityEmb", Nd4j.randn(DataType.DOUBLE, numEntities, dim).muli(0.3));
        SDVariable relEmb    = sd.var("relEmb",    Nd4j.randn(DataType.DOUBLE, numRelations, dim).muli(0.3));

        // Triple indices are fed each step (positives fixed, negatives resampled).
        SDVariable hIdx  = sd.placeHolder("hIdx",  DataType.INT32, -1);
        SDVariable rIdx  = sd.placeHolder("rIdx",  DataType.INT32, -1);
        SDVariable tIdx  = sd.placeHolder("tIdx",  DataType.INT32, -1);
        SDVariable nhIdx = sd.placeHolder("nhIdx", DataType.INT32, -1);
        SDVariable ntIdx = sd.placeHolder("ntIdx", DataType.INT32, -1);

        SDVariable posScore = sd.graph().distMult(
                sd.gather(entityEmb, hIdx, 0), sd.gather(relEmb, rIdx, 0), sd.gather(entityEmb, tIdx, 0));
        SDVariable negScore = sd.graph().distMult(
                sd.gather(entityEmb, nhIdx, 0), sd.gather(relEmb, rIdx, 0), sd.gather(entityEmb, ntIdx, 0));
        SDVariable loss = sd.graph().marginRankingLoss(posScore, negScore, 1.0);
        sd.setLossVariables(loss);

        // Fixed positive index columns.
        INDArray posH = Nd4j.createFromArray(KgeTripleSampler.column(triples, 0));
        INDArray posR = Nd4j.createFromArray(KgeTripleSampler.column(triples, 1));
        INDArray posT = Nd4j.createFromArray(KgeTripleSampler.column(triples, 2));

        double lr = 0.1;
        System.out.println("Training DistMult on a " + numEntities + "-entity knowledge graph...");
        for (int epoch = 0; epoch <= 400; epoch++) {
            // Resample one corrupted negative per positive each epoch (filtered).
            int[][] negs = KgeTripleSampler.corrupt(triples, numEntities, numRelations, 1, known, 100L + epoch);
            Map<String, INDArray> feed = new HashMap<>();
            feed.put("hIdx", posH);
            feed.put("rIdx", posR);
            feed.put("tIdx", posT);
            feed.put("nhIdx", Nd4j.createFromArray(KgeTripleSampler.column(negs, 0)));
            feed.put("ntIdx", Nd4j.createFromArray(KgeTripleSampler.column(negs, 2)));

            Map<String, INDArray> grads = sd.calculateGradients(feed, "entityEmb", "relEmb");
            entityEmb.getArr().subi(grads.get("entityEmb").mul(lr));
            relEmb.getArr().subi(grads.get("relEmb").mul(lr));

            if (epoch % 80 == 0) {
                double l = sd.output(feed, loss.name()).get(loss.name()).getDouble(0);
                System.out.printf("  epoch %3d   margin loss = %.4f%n", epoch, l);
            }
        }

        // ---- Evaluate: for each triple (h, r, ?), score every candidate tail and rank the true one ----
        INDArray E = entityEmb.getArr();
        INDArray Rl = relEmb.getArr();
        double[][] scores = new double[triples.length][numEntities];
        int[] trueTails = new int[triples.length];
        for (int i = 0; i < triples.length; i++) {
            int[] tr = triples[i];
            INDArray hr = E.getRow(tr[0]).mul(Rl.getRow(tr[1]));          // h ⊙ r  [dim]
            for (int e = 0; e < numEntities; e++) {
                scores[i][e] = E.getRow(e).mul(hr).sumNumber().doubleValue();  // DistMult score for tail e
            }
            trueTails[i] = tr[2];
        }
        KgeEvaluation.Metrics m = KgeEvaluation.evaluate(scores, trueTails, new int[]{1, 3, 10});
        System.out.println("\nTail-ranking over the training triples: " + m);
        System.out.println("(MRR / Hits@K near 1.0 means the model ranks each true tail at or near the top.)");
    }
}
