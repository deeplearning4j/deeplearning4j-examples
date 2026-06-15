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

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * SameDiff Loss Functions (sd.loss namespace) - Complete API Reference
 *
 * This example covers all built-in loss functions:
 *
 *   1. absoluteDifference - L1 loss (MAE)
 *   2. meanSquaredError - L2 loss (MSE)
 *   3. huberLoss - Smooth L1 (robust to outliers)
 *   4. logLoss - Binary cross-entropy
 *   5. sigmoidCrossEntropy - Logits-based binary cross-entropy
 *   6. softmaxCrossEntropy - Multi-class cross-entropy (one-hot labels)
 *   7. sparseSoftmaxCrossEntropy - Multi-class cross-entropy (integer labels)
 *   8. hingeLoss - SVM-style margin loss
 *   9. cosineDistance - Direction-based loss
 *  10. logPoisson - Poisson regression loss
 *  11. meanPairwiseSquaredError - Pairwise MSE
 *  12. l2Loss - L2 regularization term
 *  13. contrastiveLoss - CLIP-style contrastive learning loss
 *  14. ctcLoss - Connectionist Temporal Classification
 *
 * Key concepts:
 *   - LossReduce enum: NONE, SUM, MEAN_BY_WEIGHT, MEAN_BY_NONZERO_WEIGHT_COUNT
 *   - Optional per-example or per-output weights
 *   - Label smoothing for cross-entropy losses
 */
public class LossOpsExample {

    public static void main(String[] args) {

        int batchSize = 4;
        int numClasses = 5;

        // ============================================================
        // 1. ABSOLUTE DIFFERENCE (L1 / MAE Loss)
        // ============================================================
        System.out.println("=== Absolute Difference (L1 / MAE) ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, batchSize, numClasses);
            SDVariable predictions = sd.placeHolder("predictions", DataType.FLOAT, batchSize, numClasses);

            // Without weights (null), with MEAN reduction
            SDVariable loss = sd.loss().absoluteDifference("mae_loss", labels, predictions, null,
                    LossReduce.MEAN_BY_WEIGHT);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("labels", Nd4j.rand(DataType.FLOAT, batchSize, numClasses));
            ph.put("predictions", Nd4j.rand(DataType.FLOAT, batchSize, numClasses));

            INDArray result = sd.output(ph, "mae_loss").get("mae_loss");
            System.out.println("  MAE loss (scalar): " + result);
            System.out.println("  Formula: mean(|labels - predictions|)");
        }

        // ============================================================
        // 2. MEAN SQUARED ERROR (L2 / MSE Loss)
        // ============================================================
        System.out.println("\n=== Mean Squared Error (MSE) ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, batchSize, numClasses);
            SDVariable predictions = sd.placeHolder("predictions", DataType.FLOAT, batchSize, numClasses);

            // With per-example weights
            SDVariable weights = sd.var("weights", Nd4j.create(new float[]{1, 2, 1, 3}).reshape(batchSize, 1));

            SDVariable mse = sd.loss().meanSquaredError("mse_loss", labels, predictions, weights,
                    LossReduce.MEAN_BY_WEIGHT);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("labels", Nd4j.rand(DataType.FLOAT, batchSize, numClasses));
            ph.put("predictions", Nd4j.rand(DataType.FLOAT, batchSize, numClasses));

            INDArray result = sd.output(ph, "mse_loss").get("mse_loss");
            System.out.println("  Weighted MSE loss: " + result);
            System.out.println("  Formula: sum(weights * (labels - predictions)^2) / sum(weights)");
        }

        // ============================================================
        // 3. HUBER LOSS (Smooth L1)
        // ============================================================
        System.out.println("\n=== Huber Loss (Smooth L1) ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, batchSize, numClasses);
            SDVariable predictions = sd.placeHolder("predictions", DataType.FLOAT, batchSize, numClasses);

            // delta controls the transition from quadratic to linear
            double delta = 1.0;
            SDVariable huber = sd.loss().huberLoss("huber_loss", labels, predictions, null,
                    LossReduce.SUM, delta);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("labels", Nd4j.rand(DataType.FLOAT, batchSize, numClasses));
            ph.put("predictions", Nd4j.rand(DataType.FLOAT, batchSize, numClasses));

            INDArray result = sd.output(ph, "huber_loss").get("huber_loss");
            System.out.println("  Huber loss (delta=" + delta + "): " + result);
            System.out.println("  |error| < delta: 0.5 * error^2  (quadratic)");
            System.out.println("  |error| >= delta: delta * |error| - 0.5 * delta^2  (linear)");
        }

        // ============================================================
        // 4. LOG LOSS (Binary Cross-Entropy)
        // ============================================================
        System.out.println("\n=== Log Loss (Binary Cross-Entropy) ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, batchSize, 1);
            SDVariable predictions = sd.placeHolder("predictions", DataType.FLOAT, batchSize, 1);

            // Simple form (no weights, no epsilon)
            SDVariable logLoss = sd.loss().logLoss("log_loss", labels, predictions);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("labels", Nd4j.create(new float[]{1, 0, 1, 0}).reshape(batchSize, 1));
            ph.put("predictions", Nd4j.create(new float[]{0.9f, 0.1f, 0.8f, 0.2f}).reshape(batchSize, 1));

            INDArray result = sd.output(ph, "log_loss").get("log_loss");
            System.out.println("  Binary CE loss: " + result);
            System.out.println("  Formula: -mean(y*log(p) + (1-y)*log(1-p))");
        }

        // ============================================================
        // 5. SIGMOID CROSS-ENTROPY (from logits)
        // ============================================================
        System.out.println("\n=== Sigmoid Cross-Entropy ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, batchSize, 1);
            SDVariable logits = sd.placeHolder("logits", DataType.FLOAT, batchSize, 1);

            // Label smoothing reduces overconfidence
            double labelSmoothing = 0.1;
            SDVariable sigCE = sd.loss().sigmoidCrossEntropy("sig_ce", labels, logits, null,
                    LossReduce.MEAN_BY_WEIGHT, labelSmoothing);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("labels", Nd4j.create(new float[]{1, 0, 1, 0}).reshape(batchSize, 1));
            ph.put("logits", Nd4j.create(new float[]{2.0f, -1.5f, 1.0f, -2.0f}).reshape(batchSize, 1));

            INDArray result = sd.output(ph, "sig_ce").get("sig_ce");
            System.out.println("  Sigmoid CE (smoothing=" + labelSmoothing + "): " + result);
            System.out.println("  Numerically stable: operates on logits, not probabilities");
        }

        // ============================================================
        // 6. SOFTMAX CROSS-ENTROPY (multi-class, one-hot labels)
        // ============================================================
        System.out.println("\n=== Softmax Cross-Entropy ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable oneHotLabels = sd.placeHolder("labels", DataType.FLOAT, batchSize, numClasses);
            SDVariable logits = sd.placeHolder("logits", DataType.FLOAT, batchSize, numClasses);

            SDVariable softmaxCE = sd.loss().softmaxCrossEntropy("softmax_ce", oneHotLabels, logits,
                    null, LossReduce.MEAN_BY_WEIGHT, 0.0);

            // Create one-hot labels: classes [0, 2, 4, 1]
            INDArray labelsArr = Nd4j.zeros(DataType.FLOAT, batchSize, numClasses);
            labelsArr.putScalar(0, 0, 1);
            labelsArr.putScalar(1, 2, 1);
            labelsArr.putScalar(2, 4, 1);
            labelsArr.putScalar(3, 1, 1);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("labels", labelsArr);
            ph.put("logits", Nd4j.randn(DataType.FLOAT, batchSize, numClasses));

            INDArray result = sd.output(ph, "softmax_ce").get("softmax_ce");
            System.out.println("  Softmax CE: " + result);
            System.out.println("  Internally: -sum(oneHot * log(softmax(logits)))");
        }

        // ============================================================
        // 7. SPARSE SOFTMAX CROSS-ENTROPY (integer labels)
        // ============================================================
        System.out.println("\n=== Sparse Softmax Cross-Entropy ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable logits = sd.placeHolder("logits", DataType.FLOAT, batchSize, numClasses);
            SDVariable labels = sd.placeHolder("labels", DataType.INT, batchSize);

            // Integer class indices instead of one-hot
            SDVariable sparseCE = sd.loss().sparseSoftmaxCrossEntropy("sparse_ce", logits, labels);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("logits", Nd4j.randn(DataType.FLOAT, batchSize, numClasses));
            ph.put("labels", Nd4j.create(new int[]{0, 2, 4, 1}).reshape(batchSize).castTo(DataType.INT));

            INDArray result = sd.output(ph, "sparse_ce").get("sparse_ce");
            System.out.println("  Sparse softmax CE: " + result);
            System.out.println("  More memory-efficient than one-hot encoding");
        }

        // ============================================================
        // 8. HINGE LOSS (SVM-style)
        // ============================================================
        System.out.println("\n=== Hinge Loss ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, batchSize, 1);
            SDVariable predictions = sd.placeHolder("predictions", DataType.FLOAT, batchSize, 1);

            SDVariable hinge = sd.loss().hingeLoss("hinge_loss", labels, predictions,
                    null, LossReduce.MEAN_BY_WEIGHT);

            Map<String, INDArray> ph = new HashMap<>();
            // Labels should be 0 or 1
            ph.put("labels", Nd4j.create(new float[]{1, 0, 1, 0}).reshape(batchSize, 1));
            ph.put("predictions", Nd4j.create(new float[]{0.8f, 0.3f, 0.6f, 0.7f}).reshape(batchSize, 1));

            INDArray result = sd.output(ph, "hinge_loss").get("hinge_loss");
            System.out.println("  Hinge loss: " + result);
            System.out.println("  Formula: max(0, 1 - labels * predictions)");
        }

        // ============================================================
        // 9. COSINE DISTANCE LOSS
        // ============================================================
        System.out.println("\n=== Cosine Distance Loss ===");
        {
            SameDiff sd = SameDiff.create();
            int embedDim = 8;
            SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, batchSize, embedDim);
            SDVariable predictions = sd.placeHolder("predictions", DataType.FLOAT, batchSize, embedDim);

            // dim=1 means cosine distance along the embedding dimension
            SDVariable cosineLoss = sd.loss().cosineDistance("cosine_loss", labels, predictions,
                    null, LossReduce.MEAN_BY_WEIGHT, 1);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("labels", Nd4j.rand(DataType.FLOAT, batchSize, embedDim));
            ph.put("predictions", Nd4j.rand(DataType.FLOAT, batchSize, embedDim));

            INDArray result = sd.output(ph, "cosine_loss").get("cosine_loss");
            System.out.println("  Cosine distance loss: " + result);
            System.out.println("  = 1 - cosine_similarity(labels, predictions)");
        }

        // ============================================================
        // 10. LOG POISSON LOSS
        // ============================================================
        System.out.println("\n=== Log Poisson Loss ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, batchSize, 1);
            SDVariable predictions = sd.placeHolder("predictions", DataType.FLOAT, batchSize, 1);

            // full=true adds the Stirling approximation for the log factorial term
            SDVariable poisson = sd.loss().logPoisson("poisson_loss", labels, predictions,
                    null, LossReduce.MEAN_BY_WEIGHT, true);

            Map<String, INDArray> ph = new HashMap<>();
            // Count data (Poisson targets are non-negative integers)
            ph.put("labels", Nd4j.create(new float[]{3, 0, 5, 2}).reshape(batchSize, 1));
            ph.put("predictions", Nd4j.create(new float[]{1.0f, 0.5f, 1.5f, 0.8f}).reshape(batchSize, 1));

            INDArray result = sd.output(ph, "poisson_loss").get("poisson_loss");
            System.out.println("  Log Poisson loss: " + result);
            System.out.println("  Formula: exp(predictions) - labels * predictions");
        }

        // ============================================================
        // 11. L2 LOSS (Regularization)
        // ============================================================
        System.out.println("\n=== L2 Loss (Regularization) ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable weights = sd.var("weights", Nd4j.randn(DataType.FLOAT, 100, 50).muli(0.1));

            // L2 loss = sum(x^2) / 2
            SDVariable l2 = sd.loss().l2Loss("l2_reg", weights);

            INDArray result = sd.output(Collections.emptyMap(), "l2_reg").get("l2_reg");
            System.out.println("  L2 regularization: " + result);
            System.out.println("  Formula: sum(weights^2) / 2");
            System.out.println("  Typically added to main loss with a lambda coefficient");
        }

        // ============================================================
        // 12. CONTRASTIVE LOSS (CLIP-style)
        // ============================================================
        System.out.println("\n=== Contrastive Loss ===");
        {
            SameDiff sd = SameDiff.create();
            int embedDim = 16;
            SDVariable imageEmbed = sd.placeHolder("imageEmbed", DataType.FLOAT, batchSize, embedDim);
            SDVariable textEmbed = sd.placeHolder("textEmbed", DataType.FLOAT, batchSize, embedDim);

            // CLIP-style contrastive: temperature controls sharpness
            SDVariable contrastive = sd.loss().contrastiveLoss("contrastive", imageEmbed, textEmbed,
                    0.07);

            // Simple form: default temperature
            SDVariable contrastiveSimple = sd.loss().contrastiveLoss("contrastive_simple",
                    imageEmbed, textEmbed);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("imageEmbed", Nd4j.rand(DataType.FLOAT, batchSize, embedDim));
            ph.put("textEmbed", Nd4j.rand(DataType.FLOAT, batchSize, embedDim));

            INDArray result = sd.output(ph, "contrastive").get("contrastive");
            System.out.println("  Contrastive loss (temp=0.07): " + result);
            System.out.println("  Aligns matched image-text pairs, pushes apart unmatched");
        }

        // ============================================================
        // 13. LOSS REDUCTION MODES
        // ============================================================
        System.out.println("\n=== Loss Reduction Modes ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, batchSize, numClasses);
            SDVariable predictions = sd.placeHolder("predictions", DataType.FLOAT, batchSize, numClasses);

            // NONE: per-element loss (same shape as input)
            SDVariable lossNone = sd.loss().meanSquaredError("mse_none", labels, predictions,
                    null, LossReduce.NONE);

            // SUM: sum of all losses -> scalar
            SDVariable lossSum = sd.loss().meanSquaredError("mse_sum", labels, predictions,
                    null, LossReduce.SUM);

            // MEAN_BY_WEIGHT: sum(w*loss) / sum(w) -> scalar
            SDVariable lossMean = sd.loss().meanSquaredError("mse_mean", labels, predictions,
                    null, LossReduce.MEAN_BY_WEIGHT);

            // MEAN_BY_NONZERO_WEIGHT_COUNT: sum(w*loss) / count(w != 0) -> scalar
            SDVariable lossMeanNZ = sd.loss().meanSquaredError("mse_mean_nz", labels, predictions,
                    null, LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("labels", Nd4j.rand(DataType.FLOAT, batchSize, numClasses));
            ph.put("predictions", Nd4j.rand(DataType.FLOAT, batchSize, numClasses));

            Map<String, INDArray> results = sd.output(ph,
                    "mse_none", "mse_sum", "mse_mean", "mse_mean_nz");

            System.out.println("  NONE shape:  " + java.util.Arrays.toString(results.get("mse_none").shape()));
            System.out.println("  SUM value:   " + results.get("mse_sum"));
            System.out.println("  MEAN value:  " + results.get("mse_mean"));
            System.out.println("  MEAN_NZ:     " + results.get("mse_mean_nz"));
        }

        // ============================================================
        // 14. USING LOSS IN A TRAINING GRAPH
        // ============================================================
        System.out.println("\n=== Loss in Training Graph ===");
        {
            SameDiff sd = SameDiff.create();
            int nIn = 10, nOut = 5;

            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, nIn);
            SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, -1, nOut);

            // Simple linear model
            SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, nIn, nOut).muli(0.1));
            SDVariable b = sd.var("b", Nd4j.zeros(DataType.FLOAT, nOut));
            SDVariable logits = input.mmul(w).add(b);

            // Softmax cross-entropy as the training loss
            SDVariable loss = sd.loss().softmaxCrossEntropy("loss", labels, logits,
                    null, LossReduce.MEAN_BY_WEIGHT, 0.0);

            // Add L2 regularization
            SDVariable l2Reg = sd.loss().l2Loss("l2_reg", w);
            double lambda = 0.001;
            SDVariable totalLoss = loss.add("total_loss", l2Reg.mul(lambda));

            // Mark as loss for training
            totalLoss.markAsLoss();

            // Compute gradients
            Map<String, INDArray> ph = new HashMap<>();
            ph.put("input", Nd4j.rand(DataType.FLOAT, batchSize, nIn));
            INDArray labelsArr = Nd4j.zeros(DataType.FLOAT, batchSize, nOut);
            for (int i = 0; i < batchSize; i++) {
                labelsArr.putScalar(i, i % nOut, 1.0f);
            }
            ph.put("labels", labelsArr);

            Map<String, INDArray> grads = sd.calculateGradients(ph, "w", "b");
            System.out.println("  Weight gradient shape: " + java.util.Arrays.toString(grads.get("w").shape()));
            System.out.println("  Bias gradient shape:   " + java.util.Arrays.toString(grads.get("b").shape()));

            INDArray lossVal = sd.output(ph, "total_loss").get("total_loss");
            System.out.println("  Total loss (CE + L2): " + lossVal);
        }

        System.out.println("\nAll loss functions demonstrated successfully.");
    }
}
