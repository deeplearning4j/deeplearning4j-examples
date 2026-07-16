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

package org.deeplearning4j.examples.quickstart.features.evaluation;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.EvaluationCalibration;
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.evaluation.custom.CustomEvaluation;
import org.nd4j.evaluation.custom.EvaluationLambda;
import org.nd4j.evaluation.custom.MergeLambda;
import org.nd4j.evaluation.curves.PrecisionRecallCurve;
import org.nd4j.evaluation.curves.RocCurve;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.evaluation.curves.Histogram;
import org.nd4j.evaluation.curves.ReliabilityDiagram;

import java.util.List;

/**
 * Evaluation Metrics in DL4J - Complete API Reference
 *
 * This example covers all evaluation capabilities:
 *
 *   1. Standard Evaluation - Accuracy, precision, recall, F1
 *   2. ROC - Binary ROC curves, AUROC, AUPRC
 *   3. ROCMultiClass - Per-class ROC for multi-class problems
 *   4. EvaluationCalibration - Reliability diagrams, calibration metrics
 *   5. CustomEvaluation - User-defined metrics with EvaluationLambda
 *
 * Key classes:
 *   - org.nd4j.evaluation.classification.Evaluation
 *   - org.nd4j.evaluation.classification.ROC
 *   - org.nd4j.evaluation.classification.ROCMultiClass
 *   - org.nd4j.evaluation.classification.EvaluationCalibration
 *   - org.nd4j.evaluation.custom.CustomEvaluation
 *   - org.nd4j.evaluation.custom.EvaluationLambda
 */
public class EvaluationMetricsExample {

    public static void main(String[] args) throws Exception {

        // Build a simple MNIST classifier for evaluation
        int batchSize = 64;
        DataSetIterator trainIter = new MnistDataSetIterator(batchSize, true, 42);
        DataSetIterator testIter = new MnistDataSetIterator(batchSize, false, 42);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-3))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder().nIn(784).nOut(128).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(128).nOut(10).activation(Activation.SOFTMAX).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Quick training (1 epoch for demo)
        model.fit(trainIter);

        // ============================================================
        // 1. STANDARD EVALUATION
        // ============================================================
        System.out.println("=== Standard Evaluation ===");
        {
            Evaluation eval = model.evaluate(testIter);
            testIter.reset();

            System.out.println("  Accuracy:  " + eval.accuracy());
            System.out.println("  Precision: " + eval.precision());
            System.out.println("  Recall:    " + eval.recall());
            System.out.println("  F1 Score:  " + eval.f1());

            // Per-class metrics
            System.out.println("\n  Per-class precision:");
            for (int i = 0; i < 10; i++) {
                System.out.printf("    Class %d: precision=%.4f recall=%.4f f1=%.4f%n",
                        i, eval.precision(i), eval.recall(i), eval.f1(i));
            }

            // Confusion matrix
            System.out.println("\n  Confusion matrix:");
            System.out.println(eval.confusionMatrix());
        }

        // ============================================================
        // 2. ROC - Binary Classification Metrics
        // ============================================================
        System.out.println("\n=== ROC (Binary) ===");
        {
            // Exact ROC mode (thresholdSteps=0)
            ROC roc = new ROC();

            // Approximate mode (fixed threshold steps, faster for large datasets)
            ROC rocApprox = new ROC(100);

            // With options
            ROC rocFull = new ROC(0, true, 2048);

            // Evaluate: for binary ROC, use column 1 as positive class probability
            testIter.reset();
            while (testIter.hasNext()) {
                DataSet batch = testIter.next();
                INDArray predictions = model.output(batch.getFeatures());
                INDArray labels = batch.getLabels();
                // Binary ROC: pick class 0 vs rest for demo
                INDArray binaryPred = predictions.getColumn(0);
                INDArray binaryLabel = labels.getColumn(0);
                roc.eval(binaryLabel, binaryPred);
            }

            System.out.println("  AUROC (class 0 vs rest): " + roc.calculateAUC());
            System.out.println("  AUPRC: " + roc.calculateAUCPR());

            // Get curve data points
            RocCurve rocCurve = roc.getRocCurve();
            System.out.println("  ROC curve points: " + rocCurve.numPoints());

            PrecisionRecallCurve prCurve = roc.getPrecisionRecallCurve();
            System.out.println("  PR curve points: " + prCurve.numPoints());

            // Score for metric (useful for model selection)
            System.out.println("  AUROC metric: " + roc.scoreForMetric(ROC.Metric.AUROC));
            System.out.println("  AUPRC metric: " + roc.scoreForMetric(ROC.Metric.AUPRC));
        }

        // ============================================================
        // 3. ROC MULTI-CLASS - Per-class ROC
        // ============================================================
        System.out.println("\n=== ROC Multi-Class ===");
        {
            ROCMultiClass rocMC = new ROCMultiClass();

            testIter.reset();
            while (testIter.hasNext()) {
                DataSet batch = testIter.next();
                INDArray predictions = model.output(batch.getFeatures());
                rocMC.eval(batch.getLabels(), predictions);
            }

            System.out.println("  Per-class AUROC:");
            for (int i = 0; i < 10; i++) {
                System.out.printf("    Class %d: AUROC=%.4f  AUPRC=%.4f%n",
                        i, rocMC.calculateAUC(i), rocMC.calculateAUCPR(i));
            }

            // Average AUROC
            double avgAuroc = 0;
            for (int i = 0; i < 10; i++) {
                avgAuroc += rocMC.calculateAUC(i);
            }
            System.out.println("  Average AUROC: " + (avgAuroc / 10));
        }

        // ============================================================
        // 4. EVALUATION CALIBRATION - Model confidence analysis
        // ============================================================
        System.out.println("\n=== Evaluation Calibration ===");
        {
            // Constructor: (reliabilityDiagBins, histogramBins, excludeEmptyBins)
            EvaluationCalibration evalCal = new EvaluationCalibration(10, 50, true);

            testIter.reset();
            while (testIter.hasNext()) {
                DataSet batch = testIter.next();
                INDArray predictions = model.output(batch.getFeatures());
                evalCal.eval(batch.getLabels(), predictions);
            }

            System.out.println("  Number of classes: " + evalCal.numClasses());

            // Label and prediction distribution
            int[] labelCounts = evalCal.getLabelCountsEachClass();
            int[] predCounts = evalCal.getPredictionCountsEachClass();
            System.out.println("\n  Class distribution:");
            for (int i = 0; i < 10; i++) {
                System.out.printf("    Class %d: labels=%d  predictions=%d%n",
                        i, labelCounts[i], predCounts[i]);
            }

            // Reliability diagram (calibration curve) for each class
            // Perfect calibration: predicted probability matches actual frequency
            System.out.println("\n  Reliability diagram (class 0):");
            ReliabilityDiagram reliabilityDiag = evalCal.getReliabilityDiagram(0);
            System.out.println("    Points: " + reliabilityDiag.numPoints());

            // Residual and probability histograms
            Histogram residualHist = evalCal.getResidualPlotAllClasses();
            System.out.println("  Residual histogram bins: " + residualHist.numPoints());

            Histogram probHist = evalCal.getProbabilityHistogramAllClasses();
            System.out.println("  Probability histogram bins: " + probHist.numPoints());
        }

        // ============================================================
        // 5. CUSTOM EVALUATION - User-defined metrics
        // ============================================================
        System.out.println("\n=== Custom Evaluation ===");
        {
            // Define a custom evaluation lambda that computes top-K accuracy
            EvaluationLambda<Double> topKAccuracyLambda = (labels, predictions, mask, metadata) -> {
                int k = 3; // Top-3 accuracy
                int batchSz = (int) labels.size(0);
                int correct = 0;
                for (int i = 0; i < batchSz; i++) {
                    int trueClass = labels.getRow(i).argMax().getInt(0);
                    // Get top-k prediction indices
                    INDArray sorted = predictions.getRow(i).dup();
                    for (int j = 0; j < k; j++) {
                        int maxIdx = sorted.argMax().getInt(0);
                        if (maxIdx == trueClass) {
                            correct++;
                            break;
                        }
                        sorted.putScalar(maxIdx, Float.NEGATIVE_INFINITY);
                    }
                }
                return (double) correct / batchSz;
            };

            // Merge lambda: concatenate results from parallel evaluations
            MergeLambda<Double> mergeLambda = CustomEvaluation.mergeConcatenate();

            // Create custom evaluation
            CustomEvaluation<Double> customEval = new CustomEvaluation<>(topKAccuracyLambda, mergeLambda);

            // Define metric: average of all batch results
            CustomEvaluation.Metric<Double> top3AccMetric = CustomEvaluation.Metric.doubleAverage(false);

            // Evaluate
            testIter.reset();
            while (testIter.hasNext()) {
                DataSet batch = testIter.next();
                INDArray predictions = model.output(batch.getFeatures());
                customEval.eval(batch.getLabels(), predictions, null, null);
            }

            double top3Accuracy = customEval.getValue(top3AccMetric);
            System.out.println("  Top-3 accuracy: " + top3Accuracy);

            // Another custom metric: confidence gap (max prob - second max prob)
            EvaluationLambda<Double> confidenceGapLambda = (labels, predictions, mask, metadata) -> {
                double totalGap = 0;
                int batchSz = (int) predictions.size(0);
                for (int i = 0; i < batchSz; i++) {
                    INDArray row = predictions.getRow(i).dup();
                    int maxIdx = row.argMax().getInt(0);
                    double maxProb = row.getDouble(maxIdx);
                    row.putScalar(maxIdx, Float.NEGATIVE_INFINITY);
                    double secondMax = row.maxNumber().doubleValue();
                    totalGap += (maxProb - secondMax);
                }
                return totalGap / batchSz;
            };

            CustomEvaluation<Double> gapEval = new CustomEvaluation<>(confidenceGapLambda,
                    CustomEvaluation.mergeConcatenate());

            testIter.reset();
            while (testIter.hasNext()) {
                DataSet batch = testIter.next();
                INDArray predictions = model.output(batch.getFeatures());
                gapEval.eval(batch.getLabels(), predictions, null, null);
            }

            double avgGap = gapEval.getValue(CustomEvaluation.Metric.doubleAverage(false));
            System.out.println("  Average confidence gap: " + String.format("%.4f", avgGap));
            System.out.println("  (higher = model more decisive between top-2 predictions)");
        }

        // ============================================================
        // 6. COMBINING EVALUATIONS
        // ============================================================
        System.out.println("\n=== Combined Evaluation Pass ===");
        {
            // Evaluate all metrics in a single pass over the test set
            Evaluation eval = new Evaluation(10);
            ROCMultiClass rocMC = new ROCMultiClass();
            EvaluationCalibration evalCal = new EvaluationCalibration();

            testIter.reset();
            while (testIter.hasNext()) {
                DataSet batch = testIter.next();
                INDArray predictions = model.output(batch.getFeatures());
                INDArray labels = batch.getLabels();

                // All evaluators can process the same predictions
                eval.eval(labels, predictions);
                rocMC.eval(labels, predictions);
                evalCal.eval(labels, predictions);
            }

            System.out.println("  Overall accuracy:   " + String.format("%.4f", eval.accuracy()));
            System.out.println("  Weighted F1:        " + String.format("%.4f", eval.f1()));

            double macroAuroc = 0;
            for (int i = 0; i < 10; i++) {
                macroAuroc += rocMC.calculateAUC(i);
            }
            System.out.println("  Macro-avg AUROC:    " + String.format("%.4f", macroAuroc / 10));
            System.out.println("  Calibration classes: " + evalCal.numClasses());
        }

        System.out.println("\nAll evaluation metrics demonstrated successfully.");
    }
}
