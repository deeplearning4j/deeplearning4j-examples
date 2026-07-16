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

package org.deeplearning4j.examples.quickstart.features.listeners;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.*;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.evaluation.classification.Evaluation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Training Listeners and Checkpointing Example.
 *
 * DL4J provides training listeners for monitoring, evaluation, and checkpointing
 * during model training. This example demonstrates:
 *
 * <h3>Core Listeners:</h3>
 * <ul>
 *   <li>ScoreIterationListener — Log training loss every N iterations</li>
 *   <li>PerformanceListener — Track throughput (samples/sec, batches/sec, GC, ETL)</li>
 *   <li>EvaluativeListener — Run evaluation on test set at intervals</li>
 *   <li>CollectScoresListener — Collect scores for later analysis/plotting</li>
 *   <li>CheckpointListener — Save model checkpoints to disk</li>
 * </ul>
 *
 * <h3>CheckpointListener Features:</h3>
 * <pre>
 * CheckpointListener.builder(checkpointDir)
 *   .saveEveryEpoch()                    // After each epoch
 *   .saveEveryNEpochs(5)                 // Every 5 epochs
 *   .saveEveryNIterations(1000)          // Every 1000 iterations
 *   .saveEvery(30, TimeUnit.MINUTES)     // Time-based
 *   .keepAll()                           // Keep all checkpoints
 *   .keepLast(3)                         // Keep only last 3
 *   .keepLastAndEvery(3, 10)             // Keep last 3 + every 10th
 *   .deleteExisting(true)               // Clean up old checkpoints
 *   .logSaving(true)                    // Log checkpoint events
 *   .build()
 *
 * // Loading checkpoints
 * List&lt;Checkpoint&gt; available = CheckpointListener.availableCheckpoints(dir);
 * Checkpoint last = CheckpointListener.lastCheckpoint(dir);
 * MultiLayerNetwork loaded = CheckpointListener.loadLastCheckpointMLN(dir);
 * ComputationGraph loaded = CheckpointListener.loadLastCheckpointCG(dir);
 * </pre>
 */
public class TrainingListenersExample {
    private static final Logger log = LoggerFactory.getLogger(TrainingListenersExample.class);

    public static void main(String[] args) throws Exception {
        int batchSize = 64;
        int nEpochs = 2;
        int seed = 123;

        log.info("Load data...");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);

        log.info("Build model...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                .layer(new DenseLayer.Builder().nIn(784).nOut(256)
                        .activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nOut(128)
                        .activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10).activation(Activation.SOFTMAX).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // =====================================================================
        // 1. ScoreIterationListener — Log loss every N iterations
        // =====================================================================
        log.info("=== Listener: ScoreIterationListener ===");
        // Logs the training score (loss) every N iterations
        ScoreIterationListener scoreListener = new ScoreIterationListener(50);

        // =====================================================================
        // 2. PerformanceListener — Track throughput and system metrics
        // =====================================================================
        log.info("=== Listener: PerformanceListener ===");
        // Reports samples/sec, batches/sec, and optionally GC stats + ETL time
        PerformanceListener perfListener = new PerformanceListener.Builder()
                .reportScore(true)       // Include training score
                .reportSample(true)      // Samples per second
                .reportBatch(true)       // Batches per second
                .reportIteration(true)   // Iteration number
                .reportTime(true)        // Wall clock time
                .reportETL(true)         // Extract-Transform-Load time
                .setFrequency(50)        // Report every 50 iterations
                .build();

        // =====================================================================
        // 3. EvaluativeListener — Run evaluation on test set
        // =====================================================================
        log.info("=== Listener: EvaluativeListener ===");
        // Runs evaluation on the test set at specified frequency
        // InvocationType.EPOCH_END = evaluate at the end of each epoch
        EvaluativeListener evalListener = new EvaluativeListener(
                mnistTest,                    // Test dataset
                1,                            // Every 1 epoch
                InvocationType.EPOCH_END       // When to evaluate
        );

        // =====================================================================
        // 4. CollectScoresListener — Collect scores for plotting
        // =====================================================================
        log.info("=== Listener: CollectScoresListener ===");
        // Collects scores internally so they can be retrieved after training
        // for plotting loss curves, etc.
        CollectScoresIterationListener collectListener =
                new CollectScoresIterationListener(10); // Collect every 10 iterations

        // =====================================================================
        // 5. CheckpointListener — Save model checkpoints
        // =====================================================================
        log.info("=== Listener: CheckpointListener ===");

        File checkpointDir = new File(System.getProperty("java.io.tmpdir"), "dl4j-checkpoint-example");
        CheckpointListener checkpointListener = new CheckpointListener.Builder(checkpointDir)
                .saveEveryEpoch()            // Save at end of each epoch
                .keepLast(3)                 // Keep only last 3 checkpoints
                .deleteExisting(true)        // Clean up previous checkpoints
                .logSaving(true)             // Log checkpoint saves
                .build();

        // =====================================================================
        // 6. Combine all listeners and train
        // =====================================================================
        log.info("=== Training with all listeners ===");

        model.setListeners(
                scoreListener,
                perfListener,
                evalListener,
                collectListener,
                checkpointListener
        );

        model.fit(mnistTrain, nEpochs);

        // =====================================================================
        // 7. Access collected scores
        // =====================================================================
        log.info("=== Collected scores ===");
        log.info("Number of collected scores: {}", collectListener.getScoreVsIter().getScores().size());

        // =====================================================================
        // 8. Load from checkpoint
        // =====================================================================
        log.info("=== Loading from checkpoint ===");
        if (CheckpointListener.lastCheckpoint(checkpointDir) != null) {
            MultiLayerNetwork loaded = CheckpointListener.loadLastCheckpointMLN(checkpointDir);
            log.info("Loaded model from checkpoint. Params: {}", loaded.numParams());

            // Run evaluation on loaded model
            Evaluation eval = loaded.evaluate(mnistTest);
            log.info("Loaded model accuracy: {}", eval.accuracy());
        }

        log.info("Checkpoint directory: {}", checkpointDir.getAbsolutePath());
        log.info("**************** Training Listeners Example finished ********************");
    }
}
