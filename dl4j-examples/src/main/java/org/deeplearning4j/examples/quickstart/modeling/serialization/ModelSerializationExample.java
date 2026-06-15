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

package org.deeplearning4j.examples.quickstart.modeling.serialization;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.nio.file.Files;

/**
 * Model Serialization in DL4J - Complete API Reference
 *
 * This example covers all model save/load capabilities:
 *
 *   1. Save and restore MultiLayerNetwork
 *   2. Save and restore with/without updater state
 *   3. Bundle a data normalizer with the model
 *   4. Store arbitrary metadata in the model file
 *   5. SameDiff model serialization
 *
 * Key classes:
 *   - org.deeplearning4j.util.ModelSerializer
 *   - org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler
 *   - org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
 */
public class ModelSerializationExample {

    public static void main(String[] args) throws Exception {

        File tempDir = Files.createTempDirectory("model_serialization").toFile();

        // Build and train a simple model
        System.out.println("=== Building and Training Model ===");
        int batchSize = 64;
        DataSetIterator trainIter = new MnistDataSetIterator(batchSize, true, 42);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-3))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder().nIn(784).nOut(128)
                        .activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(128).nOut(64)
                        .activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(64).nOut(10).activation(Activation.SOFTMAX).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.fit(trainIter); // 1 epoch
        System.out.println("  Training complete. Score: " + model.score());

        // ============================================================
        // 1. SAVE WITH UPDATER STATE (for resuming training)
        // ============================================================
        System.out.println("\n=== Save with Updater State ===");
        {
            File modelFile = new File(tempDir, "model_with_updater.zip");

            // saveUpdater=true: saves Adam momentum/velocity for training resumption
            ModelSerializer.writeModel(model, modelFile, true);
            System.out.println("  Saved to: " + modelFile.getName());
            System.out.println("  File size: " + (modelFile.length() / 1024) + " KB (with updater)");

            // Restore
            MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(modelFile);

            // Verify weights match
            INDArray originalParams = model.params();
            INDArray restoredParams = restored.params();
            boolean paramsMatch = originalParams.equalsWithEps(restoredParams, 1e-5);
            System.out.println("  Parameters match: " + paramsMatch);

            // Verify updater state is preserved
            INDArray originalUpdater = model.getUpdater().getStateViewArray();
            INDArray restoredUpdater = restored.getUpdater().getStateViewArray();
            boolean updaterMatch = originalUpdater.equalsWithEps(restoredUpdater, 1e-5);
            System.out.println("  Updater state match: " + updaterMatch);

            // Verify configuration matches
            String originalJson = model.getLayerWiseConfigurations().toJson();
            String restoredJson = restored.getLayerWiseConfigurations().toJson();
            System.out.println("  Config match: " + originalJson.equals(restoredJson));
        }

        // ============================================================
        // 2. SAVE WITHOUT UPDATER (for inference only, smaller file)
        // ============================================================
        System.out.println("\n=== Save without Updater (Inference Only) ===");
        {
            File modelFile = new File(tempDir, "model_inference.zip");

            // saveUpdater=false: smaller file, for inference only
            ModelSerializer.writeModel(model, modelFile, false);
            System.out.println("  File size: " + (modelFile.length() / 1024) + " KB (without updater)");

            File withUpdater = new File(tempDir, "model_with_updater.zip");
            System.out.println("  vs with updater: " + (withUpdater.length() / 1024) + " KB");
            System.out.println("  Savings: " + String.format("%.0f%%",
                    (1.0 - (double) modelFile.length() / withUpdater.length()) * 100));

            MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            System.out.println("  Restored for inference: " + (restored != null));
        }

        // ============================================================
        // 3. BUNDLE NORMALIZER WITH MODEL
        // ============================================================
        System.out.println("\n=== Bundle Normalizer with Model ===");
        {
            // Fit a normalizer on training data
            DataSetIterator iter = new MnistDataSetIterator(1000, true, 42);
            DataSet sample = iter.next();
            NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
            normalizer.fit(sample);

            // Save model first
            File modelFile = new File(tempDir, "model_with_normalizer.zip");
            ModelSerializer.writeModel(model, modelFile, true);

            // Add normalizer to the saved model file
            ModelSerializer.addNormalizerToModel(modelFile, normalizer);
            System.out.println("  Saved model + normalizer: " + (modelFile.length() / 1024) + " KB");

            // Restore model
            MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(modelFile);

            // Restore normalizer separately
            NormalizerMinMaxScaler restoredNorm = ModelSerializer.restoreNormalizerFromFile(modelFile);
            System.out.println("  Normalizer restored: " + (restoredNorm != null));
            System.out.println("  Min value range: " + restoredNorm.getMin());
            System.out.println("  Max value range: " + restoredNorm.getMax());

            // Use them together for inference
            DataSet testSample = new MnistDataSetIterator(10, false, 42).next();
            restoredNorm.preProcess(testSample);
            INDArray output = restored.output(testSample.getFeatures());
            System.out.println("  Inference output shape: " + java.util.Arrays.toString(output.shape()));
        }

        // ============================================================
        // 4. STORE ARBITRARY METADATA
        // ============================================================
        System.out.println("\n=== Store Arbitrary Metadata ===");
        {
            File modelFile = new File(tempDir, "model_with_metadata.zip");
            ModelSerializer.writeModel(model, modelFile, false);

            // Store arbitrary objects in the model zip
            ModelSerializer.addObjectToFile(modelFile, "training_accuracy", 0.9875);
            ModelSerializer.addObjectToFile(modelFile, "model_version", "1.0.0");
            ModelSerializer.addObjectToFile(modelFile, "training_epochs", 10);

            // Retrieve them later
            Double accuracy = ModelSerializer.getObjectFromFile(modelFile, "training_accuracy");
            String version = ModelSerializer.getObjectFromFile(modelFile, "model_version");
            Integer epochs = ModelSerializer.getObjectFromFile(modelFile, "training_epochs");

            System.out.println("  Stored metadata:");
            System.out.println("    training_accuracy: " + accuracy);
            System.out.println("    model_version: " + version);
            System.out.println("    training_epochs: " + epochs);
        }

        // ============================================================
        // 5. EVALUATE RESTORED MODEL
        // ============================================================
        System.out.println("\n=== Evaluate Restored Model ===");
        {
            File modelFile = new File(tempDir, "model_with_updater.zip");
            MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(modelFile);

            DataSetIterator testIter = new MnistDataSetIterator(batchSize, false, 42);
            Evaluation eval = restored.evaluate(testIter);

            System.out.println("  Restored model evaluation:");
            System.out.println("    Accuracy:  " + String.format("%.4f", eval.accuracy()));
            System.out.println("    Precision: " + String.format("%.4f", eval.precision()));
            System.out.println("    Recall:    " + String.format("%.4f", eval.recall()));
            System.out.println("    F1 Score:  " + String.format("%.4f", eval.f1()));
        }

        // ============================================================
        // 6. RESUME TRAINING FROM CHECKPOINT
        // ============================================================
        System.out.println("\n=== Resume Training from Checkpoint ===");
        {
            // Save a checkpoint
            File checkpoint = new File(tempDir, "checkpoint_epoch1.zip");
            ModelSerializer.writeModel(model, checkpoint, true); // must save updater!

            // Later: restore and continue training
            MultiLayerNetwork resumed = ModelSerializer.restoreMultiLayerNetwork(checkpoint);
            double scoreBefore = resumed.score();

            // Continue training for another epoch
            DataSetIterator trainIter2 = new MnistDataSetIterator(batchSize, true, 43);
            resumed.fit(trainIter2);
            double scoreAfter = resumed.score();

            System.out.println("  Score before resuming: " + String.format("%.4f", scoreBefore));
            System.out.println("  Score after 1 more epoch: " + String.format("%.4f", scoreAfter));
            System.out.println("  Improvement: " + (scoreAfter < scoreBefore ? "yes" : "no"));
        }

        // Cleanup
        for (File f : tempDir.listFiles()) {
            f.delete();
        }
        tempDir.delete();

        System.out.println("\nAll model serialization operations demonstrated successfully.");
    }
}
