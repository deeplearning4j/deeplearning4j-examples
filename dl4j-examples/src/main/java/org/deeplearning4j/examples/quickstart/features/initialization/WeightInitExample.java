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

package org.deeplearning4j.examples.quickstart.features.initialization;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitVarScalingNormalFanIn;
import org.deeplearning4j.nn.weights.WeightInitVarScalingNormalFanOut;
import org.deeplearning4j.nn.weights.WeightInitVarScalingNormalFanAvg;
import org.deeplearning4j.nn.weights.WeightInitVarScalingUniformFanIn;
import org.deeplearning4j.nn.weights.WeightInitVarScalingUniformFanOut;
import org.deeplearning4j.nn.weights.WeightInitVarScalingUniformFanAvg;
import org.deeplearning4j.nn.weights.WeightInitIdentity;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Weight Initialization in DL4J - Complete API Reference
 *
 * Proper weight initialization is critical for training deep networks.
 * Poor initialization can lead to vanishing/exploding gradients.
 *
 * Topics covered:
 *   1. Built-in WeightInit enum values
 *   2. VAR_SCALING variants (Keras/TensorFlow compatible)
 *   3. Custom IWeightInit implementation
 *   4. Identity initialization for residual connections
 *   5. Per-layer initialization strategies
 *   6. Practical guidelines
 *
 * WeightInit enum values:
 * +------------------------------+--------------------------------------------------+
 * | WeightInit                   | Description                                      |
 * +------------------------------+--------------------------------------------------+
 * | ZERO                         | All zeros (biases, masks)                        |
 * | ONES                         | All ones (gates, scales)                         |
 * | NORMAL                       | N(0, 1/sqrt(fanIn)) = LECUN_NORMAL               |
 * | LECUN_NORMAL                 | Same as NORMAL                                   |
 * | LECUN_UNIFORM                | U[-a, a], a = 3/sqrt(fanIn)                      |
 * | XAVIER                       | Glorot normal: N(0, 2/(fanIn+fanOut))            |
 * | XAVIER_UNIFORM               | Glorot uniform: U[-a, a], a = sqrt(6/(fin+fout)) |
 * | XAVIER_FAN_IN                | Same as NORMAL (N(0, 1/sqrt(fanIn)))             |
 * | XAVIER_LEGACY                | Legacy Xavier implementation                     |
 * | RELU                         | He normal: N(0, 2/fanIn) for ReLU                |
 * | RELU_UNIFORM                 | He uniform: U[-a, a], a = sqrt(6/fanIn) for ReLU |
 * | SIGMOID_UNIFORM              | Uniform optimized for sigmoid activations        |
 * | IDENTITY                     | Identity matrix (square layers, residual)        |
 * | DISTRIBUTION                 | Custom distribution                              |
 * | VAR_SCALING_NORMAL_FAN_IN    | TruncNormal, std = sqrt(1/fanIn)                 |
 * | VAR_SCALING_NORMAL_FAN_OUT   | TruncNormal, std = sqrt(1/fanOut)                |
 * | VAR_SCALING_NORMAL_FAN_AVG   | TruncNormal, std = sqrt(2/(fanIn+fanOut))        |
 * | VAR_SCALING_UNIFORM_FAN_IN   | Uniform, a = sqrt(3/fanIn)                       |
 * | VAR_SCALING_UNIFORM_FAN_OUT  | Uniform, a = sqrt(3/fanOut)                      |
 * | VAR_SCALING_UNIFORM_FAN_AVG  | Uniform, a = sqrt(6/(fanIn+fanOut))              |
 * +------------------------------+--------------------------------------------------+
 */
public class WeightInitExample {

    public static void main(String[] args) {

        // ============================================================
        // 1. COMMON INITIALIZATIONS
        // ============================================================
        System.out.println("=== Common Weight Initializations ===");
        {
            // Xavier/Glorot: best for sigmoid/tanh activations
            MultiLayerNetwork xavierNet = buildNetwork(WeightInit.XAVIER, Activation.TANH);
            printWeightStats("Xavier + Tanh", xavierNet);

            // Xavier Uniform variant
            MultiLayerNetwork xavierUniformNet = buildNetwork(WeightInit.XAVIER_UNIFORM, Activation.TANH);
            printWeightStats("Xavier Uniform + Tanh", xavierUniformNet);

            // He/RELU: best for ReLU family activations
            MultiLayerNetwork heNet = buildNetwork(WeightInit.RELU, Activation.RELU);
            printWeightStats("He (RELU) + ReLU", heNet);

            // LeCun: good for SELU activations
            MultiLayerNetwork lecunNet = buildNetwork(WeightInit.LECUN_NORMAL, Activation.IDENTITY);
            printWeightStats("LeCun Normal", lecunNet);
        }

        // ============================================================
        // 2. VAR_SCALING VARIANTS (Keras/TensorFlow compatible)
        // ============================================================
        System.out.println("\n=== VAR_SCALING Initializations ===");
        {
            // VAR_SCALING_NORMAL_FAN_IN: equivalent to tf.keras.initializers.VarianceScaling(
            //     scale=1.0, mode='fan_in', distribution='truncated_normal')
            MultiLayerNetwork vsnfi = buildNetwork(WeightInit.VAR_SCALING_NORMAL_FAN_IN, Activation.RELU);
            printWeightStats("VarScaling Normal FanIn", vsnfi);

            // VAR_SCALING_NORMAL_FAN_OUT: scale by 1/fanOut
            MultiLayerNetwork vsnfo = buildNetwork(WeightInit.VAR_SCALING_NORMAL_FAN_OUT, Activation.RELU);
            printWeightStats("VarScaling Normal FanOut", vsnfo);

            // VAR_SCALING_NORMAL_FAN_AVG: scale by 2/(fanIn+fanOut)
            MultiLayerNetwork vsnfa = buildNetwork(WeightInit.VAR_SCALING_NORMAL_FAN_AVG, Activation.TANH);
            printWeightStats("VarScaling Normal FanAvg", vsnfa);

            // Uniform variants
            MultiLayerNetwork vsufi = buildNetwork(WeightInit.VAR_SCALING_UNIFORM_FAN_IN, Activation.RELU);
            printWeightStats("VarScaling Uniform FanIn", vsufi);
        }

        // ============================================================
        // 3. VAR_SCALING WITH CUSTOM SCALE FACTOR
        // ============================================================
        System.out.println("\n=== VAR_SCALING with Custom Scale ===");
        {
            // IWeightInit classes accept an optional scale parameter
            // VarScaling with scale=2.0 for He initialization:
            // std = sqrt(scale / fanIn) = sqrt(2/fanIn)
            IWeightInit heEquivalent = new WeightInitVarScalingNormalFanIn(2.0);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .updater(new Adam(1e-3))
                    .list()
                    .layer(new DenseLayer.Builder()
                            .nIn(784).nOut(256)
                            .activation(Activation.RELU)
                            .weightInit(heEquivalent) // IWeightInit instance
                            .build())
                    .layer(new DenseLayer.Builder()
                            .nIn(256).nOut(128)
                            .activation(Activation.RELU)
                            .weightInit(new WeightInitVarScalingNormalFanOut(1.0))
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nIn(128).nOut(10)
                            .activation(Activation.SOFTMAX)
                            .weightInit(new WeightInitVarScalingUniformFanAvg(1.0))
                            .build())
                    .build();

            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();

            System.out.println("  Layer 0: VarScaling Normal FanIn (scale=2.0) - He equivalent");
            System.out.println("  Layer 1: VarScaling Normal FanOut (scale=1.0)");
            System.out.println("  Layer 2: VarScaling Uniform FanAvg (scale=1.0)");
            printWeightStats("Custom VarScaling", model);
        }

        // ============================================================
        // 4. IDENTITY INITIALIZATION
        // ============================================================
        System.out.println("\n=== Identity Initialization ===");
        {
            // Identity init: useful for residual connections (skip connection starts as identity)
            IWeightInit identityInit = new WeightInitIdentity();

            // Identity with scale factor
            IWeightInit scaledIdentity = new WeightInitIdentity(0.1);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .updater(new Adam(1e-3))
                    .list()
                    .layer(new DenseLayer.Builder()
                            .nIn(64).nOut(64)
                            .activation(Activation.RELU)
                            .weightInit(identityInit)
                            .build())
                    .layer(new DenseLayer.Builder()
                            .nIn(64).nOut(64)
                            .activation(Activation.RELU)
                            .weightInit(scaledIdentity) // scaled identity
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nIn(64).nOut(10)
                            .activation(Activation.SOFTMAX)
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .build();

            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();

            INDArray w0 = model.getLayer(0).getParam("W");
            System.out.println("  Identity layer 0 diagonal: " +
                    w0.getDouble(0, 0) + ", " + w0.getDouble(1, 1) + ", " + w0.getDouble(2, 2));
            System.out.println("  Identity layer 0 off-diag: " + w0.getDouble(0, 1));

            INDArray w1 = model.getLayer(1).getParam("W");
            System.out.println("  Scaled(0.1) identity diagonal: " +
                    w1.getDouble(0, 0) + ", " + w1.getDouble(1, 1));
        }

        // ============================================================
        // 5. CUSTOM WEIGHT INIT
        // ============================================================
        System.out.println("\n=== Custom Weight Initialization ===");
        {
            // Implement IWeightInit for custom initialization
            // Example: orthogonal initialization (good for RNNs)
            IWeightInit orthogonalInit = new IWeightInit() {
                @Override
                public INDArray init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
                    // Generate random matrix and compute its SVD for orthogonal init
                    long rows = shape[0];
                    long cols = shape.length > 1 ? shape[1] : 1;
                    INDArray random = Nd4j.randn(DataType.FLOAT, rows, cols);

                    // Simple approximation: normalize columns
                    for (int j = 0; j < cols; j++) {
                        INDArray col = random.getColumn(j);
                        double norm = col.norm2Number().doubleValue();
                        if (norm > 0) {
                            col.divi(norm);
                        }
                    }

                    INDArray flat = random.reshape(order, shape);
                    paramView.assign(flat);
                    return paramView;
                }
            };

            // Example: constant initialization
            IWeightInit smallConstInit = new IWeightInit() {
                @Override
                public INDArray init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
                    paramView.assign(0.01);
                    return paramView.reshape(order, shape);
                }
            };

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .updater(new Adam(1e-3))
                    .list()
                    .layer(new DenseLayer.Builder()
                            .nIn(784).nOut(256)
                            .activation(Activation.RELU)
                            .weightInit(orthogonalInit) // custom init
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nIn(256).nOut(10)
                            .activation(Activation.SOFTMAX)
                            .weightInit(WeightInit.XAVIER) // built-in init
                            .build())
                    .build();

            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();

            printWeightStats("Orthogonal Custom Init", model);
        }

        // ============================================================
        // 6. PER-LAYER INITIALIZATION STRATEGIES
        // ============================================================
        System.out.println("\n=== Per-Layer Initialization Strategy ===");
        {
            // Best practice: match initialization to activation function per layer
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .updater(new Adam(1e-3))
                    .list()
                    // ReLU layers: use He initialization
                    .layer(new DenseLayer.Builder()
                            .nIn(784).nOut(512)
                            .activation(Activation.RELU)
                            .weightInit(WeightInit.RELU) // He init
                            .build())
                    .layer(new DenseLayer.Builder()
                            .nIn(512).nOut(256)
                            .activation(Activation.RELU)
                            .weightInit(WeightInit.RELU) // He init
                            .build())
                    // Tanh layer: use Xavier
                    .layer(new DenseLayer.Builder()
                            .nIn(256).nOut(128)
                            .activation(Activation.TANH)
                            .weightInit(WeightInit.XAVIER) // Glorot
                            .build())
                    // Output: Xavier or small uniform
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nIn(128).nOut(10)
                            .activation(Activation.SOFTMAX)
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .build();

            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();

            System.out.println("  Layer 0 (ReLU):    He init");
            System.out.println("  Layer 1 (ReLU):    He init");
            System.out.println("  Layer 2 (Tanh):    Xavier init");
            System.out.println("  Layer 3 (Softmax): Xavier init");
            printWeightStats("Per-Layer Strategy", model);
        }

        // ============================================================
        // GUIDELINES
        // ============================================================
        System.out.println("\n=== Initialization Guidelines ===");
        System.out.println("  +------------------+---------------------------+");
        System.out.println("  | Activation       | Recommended Init          |");
        System.out.println("  +------------------+---------------------------+");
        System.out.println("  | ReLU, Leaky ReLU | RELU (He) or RELU_UNIFORM |");
        System.out.println("  | Tanh, Sigmoid    | XAVIER or XAVIER_UNIFORM  |");
        System.out.println("  | SELU             | LECUN_NORMAL              |");
        System.out.println("  | Linear/Identity  | XAVIER or NORMAL          |");
        System.out.println("  | Softmax (output) | XAVIER                    |");
        System.out.println("  | RNN layers       | XAVIER or custom orthog.  |");
        System.out.println("  | Residual skip     | IDENTITY                  |");
        System.out.println("  | Keras compat.     | VAR_SCALING_*             |");
        System.out.println("  +------------------+---------------------------+");

        System.out.println("\nAll weight initialization examples demonstrated successfully.");
    }

    private static MultiLayerNetwork buildNetwork(WeightInit weightInit, Activation activation) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-3))
                .weightInit(weightInit)
                .list()
                .layer(new DenseLayer.Builder().nIn(784).nOut(256).activation(activation).build())
                .layer(new DenseLayer.Builder().nIn(256).nOut(128).activation(activation).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(128).nOut(10).activation(Activation.SOFTMAX).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    private static void printWeightStats(String name, MultiLayerNetwork model) {
        INDArray w0 = model.getLayer(0).getParam("W");
        System.out.printf("  %-30s  mean=%.6f  std=%.6f  min=%.4f  max=%.4f%n",
                name,
                w0.meanNumber().doubleValue(),
                w0.stdNumber().doubleValue(),
                w0.minNumber().doubleValue(),
                w0.maxNumber().doubleValue());
    }
}
