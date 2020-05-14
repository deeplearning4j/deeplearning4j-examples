/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.advanced.modelling.densenet.model;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

public class DenseNetBuilder {

    private ComputationGraphConfiguration.GraphBuilder conf;
    private int growthRate;
    private boolean useBottleNeck;

    public DenseNetBuilder(int height, int width, int channels, long seed, int growthRate, boolean useBottleNeck) {
        this.growthRate = growthRate;
        this.useBottleNeck = useBottleNeck;
        this.conf = new NeuralNetConfiguration.Builder()
            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
            .seed(seed)
            .weightInit(WeightInit.RELU)
            .activation(Activation.LEAKYRELU)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Adam(new StepSchedule(ScheduleType.EPOCH, 5e-5, 0.5, 5)))
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
            .l2(1e-4)
            .graphBuilder()
            .addInputs("input")
            .setInputTypes(InputType.convolutional(height, width, channels))
            .setOutputs("output");
    }

    public ComputationGraph getModel() {
        conf.build();
        ComputationGraph model = new ComputationGraph(conf.build());
        model.init();
        return model;
    }

    public String initLayer(int kernel, int stride, int padding, int channels) {
        String init = "initConv";
        String initPool = "initPool";
        conf.addLayer(init, new ConvolutionLayer.Builder()
            .kernelSize(kernel, kernel)
            .stride(stride, stride)
            .padding(padding, padding)
            .nIn(channels)
            .nOut(growthRate * 2)
            .build(), "input");
        conf.addLayer(initPool, new Pooling2D.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2, 2)
            .padding(0, 0)
            .build(), init);
        return initPool;
    }

    public String addTransitionLayer(String transitionName, int numIn, String... previousBlock) {
        String bnName = "bn_" + transitionName;
        String convName = "conv_" + transitionName;
        String poolName = "pool_" + transitionName;

        conf.addLayer(bnName, new BatchNormalization.Builder()
            .build(), previousBlock);
        conf.addLayer(convName, new ConvolutionLayer.Builder()
            .kernelSize(1, 1)
            .stride(1, 1)
            .padding(0, 0)
            .nOut(numIn / 2)
            .build(), bnName);
        conf.addLayer(poolName, new Pooling2D.Builder(SubsamplingLayer.PoolingType.AVG)
            .kernelSize(2, 2)
            .padding(0, 0)
            .build(), convName);
        ;

        return poolName;
    }

    private String[] addDenseLayer(boolean firstLayerInBlock, String layerName, String... previousLayers) {
        String bnName = "bn1_" + layerName;
        String convName = "conv1_" + layerName;
        String bnName2 = "bn2_" + layerName;
        String convName2 = "conv2_" + layerName;

        if (useBottleNeck) {
            conf.addLayer(bnName, new BatchNormalization.Builder()
                .build(), previousLayers);
            conf.addLayer(convName, new ConvolutionLayer.Builder()
                .kernelSize(1, 1)
                .stride(1, 1)
                .padding(0, 0)
                .nOut(growthRate * 2)
                .build(), bnName);
        }

        conf.addLayer(bnName2, new BatchNormalization.Builder()
            .build(), useBottleNeck ? new String[]{convName} : previousLayers);
        conf.addLayer(convName2, new ConvolutionLayer.Builder()
            .kernelSize(3, 3)
            .stride(1, 1)
            .padding(1, 1)
            .nOut(growthRate)
            .build(), bnName2);

        return firstLayerInBlock ? new String[]{convName2} : increaseArray(convName2, previousLayers);
    }

    public String[] addDenseBlock(int numLayers, boolean first, String blockName, String[] previousLayer) {
        String layerName = blockName + "_lay" + numLayers;
        String[] layersInput = addDenseLayer(first, layerName, previousLayer);
        --numLayers;
        if (numLayers > 0) {
            layersInput = addDenseBlock(numLayers, false, blockName, layersInput);
        }
        return first ? increaseArray(previousLayer[0], layersInput) : layersInput;
    }

    public void addOutputLayer(int height, int width, int numIn, int numLabels, String... previousLayer) {
        conf.addLayer("lastBatch", new BatchNormalization.Builder()
            .build(), previousLayer);
        conf.addLayer("GAP", new GlobalPoolingLayer.Builder()
            .poolingType(PoolingType.AVG)
            .build(), "lastBatch");
        conf.addLayer("dense", new DenseLayer.Builder()
            .nIn(numIn)
            .nOut(1024)
            .build(), "GAP");
        conf.addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(numLabels)
            .activation(Activation.SOFTMAX)
            .build(), "dense");
    }

    private String[] increaseArray(String newLayer, String... theArray) {
        String[] newArray = new String[theArray.length + 1];
        System.arraycopy(theArray, 0, newArray, 0, theArray.length);
        newArray[theArray.length] = newLayer;
        return newArray;
    }
}
