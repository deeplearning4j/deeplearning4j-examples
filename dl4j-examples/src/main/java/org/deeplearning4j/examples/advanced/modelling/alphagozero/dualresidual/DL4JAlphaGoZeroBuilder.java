/*******************************************************************************
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

package org.deeplearning4j.examples.advanced.modelling.alphagozero.dualresidual;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex.Op;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.HashMap;
import java.util.Map;


/**
 * Provides input blocks for dual residual or convolutional neural networks
 * for Go move prediction.
 *
 * @author Max Pumperla
 */
class DL4JAlphaGoZeroBuilder {

    private ComputationGraphConfiguration.GraphBuilder conf;
    private int[] strides;
    private int[] kernelSize;
    private ConvolutionMode convolutionMode;

    public DL4JAlphaGoZeroBuilder(int[] kernel, int[] strides, ConvolutionMode mode) {

        this.kernelSize = kernel;
        this.strides = strides;
        this.convolutionMode = mode;

        this.conf =  new NeuralNetConfiguration.Builder()
            .updater(new Sgd())
            .weightInit(WeightInit.LECUN_NORMAL)
            .graphBuilder().setInputTypes(InputType.convolutional(19, 19, 11));
    }

    public DL4JAlphaGoZeroBuilder() {
        this(new int[] {3, 3},  new int[] {1, 1}, ConvolutionMode.Same);
    }

    public void addInputs(String name) {
        conf.addInputs(name);
    }

    public void addOutputs(String... names) {
        conf.setOutputs(names);
    }

    public ComputationGraphConfiguration buildAndReturn() { return conf.build(); }


    /**Building block for AGZ residual blocks.
     * conv2d -> batch norm -> ReLU
     */
    public String addConvBatchNormBlock(String blockName, String inName,int nIn,
                              boolean useActivation) {
        String convName = "conv_" + blockName;
        String bnName = "batch_norm_" + blockName;
        String actName = "relu_" + blockName;

        conf.addLayer(convName, new ConvolutionLayer.Builder().kernelSize(kernelSize)
        .stride(strides).convolutionMode(convolutionMode).nIn(nIn).nOut(256).build(), inName);
        conf.addLayer(bnName, new BatchNormalization.Builder().nOut(256).build(), convName);

        if (useActivation) {
            conf.addLayer(actName, new ActivationLayer.Builder().activation(Activation.RELU).build(), bnName);
            return actName;
        } else
            return bnName;
    }

    /**Residual block for AGZ. Takes two conv-bn-relu blocks
     * and adds them to the original input.
     */
    public String addResidualBlock(int blockNumber,String inName) {
        String firstBlock = "residual_1_" + blockNumber;
        String firstOut = "relu_residual_1_" + blockNumber;
        String secondBlock = "residual_2_" + blockNumber;
        String mergeBlock = "add_" + blockNumber;
        String actBlock = "relu_" + blockNumber;

        String firstBnOut =
            addConvBatchNormBlock(firstBlock, inName, 256, true);
        String secondBnOut =
            addConvBatchNormBlock(secondBlock, firstOut, 256, false);
        conf.addVertex(mergeBlock, new ElementWiseVertex(Op.Add), firstBnOut, secondBnOut);
        conf.addLayer(actBlock, new ActivationLayer.Builder().activation(Activation.RELU).build(), mergeBlock);
        return actBlock;
    }

    /**
     * Building a tower of residual blocks.
     */
    public String addResidualTower(int numBlocks, String inName) {
        String name = inName;
        for (int i = 0; i < numBlocks; i++) {
            name = addResidualBlock(i, name);
        }
        return name;
    }

    /**
     * Building a tower of convolutional blocks.
     */
    public String addConvolutionalTower(int numBlocks, String inName) {
        String name = inName;
        for (int i = 0; i < numBlocks; i++) {
            name = addConvBatchNormBlock(String.valueOf(i), name, 256, true);
        }
        return name;
    }

    /**
     * Policy head, predicts next moves (including passing), so
     * outputs a vector of 19*19 + 1 = 362 values.
     */
    public String addPolicyHead(String inName, boolean useActivation) {
        String convName = "policy_head_conv_";
        String bnName = "policy_head_batch_norm_";
        String actName = "policy_head_relu_";
        String denseName = "policy_head_output_";

        conf.addLayer(convName, new ConvolutionLayer.Builder().kernelSize(kernelSize).stride(strides)
        .convolutionMode(convolutionMode).nOut(2).nIn(256).build(), inName);
        conf.addLayer(bnName, new BatchNormalization.Builder().nOut(2).build(), convName);
        conf.addLayer(actName, new ActivationLayer.Builder().activation(Activation.RELU).build(), bnName);
        conf.addLayer(denseName, new OutputLayer.Builder().nIn(2 * 19 * 19).nOut(19 * 19 + 1).build(), actName);

        Map<String, InputPreProcessor> preProcessorMap = new HashMap<String, InputPreProcessor>();
        preProcessorMap.put(denseName, new CnnToFeedForwardPreProcessor(19, 19, 2));
        conf.setInputPreProcessors(preProcessorMap);
        return denseName;
    }

    /**
     * Value head, estimates how valuable the current
     * board position is.
     */
    public String addValueHead(String inName, boolean useActivation) {
        String convName = "value_head_conv_";
        String bnName = "value_head_batch_norm_";
        String actName = "value_head_relu_";
        String denseName = "value_head_dense_";
        String outputName = "value_head_output_";

        conf.addLayer(convName, new ConvolutionLayer.Builder().kernelSize(kernelSize).stride(strides)
        .convolutionMode(convolutionMode).nOut(1).nIn(256).build(), inName);
        conf.addLayer(bnName, new BatchNormalization.Builder().nOut(1).build(), convName);
        conf.addLayer(actName, new ActivationLayer.Builder().activation(Activation.RELU).build(), bnName);
        conf.addLayer(denseName, new DenseLayer.Builder().nIn(19 * 19).nOut(256).build(), actName);
        Map<String, InputPreProcessor> preProcessorMap = new HashMap<String, InputPreProcessor>();
        preProcessorMap.put(denseName, new CnnToFeedForwardPreProcessor(19, 19, 1));
        conf.setInputPreProcessors(preProcessorMap);
        conf.addLayer(outputName, new OutputLayer.Builder(LossFunctions.LossFunction.XENT).activation(Activation.SIGMOID).nIn(256).nOut(1).build(), denseName);
        return outputName;
    }

}
