package org.deeplearning4j.examples.convolution.alphagozero;

import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex.Op;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;

import java.util.List;


/**
 * Provides input blocks for dual residual or convolutional neural networks
 * for Go move prediction.
 *
 * @author Max Pumperla
 */
class DL4JAlphaGoZeroBuilder {

    private ComputationGraphConfiguration.GraphBuilder conf;

    public DL4JAlphaGoZeroBuilder() {
        this.conf =  new NeuralNetConfiguration.Builder()
            .updater(new Sgd())
            .weightInit(WeightInit.LECUN_NORMAL)
            .graphBuilder().setInputTypes(InputType.convolutional(19, 19, 11));
    }

    public void addInputs(String name) {
        conf.addInputs(name);
    }

    public void addOutputs(String... names) {
        conf.setOutputs(names);
    }

    public ComputationGraphConfiguration buildAndReturn() { return conf.build(); }

    public String addConvBatchNormBlock(String blockName, String inName,int nIn,
                              boolean useActivation, int[] kernelSize,
    int[] strides, ConvolutionMode convolutionMode) {
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

    def addResidualBlock(blockNumber: Int,
                         inName: String,
                         kernelSize: List[Int] = List(3, 3),
    strides: List[Int] = List(1, 1),
    convolutionMode: ConvolutionMode = ConvolutionMode.Same): String = {
        val firstBlock = "residual_1_" + blockNumber
        val firstOut = "relu_residual_1_" + blockNumber
        val secondBlock = "residual_2_" + blockNumber
        val mergeBlock = "add_" + blockNumber
        val actBlock = "relu_" + blockNumber

        val firstBnOut =
            addConvBatchNormBlock(firstBlock, inName, 256, useActivation = true, kernelSize, strides, convolutionMode)
        val secondBnOut =
            addConvBatchNormBlock(secondBlock, firstOut, 256, useActivation = false, kernelSize, strides, convolutionMode)
        conf.addVertex(mergeBlock, new ElementWiseVertex(Op.Add), firstBnOut, secondBnOut)
        conf.addLayer(actBlock, new ActivationLayer.Builder().activation(Activation.RELU).build(), mergeBlock)
        actBlock
    }

    def addResidualTower(numBlocks: Int,
                         inName: String,
                         kernelSize: List[Int] = List(3, 3),
    strides: List[Int] = List(1, 1),
    convolutionMode: ConvolutionMode = ConvolutionMode.Same): String = {
        var name = inName
        for (i <- 0 until numBlocks)
        name = addResidualBlock(i, name, kernelSize, strides, convolutionMode)
        name
    }

    def addConvolutionalTower(numBlocks: Int,
                              inName: String,
                              kernelSize: List[Int] = List(3, 3),
    strides: List[Int] = List(1, 1),
    convolutionMode: ConvolutionMode = ConvolutionMode.Same): Unit = {
        var name = inName
        for (i <- 0 until numBlocks)
        name = addConvBatchNormBlock(i.toString, name, 256, useActivation = true, kernelSize, strides, convolutionMode)
    }

    def addPolicyHead(inName: String,
                      useActivation: Boolean = true,
                      kernelSize: List[Int] = List(3, 3),
    strides: List[Int] = List(1, 1),
    convolutionMode: ConvolutionMode = ConvolutionMode.Same): String = {
        val convName = "policy_head_conv_"
        val bnName = "policy_head_batch_norm_"
        val actName = "policy_head_relu_"
        val denseName = "policy_head_output_"

        conf.addLayer(
            convName,
            new ConvolutionLayer.Builder()
                .kernelSize(kernelSize: _*)
        .stride(strides: _*)
        .convolutionMode(convolutionMode)
            .nOut(2)
            .nIn(256)
            .build(),
            inName
    )
        conf.addLayer(bnName, new BatchNormalization.Builder().nOut(2).build(), convName)
        conf.addLayer(actName, new ActivationLayer.Builder().activation(Activation.RELU).build(), bnName)
        conf.addLayer(denseName, new OutputLayer.Builder().nIn(2 * 19 * 19).nOut(19 * 19 + 1).build(), actName)
        conf.setInputPreProcessors(
            Map[String, InputPreProcessor](denseName -> new CnnToFeedForwardPreProcessor(19, 19, 2))
    )
        denseName
    }
    def addValueHead(inName: String,
                     useActivation: Boolean = true,
                     kernelSize: List[Int] = List(1, 1),
    strides: List[Int] = List(1, 1),
    convolutionMode: ConvolutionMode = ConvolutionMode.Same): String = {
        val convName = "value_head_conv_"
        val bnName = "value_head_batch_norm_"
        val actName = "value_head_relu_"
        val denseName = "value_head_dense_"
        val outputName = "value_head_output_"

        conf.addLayer(
            convName,
            new ConvolutionLayer.Builder()
                .kernelSize(kernelSize: _*)
        .stride(strides: _*)
        .convolutionMode(convolutionMode)
            .nOut(1)
            .nIn(256)
            .build(),
            inName
    )
        conf.addLayer(bnName, new BatchNormalization.Builder().nOut(1).build(), convName)
        conf.addLayer(actName, new ActivationLayer.Builder().activation(Activation.RELU).build(), bnName)
        conf.addLayer(denseName, new DenseLayer.Builder().nIn(19 * 19).nOut(256).build(), actName)
        conf.setInputPreProcessors(
            Map[String, InputPreProcessor](denseName -> new CnnToFeedForwardPreProcessor(19, 19, 1))
    )
        conf.addLayer(outputName, new OutputLayer.Builder().nIn(256).nOut(1).build(), denseName)
        outputName
    }

}
