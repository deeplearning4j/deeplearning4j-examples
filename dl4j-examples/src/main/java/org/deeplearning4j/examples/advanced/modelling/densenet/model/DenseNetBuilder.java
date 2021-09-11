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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

    public int getGrowthRate() {
        return growthRate;
    }

    public String initLayer(int kernel, int stride, int padding, int channels) {
        ConvolutionLayer convolutionLayer = new ConvolutionLayer.Builder()
            .name("initConv")
            .kernelSize(kernel, kernel)
            .stride(stride, stride)
            .padding(padding, padding)
            .nIn(channels)
            .nOut(growthRate * 2)
            .build();
        SubsamplingLayer subsamplingLayer = new Pooling2D.Builder(SubsamplingLayer.PoolingType.MAX)
            .name("initPool")
            .kernelSize(3, 3)
            .padding(0, 0)
            .build();

        conf.addLayer(convolutionLayer.getLayerName(), convolutionLayer, "input");
        conf.addLayer(subsamplingLayer.getLayerName(), subsamplingLayer, convolutionLayer.getLayerName());

        return subsamplingLayer.getLayerName();
    }

    public String addTransitionLayer(String name, long numIn, List<String> previousLayers) {
        BatchNormalization bnLayer = new BatchNormalization.Builder()
            .name(String.format("%s_%s", name, "bn"))
            .build();
        ConvolutionLayer layer1x1 = new ConvolutionLayer.Builder()
            .name(String.format("%s_%s", name, "conv"))
            .kernelSize(1, 1)
            .stride(1, 1)
            .padding(0, 0)
            .nOut(numIn / 2)
            .build();
        SubsamplingLayer subsamplingLayer = new Pooling2D.Builder(SubsamplingLayer.PoolingType.AVG)
            .name(String.format("%s_%s", name, "pool"))
            .kernelSize(2, 2)
            .padding(0, 0)
            .build();

        conf.addLayer(bnLayer.getLayerName(), bnLayer, previousLayers.toArray(new String[1]));
        conf.addLayer(layer1x1.getLayerName(), layer1x1, bnLayer.getLayerName());
        conf.addLayer(subsamplingLayer.getLayerName(), subsamplingLayer, layer1x1.getLayerName());

        return subsamplingLayer.getLayerName();
    }

    private ConvolutionLayer addDenseLayer(String name, String... previousLayers) {
        BatchNormalization bnLayer1 = new BatchNormalization.Builder()
            .name(String.format("%s_%s", name, "bn1"))
            .build();
        ConvolutionLayer layer1x1 = new ConvolutionLayer.Builder()
            .name(String.format("%s_%s", name, "con1x1"))
            .kernelSize(1, 1)
            .stride(1, 1)
            .padding(0, 0)
            .nOut(growthRate * 4)
            .build();
        BatchNormalization bnLayer2 = new BatchNormalization.Builder()
            .name(String.format("%s_%s", name, "bn2"))
            .build();
        ConvolutionLayer layer3x3 = new ConvolutionLayer.Builder()
            .name(String.format("%s_%s", name, "con3x3"))
            .kernelSize(3, 3)
            .stride(1, 1)
            .padding(1, 1)
            .nOut(growthRate)
            .build();

        if (useBottleNeck) {
            conf.addLayer(bnLayer1.getLayerName(), bnLayer1, previousLayers);
            conf.addLayer(layer1x1.getLayerName(), layer1x1, bnLayer1.getLayerName());
            conf.addLayer(bnLayer2.getLayerName(), bnLayer2, layer1x1.getLayerName());
        } else {
            conf.addLayer(bnLayer2.getLayerName(), bnLayer2, previousLayers);
        }
        conf.addLayer(layer3x3.getLayerName(), layer3x3, bnLayer2.getLayerName());

        return layer3x3;
    }

    protected List<String> buildDenseBlock(String blockName, int numLayers, String lastLayerName) {
        List<ConvolutionLayer> layers = new ArrayList<>();
        for (int i = 0; i < numLayers; ++i) {
            layers.add(addDenseLayer(String.format("%s_%s", blockName, i), increaseArray(lastLayerName, getLayerNames(layers))));
        }
        List<String> names = new ArrayList<>(Arrays.asList(getLayerNames(layers)));
        names.add(lastLayerName);
        return names;
    }

    public void addOutputLayer(int numIn, int numLabels, String... previousLayer) {
        GlobalPoolingLayer globalPoolingLayer = new GlobalPoolingLayer.Builder()
            .name("outputGPL")
            .poolingType(PoolingType.AVG)
            .collapseDimensions(false)
            .build();
        BatchNormalization bn2 = new BatchNormalization.Builder()
            .name("outputBn")
            .build();
        ConvolutionLayer convolutionLayer2 = new ConvolutionLayer.Builder()
            .name("outputConv")
            .kernelSize(1, 1)
            .stride(1, 1)
            .padding(0, 0)
            .nOut(numIn * 2)
            .build();
        OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .name("output")
            .nOut(numLabels)
            .activation(Activation.SOFTMAX)
            .build();

        conf.addLayer(globalPoolingLayer.getLayerName(), globalPoolingLayer, previousLayer);
        conf.addLayer(bn2.getLayerName(), bn2, globalPoolingLayer.getLayerName());
        conf.addLayer(convolutionLayer2.getLayerName(), convolutionLayer2, bn2.getLayerName());
        conf.addLayer(outputLayer.getLayerName(), outputLayer, convolutionLayer2.getLayerName());
    }

    private String[] increaseArray(String newLayer, String[] theArray) {
        String[] newArray = new String[theArray.length + 1];
        System.arraycopy(theArray, 0, newArray, 0, theArray.length);
        newArray[theArray.length] = newLayer;
        return newArray;
    }

    protected String[] getLayerNames(List<ConvolutionLayer> theArray) {
        List<String> names = new ArrayList<>();
        if (theArray != null) {
            for (ConvolutionLayer convolutionLayer : theArray) {
                names.add(convolutionLayer.getLayerName());
            }
        }
        return names.toArray(new String[names.size()]);
    }
}
