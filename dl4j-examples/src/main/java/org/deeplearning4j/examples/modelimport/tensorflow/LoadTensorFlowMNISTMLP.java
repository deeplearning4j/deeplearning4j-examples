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

package org.deeplearning4j.examples.modelimport.tensorflow;

import org.nd4j.autodiff.execution.NativeGraphExecutioner;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * This example demonstrates how to loads a simple Tensorflow graph into SameDiff, nd4j's autodiff framework and execute it
 * SameDiff supports two styles of execution
 * 1) Libnd4j execution mode:
 * Similar to Tensorflow. The graph representation is converted into flat buffers and pushed down to native code.
 * 2) SameDiff execution mode:
 * Similar to Pytorch
 * <p>
 * NOTES:
 * The tensorflow import feature is currently a technology preview and in alpha
 * For now *only* the CPU backend is supported with a focus being on running inference
 * Currently a subset of lower level ops are supported i.e higher level abstractions (like tf.layers.dense) are not tested/supported
 * Expanded set of ops will be supported in the upcoming release
 * TensorFlow graphs need to be "frozen" and saved to a protobuf before import.
 * "freezing" refers to the process of converting all variables in the graph to constant and pruning the graph of all nodes/ops not necessary for inference.
 * The TensorFlow documentation on freezing graphs: https://www.tensorflow.org/extend/tool_developers/#freezing
 *
 * @author susaneraly
 */
public class LoadTensorFlowMNISTMLP {

    //Python code for this can be found in resources/import/tensorflow under generate_model.py and freeze_model_after.py
    //Input node/Placeholder in this graph is names "input"
    //Output node/op in this graph is names "output"
    public final static String BASE_DIR = "modelimport/tensorflow";

    public static void main(String[] args) throws Exception {
        final String FROZEN_MLP = new ClassPathResource(BASE_DIR + "/frozen_model.pb").getFile().getPath();

        //Load placeholder inputs and corresponding predictions generated from tensorflow
        Map<String, INDArray> inputsPredictions = readPlaceholdersAndPredictions();

        //Load the graph into samediff
        SameDiff graph = TFGraphMapper.getInstance().importGraph(new File(FROZEN_MLP));
        //libnd4j executor
        //running with input_a array expecting to get prediction_a
        graph.associateArrayWithVariable(inputsPredictions.get("input_a"), graph.variableMap().get("input"));
        NativeGraphExecutioner executioner = new NativeGraphExecutioner();
        INDArray[] results = executioner.executeGraph(graph); //returns an array of the outputs
        INDArray libnd4jPred = results[0];
        System.out.println("LIBND4J exec prediction for input_a:\n" + libnd4jPred);
        if (libnd4jPred.equals(inputsPredictions.get("prediction_a"))) {
            //this is true and therefore predictions are equal
            System.out.println("Predictions are equal to tensorflow");
        } else {
            throw new RuntimeException("Predictions don't match!");
        }

        //Now to run with the samediff executor, with input_b array expecting to get prediction_b
        SameDiff graphSD = TFGraphMapper.getInstance().importGraph(new File(FROZEN_MLP)); //Reimport graph here, necessary for the 1.0 alpha release
        graphSD.associateArrayWithVariable(inputsPredictions.get("input_b"), graph.variableMap().get("input"));
        INDArray samediffPred = graphSD.execAndEndResult();
        System.out.println("SameDiff exec prediction for input_b:\n" + samediffPred);
        if (samediffPred.equals(inputsPredictions.get("prediction_b"))) {
            //this is true and therefore predictions are equal
            System.out.println("Predictions are equal to tensorflow");
        }
        //add to graph to demonstrate pytorch like capability
        System.out.println("Adding new op to graph..");
        SDVariable linspaceConstant = graphSD.var("linspace", Nd4j.linspace(1, 10, 10));
        SDVariable totalOutput = graphSD.getVariable("output").add(linspaceConstant);
        INDArray totalOutputArr = totalOutput.eval();
        System.out.println(totalOutputArr);

    }

    //A simple helper function to load the inputs and corresponding outputs generated from tensorflow
    //Two cases: {input_a,prediction_a} and {input_b,prediction_b}
    protected static Map<String, INDArray> readPlaceholdersAndPredictions() throws IOException {
        String[] toReadList = {"input_a", "input_b", "prediction_a", "prediction_b"};
        Map<String, INDArray> arraysFromPython = new HashMap<>();
        for (int i = 0; i < toReadList.length; i++) {
            String varShapePath = new ClassPathResource(BASE_DIR + "/" + toReadList[i] + ".shape").getFile().getPath();
            String varValuePath = new ClassPathResource(BASE_DIR + "/" + toReadList[i] + ".csv").getFile().getPath();
            int[] varShape = Nd4j.readNumpy(varShapePath, ",").data().asInt();
            float[] varContents = Nd4j.readNumpy(varValuePath).data().asFloat();
            arraysFromPython.put(toReadList[i], Nd4j.create(varContents).reshape(varShape));
        }
        return arraysFromPython;
    }
}
