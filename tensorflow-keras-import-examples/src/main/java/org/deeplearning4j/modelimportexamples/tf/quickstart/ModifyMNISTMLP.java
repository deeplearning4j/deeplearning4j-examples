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

package org.deeplearning4j.modelimportexamples.tf.quickstart;

import org.deeplearning4j.modelimportexamples.utilities.DownloaderUtility;
import org.nd4j.autodiff.execution.NativeGraphExecutioner;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

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
public class ModifyMNISTMLP {

    //Python code for this can be found in ~/dl4j-examples-data/dl4j-examples/modelimport/tensorflow under generate_model.py and freeze_model_after.py
    //Input node/Placeholder in this graph is names "input"
    //Output node/op in this graph is names "output"
    public static String dataLocalPath;


    public static void main(String[] args) throws Exception {
        dataLocalPath = DownloaderUtility.MODELIMPORT.Download() + "/tensorflow";
        final String FROZEN_MLP = new File(dataLocalPath, "frozen_model.pb").getAbsolutePath();

        //Load placeholder inputs and corresponding predictions generated from tensorflow
        List<Pair<INDArray, INDArray>> inputoutputPairs = readPlaceholdersAndPredictions();

        //Load the graph into samediff
        SameDiff graph = TFGraphMapper.importGraph(new File(FROZEN_MLP));

        //libnd4j executor
        //running with input_a array expecting to get prediction_a
        INDArray placeholderValue = inputoutputPairs.get(0).getFirst();
        INDArray TFPrediction = inputoutputPairs.get(0).getSecond();
        graph.associateArrayWithVariable(placeholderValue, graph.variableMap().get("input"));
        NativeGraphExecutioner executioner = new NativeGraphExecutioner();
        INDArray[] results = executioner.executeGraph(graph); //returns an array of the outputs
        INDArray libnd4jPrediction = results[0];
        System.out.println("LIBND4J exec prediction for input_a:\n" + libnd4jPrediction);
        if (libnd4jPrediction.equals(TFPrediction)) {
            //this is true and therefore predictions are equal
            System.out.println("Predictions are equal to tensorflow");
        } else {
            throw new RuntimeException("Predictions don't match!");
        }

        //Now to run with the samediff executor, with input_b array expecting to get prediction_b
        SameDiff graphSD = TFGraphMapper.importGraph(new File(FROZEN_MLP)); //Reimport graph here, necessary for the 1.0 alpha release
        //INDArray samediffPred = graphSD.output(inputsPredictions, "prediction_a").get("prediction_a");

        placeholderValue = inputoutputPairs.get(1).getFirst();
        TFPrediction = inputoutputPairs.get(1).getSecond();
        graphSD.associateArrayWithVariable(placeholderValue,"input");
        INDArray samediffPred = graphSD.getVariable("output").eval();
        System.out.println("SameDiff exec prediction for input_b:\n" + samediffPred);
        if (samediffPred.equals(TFPrediction)) {
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
    protected static List<Pair<INDArray, INDArray>> readPlaceholdersAndPredictions() throws IOException {
        String[] toReadList = {"_a", "_b"};
        List<Pair<INDArray, INDArray>> inputoutputPairs = new ArrayList<>();
        for (String fileSuffix : toReadList) {
            INDArray input = readNDArray("input" + fileSuffix);
            INDArray output = readNDArray("prediction" + fileSuffix);
            inputoutputPairs.add(new Pair<>(input, output));
        }
        return inputoutputPairs;
    }

    private static INDArray readNDArray(String filePrefix) throws IOException {
        int[] varShape = Nd4j.readNumpy(new File(dataLocalPath, filePrefix + ".shape").getAbsolutePath(), ",").data().asInt();
        float[] varContents = Nd4j.readNumpy(new File(dataLocalPath, filePrefix + ".csv").getAbsolutePath()).data().asFloat();
        return Nd4j.create(varContents).reshape(varShape);
    }
}
