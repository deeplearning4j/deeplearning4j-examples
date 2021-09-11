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

package org.deeplearning4j.examples.advanced.features.customizingdl4j.lossfunctions;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.deeplearning4j.examples.quickstart.modeling.feedforward.regression.SumModel.getTrainingData;

/**
 * This is an example that illustrates how to define and instantiate a custom loss function.
 * The example uses the quickstart SumModel example as a basis
 *
 * @author susaneraly
 */
public class CustomLossUsageEx {
    public static final Random rng = new Random(12345);

    public static void main(String[] args) {
        //The neural net configuration here demonstrates how to instantiate a custom loss function
        doTraining();

        //THE FOLLOWING IS TO ILLUSTRATE A SIMPLE GRADIENT CHECK.
        //It checks the implementation against the finite difference approximation, to ensure correctness
        doGradientCheck();
    }

    public static void doTraining() {

        DataSetIterator iterator = getTrainingData(100, rng);

        //Create the network
        int numInput = 2;
        int numOutputs = 1;
        int nHidden = 10;
        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .dataType(DataType.DOUBLE) //instantiating as doubles for gradient checks
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.001, 0.95))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
                        .activation(Activation.TANH)
                        .build())
                //INSTANTIATE CUSTOM LOSS FUNCTION here as follows
                //Refer to CustomLossL1L2 class for more details on implementation
                .layer(new OutputLayer.Builder(new CustomLossDefinition())
                        .activation(Activation.IDENTITY)
                        .nIn(nHidden).nOut(numOutputs).build())
                .build()
        );
        net.init();

        //Train the network on the full data set and evaluate
        net.setListeners(new ScoreIterationListener(100));
        net.fit(iterator, 10);
        System.out.println("Training complete");
    }


    /**
     * Runs for a list of different activation functions and label sizes to ensure that the gradient is correct
     */
    public static void doGradientCheck() {
        double epsilon = 1e-3;
        int totalNFailures = 0;
        int totalTests = 0;
        double maxRelError = 1; // in %
        CustomLossDefinition lossfn = new CustomLossDefinition();
        String[] activationFns = new String[]{"identity", "relu", "tanh", "sigmoid", "softmax", "leakyrelu"};

        int[] labelLengths = new int[]{1, 2, 3, 4, 5};
        for (int i = 0; i < activationFns.length; i++) {
            System.out.println("Running checks for " + activationFns[i]);
            IActivation activation = Activation.fromString(activationFns[i]).getActivationFunction();
            List<INDArray> randomLabelList = randomValsinRightRange(activation, labelLengths);
            List<INDArray> randomPreOutputList = randomValsinRightRange(Activation.fromString("identity").getActivationFunction(), labelLengths);

            for (int j = 0; j < labelLengths.length; j++) {

                if (activation.toString().equals("softmax") && labelLengths[j] == 1) {
                    System.out.println("\tSkipping length == 1 for softmax");
                    continue;
                }
                System.out.println("\tRunning check for length " + labelLengths[j]);

                INDArray label = randomLabelList.get(j);
                INDArray preOut = randomPreOutputList.get(j);

                INDArray computedGradient = lossfn.computeGradient(label, preOut, activation, null);

                //checking gradient wrt to each feature in label
                NdIndexIterator iterPreOut = new NdIndexIterator(preOut.shape());
                while (iterPreOut.hasNext()) {
                    int[] next = ArrayUtil.toInts(iterPreOut.next());
                    double originalValue = preOut.getDouble(next);

                    //+eps,-eps changes and calc score
                    preOut.putScalar(next, originalValue + epsilon);
                    double scorePlus = lossfn.computeScore(label, preOut, activation, null, false); //not averaging
                    preOut.putScalar(next, originalValue - epsilon);
                    double scoreMinus = lossfn.computeScore(label, preOut, activation, null, false); //not averaging
                    //restore value
                    preOut.putScalar(next, originalValue);

                    double scoreDelta = scorePlus - scoreMinus;
                    double numericalGradient = scoreDelta / (2 * epsilon);

                    double analyticGradient = computedGradient.getDouble(next);

                    double relError = Math.abs(analyticGradient - numericalGradient) * 100 / (Math.abs(numericalGradient));

                    if (analyticGradient == 0.0 && numericalGradient == 0.0) relError = 0.0;
                    totalTests++;
                    if (relError > maxRelError || Double.isNaN(relError)) {
                        System.out.println("\t\tParam " + Arrays.toString(next) + " FAILED: grad= " + analyticGradient + ", numericalGrad= " + numericalGradient
                                + ", relErrorPerc= " + relError + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus);
                        totalNFailures++;
                    } else {
                        System.out.println("\t\tParam " + Arrays.toString(next) + " passed: grad= " + analyticGradient + ", numericalGrad= " + numericalGradient
                                + ", relError= " + relError + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus);
                    }
                }
            }
        }
        System.out.println("DONE! \nCompleted " + totalTests + " tests");
        if (totalNFailures > 0) {
            System.out.println("Gradient check failed for " + totalNFailures + " tests");
        } else {
            System.out.println("All checks passed");
        }
    }

    public static List<INDArray> randomValsinRightRange(IActivation activation, int[] labelSize) {
        List<INDArray> returnVals = new ArrayList<>(labelSize.length);
        for (int i = 0; i < labelSize.length; i++) {
            int aLabelSize = labelSize[i];
            INDArray someValArray = Nd4j.rand(new NormalDistribution(), 1, aLabelSize);
            returnVals.add(activation.getActivation(someValArray, false));
        }
        return returnVals;
    }
}
