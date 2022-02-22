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

package org.deeplearning4j.examples.quickstart.modeling.recurrent;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

/**
 * This example trains a RNN. When trained we only have to put the first
 * character of LEARNSTRING to the RNN, and it will recite the following chars
 *
 * @author Peter Grossmann
 */
public class MemorizeSequence {

	// define a sentence to learn.
    // Add a special character at the beginning so the RNN learns the complete string and ends with the marker.
	private static final char[] LEARNSTRING = "*Der Cottbuser Postkutscher putzt den Cottbuser Postkutschkasten.".toCharArray();

	// a list of all possible characters
	private static final List<Character> LEARNSTRING_CHARS_LIST = new ArrayList<>();

	// RNN dimensions
	private static final int HIDDEN_LAYER_WIDTH = 50;
	private static final int HIDDEN_LAYER_CONT = 2;

	public static void main(String[] args) {

		// create a dedicated list of possible chars in LEARNSTRING_CHARS_LIST
		LinkedHashSet<Character> LEARNSTRING_CHARS = new LinkedHashSet<>();
		for (char c : LEARNSTRING)
			LEARNSTRING_CHARS.add(c);
		LEARNSTRING_CHARS_LIST.addAll(LEARNSTRING_CHARS);

		// some common parameters
		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		builder.seed(123);
		builder.biasInit(0);
		builder.miniBatch(false);
		builder.updater(new RmsProp(0.001));
		builder.weightInit(WeightInit.XAVIER);

		ListBuilder listBuilder = builder.list();

		// first difference, for rnns we need to use LSTM.Builder
		for (int i = 0; i < HIDDEN_LAYER_CONT; i++) {
			LSTM.Builder hiddenLayerBuilder = new LSTM.Builder();
			hiddenLayerBuilder.nIn(i == 0 ? LEARNSTRING_CHARS.size() : HIDDEN_LAYER_WIDTH);
			hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH);
			// adopted activation function from LSTMCharModellingExample
			// seems to work well with RNNs
			hiddenLayerBuilder.activation(Activation.TANH);
			listBuilder.layer(i, hiddenLayerBuilder.build());
		}

		// we need to use RnnOutputLayer for our RNN
		RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunction.MCXENT);
		// softmax normalizes the output neurons, the sum of all outputs is 1
		// this is required for our sampleFromDistribution-function
		outputLayerBuilder.activation(Activation.SOFTMAX);
		outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH);
		outputLayerBuilder.nOut(LEARNSTRING_CHARS.size());
		listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build());

		// create network
		MultiLayerConfiguration conf = listBuilder.build();
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		/*
		 * CREATE OUR TRAINING DATA
		 */
		// create input and output arrays: SAMPLE_INDEX, INPUT_NEURON,
		// SEQUENCE_POSITION
		INDArray input = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size(), LEARNSTRING.length);
		INDArray labels = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size(), LEARNSTRING.length);
		// loop through our sample-sentence
		int samplePos = 0;
		for (char currentChar : LEARNSTRING) {
			// small hack: when currentChar is the last, take the first char as
			// nextChar - not really required. Added to this hack by adding a starter first character.
			char nextChar = LEARNSTRING[(samplePos + 1) % (LEARNSTRING.length)];
			// input neuron for current-char is 1 at "samplePos"
			input.putScalar(new int[] { 0, LEARNSTRING_CHARS_LIST.indexOf(currentChar), samplePos }, 1);
			// output neuron for next-char is 1 at "samplePos"
			labels.putScalar(new int[] { 0, LEARNSTRING_CHARS_LIST.indexOf(nextChar), samplePos }, 1);
			samplePos++;
		}
		DataSet trainingData = new DataSet(input, labels);

		// some epochs
		for (int epoch = 0; epoch < 1000; epoch++) {

			System.out.println("Epoch " + epoch);

			// train the data
			net.fit(trainingData);

			// clear current stance from the last example
			net.rnnClearPreviousState();

			// put the first character into the rrn as an initialisation
			INDArray testInit = Nd4j.zeros(1,LEARNSTRING_CHARS_LIST.size(), 1);
			testInit.putScalar(LEARNSTRING_CHARS_LIST.indexOf(LEARNSTRING[0]), 1);

			// run one step -> IMPORTANT: rnnTimeStep() must be called, not
			// output()
			// the output shows what the net thinks what should come next
			INDArray output = net.rnnTimeStep(testInit);

			// now the net should guess LEARNSTRING.length more characters
            for (char ignored : LEARNSTRING) {

                // first process the last output of the network to a concrete
                // neuron, the neuron with the highest output has the highest
                // chance to get chosen
                int sampledCharacterIdx = Nd4j.getExecutioner().exec(new ArgMax(new INDArray[]{output},false,new int[]{1}))[0].getInt(0);

                // print the chosen output
                System.out.print(LEARNSTRING_CHARS_LIST.get(sampledCharacterIdx));

                // use the last output as input
                INDArray nextInput = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size(), 1);
                nextInput.putScalar(sampledCharacterIdx, 1);
                output = net.rnnTimeStep(nextInput);

            }
			System.out.print("\n");
		}
	}
}
