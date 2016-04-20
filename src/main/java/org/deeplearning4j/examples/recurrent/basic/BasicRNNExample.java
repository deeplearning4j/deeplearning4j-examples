package org.deeplearning4j.examples.recurrent.basic;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * This example trains a RNN. WHen trained we only have to put the first
 * character of LEARNSTRING to the RNN, and it will recite the following chars
 * 
 * @author Peter Grossmann
 */
public class BasicRNNExample {

	// define a sentence to learn
	public static final char[] LEARNSTRING = "Der Cottbuser Postkutscher putzt den Cottbuser Postkutschkasten.".toCharArray();

	// a list of all possible characters
	public static final List<Character> LEARNSTRING_CHARS_LIST = new ArrayList<Character>();

	// RNN dimensions
	public static final int HIDDEN_LAYER_WIDTH = 50;
	public static final int HIDDEN_LAYER_CONT = 2;
	public static final Random r = new Random(7894);

	public static void main(String[] args) {

		// create a dedicated list of possible chars in LEARNSTRING_CHARS_LIST
		LinkedHashSet<Character> LEARNSTRING_CHARS = new LinkedHashSet<Character>();
		for (char c : LEARNSTRING)
			LEARNSTRING_CHARS.add(c);
		LEARNSTRING_CHARS_LIST.addAll(LEARNSTRING_CHARS);

		// some common parameters
		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		builder.iterations(10);
		builder.learningRate(0.001);
		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		builder.seed(123);
		builder.biasInit(0);
		builder.miniBatch(false);
		builder.updater(Updater.RMSPROP);
		builder.weightInit(WeightInit.XAVIER);

		ListBuilder listBuilder = builder.list(HIDDEN_LAYER_CONT + 1);

		// first difference, for rnns we need to use GravesLSTM.Builder
		for (int i = 0; i < HIDDEN_LAYER_CONT; i++) {
			GravesLSTM.Builder hiddenLayerBuilder = new GravesLSTM.Builder();
			hiddenLayerBuilder.nIn(i == 0 ? LEARNSTRING_CHARS.size() : HIDDEN_LAYER_WIDTH);
			hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH);
			// adopted activation function from GravesLSTMCharModellingExample
			// seems to work well with RNNs
			hiddenLayerBuilder.activation("tanh");
			listBuilder.layer(i, hiddenLayerBuilder.build());
		}

		// we need to use RnnOutputLayer for our RNN
		RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunction.MCXENT);
		// softmax normalizes the output neurons, the sum of all outputs is 1
		// this is required for our sampleFromDistribution-function
		outputLayerBuilder.activation("softmax");
		outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH);
		outputLayerBuilder.nOut(LEARNSTRING_CHARS.size());
		listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build());

		// finish builder
		listBuilder.pretrain(false);
		listBuilder.backprop(true);
		listBuilder.build();

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
			// nextChar - not really required
			char nextChar = LEARNSTRING[(samplePos + 1) % (LEARNSTRING.length)];
			// input neuron for current-char is 1 at "samplePos"
			input.putScalar(new int[] { 0, LEARNSTRING_CHARS_LIST.indexOf(currentChar), samplePos }, 1);
			// output neuron for next-char is 1 at "samplePos"
			labels.putScalar(new int[] { 0, LEARNSTRING_CHARS_LIST.indexOf(nextChar), samplePos }, 1);
			samplePos++;
		}
		DataSet trainingData = new DataSet(input, labels);

		// some epochs
		for (int epoch = 0; epoch < 100; epoch++) {
			
			System.out.println("Epoch " + epoch);
			
			// train the data
			net.fit(trainingData);

			// clear current stance from the last example
			net.rnnClearPreviousState();

			// put the first caracter into the rrn as an initialisation
			INDArray testInit = Nd4j.zeros(LEARNSTRING_CHARS_LIST.size());
			testInit.putScalar(LEARNSTRING_CHARS_LIST.indexOf(LEARNSTRING[0]), 1);

			// run one step -> IMPORTANT: rnnTimeStep() must be called, not
			// output()
			// the output shows what the net thinks what should come next
			INDArray output = net.rnnTimeStep(testInit);

			// now the net sould guess LEARNSTRING.length mor characters
			for (int j = 0; j < LEARNSTRING.length; j++) {

				// first process the last output of the network to a concrete
				// neuron, the neuron with the highest output cas the highest
				// cance to get chosen
				double[] outputProbDistribution = new double[LEARNSTRING_CHARS.size()];
				for (int k = 0; k < outputProbDistribution.length; k++) {
					outputProbDistribution[k] = output.getDouble(k);
				}
				int sampledCharacterIdx = findIndexOfHighestValue(outputProbDistribution);

				// print the chosen output
				System.out.print(LEARNSTRING_CHARS_LIST.get(sampledCharacterIdx));

				// use the last output as input
				INDArray nextInput = Nd4j.zeros(LEARNSTRING_CHARS_LIST.size());
				nextInput.putScalar(sampledCharacterIdx, 1);
				output = net.rnnTimeStep(nextInput);

			}
			System.out.print("\n");

		}

	}

	private static int findIndexOfHighestValue(double[] distribution) {
		int maxValueIndex = 0;
		double maxValue = 0;
		for (int i = 0; i < distribution.length; i++) {
			if(distribution[i] > maxValue) {
				maxValue = distribution[i];
				maxValueIndex = i;
			}
		}
		return maxValueIndex;
	}

}
