package org.deeplearning4j.examples.rnn.character;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**GravesLSTM Character modelling example
 * @author Alex Black

   Example: Train a LSTM RNN to generates text, one character at a time.
	This example is somewhat inspired by Andrej Karpathy's blog post,
	"The Unreasonable Effectiveness of Recurrent Neural Networks"
	http://karpathy.github.io/2015/05/21/rnn-effectiveness/
	
	Note that this example has not been well tuned - better performance is likely possible with better hyperparameters
	
	Some differences between this example and Karpathy's work:
	- The LSTM architectures appear to differ somewhat. GravesLSTM has peephole connections that
	  Karpathy's char-rnn implementation appears to lack. See GravesLSTM javadoc for details.
	  There are pros and cons to both architectures (addition of peephole connections is a more powerful
	  model but has more parameters per unit), though they are not radically different in practice.
	- Karpathy uses truncated backpropagation through time (BPTT) on full character
	  sequences, whereas this example uses standard (non-truncated) BPTT on partial/subset sequences.
	  Truncated BPTT is probably the preferred method of training for this sort of problem, and is configurable
      using the .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength().tBPTTBackwardLength() options
	  
	This example is set up to train on the Complete Works of William Shakespeare, downloaded
	 from Project Gutenberg. Training on other text sources should be relatively easy to implement.
 */
public class GravesLSTMCharModellingExample {
	public static void main( String[] args ) throws Exception {
		int lstmLayerSize = 200;					//Number of units in each GravesLSTM layer
		int miniBatchSize = 32;						//Size of mini batch to use when  training
		int examplesPerEpoch = 50 * miniBatchSize;	//i.e., how many examples to learn on between generating samples
		int exampleLength = 100;					//Length of each training example
		int numEpochs = 30;							//Total number of training + sample generation epochs
		int nSamplesToGenerate = 4;					//Number of samples to generate after each training epoch
		int nCharactersToSample = 300;				//Length of each sample to generate
		String generationInitialization = null;		//Optional character initialization; a random character is used if null
		// Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
		// Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
		Random rng = new Random(12345);
		
		//Get a DataSetIterator that handles vectorization of text into something we can use to train
		// our GravesLSTM network.
		CharacterIterator iter = getShakespeareIterator(miniBatchSize,exampleLength,examplesPerEpoch);
		int nOut = iter.totalOutcomes();
		
		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
			.learningRate(0.1)
			.rmsDecay(0.95)
			.seed(12345)
			.regularization(true)
			.l2(0.001)
			.list(3)
			.layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
					.updater(Updater.RMSPROP)
					.activation("tanh").weightInit(WeightInit.DISTRIBUTION)
					.dist(new UniformDistribution(-0.08, 0.08)).build())
			.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
					.updater(Updater.RMSPROP)
					.activation("tanh").weightInit(WeightInit.DISTRIBUTION)
					.dist(new UniformDistribution(-0.08, 0.08)).build())
			.layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax")        //MCXENT + softmax for classification
					.updater(Updater.RMSPROP)
					.nIn(lstmLayerSize).nOut(nOut).weightInit(WeightInit.DISTRIBUTION)
					.dist(new UniformDistribution(-0.08, 0.08)).build())
			.pretrain(false).backprop(true)
			.build();
		
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));
		
		//Print the  number of parameters in the network (and for each layer)
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for( int i=0; i<layers.length; i++ ){
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);
		
		//Do training, and then generate and print samples from network
		for( int i=0; i<numEpochs; i++ ){
			net.fit(iter);
			
			System.out.println("--------------------");
			System.out.println("Completed epoch " + i );
			System.out.println("Sampling characters from network given initialization \""+ (generationInitialization == null ? "" : generationInitialization) +"\"");
			String[] samples = sampleCharactersFromNetwork(generationInitialization,net,iter,rng,nCharactersToSample,nSamplesToGenerate);
			for( int j=0; j<samples.length; j++ ){
				System.out.println("----- Sample " + j + " -----");
				System.out.println(samples[j]);
				System.out.println();
			}
			
			iter.reset();	//Reset iterator for another epoch
		}
		
		System.out.println("\n\nExample complete");
	}

	/** Downloads Shakespeare training data and stores it locally (temp directory). Then set up and return a simple
	 * DataSetIterator that does vectorization based on the text.
	 * @param miniBatchSize Number of text segments in each training mini-batch
	 * @param exampleLength Number of characters in each text segment.
	 * @param examplesPerEpoch Number of examples we want in an 'epoch'. 
	 */
	public static CharacterIterator getShakespeareIterator(int miniBatchSize, int exampleLength, int examplesPerEpoch) throws Exception{
		//The Complete Works of William Shakespeare
		//5.3MB file in UTF-8 Encoding, ~5.4 million characters
		//https://www.gutenberg.org/ebooks/100
		String url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
		String tempDir = System.getProperty("java.io.tmpdir");
		String fileLocation = tempDir + "/Shakespeare.txt";	//Storage location from downloaded file
		File f = new File(fileLocation);
		if( !f.exists() ){
			FileUtils.copyURLToFile(new URL(url), f);
			System.out.println("File downloaded to " + f.getAbsolutePath());
		} else {
			System.out.println("Using existing text file at " + f.getAbsolutePath());
		}
		
		if(!f.exists()) throw new IOException("File does not exist: " + fileLocation);	//Download problem?
		
		char[] validCharacters = CharacterIterator.getMinimalCharacterSet();	//Which characters are allowed? Others will be removed
		return new CharacterIterator(fileLocation, Charset.forName("UTF-8"),
				miniBatchSize, exampleLength, examplesPerEpoch, validCharacters, new Random(12345),true);
	}
	
	/** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
	 * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
	 * Note that the initalization is used for all samples
	 * @param initialization String, may be null. If null, select a random character as initialization for all samples
	 * @param charactersToSample Number of characters to sample from network (excluding initialization)
	 * @param net MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
	 * @param iter CharacterIterator. Used for going from indexes back to characters
	 */
	private static String[] sampleCharactersFromNetwork( String initialization, MultiLayerNetwork net,
			CharacterIterator iter, Random rng, int charactersToSample, int numSamples ){
		//Set up initialization. If no initialization: use a random character
		if( initialization == null ){
			initialization = String.valueOf(iter.getRandomCharacter());
		}
		
		//Create input for initialization
		INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
		char[] init = initialization.toCharArray();
		for( int i=0; i<init.length; i++ ){
			int idx = iter.convertCharacterToIndex(init[i]);
			for( int j=0; j<numSamples; j++ ){
				initializationInput.putScalar(new int[]{j,idx,i}, 1.0f);
			}
		}
		
		StringBuilder[] sb = new StringBuilder[numSamples];
		for( int i=0; i<numSamples; i++ ) sb[i] = new StringBuilder(initialization);
		
		//Sample from network (and feed samples back into input) one character at a time (for all samples)
		//Sampling is done in parallel here
		net.rnnClearPreviousState();
		INDArray output = net.rnnTimeStep(initializationInput);
		output = output.tensorAlongDimension(output.size(2)-1,1,0);	//Gets the last time step output
		
		for( int i=0; i<charactersToSample; i++ ){
			//Set up next input (single time step) by sampling from previous output
			INDArray nextInput = Nd4j.zeros(numSamples,iter.inputColumns());
			//Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
			for( int s=0; s<numSamples; s++ ){
				double[] outputProbDistribution = new double[iter.totalOutcomes()];
				for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(s,j);
				int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution,rng);
				
				nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);		//Prepare next time step input
				sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));	//Add sampled character to StringBuilder (human readable output)
			}
			
			output = net.rnnTimeStep(nextInput);	//Do one time step of forward pass
		}
		
		String[] out = new String[numSamples];
		for( int i=0; i<numSamples; i++ ) out[i] = sb[i].toString();
		return out;
	}
	
	/** Given a probability distribution over discrete classes, sample from the distribution
	 * and return the generated class index.
	 * @param distribution Probability distribution over classes. Must sum to 1.0
	 */
	public static int sampleFromDistribution( double[] distribution, Random rng ){
		double d = rng.nextDouble();
		double sum = 0.0;
		for( int i=0; i<distribution.length; i++ ){
			sum += distribution[i];
			if( d <= sum ) return i;
		}
		//Should never happen if distribution is a valid probability distribution
		throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
	}
}