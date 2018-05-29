package org.deeplearning4j.examples.recurrent.character.harmonies;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.examples.recurrent.character.CharacterIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.URL;
import java.nio.charset.Charset;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;
import org.deeplearning4j.util.ModelSerializer;

import javax.swing.*;
import javax.swing.filechooser.FileFilter;

/**GravesLSTM Character modelling example
 * @author Alex Black, modified by Don Smith for learning music from symbolic harmony strings

   Example: Train a LSTM RNN to generates text, one character at a time.
	This example is somewhat inspired by Andrej Karpathy's blog post,
	"The Unreasonable Effectiveness of Recurrent Neural Networks"
	http://karpathy.github.io/2015/05/21/rnn-effectiveness/

    For more details on RNNs in DL4J, see the following:
    http://deeplearning4j.org/usingrnns
    http://deeplearning4j.org/lstm
    http://deeplearning4j.org/recurrentnetwork

 */
public class GravesLSTMForTwoPartHarmonies {
    private static boolean useInstruments = false; // Set this to true if your samples include instrument characters.
    private static String inputOutputDirectoryPath;
	public static void main( String[] args ) throws Exception {
		int lstmLayerSize = 250;					//Number of units in each GravesLSTM layer
		int miniBatchSize = 64;						//Size of mini batch to use when  training
        int exampleLength = 2000;					//Length of each training example sequence to use. This could certainly be increased
        int tbpttLength = 300;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
		int numEpochs = 16;							//Total number of training epochs
        int generateSamplesEveryNMinibatches = 5;  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
		int nSamplesToGenerate = 4;					//Number of samples to generate after each training epoch
		int nCharactersToSample = useInstruments? 3600: 1800;	//Length of each sample to generate
        double l2=0.0015;
        double learningRate = 0.05;
        IUpdater updater =  new Adam(learningRate); // new RmsProp(0.1); //
		String generationInitialization = null;
      	// Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
		// Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
		Random rng = new Random();

		//Get a DataSetIterator that handles vectorization of text into something we can use to train
		// our GravesLSTM network.
		CharacterIterator iter = getHarmoniesIterator(miniBatchSize,exampleLength);
		int nOut = iter.totalOutcomes();
        DateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd--HH-mm");
		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.seed(rng.nextLong())
			.l2(l2) // 0.001
            .weightInit(WeightInit.XAVIER)
                //.updater(new RmsProp(0.05)) // 0.1
            .updater(updater) //
			.list()
			.layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
					.activation(Activation.TANH).build())
			.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
					.activation(Activation.TANH).build())
            .layer(2, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                .activation(Activation.TANH).build())
//            .layer(3, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
//                .activation(Activation.TANH).build())
			.layer(3, new RnnOutputLayer.Builder(LossFunction.MCXENT)  //LossFunction.MCXENT
            .activation(Activation.SOFTMAX)  //MCXENT + softmax for classification
    		.nIn(lstmLayerSize).nOut(nOut).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
			.pretrain(false).backprop(true)
			.build();
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		if (false) {
            ModelSerializer.restoreMultiLayerNetwork(new File("xxxxx.zip"));
        } else {
            net.init();
        }
		net.setListeners(new ScoreIterationListener(20));

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
        int miniBatchNumber = 0;
        String identifier=
                "layerSize_" + lstmLayerSize
                + "-tbpttLength_" + tbpttLength
                + "-l2_" + l2
                + "-learningRate_"+learningRate
                + "-updater_"+ updater.getClass().getSimpleName()
                + "-" + dateFormat.format(new Date()) + ".txt";
        System.out.println(identifier);

        PrintWriter sampleWriter = new PrintWriter(inputOutputDirectoryPath + File.separator + "samples-" + identifier);
        long start=System.currentTimeMillis();
		for( int i=0; i<numEpochs; i++ ){
            while(iter.hasNext()){
                DataSet ds = iter.next();
                net.fit(ds);
                if(++miniBatchNumber % generateSamplesEveryNMinibatches == 0){
                    long seconds = (long) (0.001*(System.currentTimeMillis()-start));
                    double hours = seconds/3600.0;
                    System.out.println(hours + ": Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters" );
                    String[] samples = sampleCharactersFromNetwork(generationInitialization,net,iter,rng,nCharactersToSample,nSamplesToGenerate);
                    sampleWriter.println();
                    sampleWriter.println(hours);
                    for( int j=0; j<samples.length; j++ ){
                        System.out.println(samples[j]);
                        System.out.println();
                        sampleWriter.println(samples[j]);
                    }
                    sampleWriter.flush();
                }
            }
			iter.reset();	//Reset iterator for another epoch

            // !!!!!! CHANGE THIS BELOW IF YOU WANT TO SAVE MODEL FILES (for transfer learning) !!!!!!
            // To allow later learning set saveUpdater to true.
            // ModelSerializer.writeModel(net, "d:/tmp/harmonies/model-" + i + "-" + System.currentTimeMillis() + ".zip", true);
        }
        sampleWriter.close();

		System.out.println("\n\nExample complete");
	}
    private static File chooseInputHarmoniesFile() {
        JFileChooser chooser = new JFileChooser(System.getProperty("user.home") + File.separator + "midi-learning");
        chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        if (chooser.showDialog(null, "Choose harmonies file")!=JFileChooser.APPROVE_OPTION) {
            return null;
        } else {
            return chooser.getSelectedFile();
        }
    }
	/** Set up and return a simple DataSetIterator that does vectorization based on the text.
	 * @param miniBatchSize Number of text segments in each training mini-batch
	 * @param sequenceLength Number of characters in each text segment.
	 */
	private static CharacterIterator getHarmoniesIterator(int miniBatchSize, int sequenceLength) throws Exception{
		File f = chooseInputHarmoniesFile();
		if(f==null) {
		    System.exit(1);
		}
		inputOutputDirectoryPath = f.getParent();
		int kilobytes = (int) Math.ceil(f.length()/1024.0);
        System.out.println("Reading harmonies from " + f.getAbsolutePath() + ", length = " + kilobytes + "kb");

        char validCharactersWithInstruments[] = new char[108-32+1];
        for(int i=0;i<108-32+1;i++) {
            validCharactersWithInstruments[i]=(char) (32+i);
        }
		char[] validCharactersWithoutInstruments = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvw ".toCharArray();
        char [] validCharacters = useInstruments? validCharactersWithInstruments: validCharactersWithoutInstruments;
		return new CharacterIterator(f.getAbsolutePath(), Charset.forName("UTF-8"),
				miniBatchSize, sequenceLength, validCharacters, new Random());
	}

	/** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
	 * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
	 * Note that the initalization is used for all samples
	 * @param initialization String, may be null. If null, select a random character as initialization for all samples
	 * @param charactersToSample Number of characters to sample from network (excluding initialization)
	 * @param net MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
	 * @param iter CharacterIterator. Used for going from indexes back to characters
	 */
	private static String[] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net,
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
	    double d = 0.0;
	    double sum = 0.0;
	    for( int t=0; t<10; t++ ) {
            d = rng.nextDouble();
            sum = 0.0;
            for( int i=0; i<distribution.length; i++ ){
                sum += distribution[i];
                if( d <= sum ) return i;
            }
            //If we haven't found the right index yet, maybe the sum is slightly
            //lower than 1 due to rounding error, so try again.
        }
		//Should be extremely unlikely to happen if distribution is a valid probability distribution
		throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
	}
}
