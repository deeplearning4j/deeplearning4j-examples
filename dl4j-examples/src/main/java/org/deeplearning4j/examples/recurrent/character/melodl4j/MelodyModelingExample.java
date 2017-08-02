package org.deeplearning4j.examples.recurrent.character.melodl4j;

import org.deeplearning4j.examples.recurrent.character.CharacterIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.apache.commons.io.FileUtils;

import java.io.*;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * GravesLSTM  Symbolic melody modelling example, based closely on GravesLSTMCharModellingExample.java.
 * See the README file in this directory for documentation.
 *
 * @author Alex Black, Donald A. Smith.
 */
public class MelodyModelingExample {
    final static String inputSymbolicMelodiesFilename = "midi-melodies-bach.txt"; // Try also midi-melodies-pop.txt
    final static String tmpDir = System.getProperty("java.io.tmpdir");

    final static String symbolicMelodiesInputFilePath = tmpDir + "/" + inputSymbolicMelodiesFilename;  // Point to melodies created by Midi2MelodyStrings.java
    final static String composedMelodiesOutputFilePath = tmpDir + "/composition.txt"; // You can listen to these melodies by running PlayMelodyStrings.java against this file.

    //....
    public static void main(String[] args) throws Exception {
        String loadNetworkPath = null; //"/tmp/MelodyModel-bach.zip"; //null;
        String generationInitialization = null;        //Optional character initialization; a random character is used if null
        if (args.length == 2) {
            loadNetworkPath = args[0];
            generationInitialization = args[1];
        }

        int lstmLayerSize = 200;                    //Number of units in each GravesLSTM layer
        int miniBatchSize = 32;                     //Size of mini batch to use when training
        int exampleLength = 500; //1000; 		    //Length of each training example sequence to use.
        int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        int numEpochs = 10;                            //Total number of training epochs
        int generateSamplesEveryNMinibatches = 20;  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
        int nSamplesToGenerate = 10;                //Number of samples to generate after each training epoch
        int nCharactersToSample = 300;                //Length of each sample to generate

        // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
        // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
        Random rng = new Random(12345);
        long startTime = System.currentTimeMillis();

        System.out.println("Using " + tmpDir + " as the temporary directory");
        //Get a DataSetIterator that handles vectorization of text into something we can use to train
        // our GravesLSTM network.
        CharacterIterator iter = getMidiIterator(miniBatchSize, exampleLength);

        if (loadNetworkPath != null) {
            MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(loadNetworkPath);
            String[] samples = sampleCharactersFromNetwork(generationInitialization, net, iter, rng, nCharactersToSample, nSamplesToGenerate);
            for (String melody : samples) {
                System.out.println(melody);
                PlayMelodyStrings.playMelody(melody, 10, 48);
                System.out.println();
            }
            System.exit(0);
        }

        int nOut = iter.totalOutcomes();

        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .learningRate(0.1)
            .seed(12345)
            .regularization(true)
            .l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.RMSPROP)
            .list()
            .layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
                .activation(Activation.TANH).build())
            .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                .activation(Activation.TANH).build())
//            .layer(2, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
//                .activation(Activation.TANH).build())
            .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
                .nIn(lstmLayerSize).nOut(nOut).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
            .pretrain(false).backprop(true)
            .build();


        learn(miniBatchSize, exampleLength, numEpochs, generateSamplesEveryNMinibatches, nSamplesToGenerate, nCharactersToSample, generationInitialization, rng, startTime, iter, conf);
    }

    private static void save(CharacterIterator iter) throws IOException {
        FileOutputStream fos = new FileOutputStream("/tmp/midi-character-iterator.jobj");
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(iter);
        oos.close();
    }

    private static void learn(int miniBatchSize, int exampleLength, int numEpochs, int generateSamplesEveryNMinibatches, int nSamplesToGenerate, int nCharactersToSample, String generationInitialization, Random rng, long startTime, CharacterIterator iter, MultiLayerConfiguration conf) throws Exception {
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        //  GradientsListener listener2 = new GradientsListener(net,80);
        net.setListeners(/*listener2,*/ new ScoreIterationListener(100));

        //Print the  number of parameters in the network (and for each layer)
        Layer[] layers = net.getLayers();
        int totalNumParams = 0;
        for (int i = 0; i < layers.length; i++) {
            int nParams = layers[i].numParams();
            System.out.println("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        System.out.println("Total number of network parameters: " + totalNumParams);

        List<String> melodies = new ArrayList<>(); // Later we print them out in reverse
        // order, so that the best melodies are at the start of the file.
        //Do training, and then generate and print samples from network
        int miniBatchNumber = 0;
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            System.out.println("Starting epoch " + epoch);
            while (iter.hasNext()) {
                DataSet ds = iter.next();
                net.fit(ds);
                if (++miniBatchNumber % generateSamplesEveryNMinibatches == 0) {
                    System.out.println("---------- epoch " + epoch + " --------------");
                    System.out.println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters");
                    System.out.println("Sampling characters from network given initialization \"" + (generationInitialization == null ? "" : generationInitialization) + "\"");
                    String[] samples = sampleCharactersFromNetwork(generationInitialization, net, iter, rng, nCharactersToSample, nSamplesToGenerate);
                    for (int j = 0; j < samples.length; j++) {
                        System.out.println("----- Sample " + j + " ----- of epoch " + epoch);
                        System.out.println(samples[j]);
                        melodies.add(samples[j]);
                        System.out.println();
                    }
                }
                if (miniBatchNumber == 0) {
                    // save(iter); System.exit(0);
                }
            }
            iter.reset();    //Reset iterator for another epoch
            if (melodies.size() > 0) {
                String melody = melodies.get(melodies.size() - 1);
                int seconds = 15;
                System.out.println("\nFirst " + seconds + " seconds of " + melody);
                PlayMelodyStrings.playMelody(melody, seconds, 48);
            }
        }
        int indexOfLastPeriod = inputSymbolicMelodiesFilename.lastIndexOf('.');
        String saveFileName = inputSymbolicMelodiesFilename.substring(0, indexOfLastPeriod > 0 ? indexOfLastPeriod : inputSymbolicMelodiesFilename.length());
        ModelSerializer.writeModel(net, "/tmp/" + saveFileName + ".zip", false);

        // Write all melodies to the output file, in reverse order (so that the best melodies are at the start of the file).
        PrintWriter printWriter = new PrintWriter(composedMelodiesOutputFilePath);
        for (int i = melodies.size() - 1; i >= 0; i--) {
            printWriter.println(melodies.get(i));
        }
        printWriter.close();
        double seconds = 0.001 * (System.currentTimeMillis() - startTime);

        System.out.println("\n\nExample complete in " + seconds + " seconds");
        System.exit(0);
    }

    /**
     * Sets up and return a simple DataSetIterator that does vectorization based on the melody sample.
     *
     * @param miniBatchSize  Number of text segments in each training mini-batch
     * @param sequenceLength Number of characters in each text segment.
     */
    public static CharacterIterator getMidiIterator(int miniBatchSize, int sequenceLength) throws Exception {
        File f = new File(symbolicMelodiesInputFilePath);
        if (!f.exists()) {
            URL url = null;
            try {
                url = new URL("http://truthsite.org/music/" + inputSymbolicMelodiesFilename);
                FileUtils.copyURLToFile(url, f);
            } catch (Exception exc) {
                System.err.println("Error copying " + url + " to " + f);
                throw (exc);
            }
            if (!f.exists()) {
                throw new RuntimeException(f.getAbsolutePath() + " does not exist");
            }
            System.out.println("File downloaded to " + f.getAbsolutePath());
        } else {
            System.out.println("Using existing text file at " + f.getAbsolutePath());
        }
        char[] validCharacters = "0123456789abc!@#$%^&*(ABCDdefghijklmnopqrstuvwzyzEFGHIJKLMR".toCharArray(); //Which characters are allowed? Others will be removed
        return new CharacterIterator(symbolicMelodiesInputFilePath, Charset.forName("UTF-8"),
            miniBatchSize, sequenceLength, validCharacters, new Random(12345));
    }

    /**
     * Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
     * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
     * Note that the initalization is used for all samples
     *
     * @param initialization     String, may be null. If null, select a random character as initialization for all samples
     * @param charactersToSample Number of characters to sample from network (excluding initialization)
     * @param net                MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
     * @param iter               CharacterIterator. Used for going from indexes back to characters
     */
    public static String[] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net,
                                                       CharacterIterator iter, Random rng, int charactersToSample, int numSamples) {
        //Set up initialization. If no initialization: use a random character
        if (initialization == null) {
            initialization = String.valueOf(iter.getRandomCharacter());
        }

        //Create input for initialization
        INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
        char[] init = initialization.toCharArray();
        for (int i = 0; i < init.length; i++) {
            int idx = iter.convertCharacterToIndex(init[i]);
            for (int j = 0; j < numSamples; j++) {
                initializationInput.putScalar(new int[]{j, idx, i}, 1.0f);
            }
        }

        StringBuilder[] sb = new StringBuilder[numSamples];
        for (int i = 0; i < numSamples; i++) sb[i] = new StringBuilder(initialization);

        //Sample from network (and feed samples back into input) one character at a time (for all samples)
        //Sampling is done in parallel here
        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput);
        output = output.tensorAlongDimension(output.size(2) - 1, 1, 0);    //Gets the last time step output

        for (int i = 0; i < charactersToSample; i++) {
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(numSamples, iter.inputColumns());
            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            for (int s = 0; s < numSamples; s++) {
                double[] outputProbDistribution = new double[iter.totalOutcomes()];
                for (int j = 0; j < outputProbDistribution.length; j++)
                    outputProbDistribution[j] = output.getDouble(s, j);
                int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng);

                nextInput.putScalar(new int[]{s, sampledCharacterIdx}, 1.0f);        //Prepare next time step input
                sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));    //Add sampled character to StringBuilder (human readable output)
            }

            output = net.rnnTimeStep(nextInput);    //Do one time step of forward pass
        }

        String[] out = new String[numSamples];
        for (int i = 0; i < numSamples; i++) out[i] = sb[i].toString();
        return out;
    }

    /**
     * Given a probability distribution over discrete classes, sample from the distribution
     * and return the generated class index.
     *
     * @param distribution Probability distribution over classes. Must sum to 1.0
     */
    public static int sampleFromDistribution(double[] distribution, Random rng) {
        double d = 0.0;
        double sum = 0.0;
        for (int t = 0; t < 10; t++) {
            d = rng.nextDouble();
            sum = 0.0;
            for (int i = 0; i < distribution.length; i++) {
                sum += distribution[i];
                if (d <= sum) return i;
            }
            //If we haven't found the right index yet, maybe the sum is slightly
            //lower than 1 due to rounding error, so try again.
        }
        //Should be extremely unlikely to happen if distribution is a valid probability distribution
        throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + sum);
    }
}

