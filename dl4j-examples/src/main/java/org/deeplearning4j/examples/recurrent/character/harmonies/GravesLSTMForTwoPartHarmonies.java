package org.deeplearning4j.examples.recurrent.character.harmonies;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.examples.recurrent.character.CharacterIterator;
import org.deeplearning4j.examples.recurrent.character.LSTMCharModellingExample;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import java.io.File;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

import org.deeplearning4j.util.ModelSerializer;

import javax.swing.*;

/**
 * GravesLSTM Character modelling example for learning two-part harmonies. See the README file.
 *
 * @author Don Smith, based on Alex Black's CompGraphLSTMExample
 * <p>
 * Example: Train a LSTM RNN to generates harmony strings, one character at a time.
 * This example is somewhat inspired by Andrej Karpathy's blog post,
 * "The Unreasonable Effectiveness of Recurrent Neural Networks"
 * http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 * <p>
 * For more details on RNNs in DL4J, see the following:
 * http://deeplearning4j.org/usingrnns
 * http://deeplearning4j.org/lstm
 * http://deeplearning4j.org/recurrentnetwork
 */
public class GravesLSTMForTwoPartHarmonies {
     private static String inputOutputDirectoryPath;

    public static void main(String[] args) throws Exception {
        int lstmLayerSize = 100;     //Number of units in each GravesLSTM layer. Seems to learn better with 100 than with 200.
        int miniBatchSize = 32;                     //Size of mini batch to use when  training
        int exampleLength = 2000;                   //Length of each training example sequence to use. This could certainly be increased
        int tbpttLength = 300;                      //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        int numEpochs = 16;                         //Total number of training epochs
        int generateSamplesEveryNMinibatches = 5;  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
        int nSamplesToGenerate = 4;                    //Number of samples to generate after each training epoch
        int nCharactersToSample = 3600;    //Length of each sample to generate
        double l2 = 0.0015;
        double learningRate = 0.05;
        IUpdater updater = new Adam(learningRate); // new RmsProp(0.1); //
        String generationInitialization = null;
        // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
        // Initialization characters must all be in ' ' and MidiHarmonyUtility.PITCH_CHARACTERS_FOR_HARMONY
        Random rng = new Random();

        //Get a DataSetIterator that handles vectorization of text into something we can use to train
        // our GravesLSTM network.
        CharacterIterator iter = getHarmoniesIterator(miniBatchSize, exampleLength);
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
            .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        if (false) {
            ModelSerializer.restoreMultiLayerNetwork(new File("xxxxx.zip"));
        } else {
            net.init();
        }
        ScoreIterationListener scoreIterationListener = new ScoreIterationListener(20);
        net.addListeners(scoreIterationListener);

        //Print the  number of parameters in the network (and for each layer)
        Layer[] layers = net.getLayers();
        long totalNumParams = 0;
        for (int i = 0; i < layers.length; i++) {
            long nParams = layers[i].numParams();
            System.out.println("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        System.out.println("Total number of network parameters: " + totalNumParams);

        //Do training, and then generate and print samples from network
        int miniBatchNumber = 0;
        String identifier =
            "layerCount_" + net.getLayers().length +
            "-layerSize_" + lstmLayerSize
                + "-tbpttLength_" + tbpttLength
                + "-l2_" + l2
                + "-learningRate_" + learningRate
                + "-updater_" + updater.getClass().getSimpleName()
                + "-" + dateFormat.format(new Date());
        System.out.println(identifier);
        enableUI(net);
        PrintWriter sampleWriter = new PrintWriter(inputOutputDirectoryPath + File.separator + "samples-" + identifier + ".txt");
        long start = System.currentTimeMillis();
        for (int i = 0; i < numEpochs; i++) {
            System.out.println("\nStarting epoch " + i);
            while (iter.hasNext()) {
                DataSet ds = iter.next();
                net.fit(ds);
                if (++miniBatchNumber % generateSamplesEveryNMinibatches == 0) {
                    long seconds = (long) (0.001 * (System.currentTimeMillis() - start));
                    double hours = seconds / 3600.0;
                    System.out.println(hours + ": Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters");
                    String[] samples = sampleCharactersFromNetwork(generationInitialization, net, iter, rng, nCharactersToSample, nSamplesToGenerate);
                    sampleWriter.println();
                    sampleWriter.println(hours);
                    for (int j = 0; j < samples.length; j++) {
                        System.out.println(samples[j]);
                        System.out.println();
                        sampleWriter.println(samples[j]);
                    }
                    sampleWriter.flush();
                }
            }
            iter.reset();    //Reset iterator for another epoch

            // !!!!!! CHANGE THIS BELOW IF YOU WANT TO SAVE MODEL FILES (for playback or transfer learning).
            if (true) {
                boolean saveUpdater = false;   // To allow later learning set saveUpdater to true.
                String savePath = inputOutputDirectoryPath + "-" + identifier + ".zip";
                ModelSerializer.writeModel(net, savePath, saveUpdater);
            }
        }
        sampleWriter.close();

        System.out.println("\n\nExample complete");
    }

    private static void enableUI(MultiLayerNetwork net) {
        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Then add the StatsListener to collect this information from the network, as it trains
        net.addListeners(new StatsListener(statsStorage));
    }
    private static File chooseInputHarmoniesFile() {
        JFileChooser chooser = new JFileChooser(System.getProperty("user.home") + File.separator + "midi-learning");
        chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        if (chooser.showDialog(null, "Choose harmonies file") != JFileChooser.APPROVE_OPTION) {
            return null;
        } else {
            return chooser.getSelectedFile();
        }
    }

    /**
     * @return CharacterIterator for the purpose of getting its
     * @throws Exception
     */
    public static CharacterIterator getCharacterIteratorForPlayBack() throws Exception {
        File temp=File.createTempFile("dummie","iterator");
        PrintWriter writer = new PrintWriter(temp);
        for(int i=0;i<10;i++) {
            writer.println(" " + MidiHarmonyUtility.PITCH_CHARACTERS_FOR_HARMONY);
        }
        writer.close();
        return getHarmoniesIteratorFromFile(temp,64, 200);
    }

    /**
     * Set up and return a simple DataSetIterator that does vectorization based on the text.
     *
     * @param miniBatchSize  Number of text segments in each training mini-batch
     * @param sequenceLength Number of characters in each text segment.
     */
    private static CharacterIterator getHarmoniesIterator(int miniBatchSize, int sequenceLength) throws Exception {
        File f = chooseInputHarmoniesFile();
        if (f == null) {
            System.exit(1);
        }
        return getHarmoniesIteratorFromFile(f,miniBatchSize,sequenceLength);
    }
    private static CharacterIterator getHarmoniesIteratorFromFile(File f, int miniBatchSize, int sequenceLength) throws Exception {
        inputOutputDirectoryPath = f.getParent();
        int kilobytes = (int) Math.ceil(f.length() / 1024.0);
        System.out.println("Reading harmonies from " + f.getAbsolutePath() + ", length = " + kilobytes + "kb");

        char[] validCharacters =new char[1+MidiHarmonyUtility.PITCH_CHARACTERS_FOR_HARMONY.length()];
        validCharacters[0]=' ';
        for(int i=0;i<MidiHarmonyUtility.PITCH_CHARACTERS_FOR_HARMONY.length();i++) {
            validCharacters[i+1]= MidiHarmonyUtility.PITCH_CHARACTERS_FOR_HARMONY.charAt(i);
        }
        return new CharacterIterator(f.getAbsolutePath(), Charset.forName("UTF-8"),
            miniBatchSize, sequenceLength, validCharacters, new Random());
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
        output = output.tensorAlongDimension((int)output.size(2) - 1, 1, 0);    //Gets the last time step output

        for (int i = 0; i < charactersToSample; i++) {
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(numSamples, iter.inputColumns());
            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            for (int s = 0; s < numSamples; s++) {
                double[] outputProbDistribution = new double[iter.totalOutcomes()];
                for (int j = 0; j < outputProbDistribution.length; j++)
                    outputProbDistribution[j] = output.getDouble(s, j);
                int sampledCharacterIdx = LSTMCharModellingExample.sampleFromDistribution(outputProbDistribution, rng);

                nextInput.putScalar(new int[]{s, sampledCharacterIdx}, 1.0f);        //Prepare next time step input
                sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));    //Add sampled character to StringBuilder (human readable output)
            }

            output = net.rnnTimeStep(nextInput);    //Do one time step of forward pass
        }

        String[] out = new String[numSamples];
        for (int i = 0; i < numSamples; i++) out[i] = sb[i].toString();
        return out;
    }
}
