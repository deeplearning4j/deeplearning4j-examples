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

package org.deeplearning4j.examples.advanced.modelling.charmodelling.generatetext;

import org.deeplearning4j.examples.advanced.modelling.charmodelling.utils.CharacterIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

/**
 * This example is almost identical to the LSTMCharModellingExample, except that it utilizes the ComputationGraph
 * architecture instead of MultiLayerNetwork architecture. See the javadoc in that example for details.
 * For more details on the ComputationGraph architecture, see https://deeplearning4j.konduit.ai/models/computationgraph
 *
 * In addition to the use of the ComputationGraph a, this version has skip connections between the first and output layers,
 * in order to show how this configuration is done. In practice, this means we have the following types of connections:
 * (a) first layer -> second layer connections
 * (b) first layer -> output layer connections
 * (c) second layer -> output layer connections
 *
 * @author Alex Black
 */
@SuppressWarnings("DuplicatedCode")
public class GenerateTxtCharCompGraphModel {

    @SuppressWarnings("ConstantConditions")
    public static void main(String[] args ) throws Exception {
        int lstmLayerSize = 77;					//Number of units in each LSTM layer
        int miniBatchSize = 32;						//Size of mini batch to use when  training
        int exampleLength = 1000;					//Length of each training example sequence to use. This could certainly be increased
        int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        int numEpochs = 1;							//Total number of training epochs
        int generateSamplesEveryNMinibatches = 10;  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
        int nSamplesToGenerate = 4;					//Number of samples to generate after each training epoch
        int nCharactersToSample = 300;				//Length of each sample to generate
        String generationInitialization = null;		//Optional character initialization; a random character is used if null
        // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
        // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
        Random rng = new Random(12345);

        //Get a DataSetIterator that handles vectorization of text into something we can use to train
        // our LSTM network.
        CharacterIterator iter = GenerateTxtModel.getShakespeareIterator(miniBatchSize, exampleLength);
        int nOut = iter.totalOutcomes();

        //Set up network configuration:
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .l2(0.0001)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(0.005))
            .graphBuilder()
            .addInputs("input") //Give the input a name. For a ComputationGraph with multiple inputs, this also defines the input array orders
            //First layer: name "first", with inputs from the input called "input"
            .addLayer("first", new LSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
                .activation(Activation.TANH).build(),"input")
            //Second layer, name "second", with inputs from the layer called "first"
            .addLayer("second", new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                .activation(Activation.TANH).build(),"first")
            //Output layer, name "outputlayer" with inputs from the two layers called "first" and "second"
            .addLayer("outputLayer", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(lstmLayerSize).nOut(lstmLayerSize).build(),"second")
            .setOutputs("outputLayer")  //List the output. For a ComputationGraph with multiple outputs, this also defines the input array orders
            .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
            .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        System.out.println(net.summary());

        //Print the  number of parameters in the network (and for each layer)
        long totalNumParams = 0;
        for( int i = 0; i < net.getNumLayers(); i++) {
            long nParams = net.getLayer(i).numParams();
            System.out.println("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        System.out.println("Total number of network parameters: " + totalNumParams);

        //Do training, and then generate and print samples from network
        int miniBatchNumber = 0;
        for( int i = 0; i < numEpochs; i++) {
            while(iter.hasNext()){
                DataSet ds = iter.next();
                System.out.println("Input shape " + ds.getFeatures().shapeInfoToString());
                System.out.println("Labels " + ds.getLabels().shapeInfoToString());
                net.fit(ds);
                if(++miniBatchNumber % generateSamplesEveryNMinibatches == 0){
                    System.out.println("--------------------");
                    System.out.println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters" );
                    System.out.println("Sampling characters from network given initialization \"" + (generationInitialization == null ? "" : generationInitialization) + "\"");
                    String[] samples = sampleCharactersFromNetwork(generationInitialization,net,iter,rng,nCharactersToSample,nSamplesToGenerate);
                    for( int j = 0; j < samples.length; j++) {
                        System.out.println("----- Sample " + j + " -----");
                        System.out.println(samples[j]);
                        System.out.println();
                    }
                }
            }

            iter.reset();	//Reset iterator for another epoch
        }

        System.out.println("\n\nExample complete");
    }

    /** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
     * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
     * Note that the initialization is used for all samples
     * @param initialization String, may be null. If null, select a random character as initialization for all samples
     * @param charactersToSample Number of characters to sample from network (excluding initialization)
     * @param net MultiLayerNetwork with one or more LSTM/RNN layers and a softmax output layer
     * @param iter CharacterIterator. Used for going from indexes back to characters
     */
    private static String[] sampleCharactersFromNetwork( String initialization, ComputationGraph net,
                                                         CharacterIterator iter, Random rng, int charactersToSample, int numSamples ){
        //Set up initialization. If no initialization: use a random character
        if( initialization == null ){
            initialization = String.valueOf(iter.getRandomCharacter());
        }

        //Create input for initialization
        INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
        char[] init = initialization.toCharArray();
        for( int i=0; i<init.length; i++) {
            int idx = iter.convertCharacterToIndex(init[i]);
            for( int j = 0; j<numSamples; j++ ){
                initializationInput.putScalar(new int[]{j,idx,i}, 1.0f);
            }
        }

        StringBuilder[] sb = new StringBuilder[numSamples];
        for( int i=0; i<numSamples; i++ ) sb[i] = new StringBuilder(initialization);

        //Sample from network (and feed samples back into input) one character at a time (for all samples)
        //Sampling is done in parallel here
        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput)[0];
        output = output.tensorAlongDimension((int)output.size(2)-1,1,0);	//Gets the last time step output

        for( int i = 0; i < charactersToSample; i++ ){
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(numSamples,iter.inputColumns());
            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            for( int s=0; s<numSamples; s++ ){
                double[] outputProbDistribution = new double[iter.totalOutcomes()];
                for( int j = 0; j < outputProbDistribution.length; j++) outputProbDistribution[j] = output.getDouble(s,j);
                int sampledCharacterIdx = GenerateTxtModel.sampleFromDistribution(outputProbDistribution,rng);

                nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);		//Prepare next time step input
                sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));	//Add sampled character to StringBuilder (human readable output)
            }

            output = net.rnnTimeStep(nextInput)[0];	//Do one time step of forward pass
        }

        String[] out = new String[numSamples];
        for( int i=0; i<numSamples; i++ ) out[i] = sb[i].toString();
        return out;
    }
}
