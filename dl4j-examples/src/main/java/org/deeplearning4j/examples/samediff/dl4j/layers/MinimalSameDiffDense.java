package org.deeplearning4j.examples.samediff.dl4j.layers;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * This is a minimal fully connected custom layer using SameDiff, equivalent to DL4J's DenseLayer (minus some features)
 *
 * @author Alex Black
 */
public class MinimalSameDiffDense extends SameDiffLayer {

    private int nIn;
    private int nOut;
    private Activation activation;

    /**
     *
     * @param nIn        Number of inputs to the layer
     * @param nOut       Number of outputs - i.e., layer size
     * @param activation Activation function
     * @param weightInit Weight initialization for the weights
     */
    public MinimalSameDiffDense(int nIn, int nOut, Activation activation, WeightInit weightInit){
        this.nIn = nIn;
        this.nOut = nOut;
        this.activation = activation;
        this.weightInit = weightInit;
    }

    protected MinimalSameDiffDense(){
        //A no-arg constructor is required for JSON serialization, model saving, training on Spark etc
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        //In this method: you define the type of output/activations, if appropriate, given the type of input to the layer
        //This is used in a few methods in DL4J to calculate activation shapes, memory requirements etc
        return InputType.feedForward(nOut);
    }

    /**
     * In this method, you define the parameters and their shapes
     * For this layer, we have a weight matrix (nIn x nOut) plus a bias array
     * @param params A helper class that allows you to define the parameters
     */
    @Override
    public void defineParameters(SDLayerParams params) {
        params.addWeightParam(DefaultParamInitializer.WEIGHT_KEY, nIn, nOut);
        params.addBiasParam(DefaultParamInitializer.BIAS_KEY, 1, nOut);
    }

    /**
     * In the defineLayer method, you define the actual layer forward pass
     * For this layer, we are returning out = activationFn( input*weights + bias)
     *
     * @param sd         The SameDiff instance for this layer
     * @param layerInput A SDVariable representing the input activations for the layer
     * @param paramTable A map of parameters for the layer. These are the SDVariables corresponding to whatever you defined
     *                   in the defineParameters method
     * @return
     */
    @Override
    public SDVariable defineLayer(SameDiff sd, SDVariable layerInput, Map<String, SDVariable> paramTable, SDVariable mask) {
        SDVariable weights = paramTable.get(DefaultParamInitializer.WEIGHT_KEY);
        SDVariable bias = paramTable.get(DefaultParamInitializer.BIAS_KEY);

        SDVariable mmul = sd.mmul("mmul", layerInput, weights);
        SDVariable z = mmul.add("z", bias);
        return activation.asSameDiff("out", sd, z);
    }

    /**
     * This method is used to initialize the parameter.
     * For example, we are setting the bias parameter to 0, and using the specified DL4J weight initialization type
     * for the weights
     * @param params Map of parameters. These are the INDArrays corresponding to whatever you defined in the
     *               defineParameters method
     */
    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        params.get(DefaultParamInitializer.BIAS_KEY).assign(0);
        initWeights(nIn, nOut, weightInit, params.get(DefaultParamInitializer.WEIGHT_KEY));
    }

    //OPTIONAL methods: you can implement these if you want to support extra features
    //See the other examples for more details
//    public void setNIn(InputType inputType, boolean override)
//    public InputPreProcessor getPreProcessorForInputType(InputType inputType)
//    public void applyGlobalConfigToLayer(NeuralNetConfiguration.Builder globalConfig)



    //Setters and getters: these are required for JSON configuration serialization/deserialization
    public int getNIn() {
        return nIn;
    }

    public void setNIn(int nIn) {
        this.nIn = nIn;
    }

    public int getNOut() {
        return nOut;
    }

    public void setNOut(int nOut) {
        this.nOut = nOut;
    }

    public Activation getActivation() {
        return activation;
    }

    public void setActivation(Activation activation) {
        this.activation = activation;
    }
}
