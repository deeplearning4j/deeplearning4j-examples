package org.deeplearning4j.examples.misc.lossfunctions;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.util.*;

/**
 * Created by susaneraly on 11/9/16.
 * This is an example that illustrates how to instantiate and use a custom loss function.
 * The example is identical to the one in org.deeplearning4j.examples.feedforward.regression.RegressionSum
 * except for the custom loss function
 */
public class CustomLossExample {
    public static final int seed = 12345;
    public static final int iterations = 1;
    public static final int nEpochs = 500;
    public static final int nSamples = 1000;
    public static final int batchSize = 100;
    public static final double learningRate = 0.001;
    public static int MIN_RANGE = 0;
    public static int MAX_RANGE = 3;

    public static final Random rng = new Random(seed);

    public static void main(String[] args) {
        doTraining();

        //THE FOLLOWING IS TO ILLUSTRATE A SIMPLE GRADIENT CHECK.
        //It checks the implementation against the finite difference approximation, to ensure correctness
        //You will have to write your own gradient checks.
        //Use the code below and the following for reference.
        //  deeplearning4j/deeplearning4j-core/src/test/java/org/deeplearning4j/gradientcheck/LossFunctionGradientCheck.java
        doGradientCheck();
    }

    public static void doTraining(){

        DataSetIterator iterator = getTrainingData(batchSize,rng);

        //Create the network
        int numInput = 2;
        int numOutputs = 1;
        int nHidden = 10;
        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            .weightInit(WeightInit.XAVIER)
            .updater(new Nesterovs(0.95))
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
                .activation(Activation.TANH)
                .build())
                //INSTANTIATE CUSTOM LOSS FUNCTION here as follows
                //Refer to CustomLossL1L2 class for more details on implementation
            .layer(1, new OutputLayer.Builder(new CustomLossL1L2())
                .activation(Activation.IDENTITY)
                .nIn(nHidden).nOut(numOutputs).build())
            .pretrain(false).backprop(true).build()
        );
        net.init();
        net.setListeners(new ScoreIterationListener(100));


        //Train the network on the full data set, and evaluate in periodically
        for( int i=0; i<nEpochs; i++ ){
            iterator.reset();
            net.fit(iterator);
        }
        // Test the addition of 2 numbers (Try different numbers here)
        final INDArray input = Nd4j.create(new double[] { 0.111111, 0.3333333333333 }, new int[] { 1, 2 });
        INDArray out = net.output(input, false);
        System.out.println(out);

    }

    private static DataSetIterator getTrainingData(int batchSize, Random rand){
        double [] sum = new double[nSamples];
        double [] input1 = new double[nSamples];
        double [] input2 = new double[nSamples];
        for (int i= 0; i< nSamples; i++) {
            input1[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble();
            input2[i] =  MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble();
            sum[i] = input1[i] + input2[i];
        }
        INDArray inputNDArray1 = Nd4j.create(input1, new int[]{nSamples,1});
        INDArray inputNDArray2 = Nd4j.create(input2, new int[]{nSamples,1});
        INDArray inputNDArray = Nd4j.hstack(inputNDArray1,inputNDArray2);
        INDArray outPut = Nd4j.create(sum, new int[]{nSamples, 1});
        DataSet dataSet = new DataSet(inputNDArray, outPut);
        List<DataSet> listDs = dataSet.asList();
        Collections.shuffle(listDs,rng);
        return new ListDataSetIterator(listDs,batchSize);
    }




    public static void doGradientCheck() {
        double epsilon = 1e-3;
        int totalNFailures = 0;
        double maxRelError = 5.0; // in %
        CustomLossL1L2 lossfn = new CustomLossL1L2();
        String[] activationFns = new String[]{"identity", "softmax", "relu", "tanh", "sigmoid"};
        int[] labelSizes = new int[]{1, 2, 3, 4};
        for (int i = 0; i < activationFns.length; i++) {
            System.out.println("Running checks for "+activationFns[i]);
            IActivation activation = Activation.fromString(activationFns[i]).getActivationFunction();
            List<INDArray> labelList = makeLabels(activation,labelSizes);
            List<INDArray> preOutputList = makeLabels(new ActivationIdentity(),labelSizes);
            for (int j=0; j<labelSizes.length; j++) {
                System.out.println("\tRunning check for length " + labelSizes[j]);
                INDArray label = labelList.get(j);
                INDArray preOut = preOutputList.get(j);
                INDArray grad = lossfn.computeGradient(label,preOut,activation,null);
                NdIndexIterator iterPreOut = new NdIndexIterator(preOut.shape());
                while (iterPreOut.hasNext()) {
                    int[] next = iterPreOut.next();
                    //checking gradient with total score wrt to each output feature in label
                    double before = preOut.getDouble(next);
                    preOut.putScalar(next, before + epsilon);
                    double scorePlus = lossfn.computeScore(label, preOut, activation, null, true);
                    preOut.putScalar(next, before - epsilon);
                    double scoreMinus = lossfn.computeScore(label, preOut, activation, null, true);
                    preOut.putScalar(next, before);

                    double scoreDelta = scorePlus - scoreMinus;
                    double numericalGradient = scoreDelta / (2 * epsilon);
                    double analyticGradient = grad.getDouble(next);
                    double relError = Math.abs(analyticGradient - numericalGradient) * 100 / (Math.abs(numericalGradient));
                    if( analyticGradient == 0.0 && numericalGradient == 0.0 ) relError = 0.0;
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
        if(totalNFailures > 0) System.out.println("DONE:\n\tGradient check failed for loss function; total num failures = " + totalNFailures);
        else System.out.println("DONE:\n\tSimple gradient check passed - This is NOT exhaustive by any means");
    }

    /* This function is a utility function for the gradient check above
        It generate labels randomly in the right range depending on the activation function
        Uses a gaussian
        identity: range is anything
        relu: range is non-negative
        softmax: range is non-negative and adds up to 1
        sigmoid: range is between 0 and 1
        tanh: range is between -1 and 1

     */
    public static List<INDArray> makeLabels(IActivation activation,int[]labelSize) {
        //edge cases are label size of one for everything except softmax which is two
        //+ve and -ve values, zero and non zero values, less than zero/greater than zero
        List<INDArray> returnVals = new ArrayList<>(labelSize.length);
        for (int i=0; i< labelSize.length; i++) {
            int aLabelSize = labelSize[i];
            Random r = new Random();
            double[] someVals = new double[aLabelSize];
            double someValsSum = 0;
            for (int j=0; j<aLabelSize; j++) {
                double someVal = r.nextGaussian();
                double transformVal = 0;
                switch (activation.toString()) {
                    case "identity":
                        transformVal = someVal;
                    case "softmax":
                        transformVal = someVal;
                        break;
                    case "sigmoid":
                        transformVal = Math.sin(someVal);
                        break;
                    case "tanh":
                        transformVal = Math.tan(someVal);
                        break;
                    case "relu":
                        transformVal = someVal * someVal + 4;
                        break;
                    default:
                        throw new RuntimeException("Unknown activation function");
                }
                someVals[j] = transformVal;
                someValsSum += someVals[j];
            }
            if ("sigmoid".equals(activation.toString())) {
                for (int j=0; j<aLabelSize; j++) {
                    someVals[j] /= someValsSum;
                }
            }
            returnVals.add(Nd4j.create(someVals));
        }
        return returnVals;
    }
}
