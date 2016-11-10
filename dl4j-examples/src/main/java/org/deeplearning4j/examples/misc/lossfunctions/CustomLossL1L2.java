package org.deeplearning4j.examples.misc.lossfunctions;

import lombok.EqualsAndHashCode;
import org.apache.commons.math3.ode.MainStateJacobianProvider;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;

/**
 * Created by susaneraly on 11/8/16.
 */
@EqualsAndHashCode
public class CustomLossL1L2 implements ILossFunction {

    /* This example illustrates how to implements a custom loss function that can then be applied to training your neural net
       All loss functions have to implement the ILossFunction interface
       The loss function implemented here is:
       L = (y - y_hat)^2 +  |y - y_hat|
        y is the true label, y_hat is the predicted output
     */

    private static Logger logger = LoggerFactory.getLogger(CustomLossL1L2.class);

    /*
    Needs modification depending on your loss function
        scoreArray calculates the loss for a single data point or in other words a batch size of one
        It returns an array the shape and size of the output of the neural net.
        Each element in the array is the loss function applied to the prediction and it's true value
        scoreArray takes in:
        true labels - labels
        the input to the final/output layer of the neural network - preOutput,
        the activation function on the final layer of the neural network - activationFn
        the mask - (if there is a) mask associated with the label
     */
    private INDArray scoreArray(INDArray labels, INDArray preOutput, String activationFn, INDArray mask) {
        INDArray scoreArr;
        // This is the output of the neural network, the y_hat in the notation above
        //To obtain y_hat: pre-output is transformed by the activation function to give the output of the neural network
        INDArray output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));
        //The score is calculated as the sum of (y-y_hat)^2 + (y - y_hat)
        INDArray yMinusyHat = Transforms.abs(labels.sub(output));
        scoreArr = yMinusyHat.mul(yMinusyHat);
        scoreArr.addi(yMinusyHat);
        if (mask != null) {
            scoreArr.muliColumnVector(mask);
        }
        return scoreArr;
    }

    /*
    Remains the same for all loss functions
    Compute Score computes the average loss function across many datapoints.
    The loss for a single datapoint is summed over all output features.
     */
    @Override
    public double computeScore(INDArray labels, INDArray preOutput, String activationFn, INDArray mask, boolean average) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);

        double score = scoreArr.sumNumber().doubleValue();

        if (average) {
            score /= scoreArr.size(0);
        }

        return score;
    }

    /*
    Remains the same for all loss functions
    Compute Score computes the loss function for many datapoints.
    The loss for a single datapoint is the loss summed over all output features.
    Returns an array that is #of samples x size of the output feature
     */
    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, String activationFn, INDArray mask) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
        return scoreArr.sum(1);
    }

    /*
    Needs modification depending on your loss function
        Compute the gradient wrt to the preout (which is the input to the final layer of the neural net)
        Use the chain rule
        In this case L = (y - yhat)^2 + |y - yhat|
        dL/dyhat = -2*(y-yhat) - sign(y-yhat), sign of y - yhat = +1 if y-yhat>= 0 else -1
        dyhat/dpreout = d(Activation(preout))/dpreout = Activation'(preout)
        dL/dpreout = dL/dyhat * dyhat/dpreout
        Activation function of softmax requires special handling, see below
    */
    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, String activationFn, INDArray mask) {
        INDArray output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));
        /*
        //NOTE: There are many ways to do this same set of operations in nd4j
        //The following is the most readable for the sake of this example, not necessarily the fastest
        //Refer to the Implementation of LossL1 and LossL2 for more efficient ways
        */
        INDArray yMinusyHat = labels.sub(output);
        INDArray dldyhat = yMinusyHat.mul(-2).sub(Transforms.sign(yMinusyHat)); //d(L)/d(yhat) -> this is the line that will change with your loss function
        //INDArray dldyhat = yMinusyHat.mul(-2);

        //Everything below remains the same
        INDArray gradients;
        if ("softmax".equals(activationFn)) {
            gradients = LossUtil.dLdZsoftmaxi(dldyhat, output); //special handling for softmax
        } else {
            //dyhat/dpreout, derivative of yhat wrt preoutput
            INDArray dyhatdpreOutput = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()).derivative());
            gradients = dldyhat.muli(dyhatdpreOutput); //chain rule
        }
        //multiply with masks, always
        if (mask != null) {
            gradients.muliColumnVector(mask);
        }

        return gradients;
    }

    //remains the same for a custom loss function
    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, String activationFn, INDArray mask, boolean average) {
        return new Pair<>(
                computeScore(labels, preOutput, activationFn, mask, average),
                computeGradient(labels, preOutput, activationFn, mask));
    }


    @Override
    public String toString() {
        return "CustomLossL1L2()";
    }

    //THE FOLLOWING IS TO ILLUSTRATE A SIMPLE GRADIENT CHECK.
    //It checks the implementation against the finite difference approximation
    //References for more thorough gradient checks in dl4j:
    //  deeplearning4j/deeplearning4j-core/src/test/java/org/deeplearning4j/gradientcheck/LossFunctionGradientCheck.java
    public static void main(String[] args) throws IOException {
        doGradientCheck();
    }

    public static void doGradientCheck() {
        double epsilon = 1e-3;
        int totalNFailures = 0;
        double maxRelError = 5.0; // in %
        CustomLossL1L2 lossfn = new CustomLossL1L2();
        String[] activationFns = new String[]{"identity", "softmax", "relu", "tanh", "sigmoid"};
        //String[] activationFns = new String[]{"identity"};
        int[] labelSizes = new int[]{1, 2, 3, 4};
        //identity labels
        //relu is non-negative
        //softmax adds up to 1
        //sigmoid is between 0 and 1
        //tanh is between -1 and 1
        for (int i = 0; i < activationFns.length; i++) {
            System.out.println("Running checks for "+activationFns[i]);
            String activation = activationFns[i];
            List<INDArray> labelList = makeLabels(activation,labelSizes);
            List<INDArray> preOutputList = makeLabels("identity",labelSizes);
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
    public static List<INDArray> makeLabels(String activation,int[]labelSize) {
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
                switch (activation) {
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
                    case "reul":
                        transformVal = someVal * someVal + 4;
                        break;
                }
                    someVals[j] = transformVal;
                    someValsSum += someVals[j];
            }
            if (activation == "sigmoid") {
                for (int j=0; j<aLabelSize; j++) {
                    someVals[j] /= someValsSum;
                }
            }
            returnVals.add(Nd4j.create(someVals));
        }
        return returnVals;
    }
}

