package org.nd4j.samediff;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.impl.XavierInitScheme;

import java.util.HashMap;
import java.util.Map;

/**
 * This example shows how to implement a simple linear regression graph with a mean-squared error loss function.
 *
 * Specifically, we will implement:
 * - output = input * weights + bias
 * - loss = MSE(output, label) = 1/(nExamples * nOut) * sum_i (labels_i - out_i) ^ 2
 *
 * We will have:
 * nIn = 4
 * nOut = 1
 *
 * @author Alex Black
 */
public class Ex2_LinearRegression {

    public static void main(String[] args) {
        //How to calculate gradients, and get gradient arrays - linear regression (MSE, manually defined)

        int nIn = 4;
        int nOut = 2;

        SameDiff sd = SameDiff.create();


        //First: Let's create our placeholders. Shape: [minibatch, in/out]
        SDVariable input = sd.var("input", DataType.FLOAT, -1, nIn);
        SDVariable labels = sd.var("labels", DataType.FLOAT, -1, 1);

        //Second: let's create our variables
        SDVariable weights = sd.var("weights", new XavierInitScheme('c', nIn, nOut), DataType.FLOAT, nIn,nOut);
        SDVariable bias = sd.var("bias");


        //And define our forward pass:
        SDVariable out = input.mmul(weights).add(bias);     //Note: it's broadcast add here

        //And our loss function (done manually here for the purposes of this example):
        SDVariable difference = labels.sub(out);
        SDVariable sqDiff = sd.math().square(difference);
        SDVariable mse = sqDiff.mean("mse");

        //Let's create some mock data for this example:
        int minibatch = 3;
        Nd4j.getRandom().setSeed(12345);
        INDArray inputArr = Nd4j.rand(minibatch, nIn);
        INDArray labelArr = Nd4j.rand(minibatch, nOut);

        Map<String,INDArray> placeholderData = new HashMap<>();
        placeholderData.put("input", inputArr);
        placeholderData.put("labels", labelArr);

        //Execute forward pass:
        INDArray loss = sd.exec(placeholderData, "mse").get("mse");
        System.out.println("MSE: " + loss);

        //Calculate gradients:
        sd.execBackwards(placeholderData);

        //Get gradients for each variable:
        for(SDVariable v : new SDVariable[]{weights, bias}){
            System.out.println("Variable name: " + v.getVarName());
            System.out.println("Values:\n" + v.getArr());
            System.out.println("Gradients:\n" + v.getGradient().getArr());
        }
    }

}
