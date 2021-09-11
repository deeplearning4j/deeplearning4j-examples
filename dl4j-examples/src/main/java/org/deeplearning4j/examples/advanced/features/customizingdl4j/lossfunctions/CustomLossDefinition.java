/*******************************************************************************
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

package org.deeplearning4j.examples.advanced.features.customizingdl4j.lossfunctions;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author susaneraly
 */
public class CustomLossDefinition implements ILossFunction {

    /* This example illustrates how to implements a custom loss function that can then be applied to training your neural net
       All loss functions have to implement the ILossFunction interface
       The loss function implemented here is:
       L = (y - y_hat)^2 +  |y - y_hat|
        y is the true label, y_hat is the predicted output
     */

    private static Logger logger = LoggerFactory.getLogger(CustomLossDefinition.class);

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
    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr;
        // This is the output of the neural network, the y_hat in the notation above
        //To obtain y_hat: pre-output is transformed by the activation function to give the output of the neural network
        INDArray output = activationFn.getActivation(preOutput.dup(), true);
        //The score is calculated as the sum of (y-y_hat)^2 + |y - y_hat|
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
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
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
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
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
    */
    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray output = activationFn.getActivation(preOutput.dup(), true);
        /*
        //NOTE: There are many ways to do this same set of operations in nd4j
        //The following is the most readable for the sake of this example, not necessarily the fastest
        //Refer to the Implementation of LossL1 and LossL2 for more efficient ways
        */
        INDArray yMinusyHat = labels.sub(output);
        INDArray dldyhat = yMinusyHat.mul(-2).sub(Transforms.sign(yMinusyHat)); //d(L)/d(yhat) -> this is the line that will change with your loss function

        //Everything below remains the same
        INDArray dLdPreOut = activationFn.backprop(preOutput.dup(), dldyhat).getFirst();
        //multiply with masks, always
        if (mask != null) {
            dLdPreOut.muliColumnVector(mask);
        }

        return dLdPreOut;
    }

    //remains the same for a custom loss function
    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        return new Pair<>(
                computeScore(labels, preOutput, activationFn, mask, average),
                computeGradient(labels, preOutput, activationFn, mask));
    }

    @Override
    public String name() {
        return "CustomLossL1L2";
    }


    @Override
    public String toString() {
        return "CustomLossL1L2()";
    }

    public boolean equals(Object o) {
        if (o == this) return true;
        if (!(o instanceof CustomLossDefinition)) return false;
        final CustomLossDefinition other = (CustomLossDefinition) o;
        if (!other.canEqual((Object) this)) return false;
        return true;
    }

    public int hashCode() {
        int result = 1;
        return result;
    }

    protected boolean canEqual(Object other) {
        return other instanceof CustomLossDefinition;
    }
}

