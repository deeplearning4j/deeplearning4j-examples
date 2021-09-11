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

package org.deeplearning4j.examples.advanced.features.customizingdl4j.activationfunctions;

import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.strict.TanDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Tanh;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This is an example of how to implement a custom activation function that does not take any learnable parameters
 * Custom activation functions of this case should extend from BaseActivationFunction and implement the methods
 * shown here.
 * IMPORTANT: Do not forget gradient checks. Refer to these in the deeplearning4j repo,
 * deeplearning4j-core/src/test/java/org/deeplearning4j/gradientcheck/LossFunctionGradientCheck.java
 *
 * The form of the activation function implemented here is from https://arxiv.org/abs/1508.01292
 * "Compact Convolutional Neural Network Cascade for Face Detection" by Kalinovskii I.A. and Spitsyn V.G.
 *
 *      h(x) = 1.7159 tanh(2x/3)
 *
 * @author susaneraly
 */
public class CustomActivationDefinition extends BaseActivationFunction{

    /*
        For the forward pass:

        Transform "in" with the activation function. Best practice is to do the transform in place as shown below
        Can support different behaviour during training and test with the boolean argument
     */
    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        //Modify array "in" inplace to transform it with the activation function
        // h(x) = 1.7159*tanh(2x/3)
        Nd4j.getExecutioner().execAndReturn(new Tanh(in.muli(2/3.0)));
        in.muli(1.7159);
        return in;
    }

    /*
        For the backward pass:
        Given epsilon, the gradient at every activation node calculate the next set of gradients for the backward pass
        Best practice is to modify in place.

        Using the terminology,
            in -> linear input to the activation node
            out    -> the output of the activation node, or in other words h(out) where h is the activation function
            epsilon -> the gradient of the loss function with respect to the output of the activation node, d(Loss)/dout

                h(in) = out;
                d(Loss)/d(in) = d(Loss)/d(out) * d(out)/d(in)
                              = epsilon * h'(in)
     */
    @Override
    public Pair<INDArray,INDArray> backprop(INDArray in, INDArray epsilon) {
        //dldZ here is h'(in) in the description above
        //
        //      h(x) = 1.7159*tanh(2x/3);
        //      h'(x) = 1.7159*[tanh(2x/3)]' * 2/3
        INDArray dLdz = Nd4j.getExecutioner().exec(new TanDerivative(in.muli(2/3.0)));
        dLdz.muli(2/3.0);
        dLdz.muli(1.7159);

        //Multiply with epsilon
        dLdz.muli(epsilon);
        return new Pair<>(dLdz, null);
    }

}
