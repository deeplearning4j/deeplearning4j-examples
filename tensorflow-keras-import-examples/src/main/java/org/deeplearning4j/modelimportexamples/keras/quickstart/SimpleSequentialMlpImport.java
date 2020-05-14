/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.modelimportexamples.keras.quickstart;

import org.deeplearning4j.modelimportexamples.utilities.DownloaderUtility;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;


/**
 * Basic example for importing a Keras Sequential model into DL4J for training or inference.
 *
 * Let's say you want to create a simple MLP in Keras using the Sequential API. The model
 * takes mini-batches of vectors of length 100, has two Dense layers and predicts a total
 * of 10 categories. Such a model can be defined and serialized in HDF5 format as follows:
 *
 *
 * ```
 * from keras.models import Sequential
 * from keras.layers import Dense
 *
 * model = Sequential()
 * model.add(Dense(units=64, activation='relu', input_dim=100))
 * model.add(Dense(units=10, activation='softmax'))
 * model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
 *
 * model.save('simple_mlp.h5')
 * ```
 * This model hasn't been fit on any data yet, it's just the model definition with initial weights
 * and training configuration as provided in the compile step.
 *
 * You don't need to create this model yourself to run this example, we stored it in the resources
 * folder under 'modelimport/keras' for you.
 *
 * This example shows you how to load the saved Keras model into DL4J for further use. You could
 * either use the imported model for inference only, or train it on data in DL4J.
 *
 *
 * @author Max Pumperla
 */
public class SimpleSequentialMlpImport {

    public static String dataLocalPath;

    public static void main(String[] args) throws Exception {
        dataLocalPath = DownloaderUtility.MODELIMPORT.Download();
        final String SIMPLE_MLP = new File(dataLocalPath,"keras/simple_mlp.h5").getAbsolutePath();

        // Keras Sequential models correspond to DL4J MultiLayerNetworks. We enforce loading the training configuration
        // of the model as well. If you're only interested in inference, you can safely set this to 'false'.
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(SIMPLE_MLP, true);

        // Test basic inference on the model.
        INDArray input = Nd4j.create(256, 100);
        INDArray output = model.output(input);

        // Test basic model training.
        model.fit(input, output);

        // Sanity checks for import. First, check it optimizer is correct.
        assert model.conf().getOptimizationAlgo().equals(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);

        // The first layer is a dense layer with 100 input and 64 output units, with RELU activation
        Layer first = model.getLayer(0);
        DenseLayer firstConf = (DenseLayer) first.conf().getLayer();
        assert firstConf.getActivationFn().equals(Activation.RELU.getActivationFunction());
        assert firstConf.getNIn() == 100;
        assert firstConf.getNOut() == 64;

        // The second later is a dense layer with 64 input and 10 output units, with Softmax activation.
        Layer second = model.getLayer(1);
        DenseLayer secondConf = (DenseLayer) second.conf().getLayer();
        assert secondConf.getActivationFn().equals(Activation.SOFTMAX.getActivationFunction());
        assert secondConf.getNIn() == 64;
        assert secondConf.getNOut() == 10;

        // The loss function of the Keras model gets translated into a DL4J LossLayer, which is the final
        // layer in this MLP.
        Layer loss = model.getLayer(2);
        LossLayer lossConf = (LossLayer) loss.conf().getLayer();
        assert lossConf.getLossFn().equals(LossFunctions.LossFunction.MCXENT);
    }
}
