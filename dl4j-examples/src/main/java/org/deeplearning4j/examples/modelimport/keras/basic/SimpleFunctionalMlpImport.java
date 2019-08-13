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

package org.deeplearning4j.examples.modelimport.keras.basic;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.resources.Downloader;

import java.io.File;
import java.io.IOException;
import java.net.URL;


/**
 * Basic example for importing a Keras functional Model into DL4J for training or inference.
 *
 * Let's say you want to create a simple MLP in Keras using the functional API. The model
 * takes mini-batches of vectors of length 100, has two Dense layers and predicts a total
 * of 10 categories. Such a model can be defined and serialized in HDF5 format as follows:
 *
 *
 * ```
 * from keras.models import Model
 * from keras.layers import Dense, Input
 *
 * inputs = Input(shape=(100,))
 * x = Dense(64, activation='relu')(inputs)
 * predictions = Dense(10, activation='softmax')(x)
 * model = Model(inputs=inputs, outputs=predictions)
 * model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['accuracy'])
 *
 * model.save('simple_functional_mlp.h5')
 * ```
 *
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
public class SimpleFunctionalMlpImport {

    public static final String DATA_LOCAL_PATH;

    static {
        final String DATA_URL = "https://deeplearning4jblob.blob.core.windows.net/dl4j-examples/dl4j-examples/modelimport.zip";
        final String MD5 = "411df05aace1c9ff587e430a662ce621";
        final int DOWNLOAD_RETRIES = 10;
        final String DOWNLOAD_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "modelimport.zip");
        final String EXTRACT_DIR = FilenameUtils.concat(System.getProperty("user.home"), "dl4j-examples-data/dl4j-examples");
        DATA_LOCAL_PATH = FilenameUtils.concat(EXTRACT_DIR, "modelimport");
        if (!new File(DATA_LOCAL_PATH).exists()) {
            try {
                System.out.println("_______________________________________________________________________");
                System.out.println("Downloading data (3MB) and extracting to \n\t" + DATA_LOCAL_PATH);
                System.out.println("_______________________________________________________________________");
                Downloader.downloadAndExtract("files",
                    new URL(DATA_URL),
                    new File(DOWNLOAD_PATH),
                    new File(EXTRACT_DIR),
                    MD5,
                    DOWNLOAD_RETRIES);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            System.out.println("_______________________________________________________________________");
            System.out.println("Example data present in \n\t" + DATA_LOCAL_PATH);
            System.out.println("_______________________________________________________________________");
        }
    }


    public static void main(String[] args) throws Exception {
        final String SIMPLE_FUNCTIONAL_MLP = new File(DATA_LOCAL_PATH,"keras/simple_functional_mlp.h5").getAbsolutePath();

        // Keras functional Models correspond to DL4J ComputationGraphs. We enforce loading the training configuration
        // of the model as well. If you're only interested in inference, you can safely set this to 'false'.
        ComputationGraph model = KerasModelImport.importKerasModelAndWeights(SIMPLE_FUNCTIONAL_MLP, true);

        // Test basic inference on the model. Computation graphs take arrays of inputs and outputs, in this case of
        // length one.
        INDArray[] input = new INDArray[] {Nd4j.create(256, 100)};
        INDArray[] output = model.output(input);

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
