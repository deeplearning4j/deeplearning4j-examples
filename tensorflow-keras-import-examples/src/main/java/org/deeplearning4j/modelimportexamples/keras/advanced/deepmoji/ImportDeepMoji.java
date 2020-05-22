/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

package org.deeplearning4j.modelimportexamples.keras.advanced.deepmoji;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.net.URL;

/**
 * Test import of DeepMoji application
 *
 * @author Max Pumperla
 */
public class ImportDeepMoji {

    public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"),
        "dl4j_keras/");


    public static void main(String[] args) throws Exception {


        // First, register the Keras layer wrapped around our custom SameDiff attention layer
        KerasLayer.registerCustomLayer("AttentionWeightedAverage", KerasDeepMojiAttention.class);

        // Then, download the model from azure (check if it's cached)
        File directory = new File(DATA_PATH);
        if(!directory.exists()) directory.mkdir();

        String modelUrl = "http://blob.deeplearning4j.org/models/deepmoji.h5";
        String downloadPath = DATA_PATH + "deepmoji_model.h5";
        File cachedKerasFile = new File(downloadPath);

        if (!cachedKerasFile.exists()) {
            System.out.println("Downloading model to " + cachedKerasFile.toString());
            FileUtils.copyURLToFile(new URL(modelUrl), cachedKerasFile);
            System.out.println("Download complete");
            cachedKerasFile.deleteOnExit();
        }

        // Finally, import the model and test on artificial input data.
        ComputationGraph graph = KerasModelImport.importKerasModelAndWeights(cachedKerasFile.getAbsolutePath());;
        INDArray input = Nd4j.create(new int[] {10, 30});
        graph.output(input);
        System.out.println("Example completed.");
    }
}
