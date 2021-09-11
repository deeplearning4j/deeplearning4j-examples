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

package org.deeplearning4j.examples.advanced.features.inference;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.ParallelInference;
import org.deeplearning4j.parallelism.inference.InferenceMode;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;

/**
 * This examples shows use of ParallelInference mechanism
 * Parallel Inference takes requests from multiple threads,
 * collects them for a short while, and then queries the model for all collected requests.
 * Since the model works in parallel internally, the available resources are still fully utilized.
 *
 * Refer to: https://www.dubs.tech/guides/quickstart-with-dl4j/#parallel-inference for more information
 * @author raver119@gmail.com
 */
public class ParallelInferenceExample {

    public static void main(String[] args) throws Exception {

        // use path to your model here, or just instantiate it anywhere
        MultiLayerNetwork model =MultiLayerNetwork.load(new File("PATH_TO_YOUR_MODEL_FILE"), false);

        ParallelInference pi = new ParallelInference.Builder(model)
            // BATCHED mode is kind of optimization: if number of incoming requests is too high - PI will be batching individual queries into single batch. If number of requests will be low - queries will be processed without batching
            .inferenceMode(InferenceMode.BATCHED)

            // max size of batch for BATCHED mode. you should set this value with respect to your environment (i.e. gpu memory amounts)
            .batchLimit(32)

            // set this value to number of available computational devices, either CPUs or GPUs
            .workers(2)

            .build();


        // PLEASE NOTE: this output() call is just a placeholder, you should pass data in the same dimensionality you had during training
        INDArray result = pi.output(new float[] {0.1f, 0.1f, 0.1f, 0.2f, 0,3f });
    }
}
