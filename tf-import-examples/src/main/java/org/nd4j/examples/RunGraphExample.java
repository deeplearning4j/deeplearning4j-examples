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

package org.nd4j.examples;

import org.apache.commons.io.IOUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.examples.download.DownloaderUtility;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;

import java.io.File;
import java.io.FileInputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Runs a tensorflow graph using the tensorflow graph runner.
 *
 * @author Adam Gibson
 */
public class RunGraphExample {

    public static void main(String...args) throws Exception {
        //input name (usually with place holders)
        String dataLocalPath = DownloaderUtility.TFIMPORTEXAMPLES.Download();
        List<String> inputs = Arrays.asList("flatten_2_input");
        //load the graph from the classpath
        byte[] content = IOUtils.toByteArray(new FileInputStream(new File(dataLocalPath,"Mnist/mnist.pb")));
        DataSetIterator dataSetIterator = new MnistDataSetIterator(1,1);
        INDArray predict = dataSetIterator.next().getFeatures();
        //run the graph using nd4j
        try(GraphRunner graphRunner = new GraphRunner(content,inputs)) {
            Map<String,INDArray> inputMap = new HashMap<>();
            inputMap.put(inputs.get(0),predict);
            Map<String, INDArray> run = graphRunner.run(inputMap);
            System.out.println("Run result " + run);
        }

    }
}
