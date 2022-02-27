/* *****************************************************************************
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
package org.deeplearning4j.modelimportexamples.onnx;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.resources.Downloader;
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;

import java.io.File;
import java.net.URI;
import java.util.Arrays;
import java.util.Collections;

/**
 * Note that you should run the {@link OnnxImportSave}
 * main class first that shows how to import and download a model.
 * This class mainly shows how to use the output of that process.
 *
 * TODO:
 *  Integrate {@link ImageProcessUtils}
 *  and show training/inference on new images.
 */
public class OnnxImportLoad {

    /**
     * Load the model for training/finetuning.
     */
    public final static String MODEL_FILE_NAME = "yolov4.fb";

    public static void main(String...args) throws Exception {
        //load the imported model
        SameDiff sameDiff = SameDiff.load(new File(MODEL_FILE_NAME),true);
        //print the input names
        System.out.println(sameDiff.inputs());
        //print the shape of the input so we know what to feed the model
        System.out.println(Arrays.toString(sameDiff.getVariable(sameDiff.inputs().get(0)).getShape()));

    }

}
