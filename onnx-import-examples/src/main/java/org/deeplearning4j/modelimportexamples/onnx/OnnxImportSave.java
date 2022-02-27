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
import java.util.Collections;

public class OnnxImportSave {

    /**
     * We load yolov4 onnx model. You can always replace this with any URL found at:
     * https://github.com/onnx/models/
     *
     * The model we are focusing on in this example is the yolov4 model found here:
     * https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/yolov4
     *
     * Note that not all models may work. Please check with us on the forums before importing your model
     * and we can help you check coverage.
     * We suggest https://netron.app/ for visualizing and understanding models.
     * Models are expressed in terms of "operations" we maybe missing an operator you need for your graph.
     *
     * This can be true as well if a model has a proprietary op not found in the onnx standard.
     * This is common with pytorch.
     */
    public final static String MODEL_FILE_NAME = "yolov4.onnx";
    public final static String YOLOV4_URL = "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov4/model/" + MODEL_FILE_NAME;

    public static void main(String...args) throws Exception {
        //first create an importer which reads resource definitions such as available onnx and nd4j ops and uses those to parse files
        //annotation scanning for custom op overrides also happens here.
        //More on this can be found here: https://deeplearning4j.konduit.ai/samediff/explanation/model-import-framework
        OnnxFrameworkImporter onnxFrameworkImporter = new OnnxFrameworkImporter();
        File modelDir = new File(".","model-dir");
        modelDir.mkdirs();
        //download the model to the ./model-dir directory to load it
        Downloader.download(MODEL_FILE_NAME, URI.create(YOLOV4_URL).toURL(),new File(modelDir,MODEL_FILE_NAME),"",3);
        //use the importer to create the model
        //note we call true on a parameter called suggestDynamicVariables
        //the reason for this is we execute the graph in an eager fashion in order to compute shapes of different outputs at different points in the graph
        //not all graphs may have well defined input shapes though if this is the case, instead pass in dummy ndarrays in to the placeholder map
        SameDiff sameDiff = onnxFrameworkImporter.runImport(new File(modelDir,MODEL_FILE_NAME).getAbsolutePath(), Collections.emptyMap(), true);
        //print the graph showing that it imported
        System.out.println(sameDiff.summary());
        //save the model as samediff flatbuffers format
        //note: we recommend this because model import processes can be slow. Pre saving models allows them to be loaded later for inference.
        sameDiff.asFlatFile(new File("yolov4.fb"),true);
    }

}
