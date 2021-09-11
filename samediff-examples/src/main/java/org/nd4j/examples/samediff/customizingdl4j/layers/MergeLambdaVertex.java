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

package org.nd4j.examples.samediff.customizingdl4j.layers;

import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaVertex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

public class MergeLambdaVertex extends SameDiffLambdaVertex {
    @Override
    public SDVariable defineVertex(SameDiff sameDiff, VertexInputs inputs) {
        //2 inputs to the vertex. The VertexInputs class will dynamically add as many variables as we request from it!
        SDVariable input1 = inputs.getInput(0);
        SDVariable input2 = inputs.getInput(1);
        //SDVariable average = input1.add(input2).mul(0.5);
        SDVariable average = sameDiff.math.mergeAvg(input1,input2);
        return average;
    }
}
