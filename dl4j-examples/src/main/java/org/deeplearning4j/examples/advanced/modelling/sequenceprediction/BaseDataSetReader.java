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

package org.deeplearning4j.examples.advanced.modelling.sequenceprediction;

import org.nd4j.linalg.dataset.DataSet;

import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;

/**
 * @author WangFeng
 */
public abstract class BaseDataSetReader implements Serializable {

    protected Iterator<String> iter;
    protected Path filePath;
    private int totalExamples;
    int currentCursor;

    void doInitialize(){
        List<String> dataLines;
        try {
            dataLines = Files.readAllLines(filePath, StandardCharsets.UTF_8);
        } catch (Exception e) {
            throw new RuntimeException("loading data failed");
        }
        iter = dataLines.iterator();
        totalExamples = dataLines.size();
        currentCursor = 0;
    }

    public DataSet next(int num){
        return null;
    }

    public boolean hasNext() {
        return iter != null && iter.hasNext();
    }

    public List<String> getLabels() {
        return null;
    }

    public void reset() {
        doInitialize();
    }
    int totalExamples() {
        return totalExamples;
    }
}
