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

package org.deeplearning4j.examples.advanced.modelling.sequenceanomalydetection;


import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * @author wangfeng
 */
public class AnomalyDataSetReader {
    private int skipNumLines;
    private int skipNumColumns;
    private int longestTimeSequence;
    private int shortest;
    private Iterator<List<Writable>> iter;
    private Path filePath;
    private int totalExamples;
    private Queue<String> currentLines;

    public AnomalyDataSetReader(File file) {
        this.skipNumLines = 1;
        this.skipNumColumns = 2;
        this.longestTimeSequence = 0;
        this.shortest = 1;
        this.filePath = file.toPath();
        this.currentLines =  new LinkedList<String>();
        doInitialize();
    }
    public void doInitialize(){
        List<List<Writable>> dataLines = new ArrayList<>();
        try {
            List<String> lines = Files.readAllLines(filePath, Charset.forName("UTF-8"));
            for (int i = skipNumLines; i < lines.size(); i ++) {
                String tempStr = lines.get(i).replaceAll("\"", "");
                currentLines.offer(tempStr);
                int templength = tempStr.split(",").length - skipNumColumns;
                longestTimeSequence = longestTimeSequence < templength? templength:longestTimeSequence;
                List<Writable> dataLine = new ArrayList<>();
                String[] wary= tempStr.split(",");
                for (int j = skipNumColumns; j < wary.length; j++ ) {
                    dataLine.add(new Text(wary[j]));
                }
                dataLines.add(dataLine);
            }
        } catch (Exception e) {
            throw new RuntimeException("loading data failed");
        }
        iter = dataLines.iterator();
        totalExamples = dataLines.size();
    }

    public DataSet next(int num) {

        INDArray features = Nd4j.create(new int[]{num, shortest, longestTimeSequence}, 'f');
        INDArray featuresMask = Nd4j.ones(num, longestTimeSequence);
        for (int i = 0, k = 0; i < num && iter.hasNext(); i ++) {
            List<Writable> line= iter.next();
            int index = 0;
            for (Writable w: line) {
                features.putScalar(new int[]{i, k, index}, w.toDouble());
                ++index;
            }
            if (line.size() < longestTimeSequence) {// the default alignmentMode is ALIGN_START
                for(int step = line.size(); step < longestTimeSequence; step++) {
                    featuresMask.putScalar(i, step, 0.0D);
                }
            }
        }
        return new DataSet(features, features, featuresMask, featuresMask);
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
    public int totalExamples() {
        return totalExamples;
    }

    public Queue<String> currentLines() {
        return currentLines;
    }

}
