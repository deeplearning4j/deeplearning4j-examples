/* *****************************************************************************
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

package org.deeplearning4j.datapipelineexamples.formats.hdfs.conversion;

import org.datavec.api.records.converter.RecordReaderConverter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.hadoop.records.reader.mapfile.MapFileRecordReader;
import org.datavec.hadoop.records.writer.mapfile.MapFileRecordWriter;
import org.deeplearning4j.datapipelineexamples.utils.DownloaderUtility;

import java.io.File;
import java.util.Random;

/**
 * A simple example on how to convert a CSV (text format) to a Hadoop MapFile format. After conversion, you can use the
 * MapFileRecordReader in place of the CSV reader for your code
 *
 * Why would you want to do this?
 * 1. Performance: reading from a map file is much faster than reading from a CSV (parse once, plus reading binary
 *    format instead of text format)
 * 2. Randomization: MapFileRecordReader supports randomization of iteration order
 *
 * Note that MapFileRecordReader/Writer are in datavec-hadoop package. You will also likely need hadoop-common.
 *
 * @author Alex Black
 */
public class CSVtoMapFileConversion {

    public static void main(String[] args) throws Exception {

        //Create CSV reader
        File irisFile = new File(DownloaderUtility.IRISDATA.Download(),"iris.txt");
        RecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(new FileSplit(irisFile));

        //Create map file writer
        File mapFileLoc = new File("mapFileOutputDir");
        MapFileRecordWriter writer = new MapFileRecordWriter(mapFileLoc);

        //Convert to MapFile binary format:
        RecordReaderConverter.convert(recordReader, writer);


        //Read back in from binary MapFile, random order:
        Random rng = new Random(12345);
        RecordReader mapFileReader = new MapFileRecordReader(rng);
        mapFileReader.initialize(new FileSplit(mapFileLoc));

        //Print out:
        while(mapFileReader.hasNext()){
            System.out.println(mapFileReader.next());
        }

    }

}
