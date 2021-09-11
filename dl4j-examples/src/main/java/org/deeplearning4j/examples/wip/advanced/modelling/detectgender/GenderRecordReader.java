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

package org.deeplearning4j.examples.wip.advanced.modelling.detectgender;

/**
 * Created by KIT Solutions (www.kitsol.com) on 11/7/2016.
 */

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.InputStreamInputSplit;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.common.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.Charset;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;


/**
 * GenderRecordReader class does following job
 * - Initialize method reads .CSV file as specified in Labels in constructor
 * - It loads person name and gender data into binary converted data
 * - creates binary string iterator which can be used by RecordReaderDataSetIterator
 */

public class GenderRecordReader extends LineRecordReader {
    // list to hold labels passed in constructor
    private List<String> labels;

    // Final list that contains actual binary data generated from person name, it also contains label (1 or 0) at the end
    private List<String> names = new ArrayList<String>();

    // This String is used to convert person name to binary string seperated by comma
    private static String possibleCharacters = " abcdefghijklmnopqrstuvwxyz";

    // holds length of largest name out of all person names
    public static int maxLengthName = 88;

    // holds total number of names including both male and female names.
    // This variable is not used in PredictGenderTrain.java
    private int totalRecords = 0;

    // iterator for List "names" to be used in next() method
    private Iterator<String> iter;

    /**
     * Constructor to allow client application to pass List of possible Labels
     *
     * @param labels - List of String that client application pass all possible labels, in our case "M" and "F"
     */
    public GenderRecordReader(List<String> labels) {
        this.labels = labels;
        //this.labels = this.labels.stream().map(element -> element + ".csv").collect(Collectors.toList());
        //System.out.println("labels : " + this.labels);
    }

    /**
     * returns total number of records in List "names"
     *
     * @return - totalRecords
     */
    private int totalRecords() {
        return totalRecords;
    }


    /**
     * This function does following steps
     * - Looks for the files with the name (in specified folder) as specified in labels set in constructor
     * - File must have person name and gender of the person (M or F),
     * e.g. Deepan,M
     * Trupesh,M
     * Vinay,M
     * Ghanshyam,M
     * <p>
     * Meera,F
     * Jignasha,F
     * Chaku,F
     * <p>
     * - File for male and female names must be different, like M.csv, F.csv etc.
     * - populates all names in temporary list
     * - generate binary string for each alphabet for all person names
     * - combine binary string for all alphabets for each name
     * - find all unique alphabets to generate binary string mentioned in above step
     * - take equal number of records from all files. To do that, finds minimum record from all files, and then takes
     * that number of records from all files to keep balance between data of different labels.
     * - Note : this function uses stream() feature of Java 8, which makes processing faster. Standard method to process file takes more than 5-7 minutes.
     * using stream() takes approximately 800-900 ms only.
     * - Final converted binary data is stored List<String> names variable
     * - sets iterator from "names" list to be used in next() function
     *
     * @param split - user can pass directory containing .CSV file for that contains names of male or female
     * @throws IOException
     * @throws InterruptedException
     */
    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        if (split instanceof FileSplit) {
            URI[] locations = split.locations();
            if (locations != null && locations.length > 1) {
                List<Pair<String, List<String>>> tempNames = new ArrayList<Pair<String, List<String>>>();
                for (URI location : locations) {
                    File file = new File(location);
                    List<String> temp = this.labels.stream().filter(line -> file.getName().equals(line + ".csv")).collect(Collectors.toList());
                    if (temp.size() > 0) {
                        java.nio.file.Path path = Paths.get(file.getAbsolutePath());
                        List<String> tempList = java.nio.file.Files.readAllLines(path, Charset.defaultCharset()).stream().map(element -> element.split(",")[0]).collect(Collectors.toList());
                        Pair<String, List<String>> tempPair = new Pair<String, List<String>>(temp.get(0), tempList);
                        tempNames.add(tempPair);
                    } else {
                        if (!file.getName().equals("PredictGender.net"))
                            throw new InterruptedException("File missing for any of the specified labels");
                    }
                }


                Pair<String, List<String>> tempPair = tempNames.get(0);
                int minSize = tempPair.getSecond().size();
                for (int i = 1; i < tempNames.size(); i++) {
                    if (minSize > tempNames.get(i).getSecond().size())
                        minSize = tempNames.get(i).getSecond().size();
                }

                List<Pair<String, List<String>>> oneMoreTempNames = new ArrayList<Pair<String, List<String>>>();
                for (int i = 0; i < tempNames.size(); i++) {
                    int diff = Math.abs(minSize - tempNames.get(i).getSecond().size());
                    List<String> tempList = new ArrayList<String>();

                    if (tempNames.get(i).getSecond().size() > minSize) {
                        tempList = tempNames.get(i).getSecond();
                        tempList = tempList.subList(0, tempList.size() - diff);
                    } else
                        tempList = tempNames.get(i).getSecond();
                    Pair<String, List<String>> tempNewPair = new Pair<String, List<String>>(tempNames.get(i).getFirst(), tempList);
                    oneMoreTempNames.add(tempNewPair);
                }
                tempNames.clear();

                List<Pair<String, List<String>>> secondMoreTempNames = new ArrayList<Pair<String, List<String>>>();

                for (int i = 0; i < oneMoreTempNames.size(); i++) {
                    int gender = oneMoreTempNames.get(i).getFirst().equals("M") ? 1 : 0;
                    List<String> secondList = oneMoreTempNames.get(i).getSecond().stream().map(element -> getBinaryString(element.split(",")[0], gender)).collect(Collectors.toList());
                    Pair<String, List<String>> secondTempPair = new Pair<String, List<String>>(oneMoreTempNames.get(i).getFirst(), secondList);
                    secondMoreTempNames.add(secondTempPair);
                }
                oneMoreTempNames.clear();

                for (int i = 0; i < secondMoreTempNames.size(); i++) {
                    names.addAll(secondMoreTempNames.get(i).getSecond());
                }
                secondMoreTempNames.clear();
                this.totalRecords = names.size();
                Collections.shuffle(names);
                this.iter = names.iterator();
            } else
                throw new InterruptedException("File missing for any of the specified labels");
        } else if (split instanceof InputStreamInputSplit) {
            System.out.println("InputStream Split found...Currently not supported");
            throw new InterruptedException("File missing for any of the specified labels");
        }
    }


    /**
     * - takes one record at a time from names list using iter iterator
     * - stores it into Writable List and returns it
     *
     * @return
     */
    @Override
    public List<Writable> next() {
        if (iter.hasNext()) {
            List<Writable> ret = new ArrayList<>();
            String currentRecord = iter.next();
            String[] temp = currentRecord.split(",");
            for (int i = 0; i < temp.length; i++) {
                ret.add(new DoubleWritable(Double.parseDouble(temp[i])));
            }
            return ret;
        } else
            throw new IllegalStateException("no more elements");
    }

    @Override
    public boolean hasNext() {
        if (iter != null) {
            return iter.hasNext();
        }
        throw new IllegalStateException("Indeterminant state: record must not be null, or a file iterator must exist");
    }

    @Override
    public void close() throws IOException {

    }

    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    @Override
    public Configuration getConf() {
        return conf;
    }

    @Override
    public void reset() {
        this.iter = names.iterator();
    }

    /**
     * This function gives binary string for full name string
     * - It uses "PossibleCharacters" string to find the decimal equivalent to any alphabet from it
     * - generate binary string for each alphabet
     * - left pads binary string for each alphabet to make it of size 5
     * - combine binary string for all alphabets of a name
     * - Right pads complete binary string to make it of size that is the size of largest name to keep all name length of equal size
     * - appends label value (1 or 0 for male or female respectively)
     *
     * @param name   - person name to be converted to binary string
     * @param gender - variable to decide value of label to be added to name's binary string at the end of the string
     * @return
     */
    private String getBinaryString(String name, int gender) {
        return nameToBinary(name) + "," + String.valueOf(gender);
    }

    public static String nameToBinary(String name) {
        String binaryString = "";
        for (int j = 0; j < name.length(); j++) {
            String fs = org.apache.commons.lang3.StringUtils.leftPad(Integer.toBinaryString(possibleCharacters.indexOf(name.charAt(j))), 5, "0");
            binaryString = binaryString + fs;
        }
        binaryString = org.apache.commons.lang3.StringUtils.rightPad(binaryString, maxLengthName * 5, "0");
        binaryString = binaryString.replaceAll(".(?!$)", "$0,");
        return binaryString;

    }

}
