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

package org.datavec.inputsplit;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.nd4j.resources.Downloader;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Iterator;
import java.util.Random;

/**
 * {@link org.datavec.api.split.BaseInputSplit} and its implementation provides a
 * {@link org.datavec.api.split.BaseInputSplit#sample(PathFilter, double...)} method that is very useful for generating
 * several {@link org.datavec.api.split.InputSplit}s from the main split.
 * <p>
 * This can be used for dividing your dataset into several subsets. For example, into training, validation and testing.
 * <p>
 * The {@link PathFilter} is useful for filtering the main split before generating the input splits array.
 * The second argument is a list of weights, which indicate a percentage of each input split.
 * <p>
 * The samples are divided in the following way -> totalSamples * (weight1/totalWeightSum, weight2/totalWeightSum, ...,
 * weightN/totalWeightSum)
 * <p>
 * {@link PathFilter} has two default implementations,
 * {@link org.datavec.api.io.filters.RandomPathFilter} that simple randomizes the order of paths in an array.
 * and
 * {@link org.datavec.api.io.filters.BalancedPathFilter} that randomizes the order of paths in an array and removes
 * paths randomly to have the same number of paths for each label. Further interlaces the paths on output based on
 * their labels, to obtain easily optimal batches for training.
 * <p>
 * Their usages are shown here.
 */
public class Ex05_SamplingBaseInputSplitExample {

    public static final String DATA_LOCAL_PATH;

    static {
        final String DATA_URL = "https://deeplearning4jblob.blob.core.windows.net/dl4j-examples/datavec-examples/inputsplit.zip";
        final String MD5 = "f316b5274bab3b0f568eded9bee1c67f";
        final int DOWNLOAD_RETRIES = 10;
        final String DOWNLOAD_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "inputsplit.zip");
        final String EXTRACT_DIR = FilenameUtils.concat(System.getProperty("user.home"), "dl4j-examples-data/datavec-examples");
        DATA_LOCAL_PATH = FilenameUtils.concat(EXTRACT_DIR, "inputsplit");
        if (!new File(DATA_LOCAL_PATH).exists()) {
            try {

                System.out.println("_______________________________________________________________________");
                System.out.println("Downloading data (128KB) and extracting to \n\t" + DATA_LOCAL_PATH);
                System.out.println("_______________________________________________________________________");
                Downloader.downloadAndExtract("files",
                    new URL(DATA_URL),
                    new File(DOWNLOAD_PATH),
                    new File(EXTRACT_DIR),
                    MD5,
                    DOWNLOAD_RETRIES);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            System.out.println("_______________________________________________________________________");
            System.out.println("Example data present in \n\t" + DATA_LOCAL_PATH);
            System.out.println("_______________________________________________________________________");
        }

    }

    public static void main(String[] args) throws Exception {
        FileSplit fileSplit = new FileSplit(new File(DATA_LOCAL_PATH, "files"));

        //Sampling with a RandomPathFilter
        InputSplit[] inputSplits1 = fileSplit.sample(
            new RandomPathFilter(new Random(123), null),
            10, 10, 10, 10, 10);

        System.out.println(String.format(("Random filtered splits -> Total(%d) = Splits of (%s)"), fileSplit.length(),
            String.join(" + ", () -> new InputSplitLengthIterator(inputSplits1))));

        //Sampling with a BalancedPathFilter
        InputSplit[] inputSplits2 = fileSplit.sample(
            new BalancedPathFilter(new Random(123), null, new ParentPathLabelGenerator()),
            10, 10, 10, 10, 10);

        System.out.println(String.format(("Balanced Splits are: %s"),
            String.join(" + ", () -> new InputSplitLengthIterator(inputSplits2))));
    }

    private static class InputSplitLengthIterator implements Iterator<CharSequence> {

        InputSplit[] inputSplits;
        int i;

        public InputSplitLengthIterator(InputSplit[] inputSplits) {
            this.inputSplits = inputSplits;
            this.i = 0;
        }

        @Override
        public boolean hasNext() {
            return i < inputSplits.length;
        }

        @Override
        public CharSequence next() {
            return String.valueOf(inputSplits[i++].length());
        }
    }
}
