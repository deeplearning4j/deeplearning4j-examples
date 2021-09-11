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

package org.deeplearning4j.examples.advanced.features.transferlearning.iterators;

import org.apache.commons.io.FileUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.common.util.ArchiveUtils;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Random;

/**
 * Automatically downloads the dataset from
 * http://download.tensorflow.org/example_images/flower_photos.tgz
 * and untar's it to the users home directory
 * @author susaneraly on 3/9/17.
 */
public class FlowerDataSetIterator {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FlowerDataSetIterator.class);

    private static final String DATA_DIR = new File(System.getProperty("user.home")) + "/dl4j-examples-data/dl4j-examples";
    private static final String DATA_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz";
    private static final String FLOWER_DIR = DATA_DIR + "/flower_photos";

    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final Random rng  = new Random(13);

    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;
    private static final int numClasses = 5;

    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static InputSplit trainData,testData;
    private static int batchSize;

    public static DataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData);

    }

    public static DataSetIterator testIterator() throws IOException {
        return makeIterator(testData);

    }

    public static void setup(int batchSizeArg, int trainPerc) {
        try {
            downloadAndUntar();
        } catch (IOException e) {
            e.printStackTrace();
            log.error("IOException : ", e);
        }
        batchSize = batchSizeArg;
        File parentDir = new File(FLOWER_DIR);
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        if (trainPerc >= 100) {
            throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%. Test percentage = 100 - training percentage, has to be greater than 0");
        }
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 100-trainPerc);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];
    }

    private static DataSetIterator makeIterator(InputSplit split) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        recordReader.initialize(split);
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        iter.setPreProcessor( new VGG16ImagePreProcessor());
        return iter;
    }

    private static void downloadAndUntar() throws IOException {
         File rootFile = new File(DATA_DIR);
         if (!rootFile.exists()) {
             rootFile.mkdir();
         }
         File tarFile = new File(DATA_DIR, "flower_photos.tgz");
         if (!tarFile.isFile()) {
             log.info("Downloading the flower dataset from "+DATA_URL+ "...");
             FileUtils.copyURLToFile(
                     new URL(DATA_URL),
                     tarFile);
         }
         ArchiveUtils.unzipFileTo(tarFile.getAbsolutePath(), rootFile.getAbsolutePath());
    }
}
