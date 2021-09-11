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

package org.deeplearning4j.examples.wip.quickstart.features.knn;

import org.apache.commons.io.IOUtils;
import org.datavec.image.loader.CifarLoader;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.time.Instant;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

import static java.time.temporal.ChronoUnit.MILLIS;
import static org.datavec.image.loader.CifarLoader.*;
import static org.nd4j.linalg.ops.transforms.Transforms.*;

/**
 * Issue Summary: Takes too long to be a realistic example. Needs to be updated to something that runs quicker.
 * Might also be a good idea to update knn documentation and do a pass on performance.
 *
 * Github Issue: https://github.com/eclipse/deeplearning4j-examples/issues/976
 *
 * Simple example for Nearest Neighbor classification.
 * Please @see <a href="http://cs231n.github.io/classification/#nn">cs123</a> for explanation.
 * Using Manhattan distance for L1 and Euclidean for L2.
 */
public class NearestNeighbor {

    private static Logger log = LoggerFactory.getLogger(NearestNeighbor.class);

    public static void main(String[] args) throws IOException {
        processCifar10Images();
    }

    private static void processCifar10Images() throws IOException {
        Map<INDArray, Byte> trainingMap = readTrainingData();

        CifarLoader cifarLoader = new CifarLoader(false);
        final byte[] testImageData = IOUtils.toByteArray(cifarLoader.getInputStream());
        int imageLen = HEIGHT * WIDTH * CHANNELS;
        Random random = new Random(100);
        final int numberOfEpochs = 10;
        long timeTaken = 0;
        for (int epochIndex = 0; epochIndex < numberOfEpochs; epochIndex++) {
            log.info("Epoch " + epochIndex);
            final Instant start = Instant.now();
            float l1MatchCount = 0f, l2MatchCount = 0f;
            //Test Random 20 images
            final int numberOfImagesToTest = 20;
            for (int i = 0; i < numberOfImagesToTest; i++) {
                int imageIndex = random.nextInt(10000) * (imageLen + 1);
                final byte[] imageByteArray = Arrays.copyOfRange(testImageData, imageIndex + 1, imageIndex + (imageLen + 1));
                final double[] imageDoubles = IntStream.range(0, imageByteArray.length).mapToDouble(idx -> imageByteArray[idx]).toArray();
                final INDArray testImage = abs(Nd4j.create(imageDoubles));
                final Byte testLabel = testImageData[imageIndex];
                l1MatchCount += trainingMap.entrySet().stream()
                            .min((o1, o2) -> {
                                final double o1Difference = manhattanDistance(testImage, o1.getKey());
                                final double o2Difference = manhattanDistance(testImage, o2.getKey());
                                return (int) Math.abs(o1Difference - o2Difference);
                            })
                            .map(entry -> entry.getValue().equals(testLabel) ? 1 : 0).get();
                l2MatchCount += trainingMap.entrySet().stream()
                    .min((o1, o2) -> {
                        final double o1Difference = euclideanDistance(testImage, o1.getKey());
                        final double o2Difference = euclideanDistance(testImage, o2.getKey());
                        return (int) Math.abs(o1Difference - o2Difference);
                    })
                    .map(entry -> entry.getValue().equals(testLabel) ? 1 : 0).get();
            }
            log.info("Manhattan distance accuracy = " + (l1MatchCount / 20f) * 100f + "%");
            log.info("Euclidean distance accuracy = " + (l2MatchCount / 20f) * 100f + "%");
            timeTaken += MILLIS.between(start, Instant.now());
        }
        log.info("Average time = " + timeTaken/numberOfEpochs);
    }

    @NotNull
    private static Map<INDArray, Byte> readTrainingData() throws IOException {
        log.info("Reading training data.");
        Map<INDArray, Byte> trainingMap = new HashMap<>();
        CifarLoader cifarLoader = new CifarLoader(true);
        byte[] trainingImageData = IOUtils.toByteArray(cifarLoader.getInputStream());
        int imageLen = HEIGHT * WIDTH * CHANNELS;
        for (int imageIndex = 0; imageIndex < trainingImageData.length; imageIndex += (imageLen + 1)) {
            final byte[] imageByteArray = Arrays.copyOfRange(trainingImageData, imageIndex + 1, imageIndex + (imageLen + 1));
            final double[] imageDoubles = IntStream.range(0, imageByteArray.length).mapToDouble(idx -> imageByteArray[idx]).toArray();
            trainingMap.put(abs(Nd4j.create(imageDoubles)), trainingImageData[imageIndex]);
        }
        log.info("Training data read.");
        return trainingMap;
    }

}
