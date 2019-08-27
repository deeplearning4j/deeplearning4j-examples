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

package org.deeplearning4j.examples.multigpu.vgg16;

import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.fetchers.TinyImageNetFetcher;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold.AdaptiveThresholdAlgorithm;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.slf4j.Logger;


/**
 * This is modified version of original LenetMnistExample, made compatible with multi-gpu environment
 * for the TinyImageNet dataset and VGG16.
 *
 * @author Justin Long (crockpotveggies)
 */
public class MultiGpuVGG16TinyImageNetExample {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(MultiGpuVGG16TinyImageNetExample.class);

    // for GPU you usually want to have higher batchSize
    public static int batchSize = 16;
    public static int nEpochs = 10;
    public static int seed = 123;

    public static void main(String[] args) throws Exception {

        VGG16 zooModel = VGG16.builder()
            .numClasses(TinyImageNetFetcher.NUM_LABELS)
            .seed(seed)
            .inputShape(new int[]{TinyImageNetFetcher.INPUT_CHANNELS, 224, 224})
            .updater(new AdaDelta())
            .workspaceMode(WorkspaceMode.ENABLED)
            .cacheMode(CacheMode.DEVICE)
            .build();
        ComputationGraph vgg16 = zooModel.init();
        vgg16.setListeners(new PerformanceListener(1, true));

        log.info("Loading data....");
        DataSetIterator trainIter = new TinyImageNetDataSetIterator(batchSize, new int[]{224,224}, DataSetType.TRAIN);
        DataSetIterator testIter = new TinyImageNetDataSetIterator(batchSize, new int[]{224,224}, DataSetType.TEST);

        log.info("Fitting normalizer...");
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        // ParallelWrapper will take care of load balancing between GPUs.
        ParallelWrapper wrapper = new ParallelWrapper.Builder(vgg16)
            // DataSets prefetching options. Set this value with respect to number of actual devices
            .prefetchBuffer(24)

            // set number of workers equal to number of available devices
            .workers(Nd4j.getAffinityManager().getNumberOfDevices())

            // use gradient sharing, a more effective distributed training method
            .trainingMode(ParallelWrapper.TrainingMode.SHARED_GRADIENTS)

            .thresholdAlgorithm(new AdaptiveThresholdAlgorithm())

            .build();

        log.info("Train model....");
        long timeX = System.currentTimeMillis();

        for( int i=0; i<nEpochs; i++ ) {
            long time = System.currentTimeMillis();
            wrapper.fit(trainIter);
            time = System.currentTimeMillis() - time;
            log.info("*** Completed epoch {}, time: {} ***", i, time);
        }
        long timeY = System.currentTimeMillis();

        log.info("*** Training complete, time: {} ***", (timeY - timeX));

        log.info("Evaluate model....");
        Evaluation eval = vgg16.evaluate(testIter);
        log.info(eval.stats());

        log.info("****************Example finished********************");
    }
}
