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

package org.deeplearning4j.examples.advanced.modelling.densenet;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.advanced.modelling.densenet.imageUtils.BlurTransform;
import org.deeplearning4j.examples.advanced.modelling.densenet.imageUtils.NoiseTransform;
import org.deeplearning4j.examples.advanced.modelling.densenet.model.DenseNetModel;
import org.deeplearning4j.examples.utils.DownloaderUtility;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.CheckpointListener;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.common.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class DenseNetMain {
    private static final Logger log = LoggerFactory.getLogger(org.deeplearning4j.examples.advanced.modelling.densenet.DenseNetMain.class);

    private static final String MODEL_PATH = FilenameUtils.concat(System.getProperty("user.home") + "/Desktop", "dl4jModel/");

    private static InputSplit trainingData;
    private static InputSplit validationData;

    private static final int height = 227;
    private static final int width = 227;
    private static final int channels = 3;
    private static final int batchSize = 32;
    private static final int outputNum = 4;
    private static final int numEpochs = 1000;
    private static final double splitTrainTest = 0.8;

    public static String dataLocalPath;


    public static void main(String[] args) throws Exception {
        dataLocalPath = DownloaderUtility.ANIMALS.Download();
        File mainPath = new File(dataLocalPath);
        Random random = new Random(1234);

        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, random);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        DataNormalization dataNormalization = new ImagePreProcessingScaler(0, 1);

        BalancedPathFilter pathFilter = new BalancedPathFilter(random, labelMaker, 0);
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        trainingData = inputSplit[0];
        validationData = inputSplit[1];


        log.info("BUILD MODEL");
        ComputationGraph computationGraph = DenseNetModel.getInstance().buildNetwork(432545609, channels, outputNum, width, height);
        setListeners(computationGraph, dataNormalization, labelMaker, 1);

        log.info("TRAIN MODEL");
        trainData(dataNormalization, labelMaker, computationGraph);
    }

    private static ImageTransform getImageTransform() {
        Random random = new Random(1234);
        ImageTransform blur = new BlurTransform(random, 3, 3);
        ImageTransform noise = new NoiseTransform(random, (int) (height * width * 0.1));
        ImageTransform show = new ShowImageTransform("Display");

        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
            new Pair<>(blur, 0.5),
            new Pair<>(noise, 0.5),
            new Pair<>(show, 1.0)
        );
        return new PipelineImageTransform(pipeline, false);
    }

    private static void trainData(DataNormalization dataNormalization, ParentPathLabelGenerator labelMaker, ComputationGraph computationGraph) {
        try {
            ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
            recordReader.initialize(trainingData, getImageTransform());
            DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
            dataNormalization.fit(dataSetIterator);
            dataSetIterator.setPreProcessor(dataNormalization);
            computationGraph.fit(dataSetIterator, numEpochs);
            dataSetIterator.reset();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void setListeners(ComputationGraph computationGraph, DataNormalization dataNormalization, ParentPathLabelGenerator labelMaker, int epochs) {
        try {
            UIServer uiServer = UIServer.getInstance();

            ImageRecordReader trainingRecordReader = new ImageRecordReader(height, width, channels, labelMaker);
            trainingRecordReader.initialize(trainingData, getImageTransform());
            DataSetIterator trainingDataSetIterator = new RecordReaderDataSetIterator(trainingRecordReader, batchSize, 1, outputNum);
            dataNormalization.fit(trainingDataSetIterator);
            trainingDataSetIterator.setPreProcessor(dataNormalization);
            EvaluativeListener evaluativeTrainingListener = new EvaluativeListener(trainingDataSetIterator, epochs, InvocationType.EPOCH_END, new Evaluation(outputNum));

            ImageRecordReader validationRecordReader = new ImageRecordReader(height, width, channels, labelMaker);
            validationRecordReader.initialize(validationData, null);
            DataSetIterator validationDataSetIterator = new RecordReaderDataSetIterator(validationRecordReader, batchSize, 1, outputNum);
            dataNormalization.fit(validationDataSetIterator);
            validationDataSetIterator.setPreProcessor(dataNormalization);
            EvaluativeListener evaluativeValidationListener = new EvaluativeListener(validationDataSetIterator, epochs, InvocationType.EPOCH_END, new Evaluation(outputNum));

            StatsStorage statsStorage = new InMemoryStatsStorage();
            StatsListener statsListener = new StatsListener(statsStorage);

            ScoreIterationListener scoreIterationListener = new ScoreIterationListener(1);

            File model = new File(MODEL_PATH);
            boolean newRun = false;
            if (!model.exists()) {
                newRun = model.mkdir();
            }
            CheckpointListener checkpointListener = new CheckpointListener.Builder(model)
                .keepAll()
                .deleteExisting(!newRun)
                .saveEveryNEpochs(epochs)
                .build();

            uiServer.attach(statsStorage);
            computationGraph.setListeners(evaluativeValidationListener, statsListener, scoreIterationListener, checkpointListener);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
