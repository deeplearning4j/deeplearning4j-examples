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

package org.deeplearning4j.examples.dataexamples;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

/**
 * This code example is featured in this youtube video
 * https://www.youtube.com/watch?v=ECA6y6ahH5E
 *
 * This differs slightly from the Video Example,
 * The Video example had the data already downloaded
 * This example includes code that downloads the data
 *
 * Data is downloaded from
 * wget http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
 * followed by tar xzvf mnist_png.tar.gz
 *
 * This examples builds on the MnistImagePipelineExample
 * by adding a Neural Net
 */
public class MnistImagePipelineExampleAddNeuralNet {
  private static Logger log = LoggerFactory.getLogger(MnistImagePipelineExampleAddNeuralNet.class);

  /** Data URL for downloading */
  public static final String DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

  /** Location to save and extract the training/testing data */
  public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_Mnist/");

  public static void main(String[] args) throws Exception {
    // image information
    // 28 * 28 grayscale
    // grayscale implies single channel
    int height = 28;
    int width = 28;
    int channels = 1;
    int rngseed = 123;
    Random randNumGen = new Random(rngseed);
    int batchSize = 128;
    int outputNum = 10;
    int numEpochs = 1;

    /*
    This class downloadData() downloads the data
    stores the data in java's tmpdir 15MB download compressed
    It will take 158MB of space when uncompressed
    The data can be downloaded manually here
    http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
    */
    MnistImagePipelineExample.downloadData();

    // Define the File Paths
    File trainData = new File(DATA_PATH + "/mnist_png/training");
    File testData = new File(DATA_PATH + "/mnist_png/testing");

    // Define the FileSplit(PATH, ALLOWED FORMATS,random)
    FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
    FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

    // Extract the parent path as the image label
    ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

    ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

    // Initialize the record reader
    // add a listener, to extract the name
    recordReader.initialize(train);
    //recordReader.setListeners(new LogRecordListener());

    // DataSet Iterator
    DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

    // Scale pixel values to 0-1
    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    scaler.fit(dataIter);
    dataIter.setPreProcessor(scaler);

    // Build Our Neural Network
    log.info("BUILD MODEL");
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(rngseed)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(new Nesterovs(0.006, 0.9))
        .l2(1e-4)
        .list()
        .layer(0, new DenseLayer.Builder()
            .nIn(height * width)
            .nOut(100)
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .build())
        .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nIn(100)
            .nOut(outputNum)
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.XAVIER)
            .build())
        .setInputType(InputType.convolutional(height, width, channels))
        .build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);

    // The Score iteration Listener will log
    // output to show how well the network is training
    model.setListeners(new ScoreIterationListener(10));

    log.info("TRAIN MODEL");
    for (int i = 0; i < numEpochs; i++) {
      model.fit(dataIter);
    }

    log.info("EVALUATE MODEL");
    recordReader.reset();

    // The model trained on the training dataset split
    // now that it has trained we evaluate against the
    // test data of images the network has not seen

    recordReader.initialize(test);
    DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
    scaler.fit(testIter);
    testIter.setPreProcessor(scaler);

    /*
    log the order of the labels for later use
    In previous versions the label order was consistent, but random
    In current verions label order is lexicographic
    preserving the RecordReader Labels order is no
    longer needed left in for demonstration
    purposes
    */
    log.info(recordReader.getLabels().toString());

    // Create Eval object with 10 possible classes
    Evaluation eval = new Evaluation(outputNum);

    // Evaluate the network
    while (testIter.hasNext()) {
      DataSet next = testIter.next();
      INDArray output = model.output(next.getFeatures());
      // Compare the Feature Matrix from the model
      // with the labels from the RecordReader
      eval.eval(next.getLabels(), output);
    }

    log.info(eval.stats());
  }

}
