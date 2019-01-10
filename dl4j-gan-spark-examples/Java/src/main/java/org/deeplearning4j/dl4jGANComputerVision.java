/*
License

Copyright 2019 Hamaad Musharaf Shah

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
*/

package org.deeplearning4j;

import java.io.*;
import java.util.*;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.*;
import org.deeplearning4j.nn.weights.*;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.util.ModelSerializer;

import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class dl4jGANComputerVision {
    private static final Logger log = LoggerFactory.getLogger(dl4jGANComputerVision.class);

    private static final int batchSizePerWorker = 200;
    private static final int batchSizePred = 500;
    private static final int labelIndex = 784;
    private static final int numClasses = 10; // Using Softmax.
    private static final int numClassesDis = 1; // Using Sigmoid.
    private static final int numFeatures = 784;
    private static final int numIterations = 10000;
    private static final int numGenSamples = 10; // This will be a grid so effectively we get {numGenSamples * numGenSamples} samples.
    private static final int numLinesToSkip = 0;
    private static final int numberOfTheBeast = 666;
    private static final int printEvery = 100;
    private static final int saveEvery = 100;
    private static final int tensorDimOneSize = 28;
    private static final int tensorDimTwoSize = 28;
    private static final int tensorDimThreeSize = 1;
    private static final int zSize = 2;

    private static final double dis_learning_rate = 0.002;
    private static final double frozen_learning_rate = 0.0;
    private static final double gen_learning_rate = 0.004;

    private static final String delimiter = ",";
    private static final String resPath = "/Users/samson/Projects/gan_deeplearning4j/outputs/computer_vision/";
    private static final String newLine = "\n";
    private static final String dataSetName = "mnist";

    private static final boolean useGpu = true;

    public static void main(String[] args) throws Exception {
        new dl4jGANComputerVision().GAN(args);
    }

    private void GAN(String[] args) throws Exception {
        for (int i = 0; i < args.length; i++) {
            System.out.println(args[i]);
        }

        if (useGpu) {
            System.out.println("Setting up CUDA environment!");
            Nd4j.setDataType(DataBuffer.Type.FLOAT);

            CudaEnvironment.getInstance().getConfiguration()
                    .allowMultiGPU(true)
                    .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)
                    .allowCrossDeviceAccess(true)
                    .setVerbose(true);
        }

        System.out.println(Nd4j.getBackend());
        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        log.info("Unfrozen discriminator!");
        ComputationGraph dis = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .seed(numberOfTheBeast)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .l2(0.0001)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("dis_input_layer_0")
                .setInputTypes(InputType.convolutionalFlat(tensorDimOneSize, tensorDimTwoSize, tensorDimThreeSize))
                .addLayer("dis_batch_layer_1", new BatchNormalization.Builder()
                        .updater(new RmsProp(dis_learning_rate, 1e-8, 1e-8))
                        .build(), "dis_input_layer_0")
                .addLayer("dis_conv2d_layer_2", new ConvolutionLayer.Builder(5, 5)
                        .stride(2, 2)
                        .updater(new RmsProp(dis_learning_rate, 1e-8, 1e-8))
                        .nIn(1)
                        .nOut(64)
                        .build(), "dis_batch_layer_1")
                .addLayer("dis_maxpool_layer_3", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build(), "dis_conv2d_layer_2")
                .addLayer("dis_conv2d_layer_4", new ConvolutionLayer.Builder(5, 5)
                        .stride(2, 2)
                        .updater(new RmsProp(dis_learning_rate, 1e-8, 1e-8))
                        .nIn(64)
                        .nOut(128)
                        .build(), "dis_maxpool_layer_3")
                .addLayer("dis_maxpool_layer_5", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build(), "dis_conv2d_layer_4")
                .addLayer("dis_dense_layer_6", new DenseLayer.Builder()
                        .updater(new RmsProp(dis_learning_rate, 1e-8, 1e-8))
                        .nOut(1024)
                        .build(), "dis_maxpool_layer_5")
                .addLayer("dis_output_layer_7", new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .updater(new RmsProp(dis_learning_rate, 1e-8, 1e-8))
                        .nOut(numClassesDis)
                        .activation(Activation.SIGMOID)
                        .build(), "dis_dense_layer_6")
                .setOutputs("dis_output_layer_7")
                .build());
        dis.init();
        System.out.println(dis.summary());
        System.out.println(Arrays.toString(dis.output(Nd4j.randn(numGenSamples, numFeatures))[0].shape()));

        log.info("Frozen generator!");
        ComputationGraph gen = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .seed(numberOfTheBeast)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .l2(0.0001)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("gen_input_layer_0")
                .setInputTypes(InputType.feedForward(zSize))
                .addLayer("gen_batch_1", new BatchNormalization.Builder()
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .build(), "gen_input_layer_0")
                .addLayer("gen_dense_layer_2", new DenseLayer.Builder()
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .nOut(1024)
                        .build(), "gen_batch_1")
                .addLayer("gen_dense_layer_3", new DenseLayer.Builder()
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .nOut(7 * 7 * 128)
                        .build(), "gen_dense_layer_2")
                .addLayer("gen_batch_4", new BatchNormalization.Builder()
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .build(), "gen_dense_layer_3")
                .inputPreProcessor("gen_deconv2d_5", new FeedForwardToCnnPreProcessor(7, 7, 128))
                .addLayer("gen_deconv2d_5", new Upsampling2D.Builder(2)
                        .build(), "gen_batch_4")
                .addLayer("gen_conv2d_6", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .padding(2, 2)
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .nIn(128)
                        .nOut(64)
                        .build(), "gen_deconv2d_5")
                .addLayer("gen_deconv2d_7", new Upsampling2D.Builder(2)
                        .build(), "gen_conv2d_6")
                .addLayer("gen_conv2d_8", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .padding(2, 2)
                        .activation(Activation.SIGMOID)
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .nIn(64)
                        .nOut(1)
                        .build(), "gen_deconv2d_7")
                .setOutputs("gen_conv2d_8")
                .build());
        gen.init();
        System.out.println(gen.summary());
        System.out.println(Arrays.toString(gen.output(Nd4j.randn(numGenSamples, zSize))[0].shape()));

        log.info("GAN with unfrozen generator and frozen discriminator!");
        ComputationGraph gan = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .seed(numberOfTheBeast)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .l2(0.0001)
                .graphBuilder()
                .addInputs("gan_input_layer_0")
                .setInputTypes(InputType.feedForward(zSize))
                .addLayer("gan_batch_1", new BatchNormalization.Builder()
                        .updater(new RmsProp(gen_learning_rate, 1e-8, 1e-8))
                        .build(), "gan_input_layer_0")
                .addLayer("gan_dense_layer_2", new DenseLayer.Builder()
                        .updater(new RmsProp(gen_learning_rate, 1e-8, 1e-8))
                        .nOut(1024)
                        .build(), "gan_batch_1")
                .addLayer("gan_dense_layer_3", new DenseLayer.Builder()
                        .updater(new RmsProp(gen_learning_rate, 1e-8, 1e-8))
                        .nOut(7 * 7 * 128)
                        .build(), "gan_dense_layer_2")
                .addLayer("gan_batch_4", new BatchNormalization.Builder()
                        .updater(new RmsProp(gen_learning_rate, 1e-8, 1e-8))
                        .build(), "gan_dense_layer_3")
                .inputPreProcessor("gan_deconv2d_5", new FeedForwardToCnnPreProcessor(7, 7, 128))
                .addLayer("gan_deconv2d_5", new Upsampling2D.Builder(2)
                        .build(), "gan_batch_4")
                .addLayer("gan_conv2d_6", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .padding(2, 2)
                        .updater(new RmsProp(gen_learning_rate, 1e-8, 1e-8))
                        .nIn(128)
                        .nOut(64)
                        .build(), "gan_deconv2d_5")
                .addLayer("gan_deconv2d_7", new Upsampling2D.Builder(2)
                        .build(), "gan_conv2d_6")
                .addLayer("gan_conv2d_8", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .padding(2, 2)
                        .activation(Activation.SIGMOID)
                        .updater(new RmsProp(gen_learning_rate, 1e-8, 1e-8))
                        .nIn(64)
                        .nOut(1)
                        .build(), "gan_deconv2d_7")

                .addLayer("gan_dis_batch_layer_9", new BatchNormalization.Builder()
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .build(), "gan_conv2d_8")
                .addLayer("gan_dis_conv2d_layer_10", new ConvolutionLayer.Builder(5, 5)
                        .stride(2, 2)
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .nIn(1)
                        .nOut(64)
                        .build(), "gan_dis_batch_layer_9")
                .addLayer("gan_dis_maxpool_layer_11", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build(), "gan_dis_conv2d_layer_10")
                .addLayer("gan_dis_conv2d_layer_12", new ConvolutionLayer.Builder(5, 5)
                        .stride(2, 2)
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .nIn(64)
                        .nOut(128)
                        .build(), "gan_dis_maxpool_layer_11")
                .addLayer("gan_dis_maxpool_layer_13", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build(), "gan_dis_conv2d_layer_12")
                .addLayer("gan_dis_dense_layer_14", new DenseLayer.Builder()
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .nOut(1024)
                        .build(), "gan_dis_maxpool_layer_13")
                .addLayer("gan_dis_output_layer_15", new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .nOut(numClassesDis)
                        .activation(Activation.SIGMOID)
                        .build(), "gan_dis_dense_layer_14")
                .setOutputs("gan_dis_output_layer_15")
                .build());
        gan.init();
        System.out.println(gan.summary());
        System.out.println(Arrays.toString(gan.output(Nd4j.randn(numGenSamples, zSize))[0].shape()));

        log.info("Setting up Spark configuration!");
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[4]");
        sparkConf.setAppName("Deeplearning4j on Apache Spark: Generative Adversarial Network!");
        sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        sparkConf.set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        log.info("Setting up Synchronous Parameter Averaging!");
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)
                .averagingFrequency(10)
                .rngSeed(numberOfTheBeast)
                .workerPrefetchNumBatches(0)
                .batchSizePerWorker(batchSizePerWorker)
                .build();

        SparkComputationGraph sparkDis = new SparkComputationGraph(sc, dis, tm);
        SparkComputationGraph sparkGan = new SparkComputationGraph(sc, gan, tm);

        log.info("Computer vision deep learning model with pre-trained layers from the GAN's discriminator!");
        ComputationGraph computerVision = new TransferLearning.GraphBuilder(sparkDis.getNetwork())
                .fineTuneConfiguration(new FineTuneConfiguration.Builder()
                        .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                        .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(1.0)
                        .activation(Activation.TANH)
                        .l2(0.0001)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new RmsProp(dis_learning_rate, 1e-8, 1e-8))
                        .seed(numberOfTheBeast)
                        .build())
                .setFeatureExtractor("dis_dense_layer_6")
                .removeVertexKeepConnections("dis_output_layer_7")
                .addLayer("dis_batch", new BatchNormalization.Builder()
                        .updater(new RmsProp(dis_learning_rate, 1e-8, 1e-8))
                        .nIn(1024)
                        .nOut(1024)
                        .build(), "dis_dense_layer_6")
                .addLayer("dis_output_layer_7", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .updater(new RmsProp(dis_learning_rate, 1e-8, 1e-8))
                        .nIn(1024)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build(), "dis_batch")
                .build();
        System.out.println(computerVision.summary());
        System.out.println(Arrays.toString(computerVision.output(Nd4j.randn(numGenSamples, numFeatures))[0].shape()));

        SparkComputationGraph sparkCV = new SparkComputationGraph(sc, computerVision, tm);

        RecordReader recordReaderTrain = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReaderTrain.initialize(new FileSplit(new ClassPathResource(dataSetName + "_train.csv").getFile()));

        DataSetIterator iterTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSizePerWorker, labelIndex, numClasses);
        List<DataSet> trainDataList = new ArrayList<>();

        JavaRDD<DataSet> trainDataDis, trainDataGen, trainData;

        INDArray grid = Nd4j.linspace(-1.0, 1.0, numGenSamples);
        Collection<INDArray> z = new ArrayList<>();
        log.info("Creating some noise!");
        for (int i = 0; i < numGenSamples; i++) {
            for (int j = 0; j < numGenSamples; j++) {
                z.add(Nd4j.create(new double[]{grid.getDouble(0, i), grid.getDouble(0, j)}));
            }
        }

        int batch_counter = 0;

        DataSet trDataSet;

        RecordReader recordReaderTest = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReaderTest.initialize(new FileSplit(new ClassPathResource(dataSetName + "_test.csv").getFile()));

        DataSetIterator iterTest = new RecordReaderDataSetIterator(recordReaderTest, batchSizePred, labelIndex, numClasses);

        Collection<INDArray> outFeat;

        INDArray out;
        INDArray soften_labels_fake = Nd4j.randn(batchSizePerWorker, 1).muli(0.05);
        INDArray soften_labels_real = Nd4j.randn(batchSizePerWorker, 1).muli(0.05);

        while (iterTrain.hasNext() && batch_counter < numIterations) {
            trainDataList.clear();
            trDataSet = iterTrain.next();

            // This is real data...
            // [Fake, Real].
            trainDataList.add(new DataSet(trDataSet.getFeatures(), Nd4j.ones(batchSizePerWorker, 1).addi(soften_labels_real)));

            // ...and this is fake data.
            // [Fake, Real].
            trainDataList.add(new DataSet(gen.output(Nd4j.rand(batchSizePerWorker, zSize).muli(2.0).subi(1.0))[0], Nd4j.zeros(batchSizePerWorker, 1).addi(soften_labels_fake)));

            // Unfrozen discriminator is trying to figure itself out given a frozen generator.
            log.info("Training discriminator!");
            trainDataDis = sc.parallelize(trainDataList);
            sparkDis.fit(trainDataDis);

            // Update GAN's frozen discriminator with unfrozen discriminator.
            sparkGan.getNetwork().getLayer("gan_dis_batch_layer_9").setParam("gamma", sparkDis.getNetwork().getLayer("dis_batch_layer_1").getParam("gamma"));
            sparkGan.getNetwork().getLayer("gan_dis_batch_layer_9").setParam("beta", sparkDis.getNetwork().getLayer("dis_batch_layer_1").getParam("beta"));
            sparkGan.getNetwork().getLayer("gan_dis_batch_layer_9").setParam("mean", sparkDis.getNetwork().getLayer("dis_batch_layer_1").getParam("mean"));
            sparkGan.getNetwork().getLayer("gan_dis_batch_layer_9").setParam("var", sparkDis.getNetwork().getLayer("dis_batch_layer_1").getParam("var"));

            sparkGan.getNetwork().getLayer("gan_dis_conv2d_layer_10").setParam("W", sparkDis.getNetwork().getLayer("dis_conv2d_layer_2").getParam("W"));
            sparkGan.getNetwork().getLayer("gan_dis_conv2d_layer_10").setParam("b", sparkDis.getNetwork().getLayer("dis_conv2d_layer_2").getParam("b"));

            sparkGan.getNetwork().getLayer("gan_dis_conv2d_layer_12").setParam("W", sparkDis.getNetwork().getLayer("dis_conv2d_layer_4").getParam("W"));
            sparkGan.getNetwork().getLayer("gan_dis_conv2d_layer_12").setParam("b", sparkDis.getNetwork().getLayer("dis_conv2d_layer_4").getParam("b"));

            sparkGan.getNetwork().getLayer("gan_dis_dense_layer_14").setParam("W", sparkDis.getNetwork().getLayer("dis_dense_layer_6").getParam("W"));
            sparkGan.getNetwork().getLayer("gan_dis_dense_layer_14").setParam("b", sparkDis.getNetwork().getLayer("dis_dense_layer_6").getParam("b"));

            sparkGan.getNetwork().getLayer("gan_dis_output_layer_15").setParam("W", sparkDis.getNetwork().getLayer("dis_output_layer_7").getParam("W"));
            sparkGan.getNetwork().getLayer("gan_dis_output_layer_15").setParam("b", sparkDis.getNetwork().getLayer("dis_output_layer_7").getParam("b"));

            trainDataList.clear();
            // Tell the frozen discriminator that all the fake examples are real examples.
            // [Fake, Real].
            trainDataList.add(new DataSet(Nd4j.rand(batchSizePerWorker, zSize).muli(2.0).subi(1.0), Nd4j.ones(batchSizePerWorker, 1)));

            // Unfrozen generator is trying to fool the frozen discriminator.
            log.info("Training generator!");
            trainDataGen = sc.parallelize(trainDataList);
            sparkGan.fit(trainDataGen);

            // Update frozen generator with GAN's unfrozen generator.
            gen.getLayer("gen_batch_1").setParam("gamma", sparkGan.getNetwork().getLayer("gan_batch_1").getParam("gamma"));
            gen.getLayer("gen_batch_1").setParam("beta", sparkGan.getNetwork().getLayer("gan_batch_1").getParam("beta"));
            gen.getLayer("gen_batch_1").setParam("mean", sparkGan.getNetwork().getLayer("gan_batch_1").getParam("mean"));
            gen.getLayer("gen_batch_1").setParam("var", sparkGan.getNetwork().getLayer("gan_batch_1").getParam("var"));

            gen.getLayer("gen_dense_layer_2").setParam("W", sparkGan.getNetwork().getLayer("gan_dense_layer_2").getParam("W"));
            gen.getLayer("gen_dense_layer_2").setParam("b", sparkGan.getNetwork().getLayer("gan_dense_layer_2").getParam("b"));

            gen.getLayer("gen_dense_layer_3").setParam("W", sparkGan.getNetwork().getLayer("gan_dense_layer_3").getParam("W"));
            gen.getLayer("gen_dense_layer_3").setParam("b", sparkGan.getNetwork().getLayer("gan_dense_layer_3").getParam("b"));

            gen.getLayer("gen_batch_4").setParam("gamma", sparkGan.getNetwork().getLayer("gan_batch_4").getParam("gamma"));
            gen.getLayer("gen_batch_4").setParam("beta", sparkGan.getNetwork().getLayer("gan_batch_4").getParam("beta"));
            gen.getLayer("gen_batch_4").setParam("mean", sparkGan.getNetwork().getLayer("gan_batch_4").getParam("mean"));
            gen.getLayer("gen_batch_4").setParam("var", sparkGan.getNetwork().getLayer("gan_batch_4").getParam("var"));

            gen.getLayer("gen_conv2d_6").setParam("W", sparkGan.getNetwork().getLayer("gan_conv2d_6").getParam("W"));
            gen.getLayer("gen_conv2d_6").setParam("b", sparkGan.getNetwork().getLayer("gan_conv2d_6").getParam("b"));

            gen.getLayer("gen_conv2d_8").setParam("W", sparkGan.getNetwork().getLayer("gan_conv2d_8").getParam("W"));
            gen.getLayer("gen_conv2d_8").setParam("b", sparkGan.getNetwork().getLayer("gan_conv2d_8").getParam("b"));

            trainDataList.clear();
            trainDataList.add(trDataSet);

            log.info("Training computer vision model!");
            sparkCV.getNetwork().getLayer("dis_batch_layer_1").setParam("gamma", sparkDis.getNetwork().getLayer("dis_batch_layer_1").getParam("gamma"));
            sparkCV.getNetwork().getLayer("dis_batch_layer_1").setParam("beta", sparkDis.getNetwork().getLayer("dis_batch_layer_1").getParam("beta"));
            sparkCV.getNetwork().getLayer("dis_batch_layer_1").setParam("mean", sparkDis.getNetwork().getLayer("dis_batch_layer_1").getParam("mean"));
            sparkCV.getNetwork().getLayer("dis_batch_layer_1").setParam("var", sparkDis.getNetwork().getLayer("dis_batch_layer_1").getParam("var"));

            sparkCV.getNetwork().getLayer("dis_conv2d_layer_2").setParam("W", sparkDis.getNetwork().getLayer("dis_conv2d_layer_2").getParam("W"));
            sparkCV.getNetwork().getLayer("dis_conv2d_layer_2").setParam("b", sparkDis.getNetwork().getLayer("dis_conv2d_layer_2").getParam("b"));

            sparkCV.getNetwork().getLayer("dis_conv2d_layer_4").setParam("W", sparkDis.getNetwork().getLayer("dis_conv2d_layer_4").getParam("W"));
            sparkCV.getNetwork().getLayer("dis_conv2d_layer_4").setParam("b", sparkDis.getNetwork().getLayer("dis_conv2d_layer_4").getParam("b"));

            sparkCV.getNetwork().getLayer("dis_dense_layer_6").setParam("W", sparkDis.getNetwork().getLayer("dis_dense_layer_6").getParam("W"));
            sparkCV.getNetwork().getLayer("dis_dense_layer_6").setParam("b", sparkDis.getNetwork().getLayer("dis_dense_layer_6").getParam("b"));

            trainData = sc.parallelize(trainDataList);
            sparkCV.fit(trainData);

            batch_counter++;
            log.info("Completed Batch {}!", batch_counter);

            if ((batch_counter % printEvery) == 0) {
                out = gen.output(Nd4j.vstack(z))[0].reshape(numGenSamples * numGenSamples, numFeatures);

                FileWriter fileWriter = new FileWriter(String.format("%s%s_out_%d.csv", resPath, dataSetName, batch_counter));
                for (int i = 0; i < out.shape()[0]; i++) {
                    for (int j = 0; j < out.shape()[1]; j++) {
                        fileWriter.append(String.valueOf(out.getDouble(i, j)));
                        if (j != out.shape()[1] - 1) {
                            fileWriter.append(delimiter);
                        }
                    }
                    if (i != out.shape()[0] - 1) {
                        fileWriter.append(newLine);
                    }
                }
                fileWriter.flush();
                fileWriter.close();
            }

            if ((batch_counter % saveEvery) == 0) {
                log.info("Ensemble of deep learners for estimation of uncertainty!");

                outFeat = new ArrayList<>();
                iterTest.reset();
                while (iterTest.hasNext()) {
                    outFeat.add(sparkCV.getNetwork().output(iterTest.next().getFeatures())[0]);
                }

                INDArray toWrite = Nd4j.vstack(outFeat);
                FileWriter fileWriter = new FileWriter(String.format("%s%s_test_predictions_%d.csv", resPath, dataSetName, batch_counter));
                for (int i = 0; i < toWrite.shape()[0]; i++) {
                    for (int j = 0; j < toWrite.shape()[1]; j++) {
                        fileWriter.append(String.valueOf(toWrite.getDouble(i, j)));
                        if (j != toWrite.shape()[1] - 1) {
                            fileWriter.append(delimiter);
                        }
                    }
                    if (i != toWrite.shape()[0] - 1) {
                        fileWriter.append(newLine);
                    }
                }
                fileWriter.flush();
                fileWriter.close();
            }

            if (!iterTrain.hasNext()) {
                iterTrain.reset();
            }
        }

        log.info("Saving models!");
        ModelSerializer.writeModel(sparkDis.getNetwork(), new File(resPath + dataSetName + "_dis_model.zip"), true);
        ModelSerializer.writeModel(sparkGan.getNetwork(), new File(resPath + dataSetName + "_gan_model.zip"), true);
        ModelSerializer.writeModel(gen, new File(resPath + dataSetName + "_gen_model.zip"), true);
        ModelSerializer.writeModel(sparkCV.getNetwork(), new File(resPath + dataSetName + "_CV_model.zip"), true);

        tm.deleteTempFiles(sc);
    }
}