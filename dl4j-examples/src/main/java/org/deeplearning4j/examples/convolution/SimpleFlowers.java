package org.deeplearning4j.examples.convolution;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.nd4j.jita.conf.CudaEnvironment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

import static org.deeplearning4j.examples.dataExamples.ImagePipelineExample.randNumGen;

/**
 * Created by raver119 on 30.07.16.
 */
public class SimpleFlowers {
    private static final Logger log = LoggerFactory.getLogger(SimpleFlowers.class);

    protected static final  String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    private static class SimplePreprocessor implements DataSetPreProcessor {
        @Override
        public void preProcess(org.nd4j.linalg.dataset.api.DataSet toPreProcess){
            toPreProcess.getFeatureMatrix().divi(255);
        }
    }

    public static void main(String[] args) throws Exception {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);

        CudaEnvironment.getInstance().getConfiguration()
            .allowMultiGPU(false)
            .allowCrossDeviceAccess(true)
            .enableStatisticsGathering(false)
            .setMaximumGridSize(512)
            .setMaximumBlockSize(512)
            .setMaximumDeviceCacheableLength(1024 * 1024 * 1024L)
            .setMaximumDeviceCache(6L * 1024 * 1024 * 1024L)
            .setMaximumHostCache(6L * 1024 * 1024 * 1024L)
            .setMaximumHostCacheableLength(1024 * 1024 * 1024L)
            .setVerbose(false)
            .enableDebug(false);





        int numChannels = 3;
        int outputNum = 5;
        int iterations = 1;
        int seed = 123;

        CNNConfiguration config = new CNNConfiguration("identity",0.001,3,5,2,3,16,0.0001,5,200,200);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        File mainPath = new File("dl4j-examples/src/main/resources/flowers");

        ImageRecordReader recordReader = new ImageRecordReader(config.getImageWidth(), config.getImageHeight(), numChannels, labelMaker);
        ImageRecordReader testRecordReader = new ImageRecordReader(config.getImageWidth(), config.getImageHeight(), numChannels, labelMaker);
        FileSplit fileSplit = new FileSplit(mainPath, allowedExtensions, randNumGen);

        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

        ImageTransform transform = new MultiImageTransform(randNumGen,new ResizeImageTransform(200, 200));

        InputSplit[] inputSplit = fileSplit.sample(pathFilter,80,20);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        recordReader.initialize(trainData, transform);
        testRecordReader.initialize(testData, transform);

        RecordReaderDataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, config.getBatchSize(), -1,outputNum);
        RecordReaderDataSetIterator testDataIter = new RecordReaderDataSetIterator(testRecordReader, config.getBatchSize(), -1, outputNum);

        SimplePreprocessor SP = new SimplePreprocessor();
        dataIter.setPreProcessor(SP);
        testDataIter.setPreProcessor(SP);

        System.out.println("Build model....");
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
            .regularization(true)
            .l2(config.getL2Value())
            .learningRate(config.getLearningRate())
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list();

        int j = 0;
        int initialNOut = 32;
        while (j < config.getNumConvolutionLayers()) {

            builder = builder
                .layer(j,
                    new ConvolutionLayer.Builder(config.getFilterSize(), config.getFilterSize()).nIn(numChannels).stride(1, 1).nOut(initialNOut * (int) Math.pow(2, j)).activation(config.getActivation())
                        .padding(config.getPadding(), config.getPadding()).build())
                .layer(j + 1,
                    new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(config.getKernelSize(), config.getKernelSize()).stride(2, 2)
                        .padding(config.getPadding(), config.getPadding()).build());
            j += 2;
        }

        builder
            .layer(j, new DenseLayer.Builder().activation("relu").nOut(250).build())
            .layer(j + 1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outputNum)
                .activation("softmax").build())
            .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder, config.getImageWidth(), config.getImageHeight(), numChannels);



        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        System.out.println("Train model....");
        model.setListeners(new ScoreIterationListener(10));
/*
        ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
            .workers(2)
            .prefetchBuffer(12)
            .averagingFrequency(1000000)
            .useLegacyAveraging(true)
            .build();
*/
        for (int i = 0; i < config.getNumEpochs(); i++) {
            long time1 = System.currentTimeMillis();
            model.fit(dataIter);
            long time2 = System.currentTimeMillis();
            System.out.println("Epoch execution time: " + (time2 - time1));

            dataIter.reset();
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        while(testDataIter.hasNext()){
            DataSet ds = testDataIter.next();
            INDArray output = model.output(ds.getFeatureMatrix(), false);
            eval.eval(ds.getLabels(), output);
        }
        log.info(eval.stats());
    }

    @Data
    @AllArgsConstructor
    private static class CNNConfiguration {
        private String activation;
        private double l2Value;
        private int numConvolutionLayers;
        private int filterSize;
        private int padding;
        private int kernelSize;
        private int batchSize;
        private double learningRate;
        private int numEpochs;
        private int imageHeight;
        private int imageWidth;
    }
}
