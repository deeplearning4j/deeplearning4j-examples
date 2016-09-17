package org.deeplearning4j.examples.convolution;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.text.DecimalFormat;
import java.util.*;

/**
 * @author AlexDBlack
 */
@Slf4j
public class TinyExample {
    private static final String DATA_ROOT_DIR = System.getProperty("os.name").toLowerCase().contains("windows") ? "C:\\Users\\raver\\TinyImageNet\\" : "~/TinyImageNet/";
    private static final String LABEL_ID_FILE = DATA_ROOT_DIR + "wnids.txt";
    private static final String LABEL_NAME_FILE = DATA_ROOT_DIR + "words.txt";
    private static final String TRAIN_DIR = DATA_ROOT_DIR + "train/";
    private static final String VALIDATION_DIR = DATA_ROOT_DIR + "val/";
    private static final String VALIDATION_ANNOTATION_FILE = DATA_ROOT_DIR + "val/val_annotations.txt";
    private static final String[] allowedExtensions = new String[]{"JPEG"};

  //  private static final String OUTPUT_DIR = "~/TinyImageNet/";



    public static void main(String[] args) throws Exception {
        int inWidth = 64;
        int inHeight = 64;
        int inDepth = 3;
        int batchSize = 32;
        String afn = "leakyrelu";
        int nOut = 200;

        int nEpochs = 10;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .seed(12345)
            .regularization(true).l2(1e-4)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .learningRate(1e-2)
            .list()
            .layer(0, new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(0,0).nOut(64).build())
            .layer(1, new BatchNormalization.Builder().build())
            .layer(2, new ActivationLayer.Builder().activation(afn).build())
            .layer(3, new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(0,0).nOut(64).build())
            .layer(4, new BatchNormalization.Builder().build())
            .layer(5, new ActivationLayer.Builder().activation(afn).build())
            .layer(6, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2).stride(2,2).padding(0,0).build())
            .layer(7, new ConvolutionLayer.Builder().kernelSize(2,2).stride(2,2).padding(0,0).nOut(96).build())
            .layer(8, new BatchNormalization.Builder().build())
            .layer(9, new ActivationLayer.Builder().activation(afn).build())
            .layer(10, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3,3).stride(2,2).padding(0,0).build())
            .layer(11, new ConvolutionLayer.Builder().kernelSize(3,3).stride(2,2).padding(0,0).nOut(96).build())
            .layer(12, new BatchNormalization.Builder().build())
            .layer(13, new ActivationLayer.Builder().activation(afn).build())
            .layer(14, new DenseLayer.Builder().activation(afn).nOut(200).build())
            .layer(15, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation("softmax").nIn(200).nOut(nOut).build())
            .pretrain(false).backprop(true)
            .setInputType(InputType.convolutional(inHeight, inWidth, inDepth))
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        System.out.println("Number of parameters by layer:");
        Layer[] layers = net.getLayers();
        for( int i=0; i<layers.length; i++ ){
            System.out.println(i + "\t" + layers[i].numParams());
        }
        System.out.println("Total num params: " + net.numParams());



        Random r = new Random(12345);
        FileSplit trainSplit = new FileSplit(new File(TRAIN_DIR), allowedExtensions, r);
        FileSplit valSplit = new FileSplit(new File(VALIDATION_DIR), allowedExtensions, r);

        List<String> labelIDs = loadLabels(LABEL_ID_FILE);
        List<String> labelNames = loadLabelNames(labelIDs, LABEL_NAME_FILE);

        ImageRecordReader trainReader = new ImageRecordReader(inHeight,inWidth,inDepth,new TrainLabelGenerator(labelIDs));
        trainReader.initialize(trainSplit);
        trainReader.setLabels(labelIDs);
        ImageRecordReader testReader = new ImageRecordReader(inHeight, inWidth, inDepth, new ValidationLabelGenerator(labelIDs, VALIDATION_ANNOTATION_FILE));
        testReader.initialize(valSplit);
        testReader.setLabels(labelIDs);

        DataSetIterator trainData = new RecordReaderDataSetIterator(trainReader, batchSize, 1, nOut);
        DataSetIterator testData = new RecordReaderDataSetIterator(testReader, batchSize, 1, nOut);
        trainData.setPreProcessor(new ImagePreProcessingScaler(-1,1,8));
        testData.setPreProcessor(new ImagePreProcessingScaler(-1,1,8));


//        net.setListeners(new PerformanceListener(1,true), new HistogramIterationListener(1));
        net.setListeners(new PerformanceListener(10,true));

        for( int i=0; i<nEpochs; i++ ){
            long start = System.currentTimeMillis();
            net.fit(trainData);
            long end = System.currentTimeMillis();
            log.info("EPOCH TIME: {} ms", (end-start));
        //    postEpoch(net, testData, labelIDs, i);
        }


        //For debugging
//        List<DataSet> temp = new ArrayList<>();
//        for( int i=0; i<3; i++ ){
//            temp.add(testData.next());
//        }
//        for( int i=0; i<2; i++ ){
//            long start = System.currentTimeMillis();
//            for( int j=0; j<3; j++ ) {
//                net.fit(trainData.next());
//            }
//            long end = System.currentTimeMillis();
//            log.info("EPOCH TIME: {} ms", (end-start));
//            postEpoch(net, new ListDataSetIterator(temp,32), labelNames, i);
//        }
    }

    private static void postEpoch(MultiLayerNetwork net, DataSetIterator testData, List<String> labels, int epoch) throws IOException {
        String outputDir =  "epoch_" + epoch + "/";
        File f = new File(outputDir);
        if(f.exists()){
            f.delete();
        }
        f.mkdir();
        ModelSerializer.writeModel(net,new File(outputDir + "model.zip"), true);
        Evaluation e = net.evaluate(testData, labels);
        String evalPath = outputDir + "eval.txt";
        FileUtils.writeStringToFile(new File(evalPath), e.stats());
        DecimalFormat df = new DecimalFormat("#.####");
        System.out.println("*** At epoch " + epoch + " : accuracy = " + df.format(e.accuracy()) + ", f1 = " + df.format(e.f1()));
    }

    private static List<String> loadLabels(String path) throws IOException {
        List<String> lines = FileUtils.readLines(new File(path));
        List<String> out = new ArrayList<>(200);
        for(String s : lines){
            if(s.length() > 0){
                out.add(s);
            }
        }
        return out;
    }

    private static List<String> loadLabelNames(List<String> labelIDs, String path ) throws IOException {
        Map<String,String> indexesToNames = new HashMap<>();
        List<String> lines = FileUtils.readLines(new File(path));
        for(String s : lines){
            String[] split = s.split("\t");
            indexesToNames.put(split[0],split[1]);
        }

        List<String> out = new ArrayList<>(labelIDs.size());
        for(String s : labelIDs){
            out.add(indexesToNames.get(s));
        }
        return out;
    }

    private static class TrainLabelGenerator implements PathLabelGenerator {
        private Map<String,Integer> labelIdxs;

        public TrainLabelGenerator(List<String> labels) throws IOException {
            labelIdxs = new HashMap<>();
            int i=0;
            for(String s : labels){
                labelIdxs.put(s, i++);
            }
        }

        @Override
        public Writable getLabelForPath(String path) {
            String dirName = FilenameUtils.getBaseName(new File(path).getParentFile().getParent());
            return new Text(dirName);
        }

        @Override
        public Writable getLabelForPath(URI uri) {
            return getLabelForPath(uri.toString());
        }
    }

    private static class ValidationLabelGenerator implements PathLabelGenerator {
        private Map<String,Integer> labelIdxs;
        private Map<String,String> filenameToIndex;

        private ValidationLabelGenerator(List<String> labels, String annotationsFile) throws IOException {
            labelIdxs = new HashMap<>();
            int i=0;
            for(String s : labels){
                labelIdxs.put(s, i++);
            }
            this.filenameToIndex = loadValidationSetLabels(annotationsFile);
        }

        @Override
        public Writable getLabelForPath(String path) {
            File f = new File(path);
            String filename = f.getName();
            return new Text(filenameToIndex.get(filename));
        }

        @Override
        public Writable getLabelForPath(URI uri) {
            return getLabelForPath(uri.toString());
        }
    }

    private static Map<String,String> loadValidationSetLabels(String path) throws IOException {
        Map<String,String> validation = new HashMap<>();
        List<String> lines = FileUtils.readLines(new File(path));
        for(String s : lines){
            String[] split = s.split("\t");
            validation.put(split[0],split[1]);
        }
        return validation;
    }
}
