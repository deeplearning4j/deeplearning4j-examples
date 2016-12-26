package org.deeplearning4j.examples.misc.modelsaving;

/**
 * Created by susaneraly on 12/20/16.
 */

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.datavec.api.split.InputSplit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Random;


public class KerasModelImport {

    private static Logger logger= LoggerFactory.getLogger(KerasModelImport.class);

    private static final String TRAIN_DIR = "/Users/susaneraly/SKYMIND/kerasImport/blogPost/data/train";

    //Images are of format given by allowedExtension -
    protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    protected static final long seed = 12345;

    public static final Random rng = new Random(seed);

    protected static int height = 224;
    protected static int width = 224;
    protected static int channels = 3;

    public static int numExamples = 1000;
    public static int batchSize = 10;
    public static int numLabels = 2;
    public static final INDArray VGG_MEAN_OFFSET = Nd4j.create(new double[] {103.939,116.779,123.68});


    public static void main(String[] args) throws Exception{

        String baseDir="/Users/susaneraly/SKYMIND/kerasImport/VGG16/fromthewild";
        logger.info("load VGG...");
        ComputationGraph vggNet= Model.importFunctionalApiModel(baseDir + "/vgg16.json", baseDir + "/vgg16.h5");
        System.out.println("\n\n");
        logger.info(" loaded vgg model success!");

        DataSetIterator imageIterator = getImageIterator();

        Evaluation eval = vggNet.evaluate(imageIterator);
        logger.info(eval.stats(true));

    }

    public static DataSetIterator getImageIterator() throws Exception{

        File mainPath = new File(TRAIN_DIR);
        FileSplit fileSplit = new FileSplit(mainPath, allowedExtensions);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize);

        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        InputSplit[] inputSplit = fileSplit.sample(pathFilter,100,0);

        InputSplit trainData = inputSplit[0];
        recordReader.initialize(trainData,null);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);

        DataSetPreProcessor myPreProcessor = new processForVGG();
        dataIter.setPreProcessor(myPreProcessor);

        return dataIter;
    }

    protected static class processForVGG implements DataSetPreProcessor {
        /*
                resized_image[:,:,0] -= 103.939
                resized_image[:,:,1] -= 116.779
                resized_image[:,:,2] -= 123.68
         */

        @Override
        public void preProcess(DataSet toPreProcess) {
            INDArray features = toPreProcess.getFeatures();
            Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(features.dup(),VGG_MEAN_OFFSET,features,1));
        }

    }
}
