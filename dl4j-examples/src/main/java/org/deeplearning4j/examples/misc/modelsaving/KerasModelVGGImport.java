package org.deeplearning4j.examples.misc.modelsaving;

/**
 * @author eraly
 */

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;


public class KerasModelVGGImport {

    protected static Logger logger= LoggerFactory.getLogger(KerasModelVGGImport.class);

    //public static final String TRAIN_DIR = "/Users/susaneraly/SKYMIND/kerasImport/blogPost/data/train";
    public static final String MODEL_DIR = "/Users/susaneraly/SKYMIND/kerasImport/VGG16/saved";
    public static final String NUMPY_DIR = "/Users/susaneraly/SKYMIND/kerasImport/VGG16/deep-learning-models/validation/dogs";
    public static final String FILE_PREFIX = "/dog";
    public static final int MIN_INDEX = 0;
    public static final int MAX_INDEX = 399;
    public static final String INPUT_FILE_SUFFIX = ".csv";
    public static final String OUTPUT_FILE_SUFFIX = "Out.csv";

    public static int batchSize = 10;

    //public static final INDArray VGG_MEAN_OFFSET = Nd4j.create(new double[] {103.939,116.779,123.68});


    public static void main(String[] args) throws Exception{

        loadFromNumPy loader = new loadFromNumPy(batchSize);

        logger.info("Loading VGG...");
        ComputationGraph vggNet= KerasModelImport.importKerasModelAndWeights(MODEL_DIR + "/vgg16.json", MODEL_DIR + "/vgg16.h5");

        int incorrect = 0;
        int batchCount = 0;
        while (loader.hasNext()) {
            DataSet imageSet = loader.next();
            INDArray features = imageSet.getFeatures();
            INDArray labels = imageSet.getLabels();
            INDArray[] outputA = vggNet.output(false,features);
            INDArray output = Nd4j.concat(0,outputA);

            INDArray kerasClass = Nd4j.argMax(labels,1);
            INDArray dl4jClass = Nd4j.argMax(output,1);
            INDArray comp = kerasClass.sub(dl4jClass);
            BooleanIndexing.replaceWhere(comp,1, Conditions.notEquals(0));
            if (kerasClass != dl4jClass) incorrect+= comp.sumNumber().intValue();
            System.out.println("Keras argmax:\n"+ kerasClass);
            System.out.println("DL4J argmax:\n" + dl4jClass);

            System.out.println("The max difference difference in predictions is:");
            System.out.println(output.sub(labels).max(1));

            batchCount++;
            System.out.println("==============");
            //if (batchCount == 1) break;
        }
        System.out.println(incorrect+" predictions different from keras");

        /*
        logger.info("Loading images and building the iterator...");
        DataSetIterator imageIterator = getImageIterator();
        */

    }

    public static class loadFromNumPy {

        private int currentIndex = MIN_INDEX;
        private int batchSize = 1;

        public loadFromNumPy(int batchSize) {
            this.batchSize = batchSize;
        }

        public DataSet next() throws IOException {
            INDArray features = Nd4j.readNumpy(NUMPY_DIR + FILE_PREFIX + currentIndex + INPUT_FILE_SUFFIX, " ").reshape(1, 3, 224, 224);
            INDArray labels = Nd4j.readNumpy(NUMPY_DIR + FILE_PREFIX + currentIndex + OUTPUT_FILE_SUFFIX, " ").reshape(1, 1000);
            for (int i=currentIndex+1; i<currentIndex+batchSize;i++) {
                INDArray f = Nd4j.readNumpy(NUMPY_DIR + FILE_PREFIX + i + INPUT_FILE_SUFFIX, " ").reshape(1, 3, 224, 224);
                INDArray l = Nd4j.readNumpy(NUMPY_DIR + FILE_PREFIX + i + OUTPUT_FILE_SUFFIX, " ").reshape(1, 1000);
                features = Nd4j.concat(0,features,f);
                labels = Nd4j.concat(0,labels,l);
            }
            currentIndex = currentIndex+batchSize;
            return new org.nd4j.linalg.dataset.DataSet(features,labels);
        }

        public boolean hasNext() {
            return currentIndex+batchSize-1 < MAX_INDEX;
        }

    }

    /*
    //Currently unused
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

    //Currently unused
    protected static class processForVGG implements DataSetPreProcessor {
        //
        //      resized_image[:,:,0] -= 103.939
        //      resized_image[:,:,1] -= 116.779
        //      resized_image[:,:,2] -= 123.68
        //

        @Override
        public void preProcess(DataSet toPreProcess) {
            INDArray features = toPreProcess.getFeatures();
            Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(features.dup(),VGG_MEAN_OFFSET,features,1));
        }

    }
    */

}
