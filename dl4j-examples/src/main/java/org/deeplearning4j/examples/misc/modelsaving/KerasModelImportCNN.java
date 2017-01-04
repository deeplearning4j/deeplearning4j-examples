package org.deeplearning4j.examples.misc.modelsaving;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class KerasModelImportCNN {

    private static final Logger log = LoggerFactory.getLogger(KerasModelImportCNN.class);

    //public static final String MODEL_DIR = "/Users/susaneraly/SKYMIND/kerasImport/examples/";
    //public static final String MODEL = "mnist_model";
    //public static final String NUMPY_DIR = "/Users/susaneraly/SKYMIND/kerasImport/examples/testImages";

    public static final String MODEL_DIR = "/Users/susaneraly/SKYMIND/kerasImport/tests/";
    public static final String MODEL = "toy_cnn";
    public static final String NUMPY_DIR = "/Users/susaneraly/SKYMIND/kerasImport/tests/testImages";

    public static final String FILE_PREFIX = "/val";
    public static final int MIN_INDEX = 0;
    public static final int MAX_INDEX = 9;
    public static final String INPUT_FILE_SUFFIX = ".txt";
    public static final String OUTPUT_FILE_SUFFIX = "Out.txt";

    public static final int IMG_CH = 1;
    public static final int IMG_H = 6;
    public static final int IMG_W = 6;
    public static final int N_CLASSES = 2;
    //public static final int IMG_H = 28;
    //public static final int IMG_W = 28;
    //public static final int N_CLASSES = 10;

    public static void main(String[] args) throws Exception {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        int batchSize = 1;
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(MODEL_DIR+MODEL+".json",MODEL_DIR+MODEL+".h5");
        loadFromNumPy loader = new loadFromNumPy(batchSize);

        int incorrect = 0;
        int batchCount = 0;
        while (loader.hasNext()) {
            //if (batchCount == 0) break;
            DataSet imageSet = loader.next();
            INDArray features = imageSet.getFeatures();
            INDArray labels = imageSet.getLabels();
            INDArray output = model.output(features,false);

            INDArray kerasClass = Nd4j.argMax(labels,1);
            INDArray dl4jClass = Nd4j.argMax(output,1);
            INDArray comp = kerasClass.sub(dl4jClass);
            BooleanIndexing.replaceWhere(comp,1, Conditions.notEquals(0));
            if (kerasClass != dl4jClass) incorrect+= comp.sumNumber().intValue();

            System.out.println("Keras argmax:\n"+ kerasClass);
            System.out.println("DL4J argmax:\n" + dl4jClass);

            batchCount++;
            System.out.println("==============");
        }
        System.out.println(incorrect+" predictions different from keras");


    }

    public static class loadFromNumPy {

        private int currentIndex = MIN_INDEX;
        private int batchSize = 1;

        public loadFromNumPy(int batchSize) {
            this.batchSize = batchSize;
        }

        public DataSet next() throws IOException {
            INDArray features = Nd4j.readNumpy(NUMPY_DIR + FILE_PREFIX + currentIndex + INPUT_FILE_SUFFIX, " ").reshape(1, IMG_CH, IMG_H, IMG_W);
            INDArray labels = Nd4j.readNumpy(NUMPY_DIR + FILE_PREFIX + currentIndex + OUTPUT_FILE_SUFFIX, " ").reshape(1, N_CLASSES);
            for (int i=currentIndex+1; i<currentIndex+batchSize;i++) {
                INDArray f = Nd4j.readNumpy(NUMPY_DIR + FILE_PREFIX + i + INPUT_FILE_SUFFIX, " ").reshape(1, IMG_CH, IMG_H, IMG_W);
                INDArray l = Nd4j.readNumpy(NUMPY_DIR + FILE_PREFIX + i + OUTPUT_FILE_SUFFIX, " ").reshape(1, N_CLASSES);
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
}
