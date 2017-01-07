package org.deeplearning4j.examples.misc.modelsaving;

import org.datavec.image.loader.ImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.string.NDArrayStrings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * @author susaneraly
 */
public class VGGImportImageNet {

    protected static Logger logger= LoggerFactory.getLogger(VGGImportImageNet.class);

    public static final String MODEL_DIR = "/Users/susaneraly/SKYMIND/kerasImport/VGG16/saved";
    public static final String IMAGE_DIR = "/Users/susaneraly/SKYMIND/kerasImport/tests/imageNet/imagesMakeShift";
    public static final String FILE_PREFIX = "/val";
    public static final int MIN_INDEX = 0;
    public static final int MAX_INDEX = 4;
    public static final String INPUT_FILE_SUFFIX = ".jpg";
    public static final String OUTPUT_FILE_SUFFIX = "Out.txt";

    public static int batchSize = 1;

    //This is for BGR
    public static final INDArray VGG_MEAN_OFFSET = Nd4j.create(new double[] {123.68,116.779,103.939});

    public static void main(String[] args) throws Exception{

        getImageIterator loader = new getImageIterator(batchSize);

        logger.info("Loading VGG...");
        ComputationGraph vggNet= KerasModelImport.importKerasModelAndWeights(MODEL_DIR + "/vgg16.json", MODEL_DIR + "/vgg16.h5");

        int incorrect = 0;
        int batchCount = 0;
        while (loader.hasNext()) {

            org.nd4j.linalg.dataset.api.DataSet imageSet = loader.next();
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
            System.out.println("Keras probabilities:\n"+new NDArrayStrings(6).format(labels.max(1)));
            System.out.println("DL4J argmax:\n" + dl4jClass);
            System.out.println("DL4J probabilities:\n"+new NDArrayStrings(6).format(output.max(1)));

            INDArray absDifference = Transforms.abs(output.sub(labels)).max(1);
            INDArray percDifference = Transforms.abs(output.sub(labels).div(labels)).max(1);
            System.out.println("The absolute max difference difference in predictions is:");
            System.out.println(new NDArrayStrings(9).format(absDifference));
            System.out.println("The max percentage difference in prediction is:");
            System.out.println(new NDArrayStrings(9).format(percDifference));

            System.out.println("==============");
        }

    }

    //Currently unused
    public static class getImageIterator {

        public ImageLoader loader;
        public int cursor = 0;
        //public File dir = new File(IMAGE_DIR);
        //public Collection imageFiles = FileUtils.listFiles(dir, new WildcardFileFilter("*.jpg"), null);
        //File [] imageFilesArr = (File[]) imageFiles.toArray(new File [imageFiles.size()]);

        public int totalNum = MAX_INDEX+1;
        public int batchSize = 1;

       public getImageIterator (int batchSize) {
           this.loader = new ImageLoader(224, 224, 3);
           this.batchSize = batchSize;
       }

       public DataSet next() throws IOException {
           INDArray features = loader.toBgr(new File(IMAGE_DIR + FILE_PREFIX + cursor + INPUT_FILE_SUFFIX)).reshape(1,3,224,224);
           INDArray labels = Nd4j.readNumpy(IMAGE_DIR + FILE_PREFIX + cursor + OUTPUT_FILE_SUFFIX, " ").reshape(1, 1000);
           preProcess(features);
           cursor++;
           for (int i=cursor+1;i<cursor+batchSize && i<totalNum; i++) {
               INDArray im = loader.toBgr(new File(IMAGE_DIR + FILE_PREFIX + cursor + INPUT_FILE_SUFFIX));
               preProcess(im);
               INDArray l = Nd4j.readNumpy(IMAGE_DIR + FILE_PREFIX + cursor + OUTPUT_FILE_SUFFIX);
               features = Nd4j.concat(0,features,im);
               labels = Nd4j.concat(0,labels,l);
               cursor ++;
           }
           return new DataSet(features,labels);
       }

        public boolean hasNext() {
            return cursor <= MAX_INDEX;
        }

    }

    public static void preProcess(INDArray features) {
        Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(features.dup(),VGG_MEAN_OFFSET,features,1));
        features.reshape(1,3,224,224);
    }

}
